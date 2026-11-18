
"""DopplerDodge + joint UAV-aware planning.

Each UAV independently detects HIT/MISS using DopplerDodge.
HIT UAVs receive random movement targets.
SLSQP jointly adjusts all UAV movements to maintain UAV-UAV safety.

The planner NEVER receives the projectile's position or velocity.

Run:
    python -m baselines.dopplerdodge_coordinated --scenario all --seeds 20
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
from scipy.optimize import minimize

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import SAMPLE_RATE
from experiments.online_avoidance import (
    UAV_POSITIONS,
    UAV_RADIUS,
    PROJECTILE_RADIUS,
    PROJECTILE_SAFETY_DISTANCE,
    UAV_SAFETY_DISTANCE,
    DURATION,
    INIT_WINDOW,
    MANEUVER_DURATION,
    MAX_UAV_SPEED,
)
from baselines.dopplerdodge import (
    SCENARIOS,
    StreamingDopplerDodge,
    RandomDodgeController,
    BaselineResult,
    calibrate_threshold,
    is_threatened,
    minimum_distances,
    dodge_echo_from_history,
    print_summary,
)


@dataclass
class CoordinatedPlan:
    displacements: np.ndarray
    success: bool
    min_uav_separation: float


# ------------------------------------------------------------
# 1. UAV-UAV safety calculations
# ------------------------------------------------------------

def min_path_separation(origins, displacements):
    """Minimum separation during simultaneous linear movements."""
    closest = float("inf")

    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            relative_start = origins[i] - origins[j]
            relative_motion = displacements[i] - displacements[j]

            motion_squared = float(relative_motion @ relative_motion)

            if motion_squared > 1e-12:
                fraction = float(np.clip(
                    -relative_start @ relative_motion / motion_squared,
                    0.0,
                    1.0,
                ))
            else:
                fraction = 0.0

            distance = np.linalg.norm(
                relative_start + fraction * relative_motion
            )
            closest = min(closest, float(distance))

    return closest


def pairwise_path_margins(origins, displacements):
    """Safety margin for every UAV pair throughout the maneuver."""
    margins = []

    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            relative_start = origins[i] - origins[j]
            relative_motion = displacements[i] - displacements[j]

            motion_squared = float(relative_motion @ relative_motion)

            if motion_squared > 1e-12:
                fraction = float(np.clip(
                    -relative_start @ relative_motion / motion_squared,
                    0.0,
                    1.0,
                ))
            else:
                fraction = 0.0

            distance = np.linalg.norm(
                relative_start + fraction * relative_motion
            )

            margins.append(distance - UAV_SAFETY_DISTANCE)

    return np.asarray(margins)


# ------------------------------------------------------------
# 2. Coordinated avoidance planner
# ------------------------------------------------------------

def plan_coordinated(
    hit_predictions,
    positions,
    rng,
    min_hit_movement=0.40,
    num_starts=12,
    max_iterations=300,
):
    """Jointly plan UAV movements using only HIT/MISS and UAV positions.

    HIT UAVs receive random displacement targets.
    SLSQP adjusts all UAV movements to maintain safe separation.

    MISS UAVs may move if needed to accommodate another UAV.
    """
    origins = np.asarray(positions, dtype=float)
    hits = np.flatnonzero(hit_predictions)
    n = len(origins)

    max_displacement = MAX_UAV_SPEED * MANEUVER_DURATION
    stationary = np.zeros_like(origins)

    if not 0 < min_hit_movement <= max_displacement:
        raise ValueError("Invalid minimum HIT movement")

    # An existing UAV-UAV safety violation cannot be repaired
    # over a planning horizon that includes the starting instant.
    initial_separation = min_path_separation(origins, stationary)

    if initial_separation < UAV_SAFETY_DISTANCE:
        return CoordinatedPlan(
            stationary, False, initial_separation
        )

    if len(hits) == 0:
        return CoordinatedPlan(
            stationary, True, initial_separation
        )

    # Random movement targets for UAVs predicting HIT.
    # No projectile trajectory information is used.
    preferred = np.zeros_like(origins)

    for i in hits:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        preferred[i] = max_displacement * direction

    # Prefer remaining near the original random directions,
    # while penalizing unnecessary movement.
    def objective(flat):
        movements = flat.reshape(n, 3)

        deviation_cost = np.sum(
            (movements - preferred) ** 2
        )

        movement_cost = 0.05 * np.sum(
            np.sqrt(
                np.sum(movements ** 2, axis=1) + 1e-8
            )
        )

        return float(deviation_cost + movement_cost)

    # All UAVs must respect their movement budget.
    def speed_constraints(flat):
        movements = flat.reshape(n, 3)
        return max_displacement - np.linalg.norm(
            movements, axis=1
        )

    # HIT UAVs must actually move a minimum distance.
    def hit_constraints(flat):
        movements = flat.reshape(n, 3)
        return (
            np.linalg.norm(movements[hits], axis=1)
            - min_hit_movement
        )

    # All UAV pairs must maintain safe separation
    # at every point along their simultaneous paths.
    def separation_constraints(flat):
        movements = flat.reshape(n, 3)
        return pairwise_path_margins(origins, movements)

    constraints = [
        {"type": "ineq", "fun": speed_constraints},
        {"type": "ineq", "fun": hit_constraints},
        {"type": "ineq", "fun": separation_constraints},
    ]

    # Multiple starts improve optimization robustness.
    guesses = [preferred]

    for _ in range(num_starts - 1):
        guess = preferred.copy()

        for i in hits:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            guess[i] = max_displacement * direction

        guesses.append(guess)

    best_plan = None
    best_cost = float("inf")

    for guess in guesses:
        solution = minimize(
            objective,
            guess.ravel(),
            method="SLSQP",
            constraints=constraints,
            options={
                "maxiter": max_iterations,
                "ftol": 1e-7,
                "disp": False,
            },
        )

        movements = solution.x.reshape(n, 3)

        if not np.all(np.isfinite(movements)):
            continue

        # Explicitly validate the returned solution.
        if (
            np.min(speed_constraints(solution.x)) < -1e-5
            or np.min(hit_constraints(solution.x)) < -1e-5
            or np.min(separation_constraints(solution.x)) < -1e-5
        ):
            continue

        cost = objective(solution.x)

        if cost < best_cost:
            best_plan = movements.copy()
            best_cost = cost

    if best_plan is None:
        return CoordinatedPlan(
            stationary, False, initial_separation
        )

    return CoordinatedPlan(
        displacements=best_plan,
        success=True,
        min_uav_separation=min_path_separation(
            origins, best_plan
        ),
    )


# ------------------------------------------------------------
# 3. PyBullet simulation
# ------------------------------------------------------------

def run_coordinated_baseline(
    scenario,
    threshold,
    seed=0,
    gui=False,
    min_hit_movement=0.40,
    *,
    projectile_state=None,
    uav_positions=None,
    duration=None,
):
    """Evaluate DopplerDodge with joint UAV-aware planning."""
    state = np.asarray(SCENARIOS[scenario] if projectile_state is None else projectile_state,dtype=float).copy()
    positions = np.asarray(UAV_POSITIONS if uav_positions is None else uav_positions,dtype=float).copy()
    duration = DURATION if duration is None else float(duration)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("projectile_state must be a finite six-element vector")
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions)<1 or not np.all(np.isfinite(positions)):
        raise ValueError("uav_positions must be a finite nonempty (N,3) array")
    if not np.isfinite(duration) or duration<INIT_WINDOW:
        raise ValueError("duration must cover the observation window")
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError("threshold must be finite and nonnegative")
    rng = np.random.default_rng(seed)

    world = SimulationWorld(gui=gui, gravity=False)

    detectors = [
        StreamingDopplerDodge(threshold)
        for _ in positions
    ]

    controllers = [
        RandomDodgeController(rng)
        for _ in positions
    ]

    history_times = []
    projectile_history = []
    uav_history = []

    next_audio_index = 0
    decisions = None
    decision_time = None
    response_time = None
    plan = None

    min_projectile = float("inf")
    min_uav = float("inf")
    path_length = 0.0
    previous_uavs = None
    path_lengths = np.zeros(len(positions))
    first_motion_time = None
    detection_compute = response_compute = acoustic_compute = 0.0

    try:
        uav_ids = [
            world.create_uav(position, radius=UAV_RADIUS)
            for position in positions
        ]

        projectile_id = world.create_projectile(
            position=state[:3],
            velocity=(0, 0, 0),
            radius=PROJECTILE_RADIUS,
            mass=0,
        )

        timestamps = np.append(
            np.arange(0, duration, world.dt),
            duration,
        )

        for t in timestamps:
            t = float(t)

            # Execute planned UAV movements.
            current_uavs = world.get_uav_positions(uav_ids)

            new_uavs = np.array([
                controller.position_at(t, current_uavs[i])
                for i, controller in enumerate(controllers)
            ])

            world.set_uav_positions(uav_ids, new_uavs)

            if previous_uavs is not None:
                increments = np.linalg.norm(new_uavs-previous_uavs,axis=1)
                path_lengths += increments
                if first_motion_time is None and np.any(increments>1e-9):
                    first_motion_time = t
                path_length += float(np.sum(
                    np.linalg.norm(
                        new_uavs - previous_uavs,
                        axis=1,
                    )
                ))

            previous_uavs = new_uavs.copy()

            # Ground-truth projectile movement exists only
            # in the simulation and acoustic sensing system.
            projectile = state[:3] + state[3:] * t

            p.resetBasePositionAndOrientation(
                projectile_id,
                projectile.tolist(),
                [0, 0, 0, 1],
                physicsClientId=world.client,
            )

            history_times.append(t)
            projectile_history.append(projectile.copy())
            uav_history.append(new_uavs.copy())

            # Same causal acoustic detection as DopplerDodge.
            if decisions is None and len(history_times) >= 2:
                end_index = min(
                    int(np.floor(t * SAMPLE_RATE + 1e-8)) + 1,
                    int(round(INIT_WINDOW * SAMPLE_RATE)),
                )

                if end_index > next_audio_index:
                    audio_times = (
                        np.arange(next_audio_index, end_index)
                        / SAMPLE_RATE
                    )

                    for i, detector in enumerate(detectors):
                        acoustic_start = time.perf_counter()
                        sensor_positions = (
                            np.asarray(uav_history)[:, i, :]
                        )

                        echo = dodge_echo_from_history(
                            audio_times,
                            np.asarray(history_times),
                            np.asarray(projectile_history),
                            sensor_positions,
                        )

                        detector.add_audio(echo)
                        acoustic_compute += time.perf_counter()-acoustic_start

                    next_audio_index = end_index

            # Single synchronized avoidance decision.
            if decisions is None and t >= INIT_WINDOW - 1e-9:
                detection_start = time.perf_counter()
                decisions = [
                    detector.decide()
                    for detector in detectors
                ]
                detection_compute = time.perf_counter()-detection_start

                decision_time = t

                predicted_hits = np.array([
                    prediction == "hit"
                    for prediction in decisions
                ])

                # CRITICAL:
                # Planner gets HIT/MISS and UAV positions only.
                # It does not receive projectile state.
                response_start = time.perf_counter()
                plan = plan_coordinated(
                    predicted_hits,
                    new_uavs,
                    rng,
                    min_hit_movement=min_hit_movement,
                )

                if plan.success:
                    for i, movement in enumerate(plan.displacements):
                        if np.linalg.norm(movement) <= 1e-4:
                            continue

                        controller = controllers[i]
                        controller.start_time = t
                        controller.origin = new_uavs[i].copy()
                        controller.displacement = movement.copy()

                    if np.any(
                        np.linalg.norm(
                            plan.displacements, axis=1
                        ) > 1e-4
                    ):
                        response_time = t
                response_compute = time.perf_counter()-response_start

            # Same evaluation distances as original baseline.
            projectile_dist, uav_dist = minimum_distances(
                new_uavs,
                projectile,
            )

            min_projectile = min(
                min_projectile, projectile_dist
            )
            min_uav = min(min_uav, uav_dist)

            world.step()

    finally:
        world.close()

    if decisions is None or plan is None:
        raise RuntimeError("No avoidance decision was made")

    # Ground truth is used ONLY for evaluation.
    truth = np.array([
        is_threatened(state, position,duration)
        for position in positions
    ])

    predicted_hits = np.array([
        prediction == "hit"
        for prediction in decisions
    ])

    hit_recall = (
        float(np.mean(predicted_hits[truth]))
        if np.any(truth)
        else float("nan")
    )

    false_positive_rate = (
        float(np.mean(predicted_hits[~truth]))
        if np.any(~truth)
        else float("nan")
    )

    maneuvers_before_impact = 0
    velocity = state[3:]
    speed_squared = float(velocity @ velocity)

    for i, controller in enumerate(controllers):
        if not truth[i] or controller.start_time is None:
            continue

        if speed_squared > 1e-12:
            closest_time = float(
                (positions[i] - state[:3]) @ velocity
                / speed_squared
            )
        else:
            closest_time = 0.0

        if controller.start_time < closest_time:
            maneuvers_before_impact += 1

    projectile_collision = (
        min_projectile < UAV_RADIUS + PROJECTILE_RADIUS
    )
    uav_collision = min_uav < 2 * UAV_RADIUS

    result = BaselineResult(
        scenario=scenario,
        seed=seed,
        projectile_collision=projectile_collision,
        uav_collision=uav_collision,
        collision_free=not (
            projectile_collision or uav_collision
        ),
        safety_margin_satisfied=(
            min_projectile >= PROJECTILE_SAFETY_DISTANCE
            and min_uav >= UAV_SAFETY_DISTANCE
        ),
        min_projectile_distance=min_projectile,
        min_uav_distance=min_uav,
        total_path_length=path_length,
        number_moved=int(np.count_nonzero(path_lengths>1e-6)),
        hit_recall=hit_recall,
        false_positive_rate=false_positive_rate,
        decision_latency=decision_time,
        response_delay=(
            response_time - decision_time
            if response_time is not None
            else float("nan")
        ),
        maneuvers_before_impact=maneuvers_before_impact,
        predictions=tuple(decisions),path_lengths=tuple(float(x) for x in path_lengths),
        detection_compute_s=detection_compute,response_compute_s=response_compute,
        acoustic_generation_s=acoustic_compute,first_response_time=response_time,
        first_motion_time=first_motion_time,
    )

    reactive_moves = sum(
        path_lengths[i]>1e-6
        and not predicted_hits[i]
        for i, controller in enumerate(controllers)
    )

    return result, plan.success, reactive_moves


# ------------------------------------------------------------
# 4. Evaluation and CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DopplerDodge + coordinated UAV planning"
    )

    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-count", type=int, default=50)
    parser.add_argument(
        "--min-hit-movement",
        type=float,
        default=0.40,
    )
    parser.add_argument("--gui", action="store_true")

    args = parser.parse_args()

    if args.seeds < 1 or args.calibration_count < 1:
        parser.error("Seed and calibration counts must be positive")

    if args.threshold is not None and args.threshold < 0:
        parser.error("Threshold must be nonnegative")

    if not (
        0 < args.min_hit_movement
        <= MAX_UAV_SPEED * MANEUVER_DURATION
    ):
        parser.error("Invalid minimum HIT movement")

    if args.gui and (
        args.scenario == "all" or args.seeds > 1
    ):
        parser.error("--gui requires one scenario and one seed")

    threshold = (
        args.threshold
        if args.threshold is not None
        else calibrate_threshold(args.calibration_count)
    )

    if args.calibrate:
        print(f"\nSelected threshold: {threshold:.6f}")
        return

    scenarios = (
        list(SCENARIOS)
        if args.scenario == "all"
        else [args.scenario]
    )

    all_results = []
    all_success = []
    all_reactive = []

    for scenario in scenarios:
        results = []
        successes = []
        reactive = []

        for seed in range(args.seed, args.seed + args.seeds):
            result, success, reactive_count = (
                run_coordinated_baseline(
                    scenario,
                    threshold,
                    seed=seed,
                    gui=args.gui,
                    min_hit_movement=args.min_hit_movement,
                )
            )

            results.append(result)
            successes.append(success)
            reactive.append(reactive_count)

        print(f"\nSCENARIO: {scenario.upper()} (COORDINATED)")
        print_summary(results)
        print(f"Planning success rate:         {np.mean(successes):.1%}")
        print(f"Reactive UAVs moved:           {np.mean(reactive):.4f}")

        all_results.extend(results)
        all_success.extend(successes)
        all_reactive.extend(reactive)

    if len(scenarios) > 1:
        print("\nALL SCENARIOS COMBINED (COORDINATED)")
        print_summary(all_results)
        print(f"Planning success rate:         {np.mean(all_success):.1%}")
        print(f"Reactive UAVs moved:           {np.mean(all_reactive):.4f}")


if __name__ == "__main__":
    main()
