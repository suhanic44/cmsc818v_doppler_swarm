
"""DopplerDodge single-microphone hit/miss baseline.

Based on Garg & Roy (HotMobile 2020), Section 2.5.

Each UAV independently:
1. Processes raw acoustic echoes using cumulative drift.
2. Classifies the projectile as HIT or MISS.
3. Performs one random avoidance maneuver if HIT.

Threshold calibration and PyBullet evaluation are included below.

Run from the repository root:
    python -m baselines.dopplerdodge --calibrate
    python -m baselines.dopplerdodge --scenario direct
    python -m baselines.dopplerdodge --scenario all --seeds 20
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    _echo_from_history,
    generate_trajectory_echo,
)
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


SCENARIOS = {
    "direct": np.array([-1., 0., 0., 2., 0., 0.]),
    "faster": np.array([-1.5, 0., 0., 4., 0., 0.]),
    "diagonal": np.array([-1., -1., 0., 2., 2., 0.]),
    "offset": np.array([-1., 0.15, 0., 2., 0., 0.]),
    "none": np.array([-1., 1., 1., 2., 0., 0.]),
}

BUFFER_MS = 10.0
MIN_RMS = 0.05
CALIBRATION_SEED = 42


# ------------------------------------------------------------
# 1. Original DopplerDodge detection
# ------------------------------------------------------------

@dataclass
class DopplerDodgeResult:
    times: np.ndarray
    product_sums: np.ndarray
    cumulative_drift: np.ndarray
    score: float
    prediction: str | None


def analyze_echo(
    echo,
    sample_rate=SAMPLE_RATE,
    buffer_ms=BUFFER_MS,
    drift_threshold=None,
    min_rms=MIN_RMS,
):
    """Analyze one UAV's reflected acoustic waveform."""
    echo = np.asarray(echo, dtype=float).reshape(-1)

    if not np.all(np.isfinite(echo)):
        raise ValueError("Audio must contain only finite samples")
    if sample_rate <= 0 or buffer_ms <= 0:
        raise ValueError("sample_rate and buffer_ms must be positive")
    if drift_threshold is not None and drift_threshold < 0:
        raise ValueError("drift_threshold must be nonnegative")

    # Divide audio into equal-length buffers.
    samples_per_buffer = int(round(
        sample_rate * buffer_ms / 1000.0
    ))

    if samples_per_buffer < 2:
        raise ValueError("Buffer is too short")

    n_buffers = len(echo) // samples_per_buffer

    if n_buffers < 3:
        raise ValueError("Need at least three complete audio buffers")

    buffers = echo[:n_buffers * samples_per_buffer].reshape(
        n_buffers, samples_per_buffer
    )

    # Find the first continuous interval with a usable echo.
    rms = np.sqrt(np.mean(buffers**2, axis=1))
    active_indices = np.flatnonzero(rms >= min_rms)

    if not len(active_indices):
        raise ValueError("No reflected signal above min_rms")

    start = int(active_indices[0])
    end = start

    while end < n_buffers and rms[end] >= min_rms:
        end += 1

    if end - start < 3:
        raise ValueError("Need three consecutive active audio buffers")

    # Multiply adjacent buffers sample by sample.
    a = buffers[start:end - 1]
    b = buffers[start + 1:end]
    product_sums = np.sum(a * b, axis=1)

    # Accumulate the changes between product sums.
    changes = np.abs(np.diff(product_sums))
    cumulative_drift = np.r_[0.0, np.cumsum(changes)]
    score = float(cumulative_drift[-1])

    # Low drift = HIT; high drift = MISS.
    prediction = None
    if drift_threshold is not None:
        prediction = (
            "hit" if score <= drift_threshold else "miss"
        )

    times = (
        np.arange(start + 2, end + 1)
        * samples_per_buffer / sample_rate
    )

    return DopplerDodgeResult(
        times=times,
        product_sums=product_sums,
        cumulative_drift=cumulative_drift,
        score=score,
        prediction=prediction,
    )


class StreamingDopplerDodge:
    """Collect causal audio chunks and make one HIT/MISS decision."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.audio = []
        self.result = None

    def add_audio(self, chunk):
        self.audio.append(np.asarray(chunk, dtype=float))

    def decide(self):
        """Classify using only audio received so far."""
        if self.result is not None:
            return self.result.prediction

        audio = np.concatenate(self.audio)

        try:
            self.result = analyze_echo(
                audio,
                drift_threshold=self.threshold,
            )
            return self.result.prediction
        except ValueError as exc:
            if "No reflected signal" not in str(exc) and (
                "three consecutive active" not in str(exc)
            ):
                raise

            # No usable echo means no HIT evidence.
            return "miss"


# ------------------------------------------------------------
# 2. Threshold calibration
# ------------------------------------------------------------

def is_threatened(state, uav_position, duration=DURATION):
    """Would the projectile violate this stationary UAV's safety margin?"""
    p0 = state[:3]
    velocity = state[3:]

    speed_squared = float(np.dot(velocity, velocity))

    if speed_squared < 1e-12:
        closest_time = INIT_WINDOW
    else:
        closest_time = np.clip(
            np.dot(uav_position - p0, velocity) / speed_squared,
            INIT_WINDOW,
            duration,
        )

    closest_position = p0 + velocity * closest_time

    return (
        np.linalg.norm(closest_position - uav_position)
        < PROJECTILE_SAFETY_DISTANCE
    )


def calibration_states(count, seed=CALIBRATION_SEED):
    """Generate trajectories separate from the evaluation scenarios."""
    rng = np.random.default_rng(seed)

    for index in range(count):
        speed = rng.uniform(1.5, 4.0)
        start_x = rng.uniform(-1.8, -0.8)

        # Alternate near-collision and noncollision trajectories.
        if index % 2 == 0:
            target = UAV_POSITIONS[
                rng.integers(len(UAV_POSITIONS))
            ]
            y, z = target[1:] + rng.normal(0, 0.06, 2)
            y_velocity, z_velocity = rng.normal(0, 0.1, 2)
        else:
            y, z = rng.uniform(-1.0, 3.0, 2)
            y_velocity, z_velocity = rng.normal(0, 0.2, 2)

        yield np.array([
            start_x, y, z,
            speed, y_velocity, z_velocity,
        ])


def calibrate_threshold(count=50, seed=CALIBRATION_SEED):
    """Choose a threshold maximizing balanced HIT/MISS accuracy."""
    scores = []
    labels = []

    physics_times = np.arange(0, INIT_WINDOW + 1e-9, 1 / 240)
    audio_times = np.arange(0, INIT_WINDOW, 1 / SAMPLE_RATE)

    for state in calibration_states(count, seed):
        projectile_positions = (
            state[:3]
            + physics_times[:, None] * state[3:]
        )

        for uav_position in UAV_POSITIONS:
            echo = generate_trajectory_echo(
                audio_times,
                physics_times,
                projectile_positions,
                uav_position,
            )

            try:
                result = analyze_echo(echo)
            except ValueError as exc:
                if "No reflected signal" in str(exc) or (
                    "three consecutive active" in str(exc)
                ):
                    continue
                raise

            scores.append(result.score)
            labels.append(is_threatened(state, uav_position))

    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)

    if not np.any(labels) or not np.any(~labels):
        raise RuntimeError(
            "Calibration requires both HIT and MISS examples."
        )

    # Evaluate every possible threshold between observed scores.
    unique_scores = np.unique(scores)
    candidates = np.r_[
        0.0,
        (unique_scores[:-1] + unique_scores[1:]) / 2,
        unique_scores[-1] + 1.0,
    ]

    best_threshold = 0.0
    best_accuracy = -1.0
    best_recall = -1.0

    for threshold in candidates:
        predicted = scores <= threshold

        recall = np.mean(predicted[labels])
        specificity = np.mean(~predicted[~labels])
        balanced_accuracy = (recall + specificity) / 2

        if (
            balanced_accuracy > best_accuracy + 1e-12
            or (
                abs(balanced_accuracy - best_accuracy) <= 1e-12
                and recall > best_recall
            )
        ):
            best_threshold = float(threshold)
            best_accuracy = float(balanced_accuracy)
            best_recall = float(recall)

    predicted = scores <= best_threshold

    print("\nTHRESHOLD CALIBRATION")
    print(f"Trajectories:       {count}")
    print(f"Usable UAV samples: {len(scores)}")
    print(f"Threshold:          {best_threshold:.4f}")
    print(f"Balanced accuracy:  {best_accuracy:.1%}")
    print(f"HIT recall:         {np.mean(predicted[labels]):.1%}")
    print(f"False-positive rate:{np.mean(predicted[~labels]):.1%}")

    return best_threshold


# ------------------------------------------------------------
# 3. Independent random avoidance
# ------------------------------------------------------------

class RandomDodgeController:
    """Each UAV independently chooses one random avoidance direction."""

    def __init__(self, rng):
        self.rng = rng
        self.start_time = None
        self.origin = None
        self.displacement = np.zeros(3)

    def trigger(self, time, position):
        direction = self.rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        self.start_time = time
        self.origin = np.asarray(position, dtype=float).copy()

        self.displacement = (
            direction * MAX_UAV_SPEED * MANEUVER_DURATION
        )

    def position_at(self, time, current_position):
        if self.start_time is None:
            return current_position

        fraction = np.clip(
            (time - self.start_time) / MANEUVER_DURATION,
            0.0,
            1.0,
        )

        return self.origin + fraction * self.displacement


# ------------------------------------------------------------
# 4. PyBullet evaluation
# ------------------------------------------------------------

@dataclass
class BaselineResult:
    scenario: str
    seed: int
    projectile_collision: bool
    uav_collision: bool
    collision_free: bool
    safety_margin_satisfied: bool
    min_projectile_distance: float
    min_uav_distance: float
    total_path_length: float
    number_moved: int
    hit_recall: float
    false_positive_rate: float
    decision_latency: float
    response_delay: float
    maneuvers_before_impact: int
    predictions: tuple = ()
    path_lengths: tuple = ()
    detection_compute_s: float = 0.0
    response_compute_s: float = 0.0
    acoustic_generation_s: float = 0.0
    first_response_time: float | None = None
    first_motion_time: float | None = None


def minimum_distances(uavs, projectile):
    """Return minimum center-to-center distances."""
    projectile_dist = float(np.min(
        np.linalg.norm(uavs - projectile, axis=1)
    ))

    uav_dist = float("inf")

    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            uav_dist = min(
                uav_dist,
                float(np.linalg.norm(uavs[i] - uavs[j])),
            )

    return projectile_dist, uav_dist


def run_baseline(
    scenario,
    threshold,
    seed=0,
    gui=False,
    *,
    projectile_state=None,
    uav_positions=None,
    duration=None,
):
    """Run one full PyBullet trial with independent UAV decisions."""
    state = np.asarray(SCENARIOS[scenario] if projectile_state is None else projectile_state, dtype=float).copy()
    positions = np.asarray(UAV_POSITIONS if uav_positions is None else uav_positions, dtype=float).copy()
    duration = DURATION if duration is None else float(duration)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("projectile_state must be a finite six-element vector")
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1 or not np.all(np.isfinite(positions)):
        raise ValueError("uav_positions must be a finite nonempty (N,3) array")
    if not np.isfinite(duration) or duration < INIT_WINDOW:
        raise ValueError("duration must cover the observation window")
    if not np.isfinite(threshold) or threshold < 0:
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

    min_projectile = float("inf")
    min_uav = float("inf")
    path_length = 0.0
    previous_uavs = None
    path_lengths = np.zeros(len(positions))
    detection_compute = response_compute = acoustic_compute = 0.0
    first_motion_time = None

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

            # Execute any triggered avoidance maneuvers.
            current_uavs = world.get_uav_positions(uav_ids)

            new_uavs = np.array([
                controller.position_at(t, current_uavs[i])
                for i, controller in enumerate(controllers)
            ])

            world.set_uav_positions(uav_ids, new_uavs)

            if previous_uavs is not None:
                increments = np.linalg.norm(new_uavs-previous_uavs,axis=1)
                path_lengths += increments
                if first_motion_time is None and np.any(increments > 1e-9):
                    first_motion_time = t
                path_length += float(np.sum(
                    np.linalg.norm(
                        new_uavs - previous_uavs,
                        axis=1,
                    )
                ))

            previous_uavs = new_uavs.copy()

            # Projectile follows its prescribed constant-velocity path.
            projectile = state[:3] + state[3:] * t

            p.resetBasePositionAndOrientation(
                projectile_id,
                projectile.tolist(),
                [0, 0, 0, 1],
                physicsClientId=world.client,
            )

            # Record physics history for causal acoustic generation.
            history_times.append(t)
            projectile_history.append(projectile.copy())
            uav_history.append(new_uavs.copy())

            # Generate newly available raw acoustic samples.
            if decisions is None and len(history_times) >= 2:
                end_index = min(
                    int(np.floor(t * SAMPLE_RATE + 1e-8)) + 1,
                    int(round(INIT_WINDOW * SAMPLE_RATE)),
                )

                if end_index > next_audio_index:
                    sample_indices = np.arange(
                        next_audio_index, end_index
                    )
                    audio_times = sample_indices / SAMPLE_RATE

                    for i, detector in enumerate(detectors):
                        acoustic_start = time.perf_counter()
                        sensor_positions = np.array([
                            positions[i] for positions in uav_history
                        ])

                        echo = _echo_from_history(
                            audio_times,
                            np.asarray(history_times),
                            np.asarray(projectile_history),
                            sensor_positions,
                        )

                        detector.add_audio(echo)
                        acoustic_compute += time.perf_counter()-acoustic_start

                    next_audio_index = end_index

            # Make one independent decision after the observation window.
            if decisions is None and t >= INIT_WINDOW - 1e-9:
                decision_start = time.perf_counter()
                decisions = [
                    detector.decide()
                    for detector in detectors
                ]
                detection_compute = time.perf_counter()-decision_start
                decision_time = t

                response_start = time.perf_counter()
                for i, prediction in enumerate(decisions):
                    if prediction == "hit":
                        controllers[i].trigger(t, new_uavs[i])
                response_compute = time.perf_counter()-response_start

                if any(d == "hit" for d in decisions):
                    response_time = t

            # Evaluate separation at every physics timestep.
            projectile_dist, uav_dist = minimum_distances(
                new_uavs, projectile
            )

            min_projectile = min(
                min_projectile, projectile_dist
            )
            min_uav = min(min_uav, uav_dist)

            world.step()

    finally:
        world.close()

    if decisions is None:
        raise RuntimeError("Simulation ended before HIT/MISS decisions")

    # Ground-truth labels are used for evaluation only.
    truth = np.array([
        is_threatened(state, position, duration)
        for position in positions
    ])

    predicted_hits = np.array([
        prediction == "hit"
        for prediction in decisions
    ])

    hit_recall = (
        float(np.mean(predicted_hits[truth]))
        if np.any(truth) else float("nan")
    )

    false_positive_rate = (
        float(np.mean(predicted_hits[~truth]))
        if np.any(~truth) else float("nan")
    )

    # Count maneuvers initiated before the nominal closest approach.
    maneuvers_before_impact = 0
    velocity = state[3:]
    speed_squared = float(np.dot(velocity, velocity))

    for i, controller in enumerate(controllers):
        if not truth[i] or controller.start_time is None:
            continue

        closest_time = (
            np.dot(positions[i] - state[:3], velocity)
            / speed_squared
            if speed_squared > 1e-12 else 0.0
        )

        if controller.start_time < closest_time:
            maneuvers_before_impact += 1

    projectile_collision = (
        min_projectile < UAV_RADIUS + PROJECTILE_RADIUS
    )
    uav_collision = min_uav < 2 * UAV_RADIUS

    return BaselineResult(
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
        number_moved=int(np.count_nonzero(path_lengths > 1e-6)),
        hit_recall=hit_recall,
        false_positive_rate=false_positive_rate,
        decision_latency=decision_time,
        response_delay=(
            response_time - decision_time
            if response_time is not None
            else float("nan")
        ),
        maneuvers_before_impact=maneuvers_before_impact,
        predictions=tuple(decisions),
        path_lengths=tuple(float(x) for x in path_lengths),
        detection_compute_s=detection_compute,
        response_compute_s=response_compute,
        acoustic_generation_s=acoustic_compute,
        first_response_time=response_time,
        first_motion_time=first_motion_time,
    )


# ------------------------------------------------------------
# 5. Reporting and CLI
# ------------------------------------------------------------

def mean_metric(results, name):
    values = np.array([
        float(getattr(result, name))
        for result in results
    ])

    if not np.any(np.isfinite(values)):
        return float("nan")

    return float(np.mean(values[np.isfinite(values)]))


def print_summary(results):
    print("\n" + "=" * 58)
    print("DOPPLERDODGE EVALUATION")
    print("=" * 58)

    print(f"Trials:                       {len(results)}")

    metrics = [
        ("Collision-free success rate", "collision_free", "%"),
        ("Projectile-UAV collision rate", "projectile_collision", "%"),
        ("UAV-UAV collision rate", "uav_collision", "%"),
        ("Safety-margin success rate", "safety_margin_satisfied", "%"),
        ("Minimum projectile distance", "min_projectile_distance", "m"),
        ("Minimum UAV distance", "min_uav_distance", "m"),
        ("Total UAV path length", "total_path_length", "m"),
        ("UAVs moved", "number_moved", ""),
        ("HIT recall", "hit_recall", "%"),
        ("False-positive rate", "false_positive_rate", "%"),
        ("Decision latency", "decision_latency", "s"),
        ("Response delay", "response_delay", "s"),
        ("Maneuvers before impact", "maneuvers_before_impact", ""),
    ]

    for label, field, unit in metrics:
        value = mean_metric(results, field)

        if unit == "%":
            formatted = f"{value:.1%}"
        else:
            formatted = f"{value:.4f} {unit}".strip()

        print(f"{label + ':':30s} {formatted}")


def main():
    parser = argparse.ArgumentParser(
        description="Independent DopplerDodge avoidance baseline"
    )

    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-count", type=int, default=50)
    parser.add_argument("--gui", action="store_true")

    args = parser.parse_args()

    if args.seeds < 1 or args.calibration_count < 1:
        parser.error("--seeds and --calibration-count must be positive")
    if args.threshold is not None and args.threshold < 0:
        parser.error("--threshold must be nonnegative")
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

    for scenario in scenarios:
        results = []

        for seed in range(args.seed, args.seed + args.seeds):
            result = run_baseline(
                scenario=scenario,
                threshold=threshold,
                seed=seed,
                gui=args.gui,
            )
            results.append(result)

        print(f"\nSCENARIO: {scenario.upper()}")
        print_summary(results)
        all_results.extend(results)

    if len(scenarios) > 1:
        print("\nALL SCENARIOS COMBINED")
        print_summary(all_results)


if __name__ == "__main__":
    main()
