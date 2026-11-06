"""

End-to-end acoustic trajectory estimation and UAV avoidance.
Run:
    python -m experiments.online_avoidance
    python -m experiments.online_avoidance --gui --realtime
The projectile's ground truth is used for simulation and
evaluation only. Planning uses the EKF estimate.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import MovingStreamingAcousticSensor
from experiments.online_estimation import OnlineEstimationPipeline
from estimation.batch_estimator import (
    predict_doppler,
    DEFAULT_MEASUREMENT_LAG,
)
from planning.avoidance_planner import plan_avoidance


UAV_POSITIONS = np.array([
    [0., 0., 0.],
    [2., 0., 0.],
    [0., 2., 0.],
    [0., 0., 2.],
])
PROJECTILE_STATE = np.array([
    -1., 0., 0.,
    2., 0., 0.,
])
UAV_RADIUS = 0.05
PROJECTILE_RADIUS = 0.02
SAFETY_MARGIN = 0.03
UAV_SEPARATION_MARGIN = 0.05
PROJECTILE_SAFETY_DISTANCE = (
    UAV_RADIUS + PROJECTILE_RADIUS + SAFETY_MARGIN
)
UAV_SAFETY_DISTANCE = (
    2 * UAV_RADIUS + UAV_SEPARATION_MARGIN
)
DURATION = 1.0
INIT_WINDOW = 0.2
MANEUVER_DURATION = 0.15
MAX_UAV_SPEED = 3.0
REPLAN_INTERVAL = 0.10


@dataclass
class OnlineAvoidanceResult:
    planning_attempts: int
    successful_plans: int
    first_plan_time: float | None
    initialized: bool
    minimum_projectile_separation: float
    minimum_uav_separation: float
    collision_free: bool
    safety_margin_satisfied: bool
    final_uav_positions: np.ndarray
    measurement_count: int
    ekf_update_count: int
    estimate_times: np.ndarray
    position_errors: np.ndarray
    velocity_errors: np.ndarray
    doppler_rmse: float
    moving_doppler_rmse: float
    moving_doppler_samples: int


def rmse(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def minimum_separations(uav_positions, projectile_position):
    projectile_distances = np.linalg.norm(
        uav_positions - projectile_position[None, :],
        axis=1,
    )
    min_projectile = float(np.min(projectile_distances))
    min_uav = float("inf")
    for i in range(len(uav_positions)):
        for j in range(i + 1, len(uav_positions)):
            distance = np.linalg.norm(
                uav_positions[i] - uav_positions[j]
            )
            min_uav = min(min_uav, float(distance))
    return min_projectile, min_uav


class UAVMotionHistory:
    """Store UAV positions and velocities at physics timestamps."""
    def __init__(self):
        self.times = []
        self.positions = []
        self.velocities = []
    def add(self, t, positions, velocities):
        t = float(t)
        if self.times and t <= self.times[-1]:
            raise ValueError(
                "Motion-history timestamps must increase."
            )
        self.times.append(t)
        self.positions.append(
            np.asarray(positions, dtype=float).copy()
        )
        self.velocities.append(
            np.asarray(velocities, dtype=float).copy()
        )
    def at(self, t):
        if not self.times:
            raise RuntimeError("Motion history is empty.")
        times = np.asarray(self.times)
        positions = np.asarray(self.positions)
        velocities = np.asarray(self.velocities)
        t = float(np.clip(t, times[0], times[-1]))
        right = int(
            np.searchsorted(times, t, side="right")
        )
        if right == 0:
            return positions[0].copy(), velocities[0].copy()
        if right >= len(times):
            return positions[-1].copy(), velocities[-1].copy()
        left = right - 1
        alpha = (
            (t - times[left])
            / (times[right] - times[left])
        )
        position = (
            (1.0 - alpha) * positions[left]
            + alpha * positions[right]
        )
        # Use the velocity associated with the interval.
        velocity = velocities[right].copy()
        return position, velocity


def process_pipeline_measurements(
    pipeline,
    current_time,
    motion_history,
    doppler_errors,
    moving_doppler_errors,
):
    """
    Process complete synchronized measurement sets.
    Initialization uses the stationary UAV geometry.
    Subsequent EKF corrections use time-aligned geometry.
    """
    completed = sorted(
        timestamp
        for timestamp, values in pipeline.pending.items()
        if (
            len(values) == pipeline.n_uavs
            and timestamp <= current_time + 1e-6
        )
    )
    for measurement_time in completed:
        values = pipeline.pending.pop(measurement_time)
        z = np.array([
            values[i] for i in range(pipeline.n_uavs)
        ])
        pipeline.measurement_times.append(
            measurement_time
        )
        pipeline.measured_doppler.append(z)
        observation_time = (
            measurement_time - DEFAULT_MEASUREMENT_LAG
        )
        measured_uavs, measured_velocities = (
            motion_history.at(observation_time)
        )
        if not pipeline.initialized:
            if measurement_time <= pipeline.init_window:
                continue
            times = np.asarray(
                pipeline.measurement_times
            )
            measurements = np.asarray(
                pipeline.measured_doppler
            )
            K = int(np.searchsorted(
                times,
                pipeline.init_window,
                side="right",
            ))
            if K < 2:
                raise RuntimeError(
                    "Insufficient batch initialization data."
                )
            initial_uavs, _ = motion_history.at(0.0)
            start = time.perf_counter()
            pipeline.ekf.initialize(
                times=times[:K],
                uav_positions=initial_uavs,
                measured_doppler=measurements[:K],
                initial_covariance=np.eye(6),
            )
            print(
                "Batch initial state:",
                np.round(pipeline.ekf.x, 4)
            )
            pipeline.initialization_runtime = (
                time.perf_counter() - start
            )
            pipeline.initialization_measurements = K
            pipeline.ekf.predict(times[K - 1])
            pipeline.initialized = True
        dt = measurement_time - pipeline.ekf.t
        if dt < -1e-9:
            raise RuntimeError(
                "Measurements arrived out of order."
            )
        start = time.perf_counter()
        pipeline.ekf.predict(max(0.0, dt))
        # Innovation diagnostic, before EKF correction.
        predicted = predict_doppler(
            pipeline.ekf.x,
            np.array([0.0]),
            measured_uavs,
            delay_aware=True,
            measurement_lag=DEFAULT_MEASUREMENT_LAG,
            uav_velocities=measured_velocities,
        ).ravel()
        errors = z - predicted
        doppler_errors.extend(errors.tolist())
        moving_mask = (
            np.linalg.norm(measured_velocities, axis=1)
            > 1e-6
        )
        moving_doppler_errors.extend(
            errors[moving_mask].tolist()
        )
        pipeline.ekf.update(
            measured_doppler=z,
            uav_positions=measured_uavs,
            uav_velocities=measured_velocities,
        )
        pipeline.update_runtimes.append(
            time.perf_counter() - start
        )
        pipeline.estimated_times.append(
            measurement_time
        )
        pipeline.estimated_states.append(
            pipeline.ekf.x.copy()
        )


def run_online_avoidance(
    projectile_state=PROJECTILE_STATE,
    uav_positions=UAV_POSITIONS,
    duration=DURATION,
    gui=False,
    realtime=False,
    replan_interval=REPLAN_INTERVAL,
    verbose=True,
):
    projectile_state = np.asarray(
        projectile_state, dtype=float
    ).reshape(6)
    initial_uavs = np.asarray(
        uav_positions, dtype=float
    ).reshape(-1, 3)
    world = SimulationWorld(gui=gui, gravity=False)
    sensors = [
        MovingStreamingAcousticSensor(
            sensor_position=position,
            duration=duration,
        )
        for position in initial_uavs
    ]
    pipeline = OnlineEstimationPipeline(
        initial_uavs,
        init_window=INIT_WINDOW,
    )
    motion_history = UAVMotionHistory()
    doppler_errors = []
    moving_doppler_errors = []
    planning_attempts = 0
    successful_plans = 0
    first_plan_time = None
    last_plan_time = -float("inf")
    movement_start = None
    movement_origins = None
    movement_targets = None
    minimum_projectile = float("inf")
    minimum_uav = float("inf")
    position_errors = []
    velocity_errors = []
    estimate_times = []
    try:
        uav_ids = [
            world.create_uav(
                position,
                radius=UAV_RADIUS,
            )
            for position in initial_uavs
        ]
        projectile_id = world.create_projectile(
            position=projectile_state[:3],
            velocity=(0., 0., 0.),
            radius=PROJECTILE_RADIUS,
            mass=0,
        )
        if gui:
            end_position = (
                projectile_state[:3]
                + projectile_state[3:] * duration
            )
            p.addUserDebugLine(
                projectile_state[:3],
                end_position,
                [1., 0., 0.],
                lineWidth=3,
                physicsClientId=world.client,
            )
            print("Starting simulation in 2 seconds...")
            time.sleep(2.0)
        timestamps = np.append(
            np.arange(0.0, duration, world.dt),
            duration,
        )
        for step, t in enumerate(timestamps):
            t = float(t)

            # -------------------------------------------
            # 1. Execute the active UAV maneuver.
            # -------------------------------------------
            if movement_start is not None:
                world.execute_uav_movement(
                    uav_ids,
                    movement_origins,
                    movement_targets - movement_origins,
                    elapsed_time=t - movement_start,
                    maneuver_duration=MANEUVER_DURATION,
                )
            current_uavs = world.get_uav_positions(
                uav_ids
            )
            current_velocities = np.zeros_like(
                current_uavs
            )
            if movement_start is not None:
                elapsed = t - movement_start
                if 0.0 <= elapsed < MANEUVER_DURATION:
                    current_velocities = (
                        movement_targets - movement_origins
                    ) / MANEUVER_DURATION
            motion_history.add(
                t,
                current_uavs,
                current_velocities,
            )

            # -------------------------------------------
            # 2. Simulate the projectile.
            # -------------------------------------------
            true_position = (
                projectile_state[:3]
                + projectile_state[3:] * t
            )
            p.resetBasePositionAndOrientation(
                projectile_id,
                true_position.tolist(),
                [0., 0., 0., 1.],
                physicsClientId=world.client,
            )

            # -------------------------------------------
            # 3. Generate acoustic measurements.
            # -------------------------------------------
            previous_updates = len(
                pipeline.estimated_times
            )
            for i, sensor in enumerate(sensors):
                measurement_times, doppler = (
                    sensor.advance(
                        physics_time=t,
                        projectile_position=true_position,
                        sensor_position=current_uavs[i],
                    )
                )
                pipeline.add_measurements(
                    i,
                    measurement_times,
                    doppler,
                )

            # -------------------------------------------
            # 4. Update EKF with historical UAV geometry.
            # -------------------------------------------
            process_pipeline_measurements(
                pipeline,
                t,
                motion_history,
                doppler_errors,
                moving_doppler_errors,
            )
            new_updates = len(
                pipeline.estimated_times
            )

            # -------------------------------------------
            # 5. Record estimation accuracy.
            # -------------------------------------------
            for k in range(previous_updates, new_updates):
                estimate_t = pipeline.estimated_times[k]
                estimate = pipeline.estimated_states[k]
                actual = projectile_state.copy()
                actual[:3] += (
                    projectile_state[3:] * estimate_t
                )
                position_errors.append(
                    np.linalg.norm(
                        estimate[:3] - actual[:3]
                    )
                )
                velocity_errors.append(
                    np.linalg.norm(
                        estimate[3:] - actual[3:]
                    )
                )
                estimate_times.append(estimate_t)

            # -------------------------------------------
            # 6. Replan using the EKF state.
            # -------------------------------------------
            new_estimate = (
                new_updates > previous_updates
            )
            should_plan = (
                pipeline.initialized
                and new_estimate
                and t - last_plan_time
                >= replan_interval - 1e-9
            )
            if should_plan:
                last_plan_time = t
                planning_attempts += 1
                estimated_state = (
                    pipeline.ekf.x.copy()
                )
                estimate_age = max(
                    0.0,
                    t - pipeline.ekf.t,
                )
                estimated_state[:3] += (
                    estimated_state[3:] * estimate_age
                )
                if planning_attempts == 1:
                    true_state = projectile_state.copy()
                    true_state[:3] += true_state[3:] * t
                    print("\nFIRST PLANNING ATTEMPT")
                    print("Estimated state:", np.round(estimated_state, 3))
                    print("True state:     ", np.round(true_state, 3))
                remaining_horizon = duration - t
                if remaining_horizon > 0:
                    plan = plan_avoidance(
                        state=estimated_state,
                        uav_positions=current_uavs,
                        prediction_horizon=remaining_horizon,
                        maneuver_duration=MANEUVER_DURATION,
                        max_uav_speed=MAX_UAV_SPEED,
                        uav_radius=UAV_RADIUS,
                        projectile_radius=PROJECTILE_RADIUS,
                        safety_margin=SAFETY_MARGIN + 0.01,  # Planner uses a 1 cm buffer above the evaluation safety margin.
                        uav_separation_margin=(
                            UAV_SEPARATION_MARGIN
                        ),
                    )
                    if plan.success:
                        successful_plans += 1
                        if first_plan_time is None:
                            first_plan_time = t
                        movement_start = t
                        movement_origins = (
                            current_uavs.copy()
                        )
                        movement_targets = (
                            plan.target_positions.copy()
                        )
                        if verbose:
                            print(
                                f"t={t:.3f}s | "
                                f"SQP success | "
                                f"moved={plan.number_moved} | "
                                f"displacement="
                                f"{plan.total_displacement:.3f}m"
                            )
                    elif verbose:
                        print(
                            f"t={t:.3f}s | "
                            f"SQP failed: {plan.message}"
                        )

            # -------------------------------------------
            # 7. Evaluate physical separation.
            # -------------------------------------------
            projectile_distance, uav_distance = (
                minimum_separations(
                    current_uavs,
                    true_position,
                )
            )
            minimum_projectile = min(
                minimum_projectile,
                projectile_distance,
            )
            minimum_uav = min(
                minimum_uav,
                uav_distance,
            )
            if step < len(timestamps) - 1:
                world.step()
            if gui and realtime:
                time.sleep(world.dt)
        final_uavs = world.get_uav_positions(
            uav_ids
        )
        if gui:
            print(
                "\nSimulation complete. "
                "Press Enter to close."
            )
            input()
    finally:
        world.close()
    collision_free = (
        minimum_projectile
        >= UAV_RADIUS + PROJECTILE_RADIUS - 1e-6
        and minimum_uav
        >= 2 * UAV_RADIUS - 1e-6
    )
    safety_margin_satisfied = (
        minimum_projectile
        >= PROJECTILE_SAFETY_DISTANCE - 1e-6
        and minimum_uav
        >= UAV_SAFETY_DISTANCE - 1e-6
    )
    result = OnlineAvoidanceResult(
        planning_attempts=planning_attempts,
        successful_plans=successful_plans,
        first_plan_time=first_plan_time,
        initialized=pipeline.initialized,
        minimum_projectile_separation=minimum_projectile,
        minimum_uav_separation=minimum_uav,
        collision_free=collision_free,
        safety_margin_satisfied=safety_margin_satisfied,
        final_uav_positions=final_uavs,
        measurement_count=len(
            pipeline.measurement_times
        ),
        ekf_update_count=len(
            pipeline.estimated_times
        ),
        estimate_times=np.asarray(estimate_times),
        position_errors=np.asarray(position_errors),
        velocity_errors=np.asarray(velocity_errors),
        doppler_rmse=rmse(doppler_errors),
        moving_doppler_rmse=rmse(
            moving_doppler_errors
        ),
        moving_doppler_samples=len(
            moving_doppler_errors
        ),
    )
    if verbose:
        print("\n" + "=" * 55)
        print("ONLINE AVOIDANCE RESULTS")
        print("=" * 55)
        print("EKF initialized:", result.initialized)
        print("Measurements:", result.measurement_count)
        print("EKF updates:", result.ekf_update_count)
        print("Planning attempts:", result.planning_attempts)
        print("Successful plans:", result.successful_plans)
        print("First plan time:", result.first_plan_time)
        print(
            "Doppler innovation RMSE:",
            f"{result.doppler_rmse:.4f} Hz",
        )
        print(
            "Moving-UAV Doppler innovation RMSE:",
            f"{result.moving_doppler_rmse:.4f} Hz",
        )
        print(
            "Moving-UAV samples:",
            result.moving_doppler_samples,
        )
        print(
            "Position RMSE:",
            f"{rmse(result.position_errors):.4f} m",
        )
        print(
            "Velocity RMSE:",
            f"{rmse(result.velocity_errors):.4f} m/s",
        )
        print(
            "Minimum projectile separation:",
            f"{result.minimum_projectile_separation:.4f} m",
        )
        print(
            "Minimum UAV separation:",
            f"{result.minimum_uav_separation:.4f} m",
        )
        print("Collision free:", result.collision_free)
        print(
            "Safety margin satisfied:",
            result.safety_margin_satisfied,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=["direct", "faster", "diagonal", "offset", "none", "all"],
        default="direct",
    )
    args = parser.parse_args()
    scenarios = {
        "direct": np.array([-1., 0., 0., 2., 0., 0.]),
        "faster": np.array([-1.5, 0., 0., 4., 0., 0.]),
        "diagonal": np.array([-1., -1., 0., 2., 2., 0.]),
        "offset": np.array([-1., 0.15, 0., 2., 0., 0.]),
        "none": np.array([-1., 1., 1., 2., 0., 0.]),
    }
    selected = (
        scenarios.items()
        if args.scenario == "all"
        else [(args.scenario, scenarios[args.scenario])]
    )
    for name, state in selected:
        print(f"\n{'=' * 55}")
        print(f"SCENARIO: {name.upper()}")
        print(f"{'=' * 55}")
        run_online_avoidance(
            projectile_state=state,
            gui=args.gui,
            realtime=args.realtime,
        )


if __name__ == "__main__":
    main()
