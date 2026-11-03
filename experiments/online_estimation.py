
"""
Online EKF trajectory estimation using simulated acoustic measurements.
"""

import numpy as np

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    generate_trajectory_echo,
    estimate_doppler,
)
from estimation.ekf import ExtendedKalmanFilter


def synchronize_measurements(all_measurements):
    """
    Keep only timestamps for which every UAV has a valid
    Doppler measurement.
    """
    measurement_maps = []

    for times, doppler in all_measurements:
        if len(times) != len(doppler):
            raise RuntimeError("Timestamp/Doppler length mismatch.")

        rounded_times = np.round(
            np.asarray(times, dtype=float), 6
        )

        measurement_maps.append(
            dict(zip(rounded_times, doppler))
        )

    common_times = set(measurement_maps[0])

    for measurement_map in measurement_maps[1:]:
        common_times.intersection_update(measurement_map)

    measurement_times = np.array(
        sorted(common_times), dtype=float
    )

    if len(measurement_times) == 0:
        raise RuntimeError(
            "No shared Doppler measurement timestamps."
        )

    measured_doppler = np.column_stack([
        [measurement_map[t] for t in measurement_times]
        for measurement_map in measurement_maps
    ])

    if not np.all(np.isfinite(measured_doppler)):
        raise RuntimeError("Non-finite Doppler measurements.")

    print(
        "Measurements per UAV:",
        [len(m) for m in measurement_maps],
    )
    print(
        "Synchronized measurements:",
        len(measurement_times),
    )

    return measurement_times, measured_doppler


def main():
    # 1. Set up the same scenario as our previous experiment
    uav_positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])

    initial_position = np.array([3.0, 3.0, 3.0])
    initial_velocity = np.array([-2.0, -1.5, -1.0])
    duration = 0.5

    world = SimulationWorld(gui=False, gravity=False)

    try:
        for position in uav_positions:
            world.create_uav(position)

        projectile_id = world.create_projectile(
            position=initial_position,
            velocity=initial_velocity,
        )

        # 2. Record ground-truth trajectory from PyBullet
        physics_times = []
        projectile_positions = []
        projectile_velocities = []

        num_steps = int(duration / world.dt)

        for step in range(num_steps + 1):
            t = step * world.dt

            position, velocity = world.get_projectile_state(
                projectile_id
            )

            physics_times.append(t)
            projectile_positions.append(position)
            projectile_velocities.append(velocity)

            if step < num_steps:
                world.step()

        physics_times = np.array(physics_times)
        projectile_positions = np.array(projectile_positions)
        projectile_velocities = np.array(projectile_velocities)

        # 3. Generate actual simulated acoustic measurements
        audio_times = np.arange(
            0, duration, 1 / SAMPLE_RATE
        )

        all_measurements = []

        for sensor_position in uav_positions:
            echo = generate_trajectory_echo(
                audio_times,
                physics_times,
                projectile_positions,
                sensor_position,
            )

            times, doppler = estimate_doppler(
                audio_times,
                echo,
                window_duration=0.04,
                hop_duration=0.02,
            )

            all_measurements.append((times, doppler))

        # Synchronize valid measurements across all four UAVs.
        measurement_times, measured_doppler = (
            synchronize_measurements(all_measurements)
        )

        print(
            "Doppler measurement shape:",
            measured_doppler.shape,
        )

        # 4. Initialize EKF using the first 200 ms
        K = np.searchsorted(
            measurement_times, 0.2, side="right"
        )

        if K < 2 or K >= len(measurement_times):
            raise RuntimeError(
                "Insufficient measurements for EKF."
            )

        ekf = ExtendedKalmanFilter()

        ekf.initialize(
            times=measurement_times[:K],
            uav_positions=uav_positions,
            measured_doppler=measured_doppler[:K],
            initial_covariance=np.eye(6),
        )

        # Batch estimate represents the state at t=0.
        # Advance to the final initialization measurement.
        ekf.predict(measurement_times[K - 1])

        # 5. Process remaining measurements sequentially
        estimated_states = []
        true_states = []

        for k in range(K, len(measurement_times)):
            dt = (
                measurement_times[k]
                - measurement_times[k - 1]
            )

            ekf.predict(dt)
            ekf.update(
                measured_doppler[k],
                uav_positions,
            )

            estimated_states.append(ekf.x.copy())

            # Ground truth at the corresponding timestamp
            true_position = np.array([
                np.interp(
                    measurement_times[k],
                    physics_times,
                    projectile_positions[:, axis],
                )
                for axis in range(3)
            ])

            true_velocity = np.array([
                np.interp(
                    measurement_times[k],
                    physics_times,
                    projectile_velocities[:, axis],
                )
                for axis in range(3)
            ])

            true_states.append(
                np.concatenate([
                    true_position,
                    true_velocity,
                ])
            )

        estimated_states = np.array(estimated_states)
        true_states = np.array(true_states)

        # 6. Evaluate estimation accuracy
        position_errors = np.linalg.norm(
            estimated_states[:, :3] - true_states[:, :3],
            axis=1,
        )

        velocity_errors = np.linalg.norm(
            estimated_states[:, 3:] - true_states[:, 3:],
            axis=1,
        )

        print("\nFinal true state:     ", true_states[-1])
        print(
            "Final estimated state:",
            np.round(estimated_states[-1], 4),
        )

        print(
            f"Final position error: "
            f"{position_errors[-1]:.6f} m"
        )
        print(
            f"Final velocity error: "
            f"{velocity_errors[-1]:.6f} m/s"
        )
        print(
            f"Mean position error:  "
            f"{position_errors.mean():.6f} m"
        )
        print(
            f"Mean velocity error:  "
            f"{velocity_errors.mean():.6f} m/s"
        )

    finally:
        world.close()


if __name__ == "__main__":
    main()
