
"""
Validate causal Doppler extraction, batch estimation, and EKF.

Run:
    python -m experiments.validate_online_estimation
"""

import numpy as np
import matplotlib.pyplot as plt

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    generate_trajectory_echo,
    estimate_doppler,
)
from estimation.batch_estimator import (
    DEFAULT_MEASUREMENT_LAG,
    predict_doppler,
)
from estimation.ekf import ExtendedKalmanFilter


UAV_POSITIONS = np.array([
    [0., 0., 0.],
    [2., 0., 0.],
    [0., 2., 0.],
    [0., 0., 2.],
])

DURATION = 0.5
INIT_WINDOW = 0.2
DIAGNOSTIC_UPDATES = 5

TRAJECTORIES = [
    ("Original", [3., 3., 3.], [-2., -1.5, -1.]),
    ("Faster", [3., 3., 3.], [-4., -3., -2.]),
    ("Diagonal", [4., 2.5, 3.5], [-3., -1., -2.]),
    ("Different direction", [-2., 3., 3.], [2., -2., -1.]),
    ("Farther away", [5., 4., 4.], [-3., -2., -2.]),
]


def synchronize_measurements(all_measurements):
    maps = []

    for times, doppler in all_measurements:
        if len(times) != len(doppler):
            raise RuntimeError("Timestamp/Doppler length mismatch")

        rounded = np.round(
            np.asarray(times, dtype=float), 6
        )
        maps.append(dict(zip(rounded, doppler)))

    common = set(maps[0])

    for measurement_map in maps[1:]:
        common.intersection_update(measurement_map)

    times = np.array(sorted(common), dtype=float)

    if not len(times):
        raise RuntimeError("No synchronized measurements")

    measurements = np.column_stack([
        [measurement_map[t] for t in times]
        for measurement_map in maps
    ])

    if not np.all(np.isfinite(measurements)):
        raise RuntimeError("Non-finite Doppler measurements")

    print("Measurements per UAV:", [
        len(m) for m in maps
    ])
    print("Synchronized measurements:", len(times))

    return times, measurements


def generate_measurements(initial_position, initial_velocity):
    world = SimulationWorld(gui=False, gravity=False)

    try:
        for position in UAV_POSITIONS:
            world.create_uav(position)

        projectile_id = world.create_projectile(
            position=np.asarray(initial_position, dtype=float),
            velocity=np.asarray(initial_velocity, dtype=float),
        )

        physics_times = []
        positions = []
        velocities = []

        num_steps = int(DURATION / world.dt)

        for step in range(num_steps + 1):
            t = step * world.dt

            position, velocity = world.get_projectile_state(
                projectile_id
            )

            physics_times.append(t)
            positions.append(position)
            velocities.append(velocity)

            if step < num_steps:
                world.step()

        physics_times = np.asarray(physics_times)
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)

        audio_times = np.arange(
            0, DURATION, 1 / SAMPLE_RATE
        )

        all_measurements = []

        for sensor_position in UAV_POSITIONS:
            echo = generate_trajectory_echo(
                audio_times,
                physics_times,
                positions,
                sensor_position,
            )

            times, doppler = estimate_doppler(
                audio_times,
                echo,
                window_duration=0.04,
                hop_duration=0.02,
            )

            all_measurements.append((times, doppler))

        times, measured = synchronize_measurements(
            all_measurements
        )

        true_positions = np.column_stack([
            np.interp(
                times,
                physics_times,
                positions[:, axis],
            )
            for axis in range(3)
        ])

        true_velocities = np.column_stack([
            np.interp(
                times,
                physics_times,
                velocities[:, axis],
            )
            for axis in range(3)
        ])

        truth = np.column_stack([
            true_positions,
            true_velocities,
        ])

        return times, measured, truth

    finally:
        world.close()


def compare_measurement_models(times, measured, truth, name):
    old_errors = []
    new_errors = []

    for k in range(len(times)):
        # State at the actual measurement availability time.
        true_state = truth[k]

        old = predict_doppler(
            true_state,
            [0.0],
            UAV_POSITIONS,
            delay_aware=False,
            measurement_lag=0.0,
        ).ravel()

        new = predict_doppler(
            true_state,
            [0.0],
            UAV_POSITIONS,
            delay_aware=True,
            measurement_lag=DEFAULT_MEASUREMENT_LAG,
        ).ravel()

        old_errors.append(measured[k] - old)
        new_errors.append(measured[k] - new)

    old_errors = np.asarray(old_errors)
    new_errors = np.asarray(new_errors)

    old_rmse = np.sqrt(np.mean(old_errors**2))
    new_rmse = np.sqrt(np.mean(new_errors**2))

    print(
        f"Ground-truth Doppler RMSE (old model): "
        f"{old_rmse:.4f} Hz"
    )
    print(
        f"Ground-truth Doppler RMSE (causal model): "
        f"{new_rmse:.4f} Hz"
    )

    if name == "Original":
        print("\nFinal five ground-truth Doppler comparisons:")
        print(
            "Time    UAV    Old error (Hz)    "
            "Causal error (Hz)"
        )

        for k in range(
            max(0, len(times) - DIAGNOSTIC_UPDATES),
            len(times),
        ):
            for i in range(len(UAV_POSITIONS)):
                print(
                    f"{times[k]:.3f}     {i+1}       "
                    f"{old_errors[k, i]:+9.4f}          "
                    f"{new_errors[k, i]:+9.4f}"
                )

    return old_rmse, new_rmse


def evaluate_trajectory(initial_position, initial_velocity, name):
    times, measured, truth = generate_measurements(
        initial_position,
        initial_velocity,
    )

    old_rmse, new_rmse = compare_measurement_models(
        times, measured, truth, name
    )

    K = np.searchsorted(
        times, INIT_WINDOW, side="right"
    )

    if K < 2 or K >= len(times):
        raise RuntimeError(
            "Insufficient measurements for initialization"
        )

    print(f"Initialization measurements: {K}")
    print(f"EKF update measurements: {len(times) - K}")

    ekf = ExtendedKalmanFilter()

    ekf.initialize(
        times=times[:K],
        uav_positions=UAV_POSITIONS,
        measured_doppler=measured[:K],
        initial_covariance=np.eye(6),
    )

    batch_state = ekf.x.copy()

    # Advance from t=0 to last initialization timestamp.
    ekf.predict(times[K - 1])

    ekf_states = []
    batch_states = []

    for k in range(K, len(times)):
        dt = times[k] - times[k - 1]

        ekf.predict(dt)

        diagnostic_step = (
            name == "Original"
            and k >= len(times) - DIAGNOSTIC_UPDATES
        )

        if diagnostic_step:
            before = ekf.x.copy()

            predicted = predict_doppler(
                before,
                [0.0],
                UAV_POSITIONS,
                delay_aware=True,
                measurement_lag=DEFAULT_MEASUREMENT_LAG,
            ).ravel()

            innovation = measured[k] - predicted

            before_pos_error = np.linalg.norm(
                before[:3] - truth[k, :3]
            )
            before_vel_error = np.linalg.norm(
                before[3:] - truth[k, 3:]
            )

        ekf.update(measured[k], UAV_POSITIONS)

        if diagnostic_step:
            after_pos_error = np.linalg.norm(
                ekf.x[:3] - truth[k, :3]
            )
            after_vel_error = np.linalg.norm(
                ekf.x[3:] - truth[k, 3:]
            )

            print(
                f"Update {times[k]:.3f}s: "
                f"innovation={np.linalg.norm(innovation):.4f} Hz; "
                f"position {before_pos_error:.4f}"
                f"->{after_pos_error:.4f} m; "
                f"velocity {before_vel_error:.4f}"
                f"->{after_vel_error:.4f} m/s"
            )

        ekf_states.append(ekf.x.copy())

        baseline = batch_state.copy()
        baseline[:3] += baseline[3:] * times[k]
        batch_states.append(baseline)

    ekf_states = np.asarray(ekf_states)
    batch_states = np.asarray(batch_states)

    actual = truth[K:]

    result = {
        "times": times[K:],
        "ekf_position_error": np.linalg.norm(
            ekf_states[:, :3] - actual[:, :3],
            axis=1,
        ),
        "batch_position_error": np.linalg.norm(
            batch_states[:, :3] - actual[:, :3],
            axis=1,
        ),
        "ekf_velocity_error": np.linalg.norm(
            ekf_states[:, 3:] - actual[:, 3:],
            axis=1,
        ),
        "batch_velocity_error": np.linalg.norm(
            batch_states[:, 3:] - actual[:, 3:],
            axis=1,
        ),
        "old_doppler_rmse": old_rmse,
        "new_doppler_rmse": new_rmse,
    }

    for metric, unit in [
        ("position", "m"),
        ("velocity", "m/s"),
    ]:
        batch_rmse = np.sqrt(np.mean(
            result[f"batch_{metric}_error"]**2
        ))
        ekf_rmse = np.sqrt(np.mean(
            result[f"ekf_{metric}_error"]**2
        ))

        improvement = (
            100 * (1 - ekf_rmse / batch_rmse)
            if batch_rmse > 0 else 0.0
        )

        result[f"batch_{metric}_rmse"] = batch_rmse
        result[f"ekf_{metric}_rmse"] = ekf_rmse

        print(f"Batch {metric} RMSE: {batch_rmse:.4f} {unit}")
        print(f"EKF {metric} RMSE:   {ekf_rmse:.4f} {unit}")
        print(f"Improvement: {improvement:+.2f}%")

    return result


def plot_errors(result):
    for metric, unit, filename in [
        ("position", "m", "ekf_position_error.png"),
        ("velocity", "m/s", "ekf_velocity_error.png"),
    ]:
        plt.figure(figsize=(9, 5))

        plt.plot(
            result["times"],
            result[f"batch_{metric}_error"],
            "--",
            label="Batch-only",
        )

        plt.plot(
            result["times"],
            result[f"ekf_{metric}_error"],
            label="EKF",
        )

        plt.xlabel("Measurement availability time (s)")
        plt.ylabel(f"{metric.capitalize()} error ({unit})")
        plt.title(
            f"Causal Doppler EKF: "
            f"{metric.capitalize()} Error Over Time"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)

        print("Saved:", filename)

    plt.show()


def main():
    results = []

    for name, position, velocity in TRAJECTORIES:
        print("\n" + "=" * 60)
        print("Trajectory:", name)
        print("=" * 60)

        try:
            result = evaluate_trajectory(
                np.asarray(position, dtype=float),
                np.asarray(velocity, dtype=float),
                name,
            )

            results.append((name, result))

        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            print("Experiment failed:", exc)

    if not results:
        raise RuntimeError("All experiments failed")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results:
        pos_improvement = 100 * (
            1 - result["ekf_position_rmse"]
            / max(result["batch_position_rmse"], 1e-12)
        )

        vel_improvement = 100 * (
            1 - result["ekf_velocity_rmse"]
            / max(result["batch_velocity_rmse"], 1e-12)
        )

        print(
            f"{name:22s} "
            f"Position: {pos_improvement:+7.2f}%  "
            f"Velocity: {vel_improvement:+7.2f}%"
        )

    for name, result in results:
        if name == "Original":
            plot_errors(result)
            break


if __name__ == "__main__":
    main()
