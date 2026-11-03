
"""
Fully online acoustic trajectory estimation using PyBullet.

Acoustic samples are generated incrementally as physics advances.
Doppler measurements are passed directly to the estimator.

The first ~200 ms of measurements initialize the state using batch
least squares. Subsequent measurements update the EKF online.

Run:
    python -m experiments.online_estimation
    python -m experiments.online_estimation --validate
"""

import argparse
import time

import numpy as np

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    StreamingAcousticSensor,
    generate_trajectory_echo,
    estimate_doppler,
)
from estimation.ekf import ExtendedKalmanFilter


UAV_POSITIONS = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
])

INITIAL_POSITION = np.array([3.0, 3.0, 3.0])
INITIAL_VELOCITY = np.array([-2.0, -1.5, -1.0])

DURATION = 0.5
INIT_WINDOW = 0.2
N_UAVS = len(UAV_POSITIONS)


def synchronize_measurements(all_measurements):
    """
    Keep timestamps for which every UAV has a valid Doppler measurement.

    Used for the offline regression baseline.
    """
    measurement_maps = []

    for times, doppler in all_measurements:
        if len(times) != len(doppler):
            raise RuntimeError(
                "Timestamp/Doppler length mismatch."
            )

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
        sorted(common_times),
        dtype=float,
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
        raise RuntimeError(
            "Non-finite Doppler measurements."
        )

    return measurement_times, measured_doppler


def interpolate_true_state(
    measurement_time,
    physics_times,
    projectile_positions,
    projectile_velocities,
):
    """
    Interpolate the projectile's true position and velocity
    at a measurement timestamp.
    """
    position = np.array([
        np.interp(
            measurement_time,
            physics_times,
            projectile_positions[:, axis],
        )
        for axis in range(3)
    ])

    velocity = np.array([
        np.interp(
            measurement_time,
            physics_times,
            projectile_velocities[:, axis],
        )
        for axis in range(3)
    ])

    return np.concatenate([position, velocity])


class OnlineEstimationPipeline:
    """
    Receive Doppler measurements incrementally from multiple UAVs.

    Initialize the state using measurements from the first ~200 ms.
    Then perform EKF prediction and correction whenever a new
    synchronized measurement becomes available.
    """

    def __init__(
        self,
        uav_positions,
        init_window=INIT_WINDOW,
    ):
        self.uav_positions = np.asarray(
            uav_positions,
            dtype=float,
        )

        self.n_uavs = len(self.uav_positions)
        self.init_window = float(init_window)

        # Measurements waiting for all UAVs to contribute.
        self.pending = {}

        # Complete synchronized measurements.
        self.measurement_times = []
        self.measured_doppler = []

        # Trajectory estimator.
        self.ekf = ExtendedKalmanFilter()
        self.initialized = False

        # Estimated states after EKF updates.
        self.estimated_times = []
        self.estimated_states = []

        # Processing latency measurements.
        self.initialization_runtime = None
        self.update_runtimes = []

        self.initialization_measurements = 0

    def add_measurements(
        self,
        uav_index,
        times,
        doppler,
    ):
        """
        Add newly generated measurements from one UAV.

        Measurements are matched using rounded timestamps.
        """
        if len(times) != len(doppler):
            raise RuntimeError(
                "Timestamp/Doppler length mismatch."
            )

        for measurement_time, value in zip(times, doppler):
            key = round(float(measurement_time), 6)

            if key not in self.pending:
                self.pending[key] = {}

            if uav_index in self.pending[key]:
                raise RuntimeError(
                    "Duplicate UAV measurement at timestamp."
                )

            self.pending[key][uav_index] = float(value)

    def process_ready(self, current_time):
        """
        Process complete measurement sets in chronological order.

        Initialize from measurements through 200 ms, then use
        the first subsequent measurement as an EKF update.

        current_time is the current simulated physics time.
        """
        completed = sorted(
            t for t, values in self.pending.items()
            if (
                len(values) == self.n_uavs
                and t <= current_time + 1e-6
            )
        )

        for measurement_time in completed:
            values = self.pending.pop(measurement_time)

            z = np.array([
                values[i] for i in range(self.n_uavs)
            ])

            self.measurement_times.append(measurement_time)
            self.measured_doppler.append(z)

            # Batch initialization.
            if not self.initialized:
                if measurement_time <= self.init_window:
                    continue

                times = np.asarray(
                    self.measurement_times
                )

                measurements = np.asarray(
                    self.measured_doppler
                )

                K = np.searchsorted(
                    times,
                    self.init_window,
                    side="right",
                )

                if K < 2:
                    raise RuntimeError(
                        "Insufficient initialization measurements."
                    )

                start = time.perf_counter()

                self.ekf.initialize(
                    times=times[:K],
                    uav_positions=self.uav_positions,
                    measured_doppler=measurements[:K],
                    initial_covariance=np.eye(6),
                )

                self.initialization_runtime = (
                    time.perf_counter() - start
                )

                self.initialization_measurements = K

                # The batch state is referenced to t=0.
                # Match the offline baseline by predicting
                # to the last initialization measurement.
                self.ekf.predict(times[K - 1])

                self.initialized = True

            # EKF prediction and correction.
            dt = measurement_time - self.ekf.t

            if dt < -1e-9:
                raise RuntimeError(
                    "Received an out-of-order EKF measurement."
                )

            start = time.perf_counter()

            self.ekf.predict(max(0.0, dt))

            self.ekf.update(
                measured_doppler=z,
                uav_positions=self.uav_positions,
            )

            elapsed = time.perf_counter() - start

            self.update_runtimes.append(elapsed)

            self.estimated_times.append(
                measurement_time
            )

            self.estimated_states.append(
                self.ekf.x.copy()
            )

    def results(self):
        return {
            "measurement_times": np.asarray(
                self.measurement_times
            ),
            "measured_doppler": np.asarray(
                self.measured_doppler
            ),
            "estimated_times": np.asarray(
                self.estimated_times
            ),
            "estimated_states": np.asarray(
                self.estimated_states
            ),
        }


def run_streaming():
    """
    Run PyBullet, acoustic sensing, and EKF estimation
    together in one simulation loop.
    """
    world = SimulationWorld(
        gui=False,
        gravity=False,
    )

    sensors = [
        StreamingAcousticSensor(
            sensor_position=position,
            duration=DURATION,
        )
        for position in UAV_POSITIONS
    ]

    pipeline = OnlineEstimationPipeline(
        UAV_POSITIONS
    )

    physics_times = []
    projectile_positions = []
    projectile_velocities = []

    try:
        for position in UAV_POSITIONS:
            world.create_uav(position)

        projectile_id = world.create_projectile(
            position=INITIAL_POSITION,
            velocity=INITIAL_VELOCITY,
        )

        num_steps = int(DURATION / world.dt)

        for step in range(num_steps + 1):
            simulation_time = step * world.dt

            # Read the projectile's current physical state.
            position, velocity = world.get_projectile_state(
                projectile_id
            )

            physics_times.append(simulation_time)
            projectile_positions.append(position)
            projectile_velocities.append(velocity)

            # Generate only acoustic samples available
            # at the current simulation time.
            for i, sensor in enumerate(sensors):
                times, doppler = sensor.advance(
                    simulation_time,
                    position,
                )

                pipeline.add_measurements(
                    i,
                    times,
                    doppler,
                )

            # Process newly available measurements before
            # advancing the simulation again.
            pipeline.process_ready(simulation_time)

            if step < num_steps:
                world.step()

    finally:
        world.close()

    result = pipeline.results()

    result["physics_times"] = np.asarray(
        physics_times
    )

    result["projectile_positions"] = np.asarray(
        projectile_positions
    )

    result["projectile_velocities"] = np.asarray(
        projectile_velocities
    )

    result["initialization_runtime"] = (
        pipeline.initialization_runtime
    )

    result["update_runtimes"] = np.asarray(
        pipeline.update_runtimes
    )

    result["initialization_measurements"] = (
        pipeline.initialization_measurements
    )

    result["sensor_measurement_counts"] = [
        len(sensor.measurement_times)
        for sensor in sensors
    ]

    return result


def run_offline_reference(streaming_result):
    """
    Reproduce the original offline estimation pipeline
    using the same simulated projectile trajectory.
    """
    physics_times = streaming_result["physics_times"]
    positions = streaming_result["projectile_positions"]

    audio_times = np.arange(
        0,
        DURATION,
        1.0 / SAMPLE_RATE,
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

        all_measurements.append(
            (times, doppler)
        )

    measurement_times, measured_doppler = (
        synchronize_measurements(all_measurements)
    )

    K = np.searchsorted(
        measurement_times,
        INIT_WINDOW,
        side="right",
    )

    if K < 2 or K >= len(measurement_times):
        raise RuntimeError(
            "Insufficient offline reference measurements."
        )

    ekf = ExtendedKalmanFilter()

    ekf.initialize(
        times=measurement_times[:K],
        uav_positions=UAV_POSITIONS,
        measured_doppler=measured_doppler[:K],
        initial_covariance=np.eye(6),
    )

    ekf.predict(measurement_times[K - 1])

    estimated_times = []
    estimated_states = []

    for k in range(K, len(measurement_times)):
        dt = (
            measurement_times[k]
            - measurement_times[k - 1]
        )

        ekf.predict(dt)

        ekf.update(
            measured_doppler[k],
            UAV_POSITIONS,
        )

        estimated_times.append(
            measurement_times[k]
        )

        estimated_states.append(
            ekf.x.copy()
        )

    return {
        "measurement_times": measurement_times,
        "measured_doppler": measured_doppler,
        "estimated_times": np.asarray(
            estimated_times
        ),
        "estimated_states": np.asarray(
            estimated_states
        ),
    }


def print_accuracy(result):
    """
    Compare estimated trajectory states against
    the simulated projectile's ground-truth states.
    """
    estimated_times = result["estimated_times"]
    estimated_states = result["estimated_states"]

    if len(estimated_times) == 0:
        raise RuntimeError(
            "No EKF updates were performed."
        )

    true_states = np.array([
        interpolate_true_state(
            t,
            result["physics_times"],
            result["projectile_positions"],
            result["projectile_velocities"],
        )
        for t in estimated_times
    ])

    position_errors = np.linalg.norm(
        estimated_states[:, :3]
        - true_states[:, :3],
        axis=1,
    )

    velocity_errors = np.linalg.norm(
        estimated_states[:, 3:]
        - true_states[:, 3:],
        axis=1,
    )

    position_rmse = np.sqrt(
        np.mean(position_errors ** 2)
    )

    velocity_rmse = np.sqrt(
        np.mean(velocity_errors ** 2)
    )

    print(
        "\nFinal true state:     ",
        true_states[-1],
    )

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

    print(
        f"Position RMSE:        "
        f"{position_rmse:.6f} m"
    )

    print(
        f"Velocity RMSE:        "
        f"{velocity_rmse:.6f} m/s"
    )


def validate_against_offline(streaming):
    """
    Verify that streaming sensing and EKF estimation
    reproduce the existing offline baseline.
    """
    print("\n" + "=" * 60)
    print("OFFLINE REGRESSION VALIDATION")
    print("=" * 60)

    offline = run_offline_reference(streaming)

    # Verify synchronized measurement timestamps.
    measurement_times_match = np.array_equal(
        streaming["measurement_times"],
        offline["measurement_times"],
    )

    print(
        "Measurement timestamps match:",
        measurement_times_match,
    )

    if not measurement_times_match:
        raise AssertionError(
            "Streaming/offline measurement timestamps differ."
        )

    # Compare Doppler measurements.
    doppler_max_error = np.max(
        np.abs(
            streaming["measured_doppler"]
            - offline["measured_doppler"]
        )
    )

    print(
        f"Maximum Doppler difference: "
        f"{doppler_max_error:.9f} Hz"
    )

    if doppler_max_error >= 1e-3:
        raise AssertionError(
            "Streaming Doppler differs from offline baseline."
        )

    # Compare the exact EKF update timestamps.
    estimate_times_match = np.array_equal(
        streaming["estimated_times"],
        offline["estimated_times"],
    )

    print(
        "EKF update timestamps match:",
        estimate_times_match,
    )

    if not estimate_times_match:
        raise AssertionError(
            "Streaming/offline EKF update timestamps differ."
        )

    # Compare all estimated states.
    differences = (
        streaming["estimated_states"]
        - offline["estimated_states"]
    )

    position_differences = np.linalg.norm(
        differences[:, :3],
        axis=1,
    )

    velocity_differences = np.linalg.norm(
        differences[:, 3:],
        axis=1,
    )

    max_position_difference = np.max(
        position_differences
    )

    max_velocity_difference = np.max(
        velocity_differences
    )

    print(
        "Maximum position-estimate difference:",
        f"{max_position_difference:.9f} m",
    )

    print(
        "Maximum velocity-estimate difference:",
        f"{max_velocity_difference:.9f} m/s",
    )

    if (
        max_position_difference >= 1e-3
        or max_velocity_difference >= 1e-3
    ):
        raise AssertionError(
            "Streaming EKF differs from offline baseline."
        )

    print("\nRegression validation: PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Compare streaming trajectory estimation "
            "against the offline baseline."
        ),
    )

    args = parser.parse_args()

    result = run_streaming()

    print("\n" + "=" * 60)
    print("FULLY ONLINE TRAJECTORY ESTIMATION")
    print("=" * 60)

    print(
        "Measurements per UAV:",
        result["sensor_measurement_counts"],
    )

    print(
        "Synchronized measurements:",
        len(result["measurement_times"]),
    )

    print(
        "Initialization measurements:",
        result["initialization_measurements"],
    )

    print(
        "EKF update measurements:",
        len(result["estimated_times"]),
    )

    print_accuracy(result)

    # Report processing latency.
    print("\n" + "=" * 60)
    print("PROCESSING LATENCY")
    print("=" * 60)

    initialization_runtime = (
        result["initialization_runtime"]
    )

    update_runtimes = (
        result["update_runtimes"]
    )

    if initialization_runtime is None:
        raise RuntimeError(
            "Batch initialization was not performed."
        )

    if len(update_runtimes) == 0:
        raise RuntimeError(
            "No EKF update runtimes were recorded."
        )

    initialization_ms = (
        initialization_runtime * 1000
    )

    update_ms = update_runtimes * 1000

    print(
        f"Batch initialization: "
        f"{initialization_ms:.3f} ms"
    )

    print(
        f"EKF update mean:      "
        f"{update_ms.mean():.3f} ms"
    )

    print(
        f"EKF update p95:       "
        f"{np.percentile(update_ms, 95):.3f} ms"
    )

    print(
        f"EKF update maximum:   "
        f"{update_ms.max():.3f} ms"
    )

    if args.validate:
        validate_against_offline(result)


if __name__ == "__main__":
    main()
