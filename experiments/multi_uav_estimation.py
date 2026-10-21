
import numpy as np
import matplotlib.pyplot as plt

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    generate_trajectory_echo,
    estimate_doppler,
)
from estimation.batch_estimator import (
    estimate_initial_state,
    predict_doppler,
)


def main():
    np.random.seed(42)

    # --------------------------------------------------
    # 1. Create PyBullet world
    # --------------------------------------------------

    world = SimulationWorld(gui=True, gravity=False)

    try:
        uav_positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])

        for position in uav_positions:
            world.create_uav(position)

        initial_position = np.array([3.0, 3.0, 3.0])
        initial_velocity = np.array([-2.0, -1.5, -1.0])

        true_state = np.concatenate([
            initial_position,
            initial_velocity
        ])

        projectile_id = world.create_projectile(
            position=initial_position,
            velocity=initial_velocity,
            radius=0.1,
            mass=0.1
        )

        # --------------------------------------------------
        # 2. Run PyBullet and record projectile trajectory
        # --------------------------------------------------

        duration = 0.5
        num_steps = int(duration / world.dt)

        physics_times = []
        projectile_positions = []

        for step in range(num_steps + 1):
            t = step * world.dt

            position, velocity = world.get_projectile_state(
                projectile_id
            )

            physics_times.append(t)
            projectile_positions.append(position)

            if step < num_steps:
                world.step()

        physics_times = np.array(physics_times)
        projectile_positions = np.array(projectile_positions)

        expected_positions = (
            initial_position[None, :]
            + physics_times[:, None] * initial_velocity[None, :]
        )

        trajectory_error = np.linalg.norm(
            projectile_positions - expected_positions,
            axis=1
        )

        print(
            "Maximum PyBullet trajectory deviation:",
            np.max(trajectory_error),
            "m"
        )

        print(
            "PyBullet final position:",
            projectile_positions[-1]
        )

        print(
            "Expected final position:",
            expected_positions[-1]
        )

        print("PyBullet simulation complete.")
        print("Recorded physics steps:", len(physics_times))

        # --------------------------------------------------
        # 3. Generate synthetic microphone signals
        # --------------------------------------------------

        audio_times = np.arange(
            0.0,
            duration,
            1.0 / SAMPLE_RATE
        )

        all_measurements = []

        for i, sensor_position in enumerate(uav_positions):
            echo = generate_trajectory_echo(
                audio_times,
                physics_times,
                projectile_positions,
                sensor_position
            )

            measurement_times, doppler = estimate_doppler(
                audio_times,
                echo,
                window_duration=0.04,
                hop_duration=0.02
            )

            print(
                f"UAV {i}: {len(doppler)} Doppler measurements"
            )

            all_measurements.append((
                measurement_times,
                doppler
            ))

        # --------------------------------------------------
        # 4. Combine measurements from all UAVs
        # --------------------------------------------------

        reference_times = all_measurements[0][0]

        if len(reference_times) == 0:
            raise RuntimeError(
                "No valid Doppler measurements."
            )

        for times_i, _ in all_measurements:
            if not np.array_equal(times_i, reference_times):
                raise RuntimeError(
                    "UAV measurement timestamps do not align."
                )

        measured_doppler = np.column_stack([
            values for _, values in all_measurements
        ])

        # --------------------------------------------------
        # 5. Compare microphone Doppler to model prediction
        # --------------------------------------------------

        ideal_doppler = predict_doppler(
            true_state,
            reference_times,
            uav_positions
        )

        print("\nDoppler measurement comparison:")

        for i in range(len(uav_positions)):
            error = (
                measured_doppler[:, i]
                - ideal_doppler[:, i]
            )

            print(
                f"UAV {i}: "
                f"mean error = {np.mean(error):.3f} Hz, "
                f"RMSE = {np.sqrt(np.mean(error**2)):.3f} Hz"
            )

        # --------------------------------------------------
        # 6. Estimate projectile initial position and velocity
        # --------------------------------------------------

        initial_guess = np.array([
            4.0, 4.0, 4.0,
            -1.0, -1.0, -1.0
        ])

        result = estimate_initial_state(
            reference_times,
            uav_positions,
            measured_doppler,
            initial_guess
        )

        estimated_state = result.x

        print("\nGround truth:", true_state)
        print("Estimated:   ", np.round(estimated_state, 3))
        print("Optimizer success:", result.success)

        position_error = np.linalg.norm(
            estimated_state[:3] - true_state[:3]
        )

        velocity_error = np.linalg.norm(
            estimated_state[3:] - true_state[3:]
        )

        print(f"Position error: {position_error:.4f} m")
        print(f"Velocity error: {velocity_error:.4f} m/s")

        # --------------------------------------------------
        # 7. Plot PyBullet trajectory vs estimated trajectory
        # --------------------------------------------------

        estimated_trajectory = (
            estimated_state[:3][None, :]
            + physics_times[:, None]
            * estimated_state[3:][None, :]
        )

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            *projectile_positions.T,
            label="PyBullet ground truth",
            linewidth=2
        )

        ax.plot(
            *estimated_trajectory.T,
            "--",
            label="Estimated trajectory",
            linewidth=2
        )

        ax.scatter(
            *uav_positions.T,
            marker="^",
            s=90,
            label="UAVs"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.set_title(
            "Multi-UAV Acoustic Trajectory Estimation (PyBullet)"
        )

        ax.legend()
        plt.tight_layout()
        plt.show()

    finally:
        world.close()


if __name__ == "__main__":
    main()
