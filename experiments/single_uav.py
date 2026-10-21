
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from simulation.doppler import doppler_shift
from simulation.world import SimulationWorld
from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    generate_trajectory_echo,
    estimate_doppler
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["hit", "miss"],
        default="hit"
    )
    args = parser.parse_args()

    world = SimulationWorld()

    dt = 1 / 240
    duration = 3.0
    num_steps = int(duration / dt)

    sensor_position = np.array([0.0, 0.0, 2.0])
    sensor_velocity = np.zeros(3)

    start_position = np.array([
        -5.0,
        0.0 if args.scenario == "hit" else 1.0,
        2.0
    ])

    velocity = np.array([3.0, 0.0, 0.0])

    physics_times = []
    projectile_positions = []
    ideal_doppler = []

    collision_detected = False

    try:
        uav = world.create_uav(
            sensor_position.tolist()
        )
        projectile = world.create_projectile(
            start_position.tolist()
        )

        for step in range(num_steps):
            t = step * dt

            position = start_position + velocity * t

            p.resetBasePositionAndOrientation(
                projectile,
                position.tolist(),
                [0, 0, 0, 1]
            )

            p.performCollisionDetection()

            contacts = p.getClosestPoints(
                bodyA=uav,
                bodyB=projectile,
                distance=0
            )

            if contacts:
                collision_detected = True
                print(f"Collision detected at t={t:.2f}s")
                break

            # Store the actual simulated trajectory.
            physics_times.append(t)
            projectile_positions.append(position.copy())

            # Ground-truth reference for validation only.
            ideal_doppler.append(
                doppler_shift(
                    sensor_position=sensor_position,
                    sensor_velocity=sensor_velocity,
                    object_position=position,
                    object_velocity=velocity
                )
            )

            world.step()
            time.sleep(dt)

    finally:
        world.close()

    if not collision_detected:
        print("No collision detected.")

    physics_times = np.asarray(physics_times)
    projectile_positions = np.asarray(
        projectile_positions
    )
    ideal_doppler = np.asarray(ideal_doppler)

    # Generate high-frequency microphone sampling times.
    # Stop at the final valid physics sample.
    audio_times = np.arange(
        0,
        physics_times[-1],
        1 / SAMPLE_RATE
    )

    # Generate the acoustic echo from the trajectory.
    echo = generate_trajectory_echo(
        audio_times=audio_times,
        physics_times=physics_times,
        projectile_positions=projectile_positions,
        sensor_position=sensor_position
    )

    # Recover Doppler shift from the audio.
    measured_times, measured_doppler = estimate_doppler(
        audio_times,
        echo
    )

    # Plot both measurements.
    plt.figure(figsize=(10, 5))

    plt.plot(
        physics_times,
        ideal_doppler,
        label="Ideal Doppler (ground truth)",
        linewidth=2
    )

    plt.plot(
        measured_times,
        measured_doppler,
        label="Measured Doppler (microphone)",
        linestyle="--",
        linewidth=2
    )

    plt.axhline(
        0,
        color="gray",
        linestyle=":",
        linewidth=1
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Doppler shift (Hz)")
    plt.title(
        f"Ideal vs. Measured Doppler — {args.scenario.upper()}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
