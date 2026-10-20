
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from simulation.doppler import doppler_shift
from simulation.world import SimulationWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["hit", "miss"],
        default="hit"
    )
    args = parser.parse_args()

    world = SimulationWorld()

    # Simulation parameters
    dt = 1 / 240
    duration = 3.0
    num_steps = int(duration / dt)

    # UAV state (stationary for now)
    sensor_position = np.array([0.0, 0.0, 2.0])
    sensor_velocity = np.zeros(3)

    # Projectile trajectory
    start_position = np.array([
        -5.0,
        0.0 if args.scenario == "hit" else 1.0,
        2.0
    ])
    velocity = np.array([3.0, 0.0, 0.0])

    # Store ideal Doppler measurements
    times = []
    doppler_values = []

    collision_detected = False

    try:
        uav = world.create_uav(sensor_position.tolist())
        projectile = world.create_projectile(start_position.tolist())

        for step in range(num_steps):
            t = step * dt

            # Prescribed constant-velocity trajectory
            position = start_position + velocity * t

            p.resetBasePositionAndOrientation(
                projectile,
                position.tolist(),
                [0, 0, 0, 1]
            )

            # Update collision geometry
            p.performCollisionDetection()

            # Check for collision
            contacts = p.getClosestPoints(
                bodyA=uav,
                bodyB=projectile,
                distance=0
            )

            if contacts and not collision_detected:
                collision_detected = True
                print(f"Collision detected at t={t:.2f}s")

            # Calculate ideal Doppler shift
            # Only record measurements before collision.
            if not collision_detected:
                shift = doppler_shift(
                    sensor_position=sensor_position,
                    sensor_velocity=sensor_velocity,
                    object_position=position,
                    object_velocity=velocity,
                )

                times.append(t)
                doppler_values.append(shift)

            world.step()
            time.sleep(dt)

        if not collision_detected:
            print("No collision detected.")

    finally:
        world.close()

    # Plot ideal Doppler measurements
    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        doppler_values,
        label=f"{args.scenario.upper()} scenario"
    )

    plt.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Doppler shift (Hz)")
    plt.title(
        f"Ideal Doppler Shift — {args.scenario.upper()} Scenario"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
