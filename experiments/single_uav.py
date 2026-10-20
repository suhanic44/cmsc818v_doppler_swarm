
import argparse
import time

import numpy as np
import pybullet as p

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

    try:
        uav = world.create_uav([0, 0, 2])

        # Define projectile trajectory
        start_position = np.array([
            -5.0,
            0.0 if args.scenario == "hit" else 1.0,
            2.0
        ])

        velocity = np.array([3.0, 0.0, 0.0])

        projectile = world.create_projectile(
            start_position.tolist()
        )

        dt = 1 / 240
        collision_detected = False

        for step in range(720):
            t = step * dt

            # Prescribed constant-velocity trajectory
            position = start_position + velocity * t

            p.resetBasePositionAndOrientation(
                projectile,
                position.tolist(),
                [0, 0, 0, 1]
            )

            # Refresh PyBullet's collision geometry
            p.performCollisionDetection()

            # Detect geometric overlap
            contacts = p.getClosestPoints(
                bodyA=uav,
                bodyB=projectile,
                distance=0
            )

            if contacts and not collision_detected:
                collision_detected = True
                print(f"Collision detected at t={t:.2f}s")

            world.step()
            time.sleep(dt)

        if not collision_detected:
            print("No collision detected.")

    finally:
        world.close()


if __name__ == "__main__":
    main()
