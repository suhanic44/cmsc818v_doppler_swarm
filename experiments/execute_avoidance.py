
"""
Execute SQP-planned UAV avoidance in SimulationWorld.

Run:
    python -m experiments.execute_avoidance
    python -m experiments.execute_avoidance --gui --realtime
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p

from simulation.world import SimulationWorld
from planning.avoidance_planner import plan_avoidance


UAV_POSITIONS = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
])

PROJECTILE_STATE = np.array([
    -1.0, 0.0, 0.0,
    4.0, 0.0, 0.0,
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

PREDICTION_HORIZON = 1.0
MANEUVER_DURATION = 0.15
MAX_UAV_SPEED = 3.0


@dataclass
class ExecutionResult:
    planning_success: bool
    collision_free: bool
    total_displacement: float
    number_moved: int
    minimum_projectile_separation: float
    minimum_uav_separation: float
    predicted_projectile_separation: float
    predicted_uav_separation: float
    final_uav_positions: np.ndarray
    target_positions: np.ndarray
    message: str


def minimum_separations(uav_positions, projectile_position):
    """Calculate center-to-center separation distances."""

    projectile_distances = np.linalg.norm(
        uav_positions - projectile_position,
        axis=1,
    )

    minimum_projectile = float(
        np.min(projectile_distances)
    )

    minimum_uav = float("inf")

    for i in range(len(uav_positions)):
        for j in range(i + 1, len(uav_positions)):
            distance = np.linalg.norm(
                uav_positions[i] - uav_positions[j]
            )
            minimum_uav = min(minimum_uav, distance)

    return minimum_projectile, minimum_uav


def run_avoidance(
    state=PROJECTILE_STATE,
    uav_positions=UAV_POSITIONS,
    gui=False,
    realtime=False,
    prediction_horizon=PREDICTION_HORIZON,
    maneuver_duration=MANEUVER_DURATION,
    max_uav_speed=MAX_UAV_SPEED,
    verbose=True,
):
    state = np.asarray(state, dtype=float)
    uav_positions = np.asarray(uav_positions, dtype=float)

    if state.shape != (6,):
        raise ValueError("State must have shape (6,)")

    if (
        uav_positions.ndim != 2
        or uav_positions.shape[1] != 3
        or len(uav_positions) == 0
    ):
        raise ValueError("UAV positions must have shape (N, 3)")

    # Plan using the actual projectile position and velocity.
    plan = plan_avoidance(
        state=state,
        uav_positions=uav_positions,
        prediction_horizon=prediction_horizon,
        maneuver_duration=maneuver_duration,
        max_uav_speed=max_uav_speed,
        uav_radius=UAV_RADIUS,
        projectile_radius=PROJECTILE_RADIUS,
        safety_margin=SAFETY_MARGIN,
        uav_separation_margin=UAV_SEPARATION_MARGIN,
    )

    if not plan.success:
        if verbose:
            print("Planning failed:", plan.message)

        return ExecutionResult(
            planning_success=False,
            collision_free=False,
            total_displacement=0.0,
            number_moved=0,
            minimum_projectile_separation=float("nan"),
            minimum_uav_separation=float("nan"),
            predicted_projectile_separation=float("nan"),
            predicted_uav_separation=float("nan"),
            final_uav_positions=uav_positions.copy(),
            target_positions=uav_positions.copy(),
            message=plan.message,
        )

    world = SimulationWorld(gui=gui, gravity=False)

    try:
        uav_ids = [
            world.create_uav(position, radius=UAV_RADIUS)
            for position in uav_positions
        ]

        # Kinematic projectile for deterministic movement.
        projectile_id = world.create_projectile(
            position=state[:3],
            velocity=(0, 0, 0),
            radius=PROJECTILE_RADIUS,
            mass=0,
        )

        if gui:
            # Draw the projectile's full trajectory in red.
            trajectory_start = state[:3]
            trajectory_end = (
                state[:3] + state[3:] * prediction_horizon
            )

            p.addUserDebugLine(
                trajectory_start.tolist(),
                trajectory_end.tolist(),
                [1, 0, 0],
                lineWidth=3,
                physicsClientId=world.client,
            )

            # Draw planned UAV movements in green.
            for start, target in zip(
                uav_positions,
                plan.target_positions,
            ):
                p.addUserDebugLine(
                    start.tolist(),
                    target.tolist(),
                    [0, 1, 0],
                    lineWidth=4,
                    physicsClientId=world.client,
                )

            # Give the user time to inspect the initial scene.
        if gui:
            print("Starting simulation in 2 seconds...")
            time.sleep(2.0)

        minimum_projectile = float("inf")
        minimum_uav = float("inf")

        # Include t=0 and the exact horizon endpoint.
        timestamps = np.arange(
            0.0,
            prediction_horizon,
            world.dt,
        )
        timestamps = np.append(
            timestamps,
            prediction_horizon,
        )

        for index, t in enumerate(timestamps):
            # Execute the coordinated UAV maneuver.
            world.execute_uav_movement(
                uav_ids=uav_ids,
                initial_positions=uav_positions,
                displacements=plan.displacements,
                elapsed_time=t,
                maneuver_duration=maneuver_duration,
            )

            # Move projectile along its ground-truth trajectory.
            projectile_position = (
                state[:3] + state[3:] * t
            )

            p.resetBasePositionAndOrientation(
                projectile_id,
                projectile_position.tolist(),
                [0, 0, 0, 1],
                physicsClientId=world.client,
            )

            world.step()

            # Read positions from the shared simulation world.
            actual_uav_positions = world.get_uav_positions(
                uav_ids
            )

            actual_projectile_position, _ = (
                world.get_projectile_state(projectile_id)
            )

            projectile_sep, uav_sep = minimum_separations(
                actual_uav_positions,
                actual_projectile_position,
            )

            minimum_projectile = min(
                minimum_projectile,
                projectile_sep,
            )
            minimum_uav = min(
                minimum_uav,
                uav_sep,
            )

            if gui and realtime and index > 0:
                time.sleep(
                    float(timestamps[index] - timestamps[index - 1])
                )

        final_positions = world.get_uav_positions(uav_ids)

        predicted_projectile = float(
            plan.minimum_projectile_separation
        )
        predicted_uav = float(
            plan.minimum_uav_separation
        )

        tolerance = 1e-6

        continuous_safe = (
            predicted_projectile
            >= PROJECTILE_SAFETY_DISTANCE - tolerance
            and predicted_uav
            >= UAV_SAFETY_DISTANCE - tolerance
        )

        sampled_safe = (
            minimum_projectile
            >= PROJECTILE_SAFETY_DISTANCE - tolerance
            and minimum_uav
            >= UAV_SAFETY_DISTANCE - tolerance
        )

        targets_reached = np.allclose(
            final_positions,
            plan.target_positions,
            atol=tolerance,
        )

        collision_free = bool(
            continuous_safe
            and sampled_safe
            and targets_reached
        )

        result = ExecutionResult(
            planning_success=True,
            collision_free=collision_free,
            total_displacement=float(plan.total_displacement),
            number_moved=int(plan.number_moved),
            minimum_projectile_separation=minimum_projectile,
            minimum_uav_separation=minimum_uav,
            predicted_projectile_separation=predicted_projectile,
            predicted_uav_separation=predicted_uav,
            final_uav_positions=final_positions,
            target_positions=plan.target_positions,
            message=(
                "Avoidance executed successfully."
                if collision_free
                else "Safety or execution check failed."
            ),
        )

        if verbose:
            print("=" * 55)
            print("PYBULLET AVOIDANCE EXECUTION")
            print("=" * 55)
            print("Planning success:", result.planning_success)
            print("Collision-free:", result.collision_free)
            print(
                f"Total displacement: "
                f"{result.total_displacement:.4f} m"
            )
            print("UAVs moved:", result.number_moved)
            print(
                f"Predicted projectile separation: "
                f"{predicted_projectile:.4f} m"
            )
            print(
                f"Simulated projectile separation: "
                f"{minimum_projectile:.4f} m"
            )
            print(
                f"Predicted UAV separation: "
                f"{predicted_uav:.4f} m"
            )
            print(
                f"Simulated UAV separation: "
                f"{minimum_uav:.4f} m"
            )

            for i, (start, target, actual) in enumerate(
                zip(
                    uav_positions,
                    plan.target_positions,
                    final_positions,
                ),
                start=1,
            ):
                print(f"\nUAV {i}")
                print("  Start: ", np.round(start, 4))
                print("  Target:", np.round(target, 4))
                print("  Actual:", np.round(actual, 4))

            print("\n" + result.message)

        if gui:
            print("\nSimulation complete. Press Enter to close.")
            input()

        return result

    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")

    args = parser.parse_args()

    run_avoidance(
        gui=args.gui,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
