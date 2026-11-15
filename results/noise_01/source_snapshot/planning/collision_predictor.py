
"""
Geometric collision prediction for a stationary UAV swarm.

Assumptions:
    - Projectile follows a constant-velocity trajectory.
    - UAVs remain stationary during prediction.
    - UAVs and projectile use spherical collision envelopes.
    - Prediction is limited to a finite future horizon.

The projectile state is:
    [px, py, pz, vx, vy, vz]
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CollisionPrediction:
    uav_index: int
    uav_position: np.ndarray

    threatened: bool

    time_to_closest_approach: float
    minimum_separation: float

    closest_projectile_position: np.ndarray

    time_to_collision: float | None

    safety_radius: float
    clearance: float


def predict_collisions(
    state,
    uav_positions,
    prediction_horizon=1.0,
    uav_radius=0.05,
    projectile_radius=0.02,
    safety_margin=0.03,
):
    """
    Predict potential projectile collisions for each UAV.

    Args:
        state:
            Projectile state [px, py, pz, vx, vy, vz].

        uav_positions:
            Array of UAV positions with shape (N, 3).

        prediction_horizon:
            Number of seconds into the future to predict.

        uav_radius:
            Spherical UAV collision radius, in meters.

        projectile_radius:
            Spherical projectile collision radius, in meters.

        safety_margin:
            Additional required separation, in meters.

    Returns:
        List of CollisionPrediction objects, one per UAV.

    A UAV is threatened if the projectile enters its
    combined safety radius within the prediction horizon.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    uav_positions = np.asarray(
        uav_positions, dtype=float
    ).reshape(-1, 3)

    if not np.all(np.isfinite(state)):
        raise ValueError("Projectile state must be finite.")

    if not np.all(np.isfinite(uav_positions)):
        raise ValueError("UAV positions must be finite.")

    if not np.isfinite(prediction_horizon) or prediction_horizon < 0:
        raise ValueError(
            "Prediction horizon must be finite and nonnegative."
        )

    if not np.all(np.isfinite([
        uav_radius,
        projectile_radius,
        safety_margin,
    ])):
        raise ValueError(
            "Collision radii and margin must be finite."
        )

    if min(
        uav_radius,
        projectile_radius,
        safety_margin,
    ) < 0:
        raise ValueError(
            "Collision radii and margin must be nonnegative."
        )

    position = state[:3]
    velocity = state[3:]

    speed_squared = float(np.dot(velocity, velocity))

    safety_radius = (
        uav_radius
        + projectile_radius
        + safety_margin
    )

    results = []

    for i, uav_position in enumerate(uav_positions):
        # Relative projectile position with respect to UAV.
        relative_position = position - uav_position

        # Find the closest point within the future horizon.
        if speed_squared <= 1e-12:
            t_closest = 0.0
        else:
            t_unbounded = -(
                np.dot(relative_position, velocity)
                / speed_squared
            )

            t_closest = float(
                np.clip(
                    t_unbounded,
                    0.0,
                    prediction_horizon,
                )
            )

        closest_position = (
            position + velocity * t_closest
        )

        minimum_separation = float(
            np.linalg.norm(
                closest_position - uav_position
            )
        )

        clearance = (
            minimum_separation - safety_radius
        )

        # Small tolerance for floating-point boundary cases.
        threatened = (
            minimum_separation <= safety_radius + 1e-12
        )

        time_to_collision = None

        if threatened:
            initial_distance = np.linalg.norm(
                relative_position
            )

            # Already inside the safety envelope.
            if initial_distance <= safety_radius:
                time_to_collision = 0.0

            elif speed_squared > 1e-12:
                # Solve:
                # ||relative_position + velocity*t||^2
                #     = safety_radius^2
                #
                # a*t^2 + b*t + c = 0

                a = speed_squared

                b = 2.0 * np.dot(
                    relative_position,
                    velocity,
                )

                c = (
                    np.dot(
                        relative_position,
                        relative_position,
                    )
                    - safety_radius**2
                )

                discriminant = b**2 - 4.0 * a * c

                # Clamp tiny negative floating-point errors.
                discriminant = max(
                    0.0,
                    float(discriminant),
                )

                t_entry = (
                    -b - np.sqrt(discriminant)
                ) / (2.0 * a)

                time_to_collision = float(
                    np.clip(
                        t_entry,
                        0.0,
                        prediction_horizon,
                    )
                )

        results.append(
            CollisionPrediction(
                uav_index=i,
                uav_position=uav_position.copy(),
                threatened=bool(threatened),
                time_to_closest_approach=t_closest,
                minimum_separation=minimum_separation,
                closest_projectile_position=(
                    closest_position.copy()
                ),
                time_to_collision=time_to_collision,
                safety_radius=safety_radius,
                clearance=float(clearance),
            )
        )

    return results
