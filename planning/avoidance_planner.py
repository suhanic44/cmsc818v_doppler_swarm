
"""
Coordinated minimum-displacement avoidance planning.

Uses Sequential Least Squares Programming (SLSQP) to jointly
calculate avoidance targets for a UAV swarm.

Objective:
    Minimize the sum of UAV displacement distances.

Constraints:
    - Safe projectile-UAV separation throughout the maneuver.
    - Safe UAV-UAV separation throughout the maneuver.
    - Maximum UAV movement speed.

Assumptions:
    - Projectile moves at constant velocity.
    - UAVs move in straight lines at constant velocity.
    - All UAVs begin moving at the same time.
    - UAVs stop at their targets after maneuver_duration.
    - UAVs and projectile use spherical safety envelopes.
    - Gravity is disabled.

This module plans avoidance but does not execute UAV motion.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from planning.collision_predictor import predict_collisions


@dataclass
class AvoidancePlan:
    success: bool
    target_positions: np.ndarray
    displacements: np.ndarray
    total_displacement: float
    number_moved: int
    minimum_projectile_separation: float
    minimum_uav_separation: float
    message: str


def _closest_distance(a0, a1, b0, b1):
    """
    Minimum separation between two linearly moving objects
    over the same time interval.

    Object A moves from a0 to a1.
    Object B moves from b0 to b1.
    """
    relative_start = a0 - b0
    relative_motion = (a1 - a0) - (b1 - b0)

    denominator = float(np.dot(relative_motion, relative_motion))

    if denominator <= 1e-15:
        fraction = 0.0
    else:
        fraction = float(np.clip(
            -np.dot(relative_start, relative_motion) / denominator,
            0.0,
            1.0,
        ))

    return float(np.linalg.norm(
        relative_start + fraction * relative_motion
    ))


def _separation_distances(
    state,
    uav_positions,
    displacements,
    prediction_horizon,
    maneuver_duration,
):
    """
    Calculate minimum separations for each projectile-UAV
    pair and each UAV-UAV pair over the full horizon.

    Because motion is piecewise linear, closest approach
    can be calculated analytically on each interval.
    """
    position = state[:3]
    velocity = state[3:]

    n_uavs = len(uav_positions)

    # UAVs stop moving after maneuver_duration.
    times = np.unique([
        0.0,
        min(maneuver_duration, prediction_horizon),
        prediction_horizon,
    ])

    projectile_distances = np.full(n_uavs, np.inf)

    pairs = [
        (i, j)
        for i in range(n_uavs)
        for j in range(i + 1, n_uavs)
    ]

    uav_distances = np.full(len(pairs), np.inf)

    def uav_locations(t):
        fraction = min(t / maneuver_duration, 1.0)
        return uav_positions + fraction * displacements

    def projectile_location(t):
        return position + velocity * t

    if len(times) == 1:
        projectile_distances = np.linalg.norm(
            uav_positions - position,
            axis=1,
        )

        for k, (i, j) in enumerate(pairs):
            uav_distances[k] = np.linalg.norm(
                uav_positions[i] - uav_positions[j]
            )

        return projectile_distances, uav_distances

    for t0, t1 in zip(times[:-1], times[1:]):
        uav_start = uav_locations(t0)
        uav_end = uav_locations(t1)

        projectile_start = projectile_location(t0)
        projectile_end = projectile_location(t1)

        for i in range(n_uavs):
            distance = _closest_distance(
                uav_start[i],
                uav_end[i],
                projectile_start,
                projectile_end,
            )

            projectile_distances[i] = min(
                projectile_distances[i],
                distance,
            )

        for k, (i, j) in enumerate(pairs):
            distance = _closest_distance(
                uav_start[i],
                uav_end[i],
                uav_start[j],
                uav_end[j],
            )

            uav_distances[k] = min(
                uav_distances[k],
                distance,
            )

    return projectile_distances, uav_distances


def evaluate_plan(
    state,
    uav_positions,
    displacements,
    prediction_horizon=1.0,
    maneuver_duration=0.15,
):
    """
    Return minimum projectile-UAV and UAV-UAV separation
    over the entire planned maneuver and prediction horizon.
    """
    state = np.asarray(state, dtype=float)
    positions = np.asarray(uav_positions, dtype=float)
    displacements = np.asarray(displacements, dtype=float)

    projectile_distances, uav_distances = _separation_distances(
        state,
        positions,
        displacements,
        prediction_horizon,
        maneuver_duration,
    )

    return (
        float(np.min(projectile_distances)),
        float(np.min(uav_distances))
        if len(uav_distances) else float("inf"),
    )


def _avoidance_direction(velocity):
    """
    Find a unit vector perpendicular to projectile velocity.

    Lateral movement is a useful initial guess because it
    moves UAVs away from the predicted projectile path.
    """
    speed = np.linalg.norm(velocity)

    if speed <= 1e-12:
        return np.array([0.0, 1.0, 0.0])

    direction = velocity / speed

    axes = np.eye(3)
    reference = axes[np.argmin(np.abs(axes @ direction))]

    lateral = np.cross(direction, reference)

    return lateral / np.linalg.norm(lateral)


def plan_avoidance(
    state,
    uav_positions,
    prediction_horizon=1.0,
    maneuver_duration=0.15,
    max_uav_speed=3.0,
    uav_radius=0.05,
    projectile_radius=0.02,
    safety_margin=0.03,
    uav_separation_margin=0.05,
    num_starts=12,
    max_iterations=300,
):
    """
    Jointly optimize UAV avoidance targets using SLSQP.

    Returns:
        AvoidancePlan containing targets, displacements,
        safety diagnostics, and optimization status.

    A returned plan must have success=True before its
    target positions are used for UAV movement.

    This is a local nonlinear optimizer. Multiple initial
    guesses improve robustness but do not guarantee a
    globally minimum-displacement solution.
    """
    state = np.asarray(state, dtype=float)
    positions = np.asarray(uav_positions, dtype=float)

    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("State must be a finite 6D vector.")

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) == 0
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("UAV positions must have shape (N, 3).")

    parameters = [
        prediction_horizon,
        maneuver_duration,
        max_uav_speed,
        uav_radius,
        projectile_radius,
        safety_margin,
        uav_separation_margin,
    ]

    if not np.all(np.isfinite(parameters)):
        raise ValueError("Planning parameters must be finite.")

    if prediction_horizon < 0:
        raise ValueError("Prediction horizon must be nonnegative.")

    if maneuver_duration <= 0 or max_uav_speed <= 0:
        raise ValueError(
            "Maneuver duration and maximum speed must be positive."
        )

    if min(
        uav_radius,
        projectile_radius,
        safety_margin,
        uav_separation_margin,
    ) < 0:
        raise ValueError("Safety parameters must be nonnegative.")

    if num_starts < 1 or max_iterations < 1:
        raise ValueError(
            "num_starts and max_iterations must be positive."
        )

    n_uavs = len(positions)

    projectile_safety_radius = (
        uav_radius + projectile_radius + safety_margin
    )

    uav_safety_distance = (
        2.0 * uav_radius + uav_separation_margin
    )

    max_displacement = max_uav_speed * maneuver_duration

    # Determine which stationary UAVs are initially threatened.
    predictions = predict_collisions(
        state,
        positions,
        prediction_horizon=prediction_horizon,
        uav_radius=uav_radius,
        projectile_radius=projectile_radius,
        safety_margin=safety_margin,
    )

    threatened_indices = [
        prediction.uav_index
        for prediction in predictions
        if prediction.threatened
    ]

    # Safety is required from t=0 onward. If a boundary is already violated,
    # no future displacement can repair the plan, so avoid exhausting SLSQP
    # starts on a mathematically infeasible problem.
    current_projectile_distances = np.linalg.norm(
        positions - state[:3], axis=1
    )
    current_uav_distances = np.array([
        np.linalg.norm(positions[i] - positions[j])
        for i in range(n_uavs)
        for j in range(i + 1, n_uavs)
    ])
    initially_unsafe = bool(
        np.any(current_projectile_distances < projectile_safety_radius - 1e-9)
        or (
            len(current_uav_distances)
            and np.any(current_uav_distances < uav_safety_distance - 1e-9)
        )
    )
    if initially_unsafe:
        return AvoidancePlan(
            success=False,
            target_positions=positions.copy(),
            displacements=np.zeros_like(positions),
            total_displacement=0.0,
            number_moved=0,
            minimum_projectile_separation=float(
                np.min(current_projectile_distances)
            ),
            minimum_uav_separation=(
                float(np.min(current_uav_distances))
                if len(current_uav_distances) else float("inf")
            ),
            message=(
                "Safety boundary is already violated at planning time; "
                "no full-horizon feasible maneuver exists."
            ),
        )

    def validate_solution(displacements):
        projectile_distances, uav_distances = (
            _separation_distances(
                state,
                positions,
                displacements,
                prediction_horizon,
                maneuver_duration,
            )
        )

        speed_valid = np.all(
            np.linalg.norm(displacements, axis=1)
            <= max_displacement + 1e-6
        )

        projectile_valid = np.all(
            projectile_distances
            >= projectile_safety_radius - 1e-6
        )

        uav_valid = np.all(
            uav_distances >= uav_safety_distance - 1e-6
        )

        return (
            bool(speed_valid and projectile_valid and uav_valid),
            float(np.min(projectile_distances)),
            float(np.min(uav_distances))
            if len(uav_distances) else float("inf"),
        )

    zero_displacements = np.zeros_like(positions)
    zero_valid, zero_min_projectile, zero_min_uav = validate_solution(
        zero_displacements
    )
    if not threatened_indices and zero_valid:
        return AvoidancePlan(
            success=True,
            target_positions=positions.copy(),
            displacements=zero_displacements,
            total_displacement=0.0,
            number_moved=0,
            minimum_projectile_separation=zero_min_projectile,
            minimum_uav_separation=zero_min_uav,
            message="No UAV is threatened; no avoidance is required.",
        )

    # Generate deterministic starting points.
    # This makes validation runs reproducible.
    lateral = _avoidance_direction(state[3:])
    second_lateral = np.cross(state[3:], lateral)

    if np.linalg.norm(second_lateral) <= 1e-12:
        second_lateral = np.array([1.0, 0.0, 0.0])
    else:
        second_lateral /= np.linalg.norm(second_lateral)

    directions = [
        lateral,
        -lateral,
        second_lateral,
        -second_lateral,
    ]

    # Smooth approximation to total Euclidean displacement.
    epsilon = 1e-6

    def optimize(active_indices):
        """Optimize a subset, returning after the first validated solution."""
        active_indices = np.asarray(active_indices, dtype=int)

        def unpack(x):
            displacements = np.zeros_like(positions)
            displacements[active_indices] = np.asarray(x).reshape(-1, 3)
            return displacements

        def objective(x):
            lengths = np.linalg.norm(np.asarray(x).reshape(-1, 3), axis=1)
            return float(np.sum(
                np.sqrt(lengths**2 + epsilon**2) - epsilon
            ))

        def speed_constraints(x):
            return max_displacement - np.linalg.norm(
                np.asarray(x).reshape(-1, 3), axis=1
            )

        def separation_constraints(x):
            projectile_distances, uav_distances = _separation_distances(
                state,
                positions,
                unpack(x),
                prediction_horizon,
                maneuver_duration,
            )
            return np.concatenate((
                projectile_distances - projectile_safety_radius,
                uav_distances - uav_safety_distance,
            ))

        constraints = [
            {"type": "ineq", "fun": speed_constraints},
            {"type": "ineq", "fun": separation_constraints},
        ]
        guesses = [np.zeros((len(active_indices), 3))]
        for direction in directions:
            for scale in (0.5, 1.0):
                guesses.append(np.tile(
                    direction * max_displacement * scale,
                    (len(active_indices), 1),
                ))
        for start_index in range(max(0, num_starts - len(guesses))):
            guess = np.zeros((len(active_indices), 3))
            for j in range(len(active_indices)):
                guess[j] = (
                    directions[(start_index + j) % len(directions)]
                    * max_displacement
                )
            guesses.append(guess)

        for guess in guesses[:num_starts]:
            result = minimize(
                objective,
                guess.ravel(),
                method="SLSQP",
                constraints=constraints,
                options={
                    "maxiter": max_iterations,
                    "ftol": 1e-6,
                    "disp": False,
                },
            )
            candidate = unpack(result.x)
            if not np.all(np.isfinite(candidate)):
                continue
            valid, min_projectile, min_uav = validate_solution(candidate)
            if valid:
                return candidate, min_projectile, min_uav
        return None

    # Most encounters threaten only one or two UAVs. Keep every other UAV
    # fixed first, then retain the original full-swarm search as a fallback.
    optimized = optimize(threatened_indices or range(n_uavs))
    if optimized is None and len(threatened_indices) < n_uavs:
        optimized = optimize(range(n_uavs))

    if optimized is None:
        best_displacements = None
        best_min_projectile = float("inf")
        best_min_uav = float("inf")
    else:
        (
            best_displacements,
            best_min_projectile,
            best_min_uav,
        ) = optimized

    if best_displacements is None:
        # Do not return unsafe optimization targets as valid.
        return AvoidancePlan(
            success=False,
            target_positions=positions.copy(),
            displacements=np.zeros_like(positions),
            total_displacement=0.0,
            number_moved=0,
            minimum_projectile_separation=float(
                evaluate_plan(
                    state,
                    positions,
                    np.zeros_like(positions),
                    prediction_horizon,
                    maneuver_duration,
                )[0]
            ),
            minimum_uav_separation=float(
                evaluate_plan(
                    state,
                    positions,
                    np.zeros_like(positions),
                    prediction_horizon,
                    maneuver_duration,
                )[1]
            ),
            message=(
                "No feasible avoidance plan found. "
                "Do not execute the returned targets."
            ),
        )

    displacement_lengths = np.linalg.norm(
        best_displacements,
        axis=1,
    )

    return AvoidancePlan(
        success=True,
        target_positions=positions + best_displacements,
        displacements=best_displacements,
        total_displacement=float(
            np.sum(displacement_lengths)
        ),
        number_moved=int(
            np.sum(displacement_lengths > 1e-4)
        ),
        minimum_projectile_separation=best_min_projectile,
        minimum_uav_separation=best_min_uav,
        message="Feasible coordinated avoidance plan found.",
    )
