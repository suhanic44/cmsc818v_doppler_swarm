
"""
Validate geometric collision prediction.

Run:
    python -m experiments.validate_collision_prediction
"""

import numpy as np

from planning.collision_predictor import predict_collisions


UAV_POSITIONS = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
])

PREDICTION_HORIZON = 1.0
UAV_RADIUS = 0.05
PROJECTILE_RADIUS = 0.02
SAFETY_MARGIN = 0.03

SAFETY_RADIUS = (
    UAV_RADIUS + PROJECTILE_RADIUS + SAFETY_MARGIN
)


def run_prediction(
    state,
    uav_positions=UAV_POSITIONS,
    horizon=PREDICTION_HORIZON,
):
    return predict_collisions(
        state=np.asarray(state, dtype=float),
        uav_positions=uav_positions,
        prediction_horizon=horizon,
        uav_radius=UAV_RADIUS,
        projectile_radius=PROJECTILE_RADIUS,
        safety_margin=SAFETY_MARGIN,
    )


def assert_close(actual, expected, label, atol=1e-8):
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(
            f"{label}: expected {expected}, got {actual}"
        )


def test_direct_collision():
    """
    Projectile travels from x=-1 toward UAV 0 at 2 m/s.

    Closest approach occurs at 0.5 s.
    Entry into the 0.10 m safety radius occurs at 0.45 s.
    """
    state = [-1, 0, 0, 2, 0, 0]
    results = run_prediction(state)

    assert results[0].threatened

    assert_close(
        results[0].time_to_closest_approach,
        0.5,
        "Direct collision closest approach",
    )
    assert_close(
        results[0].minimum_separation,
        0.0,
        "Direct collision minimum separation",
    )
    assert_close(
        results[0].time_to_collision,
        0.45,
        "Direct collision entry time",
    )

    print("PASS: Direct collision")


def test_near_miss():
    """
    Projectile passes 0.15 m from UAV 0,
    outside the 0.10 m safety radius.
    """
    state = [-1, 0.15, 0, 2, 0, 0]
    results = run_prediction(state)

    assert not results[0].threatened
    assert results[0].time_to_collision is None

    assert_close(
        results[0].minimum_separation,
        0.15,
        "Near-miss separation",
    )
    assert_close(
        results[0].clearance,
        0.05,
        "Near-miss clearance",
    )

    print("PASS: Near miss")


def test_multiple_threatened_uavs():
    """
    Projectile travels along the x-axis from x=-1 to x=3.

    Both UAV 0 at x=0 and UAV 1 at x=2
    are threatened within the 1-second horizon.
    """
    state = [-1, 0, 0, 4, 0, 0]
    results = run_prediction(state)

    threatened_indices = [
        result.uav_index
        for result in results
        if result.threatened
    ]

    assert threatened_indices == [0, 1]

    assert_close(
        results[0].time_to_closest_approach,
        0.25,
        "First UAV closest approach",
    )
    assert_close(
        results[1].time_to_closest_approach,
        0.75,
        "Second UAV closest approach",
    )

    assert_close(
        results[0].time_to_collision,
        0.225,
        "First UAV safety entry",
    )
    assert_close(
        results[1].time_to_collision,
        0.725,
        "Second UAV safety entry",
    )

    print("PASS: Multiple threatened UAVs")


def test_moving_away():
    """
    Projectile starts to the right of UAV 0
    and continues moving farther away.
    """
    state = [1, 0, 0, 1, 0, 0]

    results = run_prediction(
        state,
        uav_positions=np.array([[0.0, 0.0, 0.0]]),
    )

    assert not results[0].threatened
    assert results[0].time_to_collision is None

    assert_close(
        results[0].time_to_closest_approach,
        0.0,
        "Moving-away closest approach",
    )
    assert_close(
        results[0].minimum_separation,
        1.0,
        "Moving-away minimum separation",
    )

    print("PASS: Projectile moving away")


def test_outside_prediction_horizon():
    """
    Projectile would hit UAV 0 after 2 seconds,
    but the prediction horizon is only 1 second.
    """
    state = [-2, 0, 0, 1, 0, 0]

    results = run_prediction(
        state,
        uav_positions=np.array([[0.0, 0.0, 0.0]]),
        horizon=1.0,
    )

    assert not results[0].threatened
    assert results[0].time_to_collision is None

    assert_close(
        results[0].time_to_closest_approach,
        1.0,
        "Horizon-limited closest approach",
    )
    assert_close(
        results[0].minimum_separation,
        1.0,
        "Horizon-limited separation",
    )

    print("PASS: Collision outside prediction horizon")


def test_stationary_projectile():
    """
    Stationary projectile remains outside the safety radius.
    """
    state = [1, 0, 0, 0, 0, 0]

    results = run_prediction(
        state,
        uav_positions=np.array([[0.0, 0.0, 0.0]]),
    )

    assert not results[0].threatened
    assert results[0].time_to_collision is None

    assert_close(
        results[0].time_to_closest_approach,
        0.0,
        "Stationary closest approach",
    )
    assert_close(
        results[0].minimum_separation,
        1.0,
        "Stationary separation",
    )

    print("PASS: Stationary projectile")


def test_already_inside_safety_radius():
    """
    Projectile starts within the UAV safety radius.
    """
    state = [0.05, 0, 0, 1, 0, 0]

    results = run_prediction(
        state,
        uav_positions=np.array([[0.0, 0.0, 0.0]]),
    )

    assert results[0].threatened

    assert_close(
        results[0].time_to_collision,
        0.0,
        "Already inside safety radius",
    )
    assert_close(
        results[0].minimum_separation,
        0.05,
        "Initial separation",
    )

    print("PASS: Already inside safety radius")


def test_tangent_to_safety_radius():
    """
    Projectile exactly touches the 0.10 m safety envelope.

    Boundary contact counts as threatened.
    """
    state = [-1, SAFETY_RADIUS, 0, 2, 0, 0]

    results = run_prediction(
        state,
        uav_positions=np.array([[0.0, 0.0, 0.0]]),
    )

    assert results[0].threatened

    assert_close(
        results[0].minimum_separation,
        SAFETY_RADIUS,
        "Tangent separation",
    )
    assert_close(
        results[0].time_to_collision,
        0.5,
        "Tangent entry time",
    )

    print("PASS: Tangent to safety radius")


def test_invalid_inputs():
    """Reject invalid states and configuration values."""
    try:
        run_prediction(
            [np.nan, 0, 0, 1, 0, 0]
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-finite state should be rejected."
        )

    try:
        predict_collisions(
            state=[0, 0, 0, 1, 0, 0],
            uav_positions=UAV_POSITIONS,
            prediction_horizon=-1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative horizon should be rejected."
        )

    print("PASS: Invalid inputs")


def main():
    print("=" * 60)
    print("COLLISION PREDICTION VALIDATION")
    print("=" * 60)

    test_direct_collision()
    test_near_miss()
    test_multiple_threatened_uavs()
    test_moving_away()
    test_outside_prediction_horizon()
    test_stationary_projectile()
    test_already_inside_safety_radius()
    test_tangent_to_safety_radius()
    test_invalid_inputs()

    print("\nAll collision prediction tests passed.")

    # Display an example for the avoidance planner.
    print("\n" + "=" * 60)
    print("EXAMPLE: MULTI-UAV COLLISION PREDICTION")
    print("=" * 60)

    state = [-1, 0, 0, 4, 0, 0]
    results = run_prediction(state)

    for result in results:
        print(f"\nUAV {result.uav_index + 1}:")
        print(f"  Threatened: {result.threatened}")
        print(
            f"  Time to closest approach: "
            f"{result.time_to_closest_approach:.4f} s"
        )
        print(
            f"  Minimum separation: "
            f"{result.minimum_separation:.4f} m"
        )
        print(
            f"  Clearance: "
            f"{result.clearance:.4f} m"
        )
        print(
            f"  Time to safety-envelope entry: "
            f"{result.time_to_collision}"
        )
        print(
            "  Projectile at closest approach:",
            np.round(
                result.closest_projectile_position,
                4,
            ),
        )


if __name__ == "__main__":
    main()
