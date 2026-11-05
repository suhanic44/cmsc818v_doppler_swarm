
"""
Validate coordinated minimum-displacement avoidance planning.

Run:
    python -m experiments.validate_avoidance_planning
"""

import numpy as np

from planning.avoidance_planner import (
    plan_avoidance,
    evaluate_plan,
)


UAV_POSITIONS = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
])

PREDICTION_HORIZON = 1.0
MANEUVER_DURATION = 0.15
MAX_UAV_SPEED = 3.0

UAV_RADIUS = 0.05
PROJECTILE_RADIUS = 0.02
SAFETY_MARGIN = 0.03
UAV_SEPARATION_MARGIN = 0.05

PROJECTILE_SAFETY_RADIUS = (
    UAV_RADIUS + PROJECTILE_RADIUS + SAFETY_MARGIN
)

UAV_SAFETY_DISTANCE = (
    2 * UAV_RADIUS + UAV_SEPARATION_MARGIN
)


def run_planner(
    state,
    uav_positions=UAV_POSITIONS,
    **kwargs,
):
    parameters = {
        "prediction_horizon": PREDICTION_HORIZON,
        "maneuver_duration": MANEUVER_DURATION,
        "max_uav_speed": MAX_UAV_SPEED,
        "uav_radius": UAV_RADIUS,
        "projectile_radius": PROJECTILE_RADIUS,
        "safety_margin": SAFETY_MARGIN,
        "uav_separation_margin": UAV_SEPARATION_MARGIN,
    }

    parameters.update(kwargs)

    return plan_avoidance(
        state=np.asarray(state, dtype=float),
        uav_positions=np.asarray(uav_positions, dtype=float),
        **parameters,
    )


def check_plan_safety(state, positions, plan):
    """
    Independently verify the resulting maneuver using
    continuous closest-approach calculations.
    """
    assert plan.success, plan.message

    min_projectile, min_uav = evaluate_plan(
        state,
        positions,
        plan.displacements,
        prediction_horizon=PREDICTION_HORIZON,
        maneuver_duration=MANEUVER_DURATION,
    )

    assert min_projectile >= PROJECTILE_SAFETY_RADIUS - 1e-6, (
        f"Unsafe projectile separation: {min_projectile}"
    )

    assert min_uav >= UAV_SAFETY_DISTANCE - 1e-6, (
        f"Unsafe UAV separation: {min_uav}"
    )

    movement_lengths = np.linalg.norm(
        plan.displacements,
        axis=1,
    )

    assert np.all(
        movement_lengths
        <= MAX_UAV_SPEED * MANEUVER_DURATION + 1e-6
    ), "UAV speed constraint violated."

    assert np.allclose(
        plan.target_positions,
        positions + plan.displacements,
    )

    assert np.isclose(
        plan.total_displacement,
        np.sum(movement_lengths),
        atol=1e-6,
    )


def test_no_collision():
    """
    Projectile passes far from all UAVs.

    Optimal solution is zero movement.
    """
    state = np.array([
        5.0, 5.0, 5.0,
        1.0, 0.0, 0.0,
    ])

    plan = run_planner(state)

    check_plan_safety(state, UAV_POSITIONS, plan)

    assert np.allclose(
        plan.displacements,
        0.0,
        atol=1e-4,
    )

    assert plan.total_displacement < 1e-4
    assert plan.number_moved == 0

    print("PASS: No collision requires no movement")


def test_single_uav_collision():
    """
    Projectile approaches UAV 0 directly.

    The planner should find a safe maneuver.
    """
    state = np.array([
        -1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
    ])

    plan = run_planner(state)

    check_plan_safety(state, UAV_POSITIONS, plan)

    assert plan.total_displacement > 0.0
    assert np.linalg.norm(plan.displacements[0]) > 1e-4

    print("PASS: Single-UAV collision avoidance")


def test_multiple_uav_collisions():
    """
    Projectile travels along the x-axis, threatening
    UAVs 0 and 1 at different times.

    The planner must keep the whole swarm safe.
    """
    state = np.array([
        -1.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
    ])

    plan = run_planner(state)

    check_plan_safety(state, UAV_POSITIONS, plan)

    assert np.linalg.norm(plan.displacements[0]) > 1e-4
    assert np.linalg.norm(plan.displacements[1]) > 1e-4

    print("PASS: Multiple-UAV coordinated avoidance")


def test_near_miss():
    """
    Projectile passes outside the safety envelope.

    No avoidance should be necessary.
    """
    state = np.array([
        -1.0, 0.2, 0.0,
        2.0, 0.0, 0.0,
    ])

    plan = run_planner(
        state,
        uav_positions=np.array([
            [0.0, 0.0, 0.0],
        ]),
    )

    check_plan_safety(
        state,
        np.array([[0.0, 0.0, 0.0]]),
        plan,
    )

    assert plan.total_displacement < 1e-4
    assert plan.number_moved == 0

    print("PASS: Near miss requires no movement")


def test_uav_separation():
    """
    Two UAVs are positioned near the projectile path.

    The resulting maneuvers must not cause the UAVs
    to collide with each other.
    """
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0],
    ])

    state = np.array([
        -1.0, 0.15, 0.0,
        2.0, 0.0, 0.0,
    ])

    plan = run_planner(
        state,
        uav_positions=positions,
    )

    check_plan_safety(state, positions, plan)

    assert plan.minimum_uav_separation >= (
        UAV_SAFETY_DISTANCE - 1e-6
    )

    print("PASS: UAV-UAV separation maintained")


def test_infeasible_maneuver():
    """
    Projectile is already inside the UAV safety radius.

    Because separation is violated at t=0, no maneuver
    can satisfy the full-horizon safety constraints.
    """
    positions = np.array([
        [0.0, 0.0, 0.0],
    ])

    state = np.array([
        0.05, 0.0, 0.0,
        1.0, 0.0, 0.0,
    ])

    plan = run_planner(
        state,
        uav_positions=positions,
    )

    assert not plan.success

    print("PASS: Infeasible maneuver identified")


def test_invalid_inputs():
    """Reject invalid states and planning parameters."""
    try:
        run_planner([
            np.nan, 0, 0, 1, 0, 0
        ])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-finite state should be rejected."
        )

    try:
        run_planner(
            [-1, 0, 0, 2, 0, 0],
            maneuver_duration=-1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative maneuver duration should be rejected."
        )

    print("PASS: Invalid inputs")


def print_example():
    """
    Display the coordinated avoidance plan for a projectile
    threatening two UAVs.
    """
    state = np.array([
        -1.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
    ])

    plan = run_planner(state)

    print("\n" + "=" * 60)
    print("EXAMPLE: COORDINATED MULTI-UAV AVOIDANCE")
    print("=" * 60)

    print("Optimization success:", plan.success)
    print("Message:", plan.message)

    if not plan.success:
        return

    print(
        f"Total UAV displacement: "
        f"{plan.total_displacement:.4f} m"
    )
    print("Number of UAVs moved:", plan.number_moved)

    print(
        f"Minimum projectile-UAV separation: "
        f"{plan.minimum_projectile_separation:.4f} m"
    )
    print(
        f"Minimum UAV-UAV separation: "
        f"{plan.minimum_uav_separation:.4f} m"
    )

    for i in range(len(UAV_POSITIONS)):
        print(f"\nUAV {i + 1}:")
        print(
            "  Original position:",
            np.round(UAV_POSITIONS[i], 4),
        )
        print(
            "  Target position:  ",
            np.round(plan.target_positions[i], 4),
        )
        print(
            "  Displacement:     ",
            np.round(plan.displacements[i], 4),
        )
        print(
            "  Movement distance:",
            f"{np.linalg.norm(plan.displacements[i]):.4f} m",
        )


def main():
    print("=" * 60)
    print("COORDINATED AVOIDANCE PLANNING VALIDATION")
    print("=" * 60)

    test_no_collision()
    test_single_uav_collision()
    test_multiple_uav_collisions()
    test_near_miss()
    test_uav_separation()
    test_infeasible_maneuver()
    test_invalid_inputs()

    print("\nAll avoidance planning tests passed.")

    print_example()


if __name__ == "__main__":
    main()
