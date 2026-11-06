
"""
Validate PyBullet avoidance execution.

Run:
    python -m experiments.validate_avoidance_execution
"""

import numpy as np

from experiments.execute_avoidance import (
    run_avoidance,
    PROJECTILE_SAFETY_DISTANCE,
    UAV_SAFETY_DISTANCE,
)


def assert_safe(result):
    assert result.planning_success, result.message
    assert result.collision_free, result.message

    assert (
        result.minimum_projectile_separation
        >= PROJECTILE_SAFETY_DISTANCE - 1e-6
    )

    assert (
        result.minimum_uav_separation
        >= UAV_SAFETY_DISTANCE - 1e-6
    )

    assert np.allclose(
        result.final_uav_positions,
        result.target_positions,
        atol=1e-6,
    )

    assert (
        result.minimum_projectile_separation
        >= result.predicted_projectile_separation - 1e-6
    )

    assert (
        result.minimum_uav_separation
        >= result.predicted_uav_separation - 1e-6
    )


def test_no_collision():
    state = np.array([
        5.0, 5.0, 5.0,
        1.0, 0.0, 0.0,
    ])

    result = run_avoidance(state=state, verbose=False)

    assert_safe(result)
    assert result.number_moved == 0
    assert result.total_displacement < 1e-4

    print("PASS: No collision requires no movement")


def test_single_uav_avoidance():
    state = np.array([
        -1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
    ])

    result = run_avoidance(state=state, verbose=False)

    assert_safe(result)
    assert result.number_moved >= 1

    print("PASS: Single-UAV avoidance")


def test_multiple_uav_avoidance():
    state = np.array([
        -1.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
    ])

    result = run_avoidance(state=state, verbose=False)

    assert_safe(result)
    assert result.number_moved >= 2

    print("PASS: Multiple-UAV coordinated avoidance")


def test_uav_separation():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0],
    ])

    state = np.array([
        -1.0, 0.15, 0.0,
        2.0, 0.0, 0.0,
    ])

    result = run_avoidance(
        state=state,
        uav_positions=positions,
        verbose=False,
    )

    assert_safe(result)

    print("PASS: UAV-UAV separation maintained")


def test_infeasible_maneuver():
    # Projectile is already inside the safety envelope.
    positions = np.array([
        [0.0, 0.0, 0.0],
    ])

    state = np.array([
        0.05, 0.0, 0.0,
        1.0, 0.0, 0.0,
    ])

    result = run_avoidance(
        state=state,
        uav_positions=positions,
        verbose=False,
    )

    assert not result.planning_success

    print("PASS: Infeasible maneuver identified")


def main():
    print("PYBULLET AVOIDANCE EXECUTION VALIDATION\n")

    test_no_collision()
    test_single_uav_avoidance()
    test_multiple_uav_avoidance()
    test_uav_separation()
    test_infeasible_maneuver()

    print("\nAll avoidance execution tests passed.")


if __name__ == "__main__":
    main()
