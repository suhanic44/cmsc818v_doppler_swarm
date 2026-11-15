
"""
Validate end-to-end online acoustic avoidance.

Run:
    python -m experiments.validate_online_avoidance
"""

import numpy as np

from experiments.online_avoidance import (
    run_online_avoidance,
    UAV_POSITIONS,
    rmse,
)


SCENARIOS = {
    "Direct approach": np.array([
        -1., 0., 0.,
        2., 0., 0.,
    ]),
    "Faster approach": np.array([
        -1.5, 0., 0.,
        4., 0., 0.,
    ]),
    "Diagonal approach": np.array([
        -1., -1., 0.,
        2., 2., 0.,
    ]),
    "Offset trajectory": np.array([
        -1., 0.15, 0.,
        2., 0., 0.,
    ]),
    "No collision": np.array([
        -1., 1., 1.,
        2., 0., 0.,
    ]),
}


def scenario_status(result):
    if result.safety_margin_satisfied:
        return "SAFE"

    if result.collision_free:
        return "COLLISION FREE (MARGIN VIOLATED)"

    return "COLLISION"


def main():
    results = {}

    for name, state in SCENARIOS.items():
        print("\n" + "=" * 65)
        print("SCENARIO:", name)
        print("=" * 65)

        result = run_online_avoidance(
            projectile_state=state,
            uav_positions=UAV_POSITIONS,
            gui=False,
            verbose=False,
        )

        results[name] = result

        # Basic integration assertions.
        assert result.initialized, (
            f"{name}: EKF failed to initialize"
        )

        assert result.ekf_update_count > 0, (
            f"{name}: No EKF updates"
        )

        assert np.all(np.isfinite(
            result.final_uav_positions
        )), (
            f"{name}: Invalid final UAV positions"
        )

        assert np.isfinite(result.doppler_rmse), (
            f"{name}: Invalid Doppler RMSE"
        )

        assert np.all(np.isfinite(
            result.position_errors
        )), (
            f"{name}: Invalid position errors"
        )

        assert np.all(np.isfinite(
            result.velocity_errors
        )), (
            f"{name}: Invalid velocity errors"
        )

        print("Measurements:", result.measurement_count)
        print("EKF updates:", result.ekf_update_count)

        print(
            "Doppler innovation RMSE:",
            f"{result.doppler_rmse:.4f} Hz",
        )

        print(
            "Moving-UAV Doppler innovation RMSE:",
            f"{result.moving_doppler_rmse:.4f} Hz",
        )

        print(
            "Moving-UAV samples:",
            result.moving_doppler_samples,
        )

        print(
            "Position RMSE:",
            f"{rmse(result.position_errors):.4f} m",
        )

        print(
            "Velocity RMSE:",
            f"{rmse(result.velocity_errors):.4f} m/s",
        )

        print(
            "Planning attempts:",
            result.planning_attempts,
        )

        print(
            "Successful plans:",
            result.successful_plans,
        )

        print(
            "First plan time:",
            result.first_plan_time,
        )

        print(
            "Minimum projectile separation:",
            f"{result.minimum_projectile_separation:.4f} m",
        )

        print(
            "Minimum UAV separation:",
            f"{result.minimum_uav_separation:.4f} m",
        )

        print(
            "Collision free:",
            result.collision_free,
        )

        print(
            "Safety margin satisfied:",
            result.safety_margin_satisfied,
        )

        print("Outcome:", scenario_status(result))

    print("\n" + "=" * 65)
    print("END-TO-END SUMMARY")
    print("=" * 65)

    print(
        f"{'Scenario':22s} "
        f"{'Pos RMSE':>10s} "
        f"{'Vel RMSE':>10s} "
        f"{'Doppler':>10s} "
        f"{'Outcome'}"
    )

    print("-" * 90)

    for name, result in results.items():
        print(
            f"{name:22s} "
            f"{rmse(result.position_errors):10.4f} "
            f"{rmse(result.velocity_errors):10.4f} "
            f"{result.doppler_rmse:10.4f} "
            f"{scenario_status(result)}"
        )

    safe_count = sum(
        result.safety_margin_satisfied
        for result in results.values()
    )

    collision_free_count = sum(
        result.collision_free
        for result in results.values()
    )

    print("\nSafety summary:")
    print(
        f"Full safety margin: "
        f"{safe_count}/{len(results)}"
    )
    print(
        f"Collision free: "
        f"{collision_free_count}/{len(results)}"
    )

    print("\nIntegration checks completed.")
    print(
        "Safety and estimation performance must be "
        "evaluated separately."
    )


if __name__ == "__main__":
    main()
