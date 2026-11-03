
"""
Validate incremental acoustic sensing against offline generation.

Run:
    python -m experiments.validate_streaming_acoustics
"""

import numpy as np

from simulation.world import SimulationWorld
from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    StreamingAcousticSensor,
    generate_trajectory_echo,
    estimate_doppler,
)


UAV_POSITIONS = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
])

DURATION = 0.5

SCENARIOS = [
    ("Original", [3, 3, 3], [-2, -1.5, -1]),
    ("Faster", [3, 3, 3], [-4, -3, -2]),
    ("Diagonal", [4, 2.5, 3.5], [-3, -1, -2]),
    ("Different direction", [-2, 3, 3], [2, -2, -1]),
    ("Farther away", [5, 4, 4], [-3, -2, -2]),
]


def run_scenario(name, initial_position, initial_velocity):
    print("\n" + "=" * 60)
    print("Trajectory:", name)
    print("=" * 60)

    world = SimulationWorld(gui=False, gravity=False)

    sensors = [
        StreamingAcousticSensor(
            sensor_position=position,
            duration=DURATION,
        )
        for position in UAV_POSITIONS
    ]

    physics_times = []
    projectile_positions = []

    try:
        for position in UAV_POSITIONS:
            world.create_uav(position)

        projectile_id = world.create_projectile(
            position=np.asarray(initial_position, dtype=float),
            velocity=np.asarray(initial_velocity, dtype=float),
        )

        num_steps = int(DURATION / world.dt)

        for step in range(num_steps + 1):
            t = step * world.dt

            position, _ = world.get_projectile_state(
                projectile_id
            )

            physics_times.append(t)
            projectile_positions.append(position)

            # Stream the current projectile observation to
            # each sensor before advancing the physics again.
            for sensor in sensors:
                sensor.advance(t, position)

            if step < num_steps:
                world.step()

    finally:
        world.close()

    physics_times = np.asarray(physics_times)
    projectile_positions = np.asarray(projectile_positions)

    audio_times = np.arange(
        0, DURATION, 1.0 / SAMPLE_RATE
    )

    all_passed = True

    for i, (sensor, sensor_position) in enumerate(
        zip(sensors, UAV_POSITIONS)
    ):
        # Offline reference uses the complete trajectory.
        offline_echo = generate_trajectory_echo(
            audio_times,
            physics_times,
            projectile_positions,
            sensor_position,
        )

        offline_times, offline_doppler = estimate_doppler(
            audio_times,
            offline_echo,
            window_duration=0.04,
            hop_duration=0.02,
        )

        streaming_times = np.asarray(
            sensor.measurement_times
        )
        streaming_doppler = np.asarray(
            sensor.doppler_values
        )

        timestamps_match = (
            len(offline_times) == len(streaming_times)
            and np.allclose(
                offline_times,
                streaming_times,
                atol=1e-9,
                rtol=0,
            )
        )

        if timestamps_match and len(offline_times) > 0:
            differences = (
                streaming_doppler - offline_doppler
            )

            rmse = np.sqrt(np.mean(differences ** 2))
            max_error = np.max(np.abs(differences))

            passed = (
                np.isfinite(rmse)
                and max_error < 1e-3
            )
        else:
            rmse = float("nan")
            max_error = float("nan")
            passed = False

        all_passed = all_passed and passed

        print(f"\nUAV {i + 1}:")
        print(f"  Offline measurements:   {len(offline_times)}")
        print(f"  Streaming measurements: {len(streaming_times)}")
        print(f"  Timestamps match:       {timestamps_match}")
        print(f"  Doppler RMSE:           {rmse:.8f} Hz")
        print(f"  Maximum difference:     {max_error:.8f} Hz")
        print(f"  Result:                 {'PASS' if passed else 'FAIL'}")

    return all_passed


def main():
    results = []

    for name, position, velocity in SCENARIOS:
        passed = run_scenario(name, position, velocity)
        results.append((name, passed))

    print("\n" + "=" * 60)
    print("STREAMING ACOUSTIC VALIDATION SUMMARY")
    print("=" * 60)

    for name, passed in results:
        print(f"{name:25s} {'PASS' if passed else 'FAIL'}")

    if not all(passed for _, passed in results):
        raise SystemExit(
            "Streaming acoustic validation failed."
        )

    print("\nAll streaming acoustic tests passed.")


if __name__ == "__main__":
    main()
