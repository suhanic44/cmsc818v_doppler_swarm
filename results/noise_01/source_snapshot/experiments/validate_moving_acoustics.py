
"""
Validate moving acoustic sensing.

Run:
    python -m experiments.validate_moving_acoustics
"""

import numpy as np

from simulation.acoustic_sensor import (
    StreamingAcousticSensor,
    MovingStreamingAcousticSensor,
)


DURATION = 0.5
DT = 1.0 / 240.0


def collect(sensor, projectile_fn, sensor_fn, moving):
    measurements = []

    for t in np.arange(0, DURATION + DT / 2, DT):
        projectile = projectile_fn(t)
        uav = sensor_fn(t)

        if moving:
            times, values = sensor.advance(
                t, projectile, uav
            )
        else:
            times, values = sensor.advance(
                t, projectile
            )

        measurements.extend(zip(times, values))

    return np.asarray(measurements)


def test_stationary_regression():
    sensor_position = np.array([0., 0., 0.])

    projectile_fn = lambda t: np.array([
        3. - 2. * t, 0., 0.
    ])
    sensor_fn = lambda t: sensor_position

    original = StreamingAcousticSensor(
        sensor_position, DURATION
    )
    moving = MovingStreamingAcousticSensor(
        sensor_position, DURATION
    )

    a = collect(
        original, projectile_fn, sensor_fn, False
    )
    b = collect(
        moving, projectile_fn, sensor_fn, True
    )

    assert a.shape == b.shape
    assert len(a) > 0

    max_error = np.max(np.abs(a[:, 1] - b[:, 1]))
    print("Stationary regression error:", max_error, "Hz")

    assert max_error < 1e-3


def test_sensor_motion():
    projectile_fn = lambda t: np.array([
        3., 0., 0.
    ])

    stationary_fn = lambda t: np.array([
        0., 0., 0.
    ])

    approaching_fn = lambda t: np.array([
        t, 0., 0.
    ])

    stationary = collect(
        MovingStreamingAcousticSensor(
            stationary_fn(0), DURATION
        ),
        projectile_fn,
        stationary_fn,
        True,
    )

    approaching = collect(
        MovingStreamingAcousticSensor(
            approaching_fn(0), DURATION
        ),
        projectile_fn,
        approaching_fn,
        True,
    )

    stationary_shift = np.median(stationary[:, 1])
    approaching_shift = np.median(approaching[:, 1])

    print("Stationary shift:", stationary_shift, "Hz")
    print("Approaching UAV shift:", approaching_shift, "Hz")

    # Moving toward a stationary reflector increases
    # the returned frequency.
    assert approaching_shift > stationary_shift + 20.


if __name__ == "__main__":
    test_stationary_regression()
    test_sensor_motion()
    print("\nMOVING ACOUSTIC TESTS: PASS")
