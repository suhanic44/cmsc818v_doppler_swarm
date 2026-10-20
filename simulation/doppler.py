
import numpy as np

SPEED_OF_SOUND = 343.0  # meters per second
SPEAKER_FREQUENCY = 10000.0  # Hz


def doppler_shift(
    sensor_position,
    sensor_velocity,
    object_position,
    object_velocity,
    frequency=SPEAKER_FREQUENCY,
    speed_of_sound=SPEED_OF_SOUND,
):
    """
    Approximate reflected Doppler shift (Hz) for a colocated
    speaker-microphone sensor.

    Positive shift means the object is approaching the sensor.
    """
    sensor_position = np.asarray(sensor_position, dtype=float)
    sensor_velocity = np.asarray(sensor_velocity, dtype=float)
    object_position = np.asarray(object_position, dtype=float)
    object_velocity = np.asarray(object_velocity, dtype=float)

    displacement = object_position - sensor_position
    distance = np.linalg.norm(displacement)

    if distance < 1e-9:
        return np.nan

    direction = displacement / distance
    relative_velocity = object_velocity - sensor_velocity

    radial_velocity = np.dot(relative_velocity, direction)

    return -2.0 * frequency * radial_velocity / speed_of_sound
