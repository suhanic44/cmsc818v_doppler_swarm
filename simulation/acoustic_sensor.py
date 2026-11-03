
import numpy as np
from scipy.signal import hilbert

SPEED_OF_SOUND = 343.0
SPEAKER_FREQUENCY = 10000.0
SAMPLE_RATE = 96000


def generate_speaker_signal(duration):
    """
    Generate a continuous sinusoidal speaker signal.

    Returns:
        times: timestamps of audio samples
        signal: emitted sound waveform
    """
    times = np.arange(0, duration, 1 / SAMPLE_RATE)
    signal = np.sin(2 * np.pi * SPEAKER_FREQUENCY * times)
    return times, signal


def generate_stationary_echo(times, projectile_distance, amplitude=0.5):
    """
    Simulate the echo from a stationary reflector.
    """
    round_trip_delay = 2 * projectile_distance / SPEED_OF_SOUND
    delayed_times = times - round_trip_delay

    echo = np.where(
        delayed_times >= 0,
        amplitude * np.sin(
            2 * np.pi * SPEAKER_FREQUENCY * delayed_times
        ),
        0.0,
    )
    return echo


def generate_moving_echo(
    times, initial_distance, approach_speed, amplitude=0.5
):
    """
    Echo from a projectile moving directly toward a stationary,
    colocated speaker/microphone.

    approach_speed > 0 means approaching.
    """
    c = SPEED_OF_SOUND

    reflection_times = (
        c * times - initial_distance
    ) / (c - approach_speed)

    reflection_distances = (
        initial_distance - approach_speed * reflection_times
    )

    emission_times = (
        reflection_times - reflection_distances / c
    )

    valid = (
        (emission_times >= 0)
        & (reflection_times >= 0)
        & (reflection_distances > 0)
    )

    echo = np.zeros_like(times)
    echo[valid] = amplitude * np.sin(
        2 * np.pi * SPEAKER_FREQUENCY * emission_times[valid]
    )
    return echo


def generate_trajectory_echo(
    audio_times,
    physics_times,
    projectile_positions,
    sensor_position,
    amplitude=0.5,
):
    """
    Generate a reflected acoustic waveform using a
    3D projectile trajectory sampled from PyBullet.

    Assumes a stationary, colocated speaker/microphone.
    """
    physics_times = np.asarray(physics_times)
    projectile_positions = np.asarray(projectile_positions)
    sensor_position = np.asarray(sensor_position)

    def position_at(query_times):
        return np.column_stack([
            np.interp(
                query_times,
                physics_times,
                projectile_positions[:, axis],
            )
            for axis in range(3)
        ])

    # Solve for reflection time:
    # t_receive = t_reflect + distance(t_reflect) / c
    reflection_times = audio_times.copy()

    for _ in range(12):
        positions = position_at(reflection_times)
        distances = np.linalg.norm(
            positions - sensor_position, axis=1
        )
        reflection_times = (
            audio_times - distances / SPEED_OF_SOUND
        )

    positions = position_at(reflection_times)
    distances = np.linalg.norm(
        positions - sensor_position, axis=1
    )

    emission_times = (
        reflection_times - distances / SPEED_OF_SOUND
    )

    valid = (
        (emission_times >= 0)
        & (reflection_times >= physics_times[0])
        & (reflection_times <= physics_times[-1])
    )

    echo = np.zeros_like(audio_times)
    echo[valid] = amplitude * np.sin(
        2 * np.pi
        * SPEAKER_FREQUENCY
        * emission_times[valid]
    )

    return echo


def estimate_doppler(
    audio_times,
    echo,
    window_duration=0.04,
    hop_duration=0.02,
):
    """
    Causal window-based Doppler extraction.

    Each window uses only samples available by its end.
    The returned timestamp is the availability time.

    The frequency estimate uses the central half of the
    completed window to reduce Hilbert boundary artifacts.

    Returns:
        measurement_times: window-end timestamps
        doppler_values: Doppler shifts in Hz
    """
    audio_times = np.asarray(audio_times, dtype=float)
    echo = np.asarray(echo, dtype=float)

    if len(audio_times) != len(echo):
        raise ValueError("Audio timestamps and echo lengths differ")

    window_size = int(round(window_duration * SAMPLE_RATE))
    hop_size = int(round(hop_duration * SAMPLE_RATE))

    if window_size < 8 or hop_size < 1:
        raise ValueError("Invalid window or hop duration")

    measurement_times = []
    doppler_values = []

    for start in range(
        0, len(echo) - window_size + 1, hop_size
    ):
        end = start + window_size
        segment = echo[start:end]

        # Reject weak or missing echoes.
        if np.max(np.abs(segment)) < 0.1:
            continue

        # Reflection padding reduces finite-window artifacts.
        pad = min(window_size // 2, 1024)
        padded = np.pad(
            segment, (pad, pad), mode="reflect"
        )

        analytic = hilbert(padded)[pad:-pad]
        envelope = np.abs(analytic)

        # Use only the central half of the window.
        margin = window_size // 4
        central_envelope = envelope[margin:-margin]

        if np.min(central_envelope) < 0.1:
            continue

        phase = np.unwrap(np.angle(analytic))
        central_phase = phase[margin:-margin]
        phase_steps = np.diff(central_phase)

        measured_frequency = (
            np.median(phase_steps)
            * SAMPLE_RATE
            / (2 * np.pi)
        )

        # The estimate is available at the final sample.
        measurement_times.append(audio_times[end - 1])

        doppler_values.append(
            measured_frequency - SPEAKER_FREQUENCY
        )

    return (
        np.asarray(measurement_times),
        np.asarray(doppler_values),
    )
