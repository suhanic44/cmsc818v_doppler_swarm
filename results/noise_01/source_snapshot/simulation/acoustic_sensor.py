
"""Causal monostatic acoustic sensing with stationary or moving UAVs."""

import numpy as np
from scipy.signal import hilbert

SPEED_OF_SOUND = 343.0
SPEAKER_FREQUENCY = 10000.0
SAMPLE_RATE = 96000


def generate_speaker_signal(duration):
    times = np.arange(0.0, duration, 1.0 / SAMPLE_RATE)
    return times, np.sin(2 * np.pi * SPEAKER_FREQUENCY * times)


def generate_stationary_echo(times, projectile_distance, amplitude=0.5):
    emission_times = np.asarray(times) - 2 * projectile_distance / SPEED_OF_SOUND
    return np.where(
        emission_times >= 0,
        amplitude * np.sin(2 * np.pi * SPEAKER_FREQUENCY * emission_times),
        0.0,
    )


def generate_moving_echo(times, initial_distance, approach_speed, amplitude=0.5):
    """1D echo for an approaching reflector and stationary speaker/mic."""
    times = np.asarray(times, dtype=float)
    c = SPEED_OF_SOUND
    reflection_times = (c * times - initial_distance) / (c - approach_speed)
    distances = initial_distance - approach_speed * reflection_times
    emission_times = reflection_times - distances / c
    valid = (emission_times >= 0) & (reflection_times >= 0) & (distances > 0)
    echo = np.zeros_like(times)
    echo[valid] = amplitude * np.sin(2 * np.pi * SPEAKER_FREQUENCY * emission_times[valid])
    return echo


def _positions_at(query_times, history_times, history_positions):
    """Interpolate known past positions; callers reject extrapolated times."""
    return np.column_stack([
        np.interp(query_times, history_times, history_positions[:, axis])
        for axis in range(3)
    ])


def _echo_from_history(
    audio_times, physics_times, projectile_positions, sensor_positions, amplitude=0.5
):
    """Echo phase is the source phase at the retarded emission time.

    t_receive = t_reflect + ||p(t_reflect) - s(t_receive)|| / c
    t_reflect = t_emit + ||p(t_reflect) - s(t_emit)|| / c
    Both are solved using only history available at the reception time.
    """
    audio_times = np.asarray(audio_times, dtype=float)
    physics_times = np.asarray(physics_times, dtype=float)
    projectile_positions = np.asarray(projectile_positions, dtype=float)
    sensor_positions = np.asarray(sensor_positions, dtype=float)
    echo = np.zeros_like(audio_times)
    if len(physics_times) < 2 or not len(audio_times):
        return echo

    receive_positions = _positions_at(audio_times, physics_times, sensor_positions)
    reflection_times = audio_times.copy()
    for _ in range(12):
        reflected_positions = _positions_at(
            reflection_times, physics_times, projectile_positions
        )
        return_distance = np.linalg.norm(reflected_positions - receive_positions, axis=1)
        reflection_times = audio_times - return_distance / SPEED_OF_SOUND

    reflected_positions = _positions_at(
        reflection_times, physics_times, projectile_positions
    )
    emission_times = reflection_times.copy()
    for _ in range(12):
        emit_positions = _positions_at(emission_times, physics_times, sensor_positions)
        outbound_distance = np.linalg.norm(reflected_positions - emit_positions, axis=1)
        emission_times = reflection_times - outbound_distance / SPEED_OF_SOUND

    valid = (
        (emission_times >= physics_times[0])
        & (reflection_times >= physics_times[0])
        & (reflection_times <= physics_times[-1])
        & (audio_times <= physics_times[-1])
    )
    echo[valid] = amplitude * np.sin(
        2 * np.pi * SPEAKER_FREQUENCY * emission_times[valid]
    )
    return echo


def generate_trajectory_echo(
    audio_times, physics_times, projectile_positions, sensor_position, amplitude=0.5
):
    """Offline echo for a fixed UAV from a sampled 3D projectile path."""
    physics_times = np.asarray(physics_times, dtype=float)
    stationary_positions = np.broadcast_to(
        np.asarray(sensor_position, dtype=float).reshape(1, 3),
        (len(physics_times), 3),
    )
    return _echo_from_history(
        audio_times,
        physics_times,
        projectile_positions,
        stationary_positions,
        amplitude=amplitude,
    )


def estimate_doppler(audio_times, echo, window_duration=0.04, hop_duration=0.02):
    """Causal Hilbert-phase frequency estimate from completed audio windows.

    Timestamp = last audio sample (availability time); measurement represents
    the central half of the preceding window, approximately 20 ms earlier.
    """
    audio_times = np.asarray(audio_times, dtype=float)
    echo = np.asarray(echo, dtype=float)
    if len(audio_times) != len(echo):
        raise ValueError("Audio timestamps and echo lengths differ")
    window_size = int(round(window_duration * SAMPLE_RATE))
    hop_size = int(round(hop_duration * SAMPLE_RATE))
    if window_size < 8 or hop_size < 1:
        raise ValueError("Invalid window or hop duration")

    measurement_times, doppler_values = [], []
    for start in range(0, len(echo) - window_size + 1, hop_size):
        end = start + window_size
        segment = echo[start:end]
        if np.max(np.abs(segment)) < 0.1:
            continue
        pad = min(window_size // 2, 1024)
        analytic = hilbert(np.pad(segment, (pad, pad), mode="reflect"))[pad:-pad]
        margin = window_size // 4
        if np.min(np.abs(analytic[margin:-margin])) < 0.1:
            continue
        central_phase = np.unwrap(np.angle(analytic))[margin:-margin]
        frequency = np.median(np.diff(central_phase)) * SAMPLE_RATE / (2 * np.pi)
        measurement_times.append(audio_times[end - 1])
        doppler_values.append(frequency - SPEAKER_FREQUENCY)
    return np.asarray(measurement_times), np.asarray(doppler_values)


class StreamingAcousticSensor:
    """Causal streaming echo for a stationary UAV (legacy interface)."""

    def __init__(
        self, sensor_position, duration, window_duration=0.04, hop_duration=0.02
    ):
        self.sensor_position = np.asarray(sensor_position, dtype=float).reshape(3)
        self.duration = float(duration)
        self.window_size = int(round(window_duration * SAMPLE_RATE))
        self.hop_size = int(round(hop_duration * SAMPLE_RATE))
        if self.window_size < 8 or self.hop_size < 1:
            raise ValueError("Invalid window or hop duration")

        self.total_samples = len(np.arange(0, self.duration, 1.0 / SAMPLE_RATE))
        self.physics_times = []
        self.projectile_positions = []
        self.next_audio_index = 0
        self.audio_start_index = 0
        self.audio_buffer = np.empty(0, dtype=float)
        self.next_window_start = 0
        self.measurement_times = []
        self.doppler_values = []

    def _generate_echo_chunk(self, audio_times):
        history_times = np.asarray(self.physics_times)
        sensor_history = np.broadcast_to(
            self.sensor_position, (len(history_times), 3)
        )
        return _echo_from_history(
            audio_times,
            history_times,
            np.asarray(self.projectile_positions),
            sensor_history,
        )

    def advance(self, physics_time, projectile_position):
        """Accept the next physics observation and return newly ready windows."""
        physics_time = float(physics_time)
        projectile_position = np.asarray(projectile_position, dtype=float).reshape(3)
        if self.physics_times and physics_time <= self.physics_times[-1]:
            raise ValueError("Physics timestamps must be strictly increasing")
        self.physics_times.append(physics_time)
        self.projectile_positions.append(projectile_position.copy())

        available_end = min(
            self.total_samples,
            int(np.floor(physics_time * SAMPLE_RATE + 1e-8)) + 1,
        )
        if available_end > self.next_audio_index:
            indices = np.arange(self.next_audio_index, available_end)
            new_audio_times = indices / SAMPLE_RATE
            new_echo = self._generate_echo_chunk(new_audio_times)
            self.audio_buffer = np.concatenate((self.audio_buffer, new_echo))
            self.next_audio_index = available_end

        new_times, new_doppler = [], []
        while self.next_window_start + self.window_size <= self.next_audio_index:
            start = self.next_window_start
            end = start + self.window_size
            local_start = start - self.audio_start_index
            local_end = end - self.audio_start_index
            segment = self.audio_buffer[local_start:local_end]
            segment_times = np.arange(start, end) / SAMPLE_RATE
            times, doppler = estimate_doppler(
                segment_times,
                segment,
                window_duration=self.window_size / SAMPLE_RATE,
                hop_duration=self.hop_size / SAMPLE_RATE,
            )
            if len(times):
                new_times.append(times[0])
                new_doppler.append(doppler[0])
                self.measurement_times.append(times[0])
                self.doppler_values.append(doppler[0])
            self.next_window_start += self.hop_size

        discard = self.next_window_start - self.audio_start_index
        if discard > 0:
            self.audio_buffer = self.audio_buffer[discard:]
            self.audio_start_index = self.next_window_start
        return np.asarray(new_times, dtype=float), np.asarray(new_doppler, dtype=float)


class MovingStreamingAcousticSensor(StreamingAcousticSensor):
    """Same monostatic echo physics, using actual historical UAV positions."""

    def __init__(
        self, sensor_position, duration, window_duration=0.04, hop_duration=0.02
    ):
        super().__init__(sensor_position, duration, window_duration, hop_duration)
        self.sensor_positions = []

    def _generate_echo_chunk(self, audio_times):
        return _echo_from_history(
            audio_times,
            np.asarray(self.physics_times),
            np.asarray(self.projectile_positions),
            np.asarray(self.sensor_positions),
        )

    def advance(self, physics_time, projectile_position, sensor_position):
        """Store UAV position *before* parent computes the acoustic samples."""
        sensor_position = np.asarray(sensor_position, dtype=float).reshape(3)
        # Validate before appending, so sensor/projectile histories stay aligned.
        if self.physics_times and physics_time <= self.physics_times[-1]:
            raise ValueError("Physics timestamps must be strictly increasing")
        self.sensor_positions.append(sensor_position.copy())
        return super().advance(physics_time, projectile_position)
