
import matplotlib.pyplot as plt
import numpy as np

from simulation.acoustic_sensor import (
    SAMPLE_RATE,
    generate_speaker_signal,
    generate_stationary_echo,
    generate_moving_echo
)

duration = 0.2

times, speaker = generate_speaker_signal(duration)

stationary_echo = generate_stationary_echo(
    times,
    projectile_distance=5.0
)

moving_echo = generate_moving_echo(
    times,
    initial_distance=5.0,
    approach_speed=3.0
)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Show a small window so individual cycles are visible
mask = (times >= 0.05) & (times <= 0.052)

axes[0].plot(times[mask], speaker[mask])
axes[0].set_title("Emitted Speaker Signal")

axes[1].plot(times[mask], stationary_echo[mask])
axes[1].set_title("Echo: Stationary Projectile")

axes[2].plot(times[mask], moving_echo[mask])
axes[2].set_title("Echo: Approaching Projectile")

for ax in axes:
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

plt.tight_layout()


# Estimate frequency from the moving echo
from scipy.signal import periodogram

# Ignore the beginning where the echo hasn't arrived
valid_echo = moving_echo[times > 0.05]

frequencies, power = periodogram(
    valid_echo,
    fs=SAMPLE_RATE,
    nfft=262144
)

# Find strongest frequency
peak_index = np.argmax(power)
measured_frequency = frequencies[peak_index]

print(f"Speaker frequency: 10000.00 Hz")
print(f"Measured echo frequency: {measured_frequency:.2f} Hz")
print(
    f"Measured Doppler shift: "
    f"{measured_frequency - 10000:.2f} Hz"
)

plt.show()
