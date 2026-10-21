
import numpy as np
import matplotlib.pyplot as plt
import kwave

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder

# 3D acoustic domain
N = 96
dx = 0.005  # 5 mm grid spacing
shape = (N, N, N)

c_air = 343.0
rho_air = 1.2
frequency = 10_000.0

kgrid = kWaveGrid(
    [N, N, N],
    [dx, dx, dx]
)

dt = 0.05 * dx / 1500.0
duration = 0.0025
Nt = int(duration / dt)
kgrid.setTime(Nt, dt)

# 3D sphere representing the projectile
# sphere_center = np.array([64, 48, 48])
sphere_center = np.array([59, 48, 48])
sphere_radius = 10  # grid cells = 5 cm

x, y, z = np.ogrid[:N, :N, :N]

# sphere_mask = (
#     (x - sphere_center[0])**2
#     + (y - sphere_center[1])**2
#     + (z - sphere_center[2])**2
#     <= sphere_radius**2
# )
sphere_mask = (
    (x - sphere_center[0])**2 +
    (y - sphere_center[1])**2 +
    (z - sphere_center[2])**2 <= 10**2
)

# Air with a contrasting spherical material.
# This is an acoustic impedance approximation,
# not an exact rigid-body boundary.
sound_speed = np.full(shape, c_air)
density = np.full(shape, rho_air)

sound_speed[sphere_mask] = 1500.0
density[sphere_mask] = 1000.0

medium = kWaveMedium(
    sound_speed=sound_speed,
    density=density
)

# A pressure source and nearby microphone
speaker_idx = (25, 48, 48)
mic_idx = (25, 51, 48)

source = kSource()
source.p_mask = np.zeros(shape, dtype=bool)
source.p_mask[speaker_idx] = True

times = np.arange(Nt) * dt

# Short 10 kHz burst
burst_duration = 0.0005
burst = (
    np.sin(2 * np.pi * frequency * times)
    * (times < burst_duration)
)

# k-Wave expects a 2D array:
# (number of pressure sources, number of time samples)
source.p = burst.reshape(1, -1).astype(np.float32)

sensor_mask = np.zeros(shape, dtype=bool)
sensor_mask[mic_idx] = True
sensor = kSensor(mask=sensor_mask)

# Run wave propagation
binary_path = str(
    kwave.BINARY_PATH / "kspaceFirstOrder-omp.exe"
)

result = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    pml_size=10,
    pml_inside=True,
    backend="cpp",
    device="cpu",
    binary_path=binary_path
)

pressure = np.asarray(result["p"]).squeeze()
np.save("pressure_sphere_closer.npy", pressure)

print("Pressure min:", np.min(pressure))
print("Pressure max:", np.max(pressure))
print("All finite:", np.all(np.isfinite(pressure)))
print("Number of samples:", len(pressure))
print("Expected samples:", Nt)

plt.figure(figsize=(10, 4))
plt.plot(times[:len(pressure)], pressure)
plt.xlabel("Time (s)")
plt.ylabel("Acoustic pressure")
plt.title("3D k-Wave microphone recording")
plt.grid(True)
plt.tight_layout()
plt.show()
