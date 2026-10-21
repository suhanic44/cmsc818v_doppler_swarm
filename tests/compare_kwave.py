import numpy as np
import matplotlib.pyplot as plt

original = np.load("pressure_with_sphere.npy")
closer = np.load("pressure_sphere_closer.npy")
baseline = np.load("pressure_without_sphere.npy")

dt = 0.05 * 0.005 / 1500.0
times = np.arange(len(original)) * dt * 1000

plt.figure(figsize=(10, 4))

plt.plot(times, original - baseline, label="Original sphere position")
plt.plot(times, closer - baseline, label="Sphere 2.5 cm closer")

plt.xlabel("Time (ms)")
plt.ylabel("Scattered pressure")
plt.legend()
plt.grid(True)
plt.show()