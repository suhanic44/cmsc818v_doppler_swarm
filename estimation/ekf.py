"""Extended Kalman filter with the same Doppler model as batch estimation."""

import numpy as np

from estimation.batch_estimator import (
    DEFAULT_MEASUREMENT_LAG,
    estimate_initial_state_geometry,
    predict_doppler,
)


class ExtendedKalmanFilter:
    def __init__(
        self,
        sigma_doppler=1.0,
        acceleration_noise=0.1,
        measurement_lag=DEFAULT_MEASUREMENT_LAG,
    ):
        self.x = None
        self.P = None
        self.t = None
        self.sigma_doppler = sigma_doppler
        self.acceleration_noise = acceleration_noise
        self.measurement_lag = measurement_lag

    def initialize(self, times, uav_positions, measured_doppler, initial_covariance):
        """Fit state at time zero using geometry-informed batch initialization."""
        result = estimate_initial_state_geometry(
            times, uav_positions, measured_doppler,
            measurement_lag=self.measurement_lag,
        )
        if not result.success:
            raise RuntimeError("Batch initialization failed")
        self.x = result.x.copy()
        self.P = np.asarray(initial_covariance, dtype=float).copy()
        self.t = 0.0
        print("Optimizer evaluations:", result.nfev)
        print("Final residual cost:", result.cost)
        print("Optimizer message:", result.message)
        return self.x

    def predict(self, dt):
        """Advance state and covariance by dt seconds (constant velocity)."""
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        self.x = F @ self.x
        G = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))
        Q = self.acceleration_noise * (G @ G.T)
        self.P = F @ self.P @ F.T + Q
        self.t += dt
        return self.x

    def update(self, measured_doppler, uav_positions, uav_velocities=None):
        """Correct state at self.t using a completed acoustic window.

        The EKF state is at window-end availability time self.t.
        Supply UAV positions and velocities at self.t - measurement_lag
        (the effective reception time, near the window center).
        """
        z = np.asarray(measured_doppler, dtype=float).ravel()
        positions = np.asarray(uav_positions, dtype=float).reshape(-1, 3)
        if uav_velocities is None:
            velocities = np.zeros_like(positions)
        else:
            velocities = np.asarray(uav_velocities, dtype=float).reshape(-1, 3)
        if len(z) != len(positions) or velocities.shape != positions.shape:
            raise ValueError("One Doppler sample and velocity per UAV required")

        def h(state):
            return predict_doppler(
                state, np.array([0.0]), positions, delay_aware=True,
                measurement_lag=self.measurement_lag,
                uav_velocities=velocities,
            ).ravel()

        expected = h(self.x)
        H = np.empty((len(z), 6))
        eps = 1e-5
        for j in range(6):
            perturbed = self.x.copy()
            perturbed[j] += eps
            H[:, j] = (h(perturbed) - expected) / eps

        R = self.sigma_doppler**2 * np.eye(len(z))
        innovation = z - expected
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innovation

        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x


def main():
    """Synthetic regression check using the shared Doppler model."""
    initial_state = np.array([3., 3., 3., -2., -1.5, -1.])
    uav_positions = np.array([
        [0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]
    ])
    times = np.linspace(0.04, 0.5, 24)
    doppler = predict_doppler(
        initial_state, times, uav_positions,
        measurement_lag=DEFAULT_MEASUREMENT_LAG,
    )
    K = np.searchsorted(times, 0.2, side="right")
    ekf = ExtendedKalmanFilter()
    ekf.initialize(times[:K], uav_positions, doppler[:K], np.eye(6))
    ekf.predict(times[K - 1])
    ekf.x += np.array([0.2, -0.2, 0.1, 0.1, -0.1, 0.1])

    estimated, truth = [], []
    for k in range(K, len(times)):
        ekf.predict(times[k] - times[k - 1])
        ekf.update(doppler[k], uav_positions)
        actual = initial_state.copy()
        actual[:3] += initial_state[3:] * times[k]
        estimated.append(ekf.x.copy())
        truth.append(actual)
    estimated, truth = np.asarray(estimated), np.asarray(truth)
    print("Final true state:", truth[-1])
    print("Final estimated state:", estimated[-1])
    print("Mean position error:", np.mean(np.linalg.norm(estimated[:, :3] - truth[:, :3], axis=1)))
    print("Mean velocity error:", np.mean(np.linalg.norm(estimated[:, 3:] - truth[:, 3:], axis=1)))


if __name__ == "__main__":
    main()
