
"""
Extended Kalman Filter (EKF) for acoustic trajectory estimation.

- Multistart batch initialization
- Constant-velocity state prediction
- Delay-aware Doppler measurement correction
- Causal acoustic window timing
"""

import numpy as np

from estimation.batch_estimator import (
    DEFAULT_MEASUREMENT_LAG,
    estimate_initial_state_multistart,
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

    def initialize(
        self,
        times,
        uav_positions,
        measured_doppler,
        initial_covariance,
    ):
        """
        Estimate the state at t=0 using the initial
        Doppler measurement window.
        """
        result = estimate_initial_state_multistart(
            times,
            uav_positions,
            measured_doppler,
            n_starts=30,
            measurement_lag=self.measurement_lag,
        )

        if not result.success:
            raise RuntimeError("Batch initialization failed")

        self.x = result.x.copy()
        self.P = np.asarray(
            initial_covariance, dtype=float
        ).copy()
        self.t = 0.0

        print("Optimizer evaluations:", result.nfev)
        print("Final residual cost:", result.cost)
        print("Optimizer message:", result.message)

        return self.x

    def predict(self, dt):
        """
        Predict the projectile's next state using
        a constant-velocity model.
        """
        if dt < 0:
            raise ValueError("dt must be nonnegative")

        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt

        self.x = F @ self.x

        # Process noise covariance.
        q = self.acceleration_noise

        G = np.vstack([
            0.5 * dt**2 * np.eye(3),
            dt * np.eye(3),
        ])

        Q = q * (G @ G.T)

        self.P = F @ self.P @ F.T + Q
        self.t += dt

        return self.x

    def update(self, measured_doppler, uav_positions):
        """
        Correct the current state estimate using Doppler.

        The state is referenced to the current measurement
        availability time. The acoustic observation describes
        an earlier time, handled by measurement_lag.
        """
        z = np.asarray(
            measured_doppler, dtype=float
        ).ravel()

        uav_positions = np.asarray(
            uav_positions, dtype=float
        )

        def measurement_model(state):
            return predict_doppler(
                state,
                np.array([0.0]),
                uav_positions,
                delay_aware=True,
                measurement_lag=self.measurement_lag,
            ).ravel()

        z_pred = measurement_model(self.x)

        # Numerical measurement Jacobian.
        H = np.zeros((len(z), 6))
        eps = 1e-5

        for j in range(6):
            x_perturbed = self.x.copy()
            x_perturbed[j] += eps

            z_perturbed = measurement_model(x_perturbed)

            H[:, j] = (
                z_perturbed - z_pred
            ) / eps

        R = (
            self.sigma_doppler**2
            * np.eye(len(z))
        )

        y = z - z_pred

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ y

        # Joseph-form covariance update.
        I = np.eye(6)
        A = I - K @ H

        self.P = (
            A @ self.P @ A.T
            + K @ R @ K.T
        )

        return self.x


def main():
    """
    Standalone synthetic EKF test.

    Measurements are generated from the same delay-aware
    model used by the estimator.
    """
    true_state = np.array([
        3., 3., 3.,
        -2., -1.5, -1.,
    ])

    uav_positions = np.array([
        [0., 0., 0.],
        [2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.],
    ])

    times = np.linspace(0.04, 0.5, 24)

    measured_doppler = predict_doppler(
        true_state,
        times,
        uav_positions,
        measurement_lag=DEFAULT_MEASUREMENT_LAG,
    )

    K = np.searchsorted(times, 0.2, side="right")

    ekf = ExtendedKalmanFilter()

    ekf.initialize(
        times=times[:K],
        uav_positions=uav_positions,
        measured_doppler=measured_doppler[:K],
        initial_covariance=np.eye(6),
    )

    ekf.predict(times[K - 1])

    # Perturb the initial estimate to test correction.
    ekf.x += np.array([
        0.2, -0.2, 0.1,
        0.1, -0.1, 0.1,
    ])

    estimated_states = []
    true_states = []

    for k in range(K, len(times)):
        dt = times[k] - times[k - 1]

        ekf.predict(dt)
        ekf.update(
            measured_doppler[k],
            uav_positions,
        )

        estimated_states.append(ekf.x.copy())

        actual = true_state.copy()
        actual[:3] += true_state[3:] * times[k]
        true_states.append(actual)

    estimated_states = np.asarray(estimated_states)
    true_states = np.asarray(true_states)

    position_errors = np.linalg.norm(
        estimated_states[:, :3] - true_states[:, :3],
        axis=1,
    )

    velocity_errors = np.linalg.norm(
        estimated_states[:, 3:] - true_states[:, 3:],
        axis=1,
    )

    print("\nFinal true state:", true_states[-1])
    print("Final estimated state:", estimated_states[-1])
    print(f"Mean position error: {position_errors.mean():.6f} m")
    print(f"Mean velocity error: {velocity_errors.mean():.6f} m/s")


if __name__ == "__main__":
    main()
