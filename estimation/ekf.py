"""
Extended Kalman Filter (EKF) Implementation
- short-window initialization: run the existing batch estimator on only the first portion of measurements
- EKF: Take that initial estimate and continuously update it as new Doppler measurements arrive

NOTES:
- simple batch initialization was NOT sufficient --> improvement: multi-start batch init
"""

import numpy as np
from estimation.batch_estimator import estimate_initial_state_multistart


class ExtendedKalmanFilter:
    def __init__(self):
        self.x = None
        self.P = None
        self.t = None

    def initialize(
        self,
        times,
        uav_positions,
        measured_doppler,
        initial_covariance,
    ):
        # Estimate state at t = 0 using the initial measurement window
        result = estimate_initial_state_multistart(
            times,
            uav_positions,
            measured_doppler,
            n_starts=30,
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


    # EKF prediction step
    def predict(self, dt):
        """
        Predict the projectile's next state using constant velocity.
        """
        # State transition matrix
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt

        # Predict position and velocity
        self.x = F @ self.x

        # Process noise covariance
        q = 0.1  # Assumed acceleration noise intensity
        G = np.vstack([
            0.5 * dt**2 * np.eye(3),
            dt * np.eye(3),
        ])
        Q = q * (G @ G.T)

        # Propagate uncertainty
        self.P = F @ self.P @ F.T + Q

        self.t += dt

        return self.x
    

    # EKF update step
    def update(self, measured_doppler, uav_positions):
        """
        Correct the predicted state using Doppler measurements.

        measured_doppler: shape (N,)
        uav_positions: shape (N, 3)
        """
        from estimation.batch_estimator import predict_doppler

        z = np.asarray(measured_doppler, dtype=float).ravel()
        uav_positions = np.asarray(uav_positions, dtype=float)

        # Predict Doppler measurements from the current state
        z_pred = predict_doppler(
            self.x,
            np.array([0.0]),
            uav_positions
        ).ravel()

        # Compute measurement Jacobian numerically
        H = np.zeros((len(z), 6))
        eps = 1e-5

        for j in range(6):
            x_perturbed = self.x.copy()
            x_perturbed[j] += eps

            z_perturbed = predict_doppler(
                x_perturbed,
                np.array([0.0]),
                uav_positions
            ).ravel()

            H[:, j] = (z_perturbed - z_pred) / eps

        # Measurement noise covariance (Hz^2)
        sigma_doppler = 1.0
        R = sigma_doppler**2 * np.eye(len(z))

        # Innovation: observed minus predicted Doppler
        y = z - z_pred

        # Kalman gain
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T

        # Correct state estimate
        self.x = self.x + K @ y

        # Update uncertainty (Joseph form)
        I = np.eye(6)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

        return self.x

def main():
    from estimation.batch_estimator import predict_doppler

    # Ground-truth initial state
    true_state = np.array([3, 3, 3, -2, -1.5, -1])

    # Four stationary UAVs
    uav_positions = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ])

    # Simulate 0.5 seconds of Doppler measurements
    times = np.linspace(0, 0.5, 51)
    measured_doppler = predict_doppler(
        true_state, times, uav_positions
    )

    # Initialize using only the first 200 ms
    K = np.searchsorted(times, 0.2, side="right")

    ekf = ExtendedKalmanFilter()
    ekf.initialize(
        times=times[:K],
        uav_positions=uav_positions,
        measured_doppler=measured_doppler[:K],
        initial_covariance=np.eye(6),
    )

    # Advance initial estimate from t=0 to t=0.2
    ekf.predict(times[K - 1])
    ekf.x += np.array([0.2, -0.2, 0.1, 0.1, -0.1, 0.1])  # adding in a little of noise to see the perturbations do affect the error

    estimated_states = [ekf.x.copy()]
    true_states = []

    initial_true_state = true_state.copy()
    initial_true_state[:3] += true_state[3:] * times[K - 1]
    true_states.append(initial_true_state)

    # Process subsequent measurements chronologically
    for k in range(K, len(times)):
        dt = times[k] - times[k - 1]

        ekf.predict(dt)
        ekf.update(measured_doppler[k], uav_positions)  # commenting this out shows EKF is improving estimates

        estimated_states.append(ekf.x.copy())

        actual = true_state.copy()
        actual[:3] += true_state[3:] * times[k]
        true_states.append(actual)

    estimated_states = np.array(estimated_states)
    true_states = np.array(true_states)

    position_errors = np.linalg.norm(
        estimated_states[:, :3] - true_states[:, :3], axis=1
    )
    velocity_errors = np.linalg.norm(
        estimated_states[:, 3:] - true_states[:, 3:], axis=1
    )

    print("\nFinal true state:     ", true_states[-1])
    print("Final estimated state:", np.round(estimated_states[-1], 4))
    print(f"Final position error: {position_errors[-1]:.6f} m")
    print(f"Final velocity error: {velocity_errors[-1]:.6f} m/s")
    print(f"Mean position error:  {position_errors.mean():.6f} m")
    print(f"Mean velocity error:  {velocity_errors.mean():.6f} m/s")


if __name__ == "__main__":
    main()