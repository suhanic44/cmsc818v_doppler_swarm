
import numpy as np
from scipy.optimize import least_squares


def predict_doppler(
    state,
    times,
    uav_positions,
    frequency=10000.0,
    c=343.0
):
    """
    Predict Doppler measurements for a constant-velocity target.

    state: [px, py, pz, vx, vy, vz] at t=0
    times: shape (K,)
    uav_positions: shape (N, 3)

    Returns: shape (K, N)

    Positive Doppler shift means the target is approaching.
    """
    p0 = state[:3]
    v = state[3:]

    # Target positions at each measurement time
    positions = (
        p0[None, :]
        + times[:, None] * v[None, :]
    )

    # Vectors from each UAV to the target
    displacement = (
        positions[:, None, :]
        - uav_positions[None, :, :]
    )

    distances = np.linalg.norm(
        displacement,
        axis=2
    )

    directions = (
        displacement
        / np.maximum(distances[..., None], 1e-8)
    )

    # Radial velocity:
    # Positive = moving away
    # Negative = approaching
    radial_velocity = np.sum(
        directions * v[None, None, :],
        axis=2
    )

    # Exact Doppler formula for a moving reflector
    # under constant radial motion.
    #
    # f_received = f_sent * (c - vr) / (c + vr)
    #
    # Doppler shift = f_received - f_sent
    return frequency * (
        (c - radial_velocity)
        / (c + radial_velocity)
        - 1.0
    )


def estimate_initial_state(
    times,
    uav_positions,
    measured_doppler,
    initial_guess,
    frequency=10000.0,
    c=343.0,
):
    """
    Estimate initial 3D position and velocity
    from Doppler measurements.
    """

    def residuals(state):
        predicted = predict_doppler(
            state,
            times,
            uav_positions,
            frequency,
            c
        )

        return (
            predicted - measured_doppler
        ).ravel()

    result = least_squares(
        residuals,
        x0=np.asarray(initial_guess, dtype=float),
        max_nfev=3000,
    )

    return result
