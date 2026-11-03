
import numpy as np
from scipy.optimize import least_squares

SPEED_OF_SOUND = 343.0
SPEAKER_FREQUENCY = 10000.0

# 40 ms audio window, evaluated approximately at its center.
DEFAULT_MEASUREMENT_LAG = 0.02


def predict_doppler(
    state,
    times,
    uav_positions,
    frequency=SPEAKER_FREQUENCY,
    c=SPEED_OF_SOUND,
    delay_aware=True,
    measurement_lag=0.0,
):
    """
    Predict Doppler measurements for a constant-velocity target.

    state: [px, py, pz, vx, vy, vz] at t=0
    times: shape (K,), relative to the state reference time
    uav_positions: shape (N, 3)

    times represent measurement availability timestamps.

    measurement_lag shifts the effective acoustic observation
    time earlier than the availability timestamp.

    When delay_aware=True, solve separately for each UAV's
    reflection time, accounting for acoustic propagation.

    Returns: shape (K, N)

    Positive Doppler shift means approaching.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    times = np.atleast_1d(
        np.asarray(times, dtype=float)
    )
    uav_positions = np.asarray(
        uav_positions, dtype=float
    ).reshape(-1, 3)

    p0 = state[:3]
    v = state[3:]

    # Effective acoustic observation time.
    observation_times = times - measurement_lag

    # One reflection time per UAV and observation.
    reflection_times = np.broadcast_to(
        observation_times[:, None],
        (len(times), len(uav_positions)),
    ).copy()

    if delay_aware:
        # Solve:
        # t_observe = t_reflect + distance(t_reflect)/c
        for _ in range(15):
            positions = (
                p0[None, None, :]
                + reflection_times[..., None] * v
            )

            distances = np.linalg.norm(
                positions - uav_positions[None, :, :],
                axis=2,
            )

            reflection_times = (
                observation_times[:, None]
                - distances / c
            )

    positions = (
        p0[None, None, :]
        + reflection_times[..., None] * v
    )

    displacement = (
        positions - uav_positions[None, :, :]
    )

    distances = np.linalg.norm(displacement, axis=2)

    directions = (
        displacement
        / np.maximum(distances[..., None], 1e-8)
    )

    # Positive radial velocity means moving away.
    radial_velocity = np.sum(
        directions * v[None, None, :],
        axis=2,
    )

    # Exact moving-reflector Doppler formula.
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
    frequency=SPEAKER_FREQUENCY,
    c=SPEED_OF_SOUND,
    measurement_lag=DEFAULT_MEASUREMENT_LAG,
):
    """
    Estimate initial 3D position and velocity
    from received Doppler measurements.
    """

    def residuals(state):
        predicted = predict_doppler(
            state,
            times,
            uav_positions,
            frequency=frequency,
            c=c,
            delay_aware=True,
            measurement_lag=measurement_lag,
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


def estimate_initial_state_multistart(
    times,
    uav_positions,
    measured_doppler,
    n_starts=30,
    position_bounds=(-5, 5),
    velocity_bounds=(-5, 5),
    seed=42,
    measurement_lag=DEFAULT_MEASUREMENT_LAG,
):
    """
    Run batch estimation from multiple random initial guesses.

    Return the solution with the lowest residual cost.
    """
    rng = np.random.default_rng(seed)
    best_result = None

    for _ in range(n_starts):
        initial_guess = np.concatenate([
            rng.uniform(*position_bounds, size=3),
            rng.uniform(*velocity_bounds, size=3),
        ])

        result = estimate_initial_state(
            times,
            uav_positions,
            measured_doppler,
            initial_guess,
            measurement_lag=measurement_lag,
        )

        if (
            best_result is None
            or result.cost < best_result.cost
        ):
            best_result = result

    return best_result
