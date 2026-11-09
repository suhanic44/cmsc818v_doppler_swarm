
"""Shared acoustic Doppler model and geometry-informed batch trajectory estimator."""

import numpy as np
from scipy.optimize import least_squares

SPEED_OF_SOUND = 343.0
SPEAKER_FREQUENCY = 10000.0
DEFAULT_MEASUREMENT_LAG = 0.02  # 40 ms audio window: effective sample at center


def predict_doppler(
    state,
    times,
    uav_positions,
    frequency=SPEAKER_FREQUENCY,
    c=SPEED_OF_SOUND,
    delay_aware=True,
    measurement_lag=0.0,
    uav_velocities=None,
):
    """Predict Doppler (Hz) from a moving reflector and monostatic UAVs.

    state: [projectile position(3), velocity(3)] at time zero.
    times: (K,) window-end availability timestamps in the state's time frame.
    uav_positions: (N, 3) at each sample's effective reception time
                   (window end minus measurement_lag).
    uav_velocities: (N, 3) at that same time, held constant over the
                    approximately millisecond-scale acoustic round trip.
    Returns: (K, N), positive for approaching motion.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    uav_positions = np.asarray(uav_positions, dtype=float).reshape(-1, 3)

    if uav_velocities is None:
        uav_velocities = np.zeros_like(uav_positions)
    else:
        uav_velocities = np.asarray(
            uav_velocities, dtype=float
        ).reshape(-1, 3)

    if uav_velocities.shape != uav_positions.shape:
        raise ValueError(
            "UAV positions and velocities must have matching shapes"
        )

    p0, v = state[:3], state[3:]
    receive_times = times[:, None] - measurement_lag
    receive_positions = uav_positions[None, :, :]

    # Return path: t_receive = t_reflect + ||p(t_reflect)-s(t_receive)|| / c.
    reflection_times = np.broadcast_to(
        receive_times, (len(times), len(uav_positions))
    ).copy()

    if delay_aware:
        for _ in range(12):
            reflected = p0 + reflection_times[..., None] * v
            distances = np.linalg.norm(
                reflected - receive_positions, axis=-1
            )
            reflection_times = receive_times - distances / c

    reflected = p0 + reflection_times[..., None] * v

    # Outbound path: t_reflect = t_emit + ||p(t_reflect)-s(t_emit)|| / c.
    emission_times = reflection_times.copy()

    if delay_aware:
        for _ in range(12):
            emitted_from = (
                receive_positions
                + (emission_times - receive_times)[..., None]
                * uav_velocities[None, :, :]
            )
            distances = np.linalg.norm(
                reflected - emitted_from, axis=-1
            )
            emission_times = reflection_times - distances / c

    emitted_from = (
        receive_positions
        + (emission_times - receive_times)[..., None]
        * uav_velocities[None, :, :]
    )

    outbound = reflected - emitted_from
    inbound = reflected - receive_positions

    outbound /= np.maximum(
        np.linalg.norm(outbound, axis=-1, keepdims=True), 1e-8
    )
    inbound /= np.maximum(
        np.linalg.norm(inbound, axis=-1, keepdims=True), 1e-8
    )

    vp_out = np.sum(outbound * v, axis=-1)
    vp_in = np.sum(inbound * v, axis=-1)
    vu_out = np.sum(
        outbound * uav_velocities[None, :, :], axis=-1
    )
    vu_in = np.sum(
        inbound * uav_velocities[None, :, :], axis=-1
    )

    received = (
        frequency
        * (c - vp_out) / (c - vu_out)
        * (c + vu_in) / (c + vp_in)
    )

    return received - frequency


def estimate_initial_state(
    times,
    uav_positions,
    measured_doppler,
    initial_guess,
    frequency=SPEAKER_FREQUENCY,
    c=SPEED_OF_SOUND,
    measurement_lag=DEFAULT_MEASUREMENT_LAG,
    max_nfev=3000,
):
    """Fit constant-velocity initial state from stationary-UAV Doppler data."""
    measured_doppler = np.asarray(measured_doppler, dtype=float)

    def residuals(state):
        prediction = predict_doppler(
            state,
            times,
            uav_positions,
            frequency=frequency,
            c=c,
            delay_aware=True,
            measurement_lag=measurement_lag,
        )

        if measured_doppler.shape != prediction.shape:
            raise ValueError(
                "Measured Doppler must have shape (times, UAVs)"
            )

        return (prediction - measured_doppler).ravel()

    return least_squares(
        residuals,
        x0=np.asarray(initial_guess, dtype=float),
        max_nfev=max_nfev,
    )


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
    """Select the lowest-residual solution among random initial guesses."""
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(n_starts):
        guess = np.concatenate((
            rng.uniform(*position_bounds, size=3),
            rng.uniform(*velocity_bounds, size=3),
        ))

        result = estimate_initial_state(
            times,
            uav_positions,
            measured_doppler,
            guess,
            measurement_lag=measurement_lag,
        )

        if best is None or result.cost < best.cost:
            best = result

    return best


def estimate_initial_state_geometry(
    times,
    uav_positions,
    measured_doppler,
    grid_points=11,
    n_refine=30,
    position_bounds=(-5, 5),
    velocity_bounds=(-5, 5),
    measurement_lag=DEFAULT_MEASUREMENT_LAG,
):
    import time
    total_start = time.perf_counter()

    """Search positions geometrically, infer velocities, then refine with full Doppler model."""
    times = np.asarray(times, dtype=float)
    uav_positions = np.asarray(
        uav_positions, dtype=float
    ).reshape(-1, 3)
    measured_doppler = np.asarray(measured_doppler, dtype=float)

    if measured_doppler.shape != (len(times), len(uav_positions)):
        raise ValueError(
            "Measured Doppler must have shape (times, UAVs)"
        )

    # Approximate projectile position at the center of the observation window.
    t_ref = float(np.mean(times) - measurement_lag)
    mean_doppler = np.mean(measured_doppler, axis=0)

    # Exact inversion of the stationary monostatic Doppler formula,
    # assuming each UAV-to-projectile direction is fixed over the window.
    radial_velocity = (
        -SPEED_OF_SOUND * mean_doppler
        / (2 * SPEAKER_FREQUENCY + mean_doppler)
    )

    grid = np.linspace(*position_bounds, grid_points)
    candidates = []

    for px in grid:
        for py in grid:
            for pz in grid:
                position = np.array([px, py, pz])

                directions = position - uav_positions
                distances = np.linalg.norm(directions, axis=1)

                # Unit directions are undefined at a UAV's location.
                if np.any(distances < 0.1):
                    continue

                unit_vectors = directions / distances[:, None]

                # u_i dot velocity = radial_velocity_i.
                velocity, _, rank, _ = np.linalg.lstsq(
                    unit_vectors,
                    radial_velocity,
                    rcond=None,
                )

                if rank < 3:
                    continue

                if np.any(velocity < velocity_bounds[0]) or np.any(
                    velocity > velocity_bounds[1]
                ):
                    continue

                # Convert candidate position at t_ref to position at t=0.
                initial_position = position - velocity * t_ref
                state = np.concatenate([initial_position, velocity])

                # Score the entire measurement window with the full model.
                predicted = predict_doppler(
                    state,
                    times,
                    uav_positions,
                    measurement_lag=measurement_lag,
                )

                cost = 0.5 * np.sum(
                    (predicted - measured_doppler) ** 2
                )

                if np.isfinite(cost):
                    candidates.append((cost, state))

    if not candidates:
        raise RuntimeError(
            "No valid geometry initialization candidates"
        )
    
    # import /time

    candidates.sort(key=lambda item: item[0])

    # Experimental parameters
    n_screen = 100
    screen_evaluations = 5
    n_final = 10

    print("\n=== TWO-STAGE INITIALIZATION ===")
    print("Total grid candidates:", len(candidates))
    print("Screening candidates:", n_screen)
    print("Screening budget:", screen_evaluations)
    print("Full refinements:", n_final)

    start_time = time.perf_counter()

    screened = []
    screening_nfev = 0

    # -------------------------------------------
    # Stage 1: Short optimization runs
    # -------------------------------------------

    for rank, (initial_cost, guess) in enumerate(
        candidates[:n_screen],
        start=1,
    ):
        result = estimate_initial_state(
            times,
            uav_positions,
            measured_doppler,
            initial_guess=guess,
            measurement_lag=measurement_lag,
            max_nfev=screen_evaluations,
        )

        screening_nfev += result.nfev

        if rank in (45, 46):
            print(
                f"DEBUG rank={rank} | "
                f"initial_cost={initial_cost:.4f} | "
                f"screened_cost={result.cost:.4f} | "
                f"screened_state={np.round(result.x, 3)}"
            )

        if np.isfinite(result.cost):
            improvement = (
                (initial_cost - result.cost)
                / max(abs(initial_cost), 1e-12)
            )

            screened.append((
                result.cost,
                rank,
                result.x.copy(),
                improvement,
            ))

    if not screened:
        raise RuntimeError(
            "No valid candidates after screening"
        )

    # -------------------------------------------
    # Select finalists using a hybrid strategy
    # -------------------------------------------

    # Half selected by lowest screened cost,
    # half by greatest relative improvement.
    n_by_cost = n_final // 2

    by_cost = sorted(
        screened,
        key=lambda item: item[0],
    )

    by_improvement = sorted(
        screened,
        key=lambda item: item[3],
        reverse=True,
    )

    finalists = []
    selected_ranks = set()

    # First: candidates with lowest screened costs
    for item in by_cost[:n_by_cost]:
        finalists.append(item)
        selected_ranks.add(item[1])

    # Second: candidates with greatest improvement
    for item in by_improvement:
        if item[1] in selected_ranks:
            continue

        finalists.append(item)
        selected_ranks.add(item[1])

        if len(finalists) >= n_final:
            break

    print("\n=== SELECTED FINALISTS ===")
    for cost, rank, state, improvement in finalists:
        print(
            f"  Original rank={rank:3d} | "
            f"Screened cost={cost:.4f} | "
            f"Improvement={improvement:.1%}"
        )

    print("===========================\n")

    # -------------------------------------------
    # Stage 2: Full nonlinear optimization
    # -------------------------------------------

    best = None
    best_rank = None
    refinement_nfev = 0

    for screened_cost, rank, guess, improvement in finalists:
        result = estimate_initial_state(
            times,
            uav_positions,
            measured_doppler,
            initial_guess=guess,
            measurement_lag=measurement_lag,
            max_nfev=3000,
        )

        refinement_nfev += result.nfev

        if best is None or result.cost < best.cost:
            best = result
            best_rank = rank

    elapsed = time.perf_counter() - start_time

    # -------------------------------------------
    # Results
    # -------------------------------------------

    print("\n=== TWO-STAGE RESULTS ===")
    print("Best original grid rank:", best_rank)
    print("Final residual cost:", best.cost)
    print("Estimated state:", np.round(best.x, 4))
    print("Screening nfev:", screening_nfev)
    print("Refinement nfev:", refinement_nfev)
    print("Total nfev:", screening_nfev + refinement_nfev)
    print(f"Screening + refinement time: {elapsed:.3f} s")
    print("=========================\n")

    total_elapsed = time.perf_counter() - total_start
    print(f"TOTAL INITIALIZATION TIME: {total_elapsed:.4f} seconds")

    return best
