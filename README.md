## Doppler Swarm: Lightweight Trajectory-Informed Collision Avoidance via Acoustic Sensing for Low-SWaP Multi-UAV Systems
CMSC818V Term Research Project -- Gili Gordiyenko & Suhani Chaudhary

***Current Project State:***

We've built and tested the full pipeline, from acoustic sensing and trajectory estimation to collision prediction and avoidance. We can now estimate an incoming projectile's trajectory from simulated Doppler measurements and move UAVs to avoid it.

- **PyBullet:** Simulate four UAVs and a moving projectile, including UAV movement.
- **Acoustic sensing:** Generate simulated microphone recordings that account for sound travel and reflection time, even while UAVs move.
- **Doppler extraction:** Process recordings to measure frequency shifts as the projectile moves.
- **Trajectory estimation:** Use geometry-informed batch least-squares optimization to estimate the projectile's initial position and velocity, then update the estimate using an EKF.
- **Collision prediction:** Predict which UAVs are at risk and when the projectile will get too close. All 9 tests passed.
- **Avoidance planning:** Use SQP optimization to find the smallest UAV movements needed to avoid collisions. All 7 tests passed.
- **End-to-end results:** Tested five projectile trajectories. Four achieved 2–7 cm position RMSE and 0.02–0.05 m/s velocity RMSE. All five avoided collisions and met safety margins, but one had a large estimation error.
- **k-Wave:** Experimented with simulating actual sound reflections, but haven't integrated this into the main simulation.

***Up Next:***

1. **Fix trajectory estimation:** Investigate why initialization fails for one of the five trajectories.
2. **Measure runtime:** Test how long initialization, EKF updates, and avoidance planning take.

***After That:***

Run experiments across different trajectories, swarm configurations, and noise levels; compare against our baselines; and evaluate collision rates, displacement, accuracy, and runtime.
