## Doppler Swarm: Lightweight Trajectory-Informed Collision Avoidance via Acoustic Sensing for Low-SWaP Multi-UAV Systems
CMSC818V Term Research Project -- Gili Gordiyenko & Suhani Chaudhary

***Current Project State:***

We've built and tested the acoustic sensing, trajectory estimation, collision prediction, and avoidance planning components. We can now estimate an incoming projectile's trajectory from simulated Doppler measurements and calculate how the UAVs should move to avoid it.

- **PyBullet:** Simulate four stationary UAVs and a moving projectile.
- **Acoustic sensing:** Generate simulated microphone recordings that account for sound travel and reflection time.
- **Doppler extraction:** Process recordings to measure frequency shifts as the projectile moves.
- **Trajectory estimation:** Use batch least-squares optimization to estimate the projectile's initial position and velocity, then continuously update the estimate using an EKF.
- **Results:** Tested five projectile trajectories, achieving roughly 7–21 cm position RMSE and 0.05–0.17 m/s velocity RMSE.
- **Collision prediction:** Predict which UAVs are at risk and when the projectile will get too close. All 9 tests passed.
- **Avoidance planning:** Use SQP optimization to find the smallest coordinated UAV movements that avoid collisions with the projectile and other UAVs. All 7 tests passed.
- **k-Wave:** Experimented with simulating actual sound reflections, but haven't integrated this into the main simulation.

***Up Next:***

1. **Test UAV avoidance in PyBullet:** Actually move the UAVs according to our planned avoidance maneuvers and verify they avoid collisions.
2. **Integrate with online EKF:** Connect acoustic sensing and trajectory estimation to the avoidance planner so the UAVs react based on estimated trajectories rather than ground truth.

***After That:***

Run experiments across different trajectories, swarm configurations, and noise levels; compare against our proposed baselines; and evaluate collision rates, displacement, accuracy, and runtime.
