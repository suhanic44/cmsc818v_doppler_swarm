## Doppler Swarm: Lightweight Trajectory-Informed Collision Avoidance via Acoustic Sensing for Low-SWaP Multi-UAV Systems
CMSC818V Term Research Project -- Gili Gordiyenko & Suhani Chaudhary 

***Current Project State:***
We run the full PyBullet simulation to record the ground-truth trajectories of the stationary UAVs and moving projectile, use those trajectories to generate synthetic acoustic signals, extract Doppler measurements from those signals, and feed all the measurements into our batch estimator to estimate the projectile’s initial 3D position and velocity
- PyBullet: We simulate four stationary UAVs and a moving projectile, and record the projectile's trajectory.
- Acoustic sensing: We use the projectile's movement to generate simulated microphone recordings, accounting for how long sound takes to travel to the projectile and reflect back.
- Doppler extraction: We process those recordings to measure the frequency shifts caused by the projectile's movement.
- Trajectory estimation: We combine Doppler measurements from all four UAVs and use least-squares optimization to estimate the projectile's initial position and velocity.
- Results: In our initial test, we got approximately 11 cm position error and 0.06 m/s velocity error.
- k-Wave: We also experimented with simulating actual sound waves reflecting off a sphere. We got reflections that changed with sphere position, but haven't integrated this into the main simulation yet.

***Up Next:***
Online estimation with Extended Kalman Filter (EKF)



