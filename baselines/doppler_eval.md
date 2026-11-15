# DopplerDodge Baseline Evaluation

## Implementation Overview

- **Acoustic detection:** Each UAV independently processes raw reflected audio using DopplerDodge's cumulative drift method to classify incoming projectiles as HIT or MISS.
- **Threshold calibration:** Determine the HIT/MISS threshold using separate labeled trajectories.
- **Random avoidance:** UAVs predicting HIT execute a random 3D maneuver; UAVs predicting MISS remain stationary.
- **Simulation:** Run in PyBullet using the same UAV configurations, projectile trajectories, and physical constraints as Doppler Swarm.
- **Evaluation:** Test the five existing scenarios (direct, faster, diagonal, offset, none), then expand to randomized trajectories and denser swarms. Repeat across multiple random seeds.

## Evaluation Metrics

- **Collision-free success rate:** Percentage of trials without any collisions.
- **Projectile–UAV collision rate:** Percentage of trials with at least one projectile–UAV collision.
- **UAV–UAV collision rate:** Percentage of trials with at least one inter-UAV collision.
- **Safety-margin success rate:** Percentage of trials maintaining required safety distances.
- **Minimum separation:** Closest projectile–UAV and UAV–UAV distances.
- **Total UAV displacement:** Sum of distances traveled by all UAVs.
- **Number of UAVs moved:** Number of UAVs executing avoidance maneuvers.
- **HIT recall:** Fraction of genuinely threatened UAVs correctly classified as HIT.
- **False-positive rate:** Fraction of nonthreatened UAVs incorrectly classified as HIT.
- **Detection latency:** Time required to produce a HIT/MISS decision.
- **Avoidance response time:** Time from detection to maneuver initiation.

## Comparison

Evaluate Doppler Swarm on the same scenarios using identical physical parameters and common safety, movement, and latency metrics. Report trajectory estimation accuracy separately for Doppler Swarm.

The shared research evaluation now includes `dopplerdodge` as a fourth method:

```powershell
python -m experiments.main_evaluation --output results/main_02 --reuse-results results/main_01
```

This reuses saved encounters and existing method rows in a new directory, adds
the baseline, and leaves the original results unchanged. A fresh run without
`--reuse-results` collects the expanded metrics for every method. See
[RESEARCH_EXPERIMENTS.md](../RESEARCH_EXPERIMENTS.md) for metric definitions,
calibration provenance, and handling of unavailable metrics in older records.
