# Main evaluation diagnostics

Source: `results/research_main_01/metrics.csv` (18000 method-trial records).

Read-only analysis; no simulations rerun.

## 1. Four-UAV Doppler Swarm failures

- Attempts: **600**
- Returned: **600**
- Safe: **369**
- Unsafe: **231**
- Physical-contact failures: **141**

### Failure distribution

| Speed (m/s) | Encounter | Safe | Unsafe | Other |
|---:|---|---:|---:|---:|
| 1 | direct_threat | 29 | 21 | 0 |
| 1 | grazing_threat | 30 | 20 | 0 |
| 1 | multiple_threats | 40 | 10 | 0 |
| 1 | safe_miss | 50 | 0 | 0 |
| 3 | direct_threat | 24 | 26 | 0 |
| 3 | grazing_threat | 26 | 24 | 0 |
| 3 | multiple_threats | 14 | 36 | 0 |
| 3 | safe_miss | 50 | 0 | 0 |
| 5 | direct_threat | 15 | 35 | 0 |
| 5 | grazing_threat | 23 | 27 | 0 |
| 5 | multiple_threats | 18 | 32 | 0 |
| 5 | safe_miss | 50 | 0 | 0 |

### Successful vs. unsuccessful trials

| Diagnostic | Safe | Unsafe |
|---|---:|---:|
| Initialization succeeded | 369/369 (100.0%) | 231/231 (100.0%) |
| No successful planner call | 0/369 (0.0%) | 0/231 (0.0%) |
| No first avoidance command | 0/369 (0.0%) | 0/231 (0.0%) |
| Command after first safety entry | 0/219 (0.0%) | 0/231 (0.0%) |
| Median position RMSE (m) | 0.208 | 1.386 |
| Median velocity RMSE (m/s) | 0.158 | 1.463 |
| Median first estimate time (s) | 0.220 | 0.220 |
| Median first command time (s) | 0.221 | 0.221 |
| Median planning attempts | 17.0 | 16.0 |
| Median successful plans | 17.0 | 16.0 |
| Median total UAV movement (m) | 0.067 | 0.032 |
| Median minimum projectile distance (m) | 0.131 | 0.048 |
| Projectile physical collisions | 0/369 (0.0%) | 141/231 (61.0%) |
| UAV-to-UAV physical collisions | 0/369 (0.0%) | 0/231 (0.0%) |

### Sample failed encounters

| Encounter ID | Speed | Category | Initialized | Position RMSE | Successful/attempted plans |
|---|---:|---|---|---:|---:|
| `s8675309_n4_direct_threat_000_v1` | 1.0 | direct_threat | True | 1.045 | 23/23 |
| `s8675309_n4_direct_threat_000_v5` | 5.0 | direct_threat | True | 5.242 | 14/14 |
| `s8675309_n4_direct_threat_001_v1` | 1.0 | direct_threat | True | 0.252 | 23/24 |
| `s8675309_n4_direct_threat_001_v3` | 3.0 | direct_threat | True | 1.446 | 16/16 |
| `s8675309_n4_direct_threat_001_v5` | 5.0 | direct_threat | True | 5.448 | 14/14 |
| `s8675309_n4_direct_threat_003_v5` | 5.0 | direct_threat | True | 3.576 | 15/15 |
| `s8675309_n4_direct_threat_004_v1` | 1.0 | direct_threat | True | 1.400 | 26/26 |
| `s8675309_n4_direct_threat_004_v3` | 3.0 | direct_threat | True | 4.206 | 17/17 |
| `s8675309_n4_direct_threat_004_v5` | 5.0 | direct_threat | True | 2.537 | 15/15 |
| `s8675309_n4_direct_threat_006_v5` | 5.0 | direct_threat | True | 0.967 | 14/14 |
| `s8675309_n4_direct_threat_007_v5` | 5.0 | direct_threat | True | 4.118 | 15/15 |
| `s8675309_n4_direct_threat_008_v3` | 3.0 | direct_threat | True | 3.546 | 15/15 |

**Interpretation:** Compare successful and failed trials to identify associations with estimation error, initialization, planning, or late commands. These associations do not by themselves establish the cause of a failure.

## 2. Coordinated DopplerDodge timeouts

The process timeout is a wall-clock computation limit, not a simulated collision deadline. Timed-out trials have unknown physical safety outcomes.

### Runtime scaling

| UAVs | Attempts | Returned | Timeouts | Errors | Median runtime (s) | P95 runtime (s) | Median planning time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 600 | 600 | 0 | 0 | 1.376 | 2.373 | 0.327 |
| 6 | 600 | 565 | 0 | 35 | 1.087 | 1.448 | 0.441 |
| 8 | 600 | 599 | 1 | 0 | 1.734 | 2.438 | 1.035 |
| 10 | 600 | 598 | 1 | 1 | 3.900 | 9.439 | 2.870 |
| 16 | 600 | 600 | 0 | 0 | 13.103 | 17.341 | 12.013 |
| 25 | 600 | 389 | 211 | 0 | 50.104 | 58.934 | 48.638 |

### 25-UAV outcomes

- Total attempts: **600**
- Returned: **389**
- Timeouts: **211**
- Errors: **0**
- Safety among returned trials: **347/389 (89.2%)**
- Successful coordinated plans among returned trials: **389/389 (100.0%)**
- Median elapsed time of timed-out attempts: **60.008 s**

### Timeouts by speed and encounter type

| Speed (m/s) | Encounter | Timeouts | Safety among returned |
|---:|---|---:|---:|
| 1 | direct_threat | 14/50 (28.0%) | 30/36 (83.3%) |
| 1 | grazing_threat | 16/50 (32.0%) | 32/34 (94.1%) |
| 1 | multiple_threats | 18/50 (36.0%) | 27/32 (84.4%) |
| 1 | safe_miss | 15/50 (30.0%) | 33/35 (94.3%) |
| 3 | direct_threat | 25/50 (50.0%) | 22/25 (88.0%) |
| 3 | grazing_threat | 25/50 (50.0%) | 24/25 (96.0%) |
| 3 | multiple_threats | 30/50 (60.0%) | 19/20 (95.0%) |
| 3 | safe_miss | 16/50 (32.0%) | 31/34 (91.2%) |
| 5 | direct_threat | 13/50 (26.0%) | 31/37 (83.8%) |
| 5 | grazing_threat | 11/50 (22.0%) | 33/39 (84.6%) |
| 5 | multiple_threats | 16/50 (32.0%) | 30/34 (88.2%) |
| 5 | safe_miss | 12/50 (24.0%) | 35/38 (92.1%) |

### Sample timed-out encounter IDs

`s8675309_n25_direct_threat_000_v1`, `s8675309_n25_direct_threat_000_v3`, `s8675309_n25_direct_threat_000_v5`, `s8675309_n25_direct_threat_001_v3`, `s8675309_n25_direct_threat_002_v1`, `s8675309_n25_direct_threat_004_v3`, `s8675309_n25_direct_threat_004_v5`, `s8675309_n25_direct_threat_005_v1`, `s8675309_n25_direct_threat_005_v3`, `s8675309_n25_direct_threat_006_v1`, `s8675309_n25_direct_threat_006_v3`, `s8675309_n25_direct_threat_006_v5`

**Interpretation:** Increasing runtime and timeout frequency with swarm size supports a computational-scaling explanation. However, terminated workers do not return detailed internal planner metrics. The saved results cannot definitively identify the exact operation responsible for each timeout.

## Before using these findings in the paper

1. Distinguish safety-margin failures from physical collisions.
2. Report computational timeouts separately from collisions.
3. Treat observed associations as possible explanations.
4. Compare movement on common successful encounters.
