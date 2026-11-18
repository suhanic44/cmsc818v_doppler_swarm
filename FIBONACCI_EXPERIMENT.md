# Supplementary experiment: Fibonacci-sphere formations

Research question: how often does acoustic trajectory estimation enable safe
coordinated avoidance, relative to the same planner with perfect projectile state?

This supplementary study fixes UAV positions on a centered Fibonacci sphere.
It evaluates safety across swarm size and projectile speed for that formation
family. The primary evaluation uses randomized volumetric formations at controlled
spacing; see [MAIN_RESEARCH_PROTOCOL.md](MAIN_RESEARCH_PROTOCOL.md).

The earlier `results/main_01` and `results/main_02` directories are now named
`results/fibonacci_01` and `results/fibonacci_02`. Their contents, manifests,
source snapshots and measurement values are unchanged. Historical provenance may
still use the original names. Use `experiments.fibonacci_evaluation` to run or
summarize this supplementary study. Shared metrics and controller instrumentation
live in `experiments/evaluation_common.py`; the primary research runner imports
those utilities independently.

## Run and read

From this directory:

```powershell
.venv\Scripts\python.exe -m experiments.fibonacci_evaluation --output results/fibonacci_03
```

The full experiment runs 2,400 encounters, each with five methods (12,000 method
trials, including the inexpensive analytic static baseline). It can take hours;
the duration depends on initialization/planner difficulty and your computer.
No full evaluation is launched automatically when developing the runner.

Every completed method trial is saved immediately. To continue an interrupted run:

```powershell
.venv\Scripts\python.exe -m experiments.fibonacci_evaluation --output results/fibonacci_03 --resume
```

Resume skips saved trials, including recorded errors. It rejects changed source,
configuration, environment, or encounter data. A truncated JSONL line after a
hard shutdown produces a parsing error rather than silently discarding evidence.
Only one process should write to an output directory.

Version 3 adds coordinated Doppler Dodge. To add it to a previous four-method
study without rerunning compatible saved methods:

```powershell
.venv\Scripts\python.exe -m experiments.fibonacci_evaluation --output results/fibonacci_03 --reuse-results results/fibonacci_02
```

The source directory is read only. Saved static/oracle/estimate/independent-Doppler
Dodge rows are imported with provenance; missing rows are executed. A complete
four-method study needs only 2,400 new coordinated-baseline trials. Three-method
version-1 studies can also be imported, adding both Dodge variants.
Controllers, encounter settings and environment must match the source study;
the changed reporting/instrumentation runner is allowed. Calibration/threshold
settings are saved in the new manifest. Existing thresholds are inherited without
recalibration, and a conflicting explicit threshold is rejected. Resume the new
directory using `--resume` without `--reuse-results`. Old studies cannot be
expanded in place under a newer protocol.

Old records do not contain per-UAV motion or first HIT/MISS decisions. Their new
metrics remain unavailable, not guessed from final displacement or first successful
command time. For the full new metric set on every method, run a fresh directory
without `--reuse-results` (all 12,000 trials). Report denominators follow the saved
manifest; unchanged older studies retain their three/four-method counts.

Summaries are generated at completion or Ctrl+C. You can regenerate them without
running simulations:

```powershell
.venv\Scripts\python.exe -m experiments.fibonacci_evaluation --output results/fibonacci_03 --summarize
```

* `report.md`: readable safety rates, intervals, and error counts, with encounter
  classes kept separate.
* `summary.csv`: one row per size/speed/class/method; includes both types of
  collision counts, runtime and movement summaries.
* `metrics.csv`: convenient trial-level table for analysis or plotting.
* `results.jsonl`: durable raw results, including individual planning runtimes.
* `encounters.jsonl`: saved projectile states, UAV positions, classification and
  analytic no-avoidance clearances. All methods use these same encounters.
* `manifest.json`: study settings, source hashes, Python/platform and packages.

The normal study uses seed 314159 and 50 repetitions per condition. The seed and
repetition count are explicit CLI options for a predeclared evaluation budget or
a separate held-out evaluation, not separate smoke/pilot modes. If you alter
them, pass the same options when resuming. Do not tune on your final evaluation.

## Fixed protocol

| Factor | Values |
|---|---|
| UAV count | 4, 6, 8, 10 |
| Projectile speed | 1, 3, 5 m/s |
| Encounter category | Direct threat, grazing threat, multiple threats, safe miss |
| Repetitions per combination | 50 |
| Formation | Centered spatial Fibonacci formation, 1 m RMS radius |
| Variation within a condition | Random rigid formation rotation, approach, target and offset |
| Nominal warning | 1.2 s to the selected target point |
| Duration | Target time + 0.4 s; multiple-threat cases also cover flight to the second target |
| Added measurement noise / network delay | None / zero for this first study |
| UAV speed / maneuver duration | Existing implementation: 3 m/s / 0.15 s |

At fixed warning, faster projectiles start farther away. This protocol measures
speed effects with equal nominal warning, not equal starting distance. The
actual earliest safety entry is recorded separately and can precede target time.

Larger swarms have the same RMS formation radius: sensor geometry, spacing, and
the number of potential victims therefore change with size. This study measures
the total effect of swarm size, not an isolated benefit from extra sensors.

Class membership is checked analytically against every stationary UAV, over the
complete encounter, before a trajectory is admitted:

1. Two or more UAVs would violate 0.10 m clearance: **multiple threats**.
2. Otherwise, a UAV would violate 0.07 m contact separation: **direct threat**.
3. Otherwise, a UAV would violate 0.10 m clearance: **grazing threat**.
4. Otherwise: **safe miss**.

Multiple-threat trajectories pass through two selected UAV centers. Single-threat
and miss trajectories use randomized perpendicular offsets. Candidate geometry
is rejected if any initial safety violation exists or the requested category is
not met at all three speeds. Nominal offset ranges are 0–0.05 m (direct),
0.075–0.095 m (grazing), and 0.12–0.40 m (miss).

Each accepted geometry is shared across speeds. Within a size/speed/category
condition, the 50 geometry draws have independent random seeds. Across speeds
the geometry draws are paired; do not treat the resulting 150 rows as independent
geometries. Balanced encounter categories are a designed challenge distribution,
not an estimate of real-world encounter prevalence.

## Methods and outcomes

* `static`: exact closest approach to stationary UAVs; analytic reference.
* `oracle`: current online planner receives true simulated projectile state from
  the start; no acoustic initialization is required.
* `estimate`: current online acoustic sensing, initialization, EKF, planning and
  movement. True state is used for simulation and scoring, not estimation.
* `dopplerdodge`: each UAV independently classifies raw acoustic echoes with the
  partner's cumulative-drift detector, then performs one seeded random 3D dodge
  if classified HIT. Uses the identical saved starting geometry, projectile
  state, duration, UAV speed and maneuver duration. No EKF, SQP, or shared sensing.
* `dopplerdodge_coordinated`: same detector, threshold, observation window and
  random seed as independent Doppler Dodge. Random preferred HIT movements are
  jointly adjusted by the partner's SLSQP planner to maintain UAV-UAV clearance.
  Its planner receives only HIT/MISS, UAV positions and a random generator,
  never projectile state. It does not constrain clearance from the unknown
  projectile trajectory. MISS UAVs may move to accommodate others. Its native
  minimum HIT movement remains 0.40 m (maximum 0.45 m), while independent Dodge
  commands 0.45 m; this compares implemented policies, not coordination alone.

Oracle is a privileged reference, not a mathematical upper bound or a globally
optimal trajectory. Its difference from estimate includes startup and update
timing. Independent trajectory-based avoidance remains a separate comparison.

Doppler Dodge's native `calibrate_threshold` is called once before any evaluation
trials, on its separate calibration trajectories (default count 50, seed 42).
That calibration uses the baseline's original formation and 1.5–4 m/s speed
distribution; it is not retuned for the evaluation swarms, classes, or speeds.
The threshold and calibration log are frozen in the manifest and reused on resume.
`--dodge-threshold VALUE` accepts an independently preselected frozen threshold;
`--dodge-calibration-count` and `--dodge-calibration-seed` configure calibration.
Declare settings before final evaluation, not after inspecting test outcomes.

The random-dodge seed is derived deterministically from study seed and encounter
ID, stored on each result row. There is one random dodge realization per encounter,
not multiple action-seed repetitions. Separate encounters provide action variation.

Coordinated Dodge also records `coordinated_plan_success` and `reactive_uavs_moved`
(MISS-classified UAVs that actually move). Failed coordinated plans are not
executed: UAVs stay stationary and the encounter is still scored. Classifications
are retained even if planning fails. Native named-scenario baseline CLIs still work.

Primary outcome: **safety clearance satisfied throughout the recorded simulation**.
Projectile contact (0.07 m) and UAV contact (0.10 m) are recorded separately from
required clearance (0.10 m projectile; 0.15 m UAV). Errors remain in the safety
denominator; collision outcomes of errored simulations are unknown, not zero.
The existing evaluator uses a 1e-6 m comparison tolerance. Online distances are
sampled at 240 Hz; static distances are analytic. This is not a continuous-time
guarantee for online motion.

Additional metrics in `metrics.csv` and `summary.csv`:

| Metric | Definition |
|---|---|
| Collision-free success rate | Fraction of all attempted trials with no projectile or UAV collisions; errors are unsuccessful |
| Projectile/UAV collision rates | Fraction among collision-evaluable trials; unknown error outcomes are excluded and coverage recorded |
| Safety-margin success rate | Fraction of all attempted trials preserving both required clearances |
| Minimum separations | Trial minima; condition-level minimum, mean, median and p95 |
| Total UAV displacement | Cumulative distance traveled, summed across UAVs, including replanning; not just endpoint displacement |
| Number moved | Distinct UAVs whose cumulative travel exceeds 1e-6 m |
| HIT recall | Sum of true-positive first decisions divided by genuinely threatened UAVs in classified trials |
| False-positive rate | Sum of false-positive first decisions divided by genuinely nonthreatened UAVs in classified trials |
| Detection latency | Simulated observation time to the first complete HIT/MISS decision vector |
| Avoidance response time | Simulated time from that decision to the first maneuver command, including failed-plan/replanning delay |

Static has no detector: HIT/FPR and detection/response latency are N/A. HIT recall
is N/A when no UAV was genuinely threatened, and FPR is N/A when none were
nonthreatened. Classification coverage and raw TP/FP/positive/negative counts
are recorded; missing decisions or imported legacy instrumentation are not zeros.

Truth labels use each UAV's analytic closest approach if the original formation
stayed stationary over the complete encounter, including threats before detection.
For oracle/estimate, the scored predictions are the first actual call to the
planner's collision predictor, even when the subsequent optimization fails.
Its existing 1 cm planning buffer remains active, so a 0.10–0.11 m predicted
pass can be called HIT but be a false positive against evaluation's 0.10 m truth.
Doppler Dodge contributes its one native independent decision per UAV. Later
replanning classifications are not substituted retrospectively for first decisions.

Response time measures command initiation, so an immediate response legitimately
has value zero. `first_motion_sim_s` separately records first nonzero movement;
the next simulation step may be later than the command. Detection/response compute
costs are recorded separately as `detection_compute_s` and
`avoidance_response_compute_s`. They are not physically applied execution delays.
For the coordinated methods, detection compute includes estimation through first
classification and that collision prediction; response compute covers planning
through first successful command. Doppler Dodge measures raw-audio classification
and random-command generation separately; these stage boundaries differ from
the EKF pipeline and must not be treated as identical audio-processing workloads.

Initialization success/time, position/velocity RMSE, plan attempts/successes,
individual planner runtimes, and whole-call wall runtime remain available.
Trajectory-estimation metrics are N/A for Doppler Dodge. RMSE covers available
estimates and must be interpreted alongside initialization failures and first-estimate time.

Movement summaries condition on safety success. Compare paired methods on
their common successful encounters before claiming a movement advantage.
Do not count low displacement on failed trials as efficient avoidance.

## Timing is separate from simulation safety

This runner temporarily wraps calls in memory to record durations, restoring all
functions afterward. It does not edit simulation, planner or estimator files.

It records:

* Estimation-processing and planning totals across the encounter.
* Those two compute totals **only through the first successful command**.
* First simulated command time.
* Acoustic `advance()` wall time separately: that function mixes synthetic audio
  generation with signal processing, so it is not pure real-sensor processing cost.

For Doppler Dodge, the analogous prefix fields contain detector classification
and random-command generation costs, and its echo-generation cost is separate.

The partial timing screen is:

`first simulated command time + prefix estimation compute + prefix planning compute + 0.15 s`

It compares this with the first stationary-swarm safety entry. Later replans are
not charged against the first command. No command means the screen fails for a
threat; misses have no deadline and receive null. No-avoidance has no controller
timing screen. Initialization time is already included in estimation processing
and is not added twice.

This is explicitly a **partial, serialized timing screen**. It excludes acoustic
processing and networking, pessimistically adds earlier measured processing to
the simulated acquisition timeline, does not model overlap/queues, and requires
the entire maneuver to finish before stationary safety entry. Passing does not
establish real-time safety; failing does not establish collision. Compute delays
do not change simulated motion. Exact deadline evidence requires per-command
causal scheduling and delayed execution in a separate timing experiment.

## Paper analysis

Plot safety success versus speed, separate panels for size and class, with one
curve per method and Wilson 95% intervals. Keep misses separate from threats.
Fifty trials per condition is a starting budget; even 50/50 successes has a lower
Wilson bound of about 93%, so this does not establish near-perfect reliability.

Use encounter IDs to compare oracle and estimate on paired trials. For inference
across speeds, resample geometry IDs rather than individual method/speed rows.
Keep development seeds separate from held-out final seeds. Report hardware and
avoid competing heavy workloads when collecting runtime measurements.

This first study supports an end-to-end comparison under idealized sensing. It
does not yet establish noise robustness, constant-density scaling, coordination's
advantage over independent planning, hardware low-power suitability, or safety
with actual compute delays. Those are subsequent research experiments, not
claims inferred from this table.
