# Focused exploratory size and spacing experiment

Question: how do the acoustic-estimated method, independent Doppler Dodge and
coordinated Doppler Dodge perform as swarm size and crowding increase? The experiment does
not assume either method will improve or deteriorate.

## Run

```powershell
.venv\Scripts\python.exe -m experiments.scaling --output results/scaling_02
```

Defaults:

| Factor | Values |
|---|---|
| UAV count | 10, 25, 50, 100 |
| RMS formation radius | 1, 2, 4 m |
| Encounter classes | Direct threat, multiple threats, safe miss |
| Geometries per size/radius/class | 3 |
| Projectile speed | 3 m/s |
| Time to selected target point | 1.2 s |
| Methods | Estimated coordinated avoidance, independent/coordinated Doppler Dodge, analytic stationary reference |
| Added noise | None |
| Controller process runtime cap | 60 seconds per trial |

This is **108 encounters, 324 controller simulations and 108 inexpensive static
references: 432 result records**. It is smaller in encounter count than the main
study; each large-swarm trial can be much more expensive. If every controller
run hits its cap, controller waiting alone would take 5.4 hours. Three geometries
per condition are for screening trends and computational limits, not establishing
paper-level reliability percentages. Larger final evaluation can use the same
protocol with a predeclared larger `--repetitions` value and a fresh directory.

The threshold is frozen from `results/fibonacci_02/manifest.json` by default. Use
`--reference OTHER_MAIN_DIRECTORY` for another completed/calibrated main study.
Only its threshold/calibration provenance are reused, not its small-swarm
encounters. An explicit `--dodge-threshold VALUE` is also supported. No calibration
or tuning uses this study's evaluation outcomes. The baseline implementation
must match the implementation that produced the reference threshold.

Examples of explicit settings:

```powershell
.venv\Scripts\python.exe -m experiments.scaling --output results/scaling_02 --swarm-sizes 10 25 50 100 --radii 1 2 4 --repetitions 3 --trial-timeout 60
.venv\Scripts\python.exe -m experiments.scaling --output results/scaling_02 --resume
.venv\Scripts\python.exe -m experiments.scaling --output results/scaling_02 --summarize
```

To add only the missing coordinated baseline to an existing scaling study:

```powershell
.venv\Scripts\python.exe -m experiments.scaling --output results/scaling_02 --reuse-results results/scaling_01
```

Use the old study's custom settings too, if any. Compatible saved methods,
including timeouts/errors, are imported into the new directory with provenance.
The original encounters and calibrated threshold are retained. For a completed
default old study, only 108 new coordinated-baseline runs are needed. The source
directory is never modified. Changed detector/estimator/planner implementations
or environment prevent reuse; the updated experiment adapters are allowed.

Resume requires the original configuration/source/environment, including custom
CLI settings if you used them. It skips completed errors and timeouts as well as
successes. To reevaluate with a larger runtime cap, use a new output directory.
An interrupted unfinished method is rerun; completed method rows are flushed
immediately. Only one writer should use a directory. Hard-crash partial JSONL
records raise an error rather than being silently deleted.

## What changing size and radius means

The existing centered Fibonacci spherical-shell formation is scaled to each RMS
radius. UAVs are around the shell, not spread throughout its interior. Increasing
UAV count at fixed radius increases crowding; increasing radius at fixed count
spreads them out. Approximately uniform spacing does not mean equal pair distances.

The data records actual minimum UAV spacing and median nearest-neighbor distance.
The reported N/(4 pi R²) is only a shell-crowding proxy, not measured volumetric
density. At R=1 m the minimum initial spacings are approximately:

| UAVs | Minimum spacing |
|---:|---:|
| 10 | 0.977 m |
| 25 | 0.617 m |
| 50 | 0.437 m |
| 100 | 0.309 m |

These all initially exceed the model's 0.15 m required UAV separation. Invalid
user configurations that violate initial clearance are rejected rather than
counted as avoidance failures. These dimensions apply to the existing 0.05 m UAV
radius model, not arbitrary full-size aircraft.

Within a size/class/repetition, the target indices, formation orientation,
projectile direction and offset are paired across radii. Candidates are retained
only if they meet the requested encounter class at every requested radius.
Thus each default condition has three geometries, and the radius comparisons
are paired; there are 36 distinct geometry identities and 108 radius-specific
encounters. They are not 108 independent random geometry draws.

Direct paths pass 0–0.05 m from one UAV; multiple-threat paths pass through two
selected UAV centers; candidate misses use 0.15–0.50 m offsets. The actual class
is checked against every stationary UAV. A candidate is regenerated if another
UAV changes that category. Projectiles start outside a sphere enclosing the
formation, approach a near-facing target, and are simulated until they exit that
sphere plus 0.2 seconds. Far-side threats are not silently omitted by ending
every trial just after the selected target.

Radius also changes sensing geometry, travel distance and encounter duration.
This is a controlled size/radius sensitivity study, not an isolated density-only
causal experiment. Nominal target time is fixed; actual first safety entry and
duration are recorded separately. No observations are discarded based on a
method's outcome.

## Comparison and runtime limits

All three controllers receive the same saved state, positions and duration. All UAVs
start stationary, move under the existing controller, and do not return to
formation. Estimator/optimizer settings and collision thresholds are unchanged.
Oracle is omitted to keep this diagnostic comparison focused and affordable.
The baseline uses a reproducible action seed, shared across the radius variants
of a geometry. Its decisions can differ across radii; an identical seed does
not force the same UAVs to trigger or guarantee identical per-UAV action draws.

Independent and coordinated Dodge share the same threshold, one-shot detector,
and action seed. The coordinated planner sees HIT/MISS and UAV positions only;
it jointly adjusts random preferred movements to maintain UAV-UAV clearance.
It can move MISS UAVs but does not know the projectile trajectory. Its unchanged
minimum HIT movement is 0.40 m versus independent Dodge's 0.45 m command. Both
SLSQP controllers are subject to the same process runtime cap. Plan success and
reactive MISS-UAV movement are recorded separately from actual encounter safety.

Each controller runs in its own spawned process. At 100 UAVs there are 4,950
UAV-pair separations to check, and the optimizer can fall back to a full-swarm
search. Acoustic generation and estimator costs also grow. A slow trial must not
block every remaining condition. The cap includes worker startup/imports as well
as the simulated encounter. `wall_runtime_s` from returned results excludes
those imports; `process_elapsed_s` includes them.

* `status=ok`: a complete result was returned; safety can be True or False.
* `status=timeout`: process cap exceeded; actual simulation safety is unknown.
* `status=error`: controller failed; actual full-encounter safety is unknown.

Summary tables separate observed safety among completed simulations from the
fraction of all attempted trials returning a safe result within the runtime cap.
Timeouts do not become observed collisions. The runtime cap is not the projectile
collision deadline. Existing first-command compute screens remain partial, and
compute time does not physically delay simulated motion.

## Outputs and interpretation

* `report.md`: compact outcomes by size, radius, encounter class and method.
* `summary.csv`: observed and budgeted safety, timeout/error counts, collisions,
  recall/FPR, movement, runtime and measured initial spacing.
* `metrics.csv`, `results.jsonl`: trial measurements; raw JSONL also retains
  per-UAV travel and individual planner-call runtimes when available.
* `paired_comparison.csv`: all three pairwise comparisons among the controllers,
  with status and safety per encounter; movement differences are calculated only
  when both methods remain safe. `method_a`/`method_b` identify each comparison.
* `scaling.png`: safe results returned within the runtime cap, with timeout counts.
  A dark/light cell can reflect computational feasibility as well as avoidance;
  consult observed safety and timeout counts separately.
* `encounters.jsonl`, `manifest.json`, `source.zip`: geometry, protocol, threshold
  provenance, environment, source hashes and an actual code snapshot.

Compare projectile versus UAV collisions: independent random dodges could avoid
the projectile while creating inter-UAV collisions. Inspect false-positive rate
and moved count on safe misses to expose unnecessary movement. Higher sensor
counts might improve estimation while worsening runtime or inter-UAV risk; the
experiment is intended to reveal those tradeoffs, not confirm a chosen winner.

This experiment lives under `experiments/scaling/` so adding it does not alter
the existing main runner's immediate-directory source scan. Existing main/noise
studies and their result files are not edited or overwritten. Avoid simultaneous
heavy runs when interpreting timing data. Large-scale live sensing also assumes
the project's idealized, separable acoustic channels; cross-talk and sound-level
limits for 100 real transmitters are outside this simulation.
