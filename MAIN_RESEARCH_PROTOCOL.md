# Main research evaluation: randomized formations at controlled spacing

Run from the directory containing `experiments`:

```powershell
.venv\Scripts\python.exe -m experiments.research_main --output results/research_main_01
```

This is the new nominal main study. It generates its own encounters and calibrates
its own frozen detector threshold; it does not need a `--source` result folder.
The supplementary `fibonacci_evaluation`, noise and scaling studies remain available. Their
results are not imported into this different protocol or overwritten.

## Design

| Factor | Default |
|---|---|
| UAV counts | 4, 6, 8, 10, 16, 25 |
| Projectile speeds | 1, 3, 5 m/s |
| Encounter categories | Direct threat, grazing threat, multiple threats, safe miss |
| Geometries per count/category | 50, each tested at all speeds |
| Initial formation | Independently randomized volumetric 3D layout |
| Median nearest-neighbor distance | 0.6 m |
| Minimum initial separation | At least 0.3 m; evaluation requires 0.15 m |
| Warning for genuine threats | 1.2 s before first stationary-swarm safety entry |
| Added noise / network delay | None / zero |
| UAV motion | Starts stationary; existing kinematic avoidance; no return-to-formation |
| Methods | Static, oracle, acoustic estimate, individual Dodge, coordinated Dodge |
| Per-controller process cap | 60 s, including startup and imports |

Total: **3,600 encounters × 5 methods = 18,000 records**. There are 1,200
independent layout/ray geometry identities, paired across three projectile speeds.
The 3,600 static references are analytic; 14,400 controller runs use PyBullet.
Counts and speeds can be explicitly configured with `--swarm-sizes`/`--speeds`;
`--repetitions` controls the final evaluation budget. There are no mini-test modes.

At the defaults, if every controller hits the cap, controller waiting alone is
240 hours. This is a large evaluation; higher counts have substantially greater
estimation/planning costs. Freeze the runtime budget before final evaluation.
A timeout is a measured computational limit, not an observed physical collision.
The runner executes only one controller process at a time; avoid competing heavy
workloads when collecting comparable timing measurements.

## Formation and encounter construction

Positions are generated from uniform proposals throughout a ball with sequential
minimum-distance rejection, then centered and scaled to exactly the selected
median nearest-neighbor distance. Samples violating the selected minimum spacing
are regenerated. This is a constrained random spatial point process, not
independent uniform points, a spherical shell, or a guaranteed optimal sensor layout.
Its algorithm is fully specified in `formation()` and its seeds are stored.

As count increases, typical neighbor spacing stays fixed and spatial extent grows.
Actual minimum spacing, median nearest-neighbor spacing, RMS radius and maximum
radius are recorded for every encounter. These dimensions describe the existing
5 cm-radius UAV model, not arbitrary real aircraft. Different counts use separate
random layouts, not nested subsets of one sensor network.

Each count/category/repetition has a newly randomized internal layout. Approaches
use isotropically sampled directions; the frontmost UAV for that direction is
selected as the first target. Multiple-threat paths select an opposite-side
second UAV and align the path through both. Perpendicular offsets are 0–0.05 m
for direct threats, 0.075–0.095 m for grazing threats, and 0.15–0.40 m for candidate
misses. This deliberately targets incoming encounters; it does not uniformly
sample every interior UAV as a primary target.

Each candidate ray is checked against ALL stationary UAVs, and retained only if
it matches the desired category. Two or more clearance violations take precedence
as a multiple threat. Direct threats physically contact one UAV; grazing threats
violate one UAV's clearance without contact; misses clear every UAV. Class
balance is an intentional challenge design, not real-world occurrence frequency.

For threats, the starting point is placed so the earliest 0.10 m safety entry
would occur exactly at the configured warning time if nobody moves. For safe
misses, nominal warning means time to passage of the selected front-side target;
there is no collision deadline. Trajectories start beyond the formation's incoming
support plane. They run past the outgoing support plane plus the nominal 0.45 m
single-maneuver budget, projectile safety clearance and 0.2 s of additional flight.
Every initial UAV is therefore passed during scoring. The end time is recorded;
this is a finite-encounter evaluation, not an indefinite safety guarantee.

The same formation and geometric ray are paired across speeds, adjusting starting
distance to maintain warning time. Every method uses exactly the same saved
encounter. Speed changes starting range and acoustic propagation geometry; this
protocol does not isolate speed at fixed starting distance. Larger formations
also have longer passage times and different sensing geometry despite matched
neighbor spacing. Claim a controlled system-level comparison, not a pure causal
effect of adding measurements alone.

## Calibration and baselines

Both Dodge variants share a single frozen threshold, calibrated before evaluation
on a separate bank generated with the same size/speed/class/spacing protocol.
Default calibration has two geometries per count/class, paired across speeds:
144 training encounters and 1,656 UAV audio samples before unusable-signal handling.
Each uses only the existing 0.2 s stationary observation window. The updated
detector uses the paper's 18 kHz carrier and Section 2.5's adjacent-buffer
multiplication, sample sums, absolute differences and cumulative sum. A one-buffer
startup guard discards the potentially partial echo-onset buffer. This guard and
the fixed 10 ms buffer/200 ms decision window are our implementation choices;
the paper does not specify a unique configuration. No FFT or trajectory estimate
is added to this detector. The shared estimator retains its existing 10 kHz audio
and numerical behavior; the carrier and boundary-roundoff fix are opt-in for Dodge.

The threshold is the smallest value covering every usable training HIT, reflecting
the paper's stated priority of avoiding false negatives. The exact search rule
is ours; the paper does not publish its threshold. Zero training false negatives
does not guarantee zero evaluation false negatives. Calibration fails if any
training HIT has no usable echo; no-signal evaluation cases predict MISS.
Calibration uses ground-truth labels, as supervised calibration should; evaluation
detectors receive only their acoustic signals. No avoidance outcomes tune it.

Calibration seed 2026 and evaluation seed 8675309 have disjoint namespaced IDs.
Store both banks and calibration scores for review. `--dodge-threshold VALUE`
can instead supply an independently selected threshold and skips calibration.
Never select the threshold from final evaluation results. Initialization/planner
budgets, process noise, movement limits and safety margins remain unchanged.

The oracle reads true projectile state immediately; it is a privileged reference,
not proof of global optimality. Acoustic estimate uses the existing initializer,
EKF and SQP planner. Individual Dodge performs one random dodge on HIT; coordinated
Dodge adjusts random preferred targets using UAV positions and HIT/MISS only,
never projectile state. Both share the action seed per encounter. Coordinated
Dodge's unchanged minimum HIT movement is 0.40 m, versus individual Dodge's 0.45 m
command, so this is a comparison of implemented policies, not coordination alone.

The proposal's accuracy, collision, movement and decision-time criteria are scored.
Its originally proposed estimated-trajectory/independent-avoidance ablation is
not equivalent to coordinated Doppler Dodge and is not silently added here.
Noise, formation-family changes and delayed-execution studies remain separate
supplementary evaluations, as agreed in the experiment plan.

## Run, resume and analyze

To refresh only both Dodge variants after this detector update, run from the
project directory in a new output directory:

```powershell
.venv\Scripts\python.exe -m experiments.research_main --output results/research_main_02 --reuse-results results/research_main_01
```

This inherits all encounter, seed and runtime settings from the source manifest;
do not pass new geometry settings. It copies the exact saved evaluation bank and
all static/oracle/estimate rows, including recorded timeouts/errors, recalibrates
on the separate training bank, and runs only `dopplerdodge` and
`dopplerdodge_coordinated`. A complete default source retains 10,800 rows and
runs 7,200 new controller trials. Original files remain unchanged. Thresholds
are never inherited. Summaries and paired comparisons include all five methods.

The new manifest records provenance separately for each method, the source results
hash, and the inherited manifest/source archive. This is a mixed-version study:
retained estimator audio is unchanged, while the Dodge baselines use the corrected
18 kHz configuration. Unrelated controller source changes and incomplete retained
method banks are rejected. Existing partial refreshes resume without recalibration:

```powershell
.venv\Scripts\python.exe -m experiments.research_main --output results/research_main_02 --resume
```

```powershell
.venv\Scripts\python.exe -m experiments.research_main --output results/research_main_01 --resume
.venv\Scripts\python.exe -m experiments.research_main --output results/research_main_01 --summarize
```

Shared metrics, controller instrumentation and bookkeeping live in
`experiments/evaluation_common.py`. The main runner does not import or execute
the supplementary Fibonacci evaluation. Completed studies can still be summarized
after a code refactor; continuing trials requires the saved source and environment.

Every completed result is flushed immediately. Each controller has its own process
and can be terminated at the runtime cap; this protects the remaining study from
a stuck large optimization. Ctrl+C retains finished rows and the unfinished method
can be rerun on resume. Errors/timeouts are saved outcomes and skipped on resume.
The current implementation and environment must match; keep any custom CLI flags
the same. Changed source/settings/inputs are rejected, not mixed silently. Use a
new directory to change a cap or algorithm. Only one writer should use a directory.
An incomplete JSONL line after a hard crash raises an error instead of silently
deleting evidence. Encounter generation/calibration happen before the first result
checkpoint; interrupting those phases requires restarting preparation.

## Outputs

* `report.md`: completed versus timed-out trials and observed/budgeted safety.
* `summary.csv`: all proposal metrics plus movement per UAV, fraction moved,
  collision rates, classification coverage, planning/initialization/response times,
  and timeout/error counts. Means/medians/p95 include observation counts.
* `metrics.csv`: per-method/per-encounter measurements and actual formation size.
* `paired_comparison.csv`: estimate versus both Dodge variants and oracle, plus
  coordinated versus individual Dodge. Movement differences require both to be safe.
* `safety_vs_speed.png`: safe returned results per attempted trial. Interpret it
  together with timeout counts and observed safety, not as a pure collision plot.
* `results.jsonl`: raw checkpointed measurements, per-UAV travel, first predictions
  and individual planning-call runtimes when available.
* `encounters.jsonl`, `calibration_encounters.jsonl`, `calibration_scores.csv`:
  inspectable evaluation/training geometry and detector calibration evidence.
* `manifest.json`, `source.zip`: declared settings, threshold, source/environment
  fingerprint, input checksums and actual code snapshot.

Budgeted safety is the fraction of attempts returning safe results within the cap.
Observed safety/collision rates use completed simulations only. Neither metric
should conceal the other; timeouts/errors have unknown physical safety outcomes.
Wilson intervals for budgeted success treat independent encounters within each
count/speed/class as samples. Across speeds, geometry is shared: use geometry-level
paired/clustered analysis rather than treating those rows as independent layouts.
Fifty encounters per condition does not establish near-perfect reliability.

Movement measures distance traveled, not only final displacement. Compare common
successful encounters and show safety rates alongside conditional efficiency.
HIT/FPR score the first decision against the stationary-swarm counterfactual;
unavailable classifications have explicit coverage, not fabricated zero scores.
Trajectory RMSE is conditional on available estimates, so inspect initialization
failures and missing-result counts too. No globally minimal movement claim follows.

The inherited first-command compute screen is partial and excludes acoustic
processing/networking. Simulation-time detection/response and measured compute
costs are separate; compute delays do not physically delay the motion. The process
cap is not a real-time collision deadline. Kinematic UAV motion, constant projectile
velocity, fixed-amplitude idealized independent echoes and sampled 240 Hz online
clearance remain model limitations. This experiment is not field validation,
continuous-time collision certification, or a claim of realistic 100-transmitter
cross-talk performance. Independent clearance/timestep and latency studies are
needed for stronger claims.
