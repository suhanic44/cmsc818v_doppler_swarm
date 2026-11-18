# Experiment 2: Doppler measurement-noise robustness

Question: how much additional Doppler measurement error can the estimator and
closed-loop avoidance tolerate, and does accounting for noise variance help?

The runner is `experiments/noise_robustness.py`, alongside the main evaluation.
It never writes into the source study's result directory. Moving the module
changes source fingerprints: already-running processes can finish, but the main
runner's strict resume check will reject manifests from before this file was
added to `experiments/`. Use a new main output with `--reuse-results` to retain
completed trials if that occurs. Older noise studies also require a fresh run
directory for execution; their saved results can still be summarized.

## Run

From the directory containing this file:

```powershell
.venv\Scripts\python.exe -m experiments.noise_robustness --source results/fibonacci_02 --output results/noise_fixed --filter-policy fixed
```

Run variance-aware separately on matching encounters and noise realizations:

```powershell
.venv\Scripts\python.exe -m experiments.noise_robustness --source results/fibonacci_02 --output results/noise_variance_aware --filter-policy variance_aware
```

Or run both together with `--filter-policy both` and a new output directory.
The default policy is `fixed`. Keep source, seed and realization count identical
for comparisons between separate directories; the injected errors are keyed
independently of policy. Both individual runs collect their own clean controls
and copy available static/oracle references. Doppler Dodge is not rerun by this
noise study. Use a source study created with the current main-evaluation code.

Resume with the same options plus `--resume`. Analyze saved trials without
running simulations:

```powershell
.venv\Scripts\python.exe -m experiments.noise_robustness --output results/noise_fixed --summarize
```

The default selects 800 encounters from the existing study: 4, 6, 8 and 10 UAVs,
3 m/s, all four threat categories, all 50 geometries per condition. The source
study need not have finished: its encounter file is created before simulation.
Completed oracle/static references are copied once; missing references are run
once. The source is not polled or changed after that snapshot. Clean estimated
controls are rerun to collect the same fixed-time prediction diagnostics as the
noisy trials. No-noise controls are not duplicated for each noise seed.

This is a substantial evaluation: **22,400 result rows** for either single policy:

* 800 static and 800 oracle references, reused when already available.
* 800 clean estimated controls.
* 800 encounters × 5 nonzero noise levels × 5 realizations × 1 filter policy
  = 20,000 noisy estimated runs.

That is up to 21,600 simulations, fewer when oracle references are reused.
`--filter-policy both` produces 40,000 noisy runs plus 2,400 controls/references,
or **42,400 records**. Running the policies in separate directories produces
44,800 records in total because their controls/references are stored in each.
Developing this runner does not launch that workload. Prefer running it after
the main evaluation finishes: simultaneous studies compete for CPU and make
wall-clock timing measurements hard to compare. Each trial is saved immediately
and summaries are produced on completion or Ctrl+C. Resume skips saved errors
as well as successes. Source/configuration/environment changes prevent resume.
After a hard crash, a malformed partial JSONL record raises an error rather
than silently deleting results. Use only one writer per output directory.

`--seed` sets the noise seed (default 161803). `--realizations` sets the number
of independent noise draws per encounter/level/policy (default 5). Set those
before final evaluation and keep the same values when resuming.
Also keep the same `--filter-policy` on resume. The expanded study is version 2;
start a fresh output directory rather than resuming an older two-size study.
Saved older results can still be summarized with `--summarize` using their
recorded sizes/policies; they are not silently expanded or overwritten.

## What the noise means

At every sensor and timestamp, before the estimator receives the measurement:

`noisy_shift_hz = extracted_shift_hz + Normal(0, additional_sigma_hz**2)`

This affects the complete initialization window and all subsequent updates.
It does not perturb true state or bypass the audio/Doppler pipeline. Timestamps,
projectile paths, swarm positions at the start, safety thresholds, motion limits,
initialization settings and process-noise settings remain unchanged. Later UAV
motion can differ because noisy estimates lead to different avoidance commands.

| Additional standard deviation | Approximate radial-velocity standard deviation |
|---:|---:|
| 0 Hz | 0 m/s additional error |
| 0.5 Hz | 0.0086 m/s |
| 1 Hz | 0.0172 m/s |
| 2 Hz | 0.0343 m/s |
| 4 Hz | 0.0686 m/s |
| 8 Hz | 0.1372 m/s |

Conversion uses the proposal's low-speed monostatic approximation,
sigma_v = 343 sigma_f / (2 × 10,000), not a guarantee of resulting full-velocity
estimation accuracy. These are deliberately prespecified synthetic sensitivity
levels, not measured microphone specifications. Zero means no ADDED noise;
extraction error and modeling approximations still exist.

Errors are independent across UAVs and measurement timestamps. The same
standard-normal draw is scaled across noise levels and shared across filter
policies for a matching encounter/realization. It is keyed by sensor and timestamp,
so different batching or early termination does not misalign random streams.
Different encounters and realizations have separate streams. No true projectile
state is used to generate measurement errors.

This does not test audio SNR, multipath, correlated errors from overlapping
windows, sensor bias, dropout, large outliers, or position/timestamp errors.
Do not describe the injected Gaussian errors as a complete real-acoustics model.

## Two filter policies, no tuning on final results

* **fixed**: retain the implementation's 1 Hz assumed measurement standard
  deviation at every added-noise level. Tests the current fixed configuration.
* **variance_aware**: assumed standard deviation becomes
  sqrt(1 Hz² + additional_sigma_hz²). Tests accounting for known injected variance.

The 1 Hz baseline is the current filter assumption, not empirically calibrated
extraction noise. The second policy is therefore variance-aware, not a claim of
perfectly matched total noise. Neither policy changes process noise, initialization
search budgets, or initial state covariance. Equal independent Gaussian error
variance on all measurements does not change the batch least-squares minimizer
by a common weighting factor; noisy values themselves change that initialization.

The clean control uses the original filter setting. Oracle is never given noise.
No policy is selected retrospectively based on which happened to perform best.

## Outcomes and files

**Primary:** full-encounter safety-clearance success; errors count as unsuccessful.
Projectile and UAV collision counts are separate. Collision outcomes of errored
simulations remain unknown rather than being reported as no collision.

**Estimation:** initialization success, first estimate actually available in
simulation time, trajectory RMSE, and errors at 0.3/0.6/0.9/1.1 s. At each fixed
time, use only the latest estimate already available, propagate it to that time,
and score position, velocity, and a 0.2 s future-position prediction. Availability
is reported alongside conditional errors. An estimate produced late cannot be
used retrospectively at an earlier checkpoint. If a run aborts, later checkpoints
are marked not reached rather than extrapolating an old state into them.

**Movement:** total UAV path length and paired changes versus the same clean
encounter, restricted to trials where both clean and noisy avoidance are safe.
Read these together with safety rates to avoid rewarding failed, motionless runs.

**Compute:** inherited component timings and partial first-command timing screen,
plus separately measured noise-generation overhead. The screen excludes acoustic
processing and networking and does not apply delays to simulated movement.
Instrumentation and synthetic noise generation add overhead; compare simulation
safety more confidently than small timing differences. Reused reference runtime
measurements were collected by the parent run under potentially different CPU load.

Outputs:

* `report.md`: readable safety overview and explicit analysis caveats.
* `summary.csv`: condition-level safety, uncertainty, paired clean differences,
  initialization failures, runtime/error percentiles and movement differences.
* `policy_comparison.csv`: produced when both policies have paired results in
  the same study; variance-aware minus fixed safety, pairing exact encounters/
  noise realizations and clustering by encounter. Separate single-policy results
  can be joined using encounter ID, sigma and realization in `metrics.csv`.
* `checkpoint_summary.csv`: fixed-time position/velocity/forecast RMSE and estimate
  availability. Errors are conditional on a usable estimate, not all trials.
* `metrics.csv`, `checkpoint_metrics.csv`: trial-level measurements for plotting.
* `safety_vs_noise.png`: safety curves and reference lines, separate panels for
  swarm size and encounter class.
* `prediction_vs_noise.png`: forecast error versus noise at 0.6 s observation time.
  Always consult the availability column before interpreting a low error.
* `results.jsonl`: append-only checkpointed raw trials, including individual
  planner calls and all fixed-time scores.
* `encounters.jsonl`, `reference_snapshot.jsonl`, `manifest.json`, `source_snapshot/`:
  immutable input copies, provenance, configuration, environment and actual source.

## Statistical interpretation

The five noise repetitions share an encounter: they are NOT five independent
geometries. Safety is averaged within an encounter and then across encounters.
95% intervals use 2,000 encounter-cluster bootstrap draws. Paired changes use
the same clean encounter and noise realizations for method comparisons. When
every cluster outcome is identical, ordinary bootstrap intervals would collapse:
the implementation instead uses conservative bounded-mean fallback intervals
(endpoint bounds for all-success/all-failure, Hoeffding bounds otherwise).
Very small partial-run summaries have little inferential value.

Keep threat categories separate from safe misses; balanced classes do not model
real-world encounter frequency. Do not pool the sixteen condition panels into a
single reliability claim. Conditional error and movement summaries can improve
simply because difficult runs failed, which is why availability and failure rates
are retained. Summary CSVs for interrupted runs reflect only completed rows.

`observed_drop_at_least_5pp` flags a prespecified five-percentage-point loss of
safety versus clean at a tested level. It is descriptive, not a significance
test or a certified tolerance threshold. Use the paired confidence interval and
the entire curve; results need not be monotonic. A threshold lies at best between
tested levels, and applies only to this geometry/speed/noise protocol. This study
uses one projectile speed deliberately; noise-by-speed interactions need their
own study before broader claims.

These encounters were already selected for the main study. If you use their
results to change the estimator, generate a new held-out main encounter bank
and run the noise study against that bank for final paper numbers.

## Verify the experiment implementation

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_noise_robustness.py
```

These development checks verify reproducible injection, sensor/seed independence,
initialization exposure, causal checkpoint scoring, clustered analysis, input
snapshots and resume. They are not the research experiment or paper results.
