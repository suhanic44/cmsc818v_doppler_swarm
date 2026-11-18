# Supplementary research evaluations (draft design)

This extends `MAIN_RESEARCH_PROTOCOL.md`. The main study remains the nominal
count/speed comparison. These studies ask separate sensitivity questions; do not
pool their trials into the main results. The values below are proposed starting
points to freeze after development pilots, before held-out evaluation.

## Common design

Start with counts **10 and 50**, speed **3 m/s**, all four encounter categories,
and **50 geometries per count/category**. This is 400 encounters per factor level
and 2,000 method trials with the same five methods as the main study. Use smaller
explicit repetition counts for development, with separate seeds and output paths.
Do not choose final levels based on which method wins. Expand to other speeds
only for a specific follow-up hypothesis, rather than crossing every factor.

Keep controller settings, movement limits, clearance thresholds, runtime cap,
and nominal 1.2 s warning fixed except for the declared factor. Inherit the frozen
Dodge threshold from the nominal main-study manifest; record its source and hash.
Do not recalibrate separately at each factor level. Any adapted filter or detector
belongs in a separately labeled adaptation experiment.

Use a new held-out geometry bank (proposed seed 910001; development seed 910002),
independent of main evaluation and calibration. Pair methods on identical saved
encounters and random actions. Pair levels on a common geometry identity wherever
the intervention permits it; store factor level separately from geometry ID.
Key perturbation seeds by geometry, sensor, simulation timestamp and realization,
not method or amplitude. Store the actual generated inputs, not only their seeds.

## Studies and proposed levels

| Study | Factor levels | Controls and interpretation |
|---|---|---|
| Crowding | Median NN spacing 0.4, 0.6, 0.9 m | Same counts, speeds, volumetric generation algorithm, absolute 0.3 m minimum separation and class construction. Extent changes with spacing; record it. |
| Formation geometry | Random planar disk, volumetric ball, spherical shell | Same count, 0.6 m median NN target, 0.3 m minimum separation, isotropic approach rule and class balance. Record RMS/max radius and threatened count; these cannot generally also be held identical. |
| Audio noise | Clean, 30, 20, 10 dB SNR | Same exact encounters, fixed detector/filter, independent noise per sensor. Apply waveform noise before feature extraction so both acoustic methods receive the intervention. |
| Warning time | 0.6, 0.9, 1.2, 1.8 s | Same formation and geometric ray; shift starting point, retain full-passage endpoint. Starting range and available acoustic history necessarily change. |
| Execution delay | 0, 0.05, 0.10, 0.20 s | Same exact encounters, 1.2 s warning and clean sensing. Queue every movement command and physically begin it after the delay, for every moving controller. |
| Acceleration mismatch | 0, 0.5, 1.0 m/s² transverse magnitude | True curved projectile trajectory; estimator retains constant-velocity model. Random fixed transverse direction per geometry. |
| Measurement dropout | 0, 0.1, 0.3 independent loss probability | Drop sensor observation frames, never fabricate zero measurements. Same underlying uniforms across probabilities. Declare frame cadence and independence model. |
| Localization mismatch | 0, 0.02, 0.05, 0.10 m coordinate SD | Fixed Gaussian position bias per UAV per encounter in controller/sensor-position metadata; physics and safety scoring retain true positions. |

These are eight one-factor sweeps, not one full factorial. Each includes its own
nominal control. Shared control runs may be reused only when input, source,
threshold, settings, horizon and random seeds match exactly and provenance is saved.
The plan generator counts controls separately and makes no reuse assumption.

For audio SNR, define signal power from the clean stationary 0.2 s observation
window per sensor, then hold noise variance fixed throughout that encounter.
Specify handling for zero-energy sensors before implementation (for example a
shared encounter reference power); save reference power and realized SNR.
Use paired independent noise realizations if the budget allows; repetitions of
noise on one geometry are not additional independent geometries.

Measurement-noise sensitivity is a separate optional estimator diagnostic:
add Doppler errors in Hz after extraction using the existing `MeasurementNoise`
utility in `noise_robustness.py`. It does not test waveform corruption of Dodge.
The existing noise runner targets an older main protocol; its outputs must not
be presented as a supplement to `research_main` without explicit integration.

## Pairing and geometry checks

For crowding, a common latent layout can be scaled across levels, but enforce the
minimum separation at every level. Select a bank feasible across all levels before
running controllers. Scaling changes clearance labels: regenerate or reject rays
using all-UAV class checks at every level, and record acceptance/rejection counts.
If common class-preserving rays cannot be obtained, use matched class-stratified
banks and report that the comparison is not encounter-paired.

For geometry, generate uniform disk-area proposals with sequential exclusion for
the planar family, the main ball sampler for volumetric, and uniform sphere-area
proposals with exclusion for shell. Center and normalize median NN spacing, then
validate minimum separation. Randomly rotate planar layouts in 3D and use isotropic
approaches for all families. A shared seed does not make different families the
same geometry. Compare class-stratified distributions and show extent/clearance
descriptors; do not claim a pure geometry effect at fixed spacing AND fixed extent.

Warning-time pairs must start outside the incoming support plane at every level;
accept/reject jointly rather than silently changing rays per warning. Audio,
delay, and dropout studies preserve the exact nominal encounter. Acceleration can
change threat labels and first entry: recompute stationary-swarm truth against
the actual trajectory and retain nominal category as separate metadata. Declare
the acceleration onset and finite horizon before implementation. Report actual
severity distributions. The oracle should receive current true state but retain
the same constant-velocity planner; a future-trajectory oracle is a separate method.

## Outcomes and analysis

Primary: budgeted safety success and observed safety, with timeout/error counts
alongside projectile and UAV collision rates. Report safety by category and count.
Secondary: minimum clearance, first decision recall/FPR and coverage, initialization
success, trajectory RMSE with availability counts, traveled distance per UAV,
fraction moved, and simulation-time response versus measured compute separately.
Movement comparisons use common safe outcomes and show their denominators.

Estimate changes from the nominal level for each method and differences in those
changes between methods. Bootstrap whole geometry identities, keeping all speeds,
levels, methods and noise realizations together. For unpaired geometry families,
resample independent layouts within each family/count/category instead. Predeclare
primary contrasts; label other comparisons exploratory. Do not interpret a failure
to find a difference as equivalence.

## Implementation order and readiness

1. Crowding: reuse main generation and bounded execution; add factor-aware IDs,
   joint bank construction, shared threshold import and grouping by spacing.
2. Warning time: reuse geometric rays, jointly validate starts, and retime encounters.
3. Geometry: add family samplers and invariant checks before running avoidance.
4. Noise and execution delays: implement common input/command hooks across all
   relevant controllers; verify the zero level reproduces nominal execution.
5. Mismatch: change true physics and controller inputs separately, then verify
   truth scoring, dropout semantics and localization paths independently.

`research_main` already exposes `--spacing`, `--min-spacing`, `--warning`, and
`--dodge-threshold`; separate invocations currently generate separate banks and
are not a substitute for a paired supplement runner. Communication delay hooks
exist in `online_avoidance`, but command-delay parity across baselines is unfinished.
None of the five supplementary study families has a complete runner here yet.

The accompanying planner writes a design and budget only; it never runs simulations:

```powershell
.venv\Scripts\python.exe -m experiments.research_supplements --output results/supplement_plan.json
.venv\Scripts\python.exe -m experiments.research_supplements --studies crowding warning --repetitions 5 --output results/supplement_pilot_plan.json
```

Before final runs, save the frozen design, encounter banks and hashes, source and
environment snapshot, threshold provenance, perturbation records, and controller
settings. Retain the main runner's incremental checkpoints and timeout semantics.
Source edits can invalidate existing main-study resume fingerprints; preserve the
original source/environment to resume an ongoing main run.
