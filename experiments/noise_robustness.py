"""Research noise study; see NOISE_EXPERIMENT.md for the fixed protocol.

python -m experiments.noise_robustness --source results/main_02 --output results/noise_fixed --filter-policy fixed
Use --resume to continue, or --summarize to analyze saved trials only.
"""

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import main_evaluation as main


SIGMAS = (0.5, 1.0, 2.0, 4.0, 8.0)  # Additional measurement-error standard deviation, Hz.
POLICIES = ("fixed", "variance_aware")
SWARM_SIZES = (4, 6, 8, 10)
CHECKPOINTS = (0.3, 0.6, 0.9, 1.1)
FORECAST_HORIZON = 0.2
BOOTSTRAPS = 2000


def stable_words(label):
    return list(np.frombuffer(hashlib.sha256(label.encode()).digest()[:16], dtype="<u4").astype(int))


class MeasurementNoise:
    """Noise keyed by measurement identity, independent of call batching/order.

    Omitting sigma/policy from the seed shares standardized errors between noise
    amplitudes and filter policies. Different sensors/timestamps get independent
    pseudorandom draws. No ground truth is used to generate measurement errors.
    """

    def __init__(self, encounter_id, master_seed, realization, sigma):
        self.words = [master_seed, *stable_words(encounter_id), realization]
        self.sigma = sigma
        self.count = 0
        self.total = 0.0
        self.squares = 0.0
        self.runtime = 0.0

    def apply(self, sensor, times, values):
        if self.sigma == 0:
            return values  # Exact no-added-noise control; don't consume random numbers.
        start = time.perf_counter()
        errors = np.array([
            self.sigma * np.random.default_rng(np.random.SeedSequence(
                [*self.words, int(sensor), int(round(float(t) * 1_000_000))]
            )).standard_normal() for t in times
        ])
        self.count += len(errors)
        self.total += float(errors.sum())
        self.squares += float(errors @ errors)
        output = np.asarray(values, dtype=float) + errors
        self.runtime += time.perf_counter() - start
        return output

    def metrics(self):
        return dict(injected_samples=self.count,
                    injected_mean_hz=self.total/self.count if self.count else None,
                    injected_rms_hz=math.sqrt(self.squares/self.count) if self.count else None,
                    noise_injection_compute_s=self.runtime)


def checkpoint_metrics(case, trace, reached_time):
    """Score the latest estimate actually available at fixed observation times."""
    truth = np.asarray(case["state"])
    scores = []
    for checkpoint in CHECKPOINTS:
        eligible = [e for e in trace if e["available_s"] <= checkpoint + 1e-9]
        score = dict(time_s=checkpoint, horizon_s=FORECAST_HORIZON,
                     status="not_reached" if checkpoint > reached_time+1e-9 else "no_estimate",
                     position_error_m=None, velocity_error_mps=None, forecast_error_m=None,
                     estimate_age_s=None)
        if checkpoint <= reached_time+1e-9 and eligible:
            latest = eligible[-1]
            state = np.asarray(latest["state"])
            predicted_now = state[:3] + (checkpoint-latest["state_time_s"])*state[3:]
            actual_now = truth[:3] + checkpoint*truth[3:]
            score.update(status="valid",
                         position_error_m=float(np.linalg.norm(predicted_now-actual_now)),
                         velocity_error_mps=float(np.linalg.norm(state[3:]-truth[3:])),
                         forecast_error_m=float(np.linalg.norm(
                             predicted_now+FORECAST_HORIZON*state[3:] -
                             (actual_now+FORECAST_HORIZON*truth[3:]))),
                         estimate_age_s=checkpoint-latest["state_time_s"])
        scores.append(score)
    return scores


def run_estimate(case, sigma, policy, realization, seed, online):
    noise = MeasurementNoise(case["encounter_id"], seed, realization, sigma)
    original_init = online.OnlineEstimationPipeline.__init__
    original_add = online.OnlineEstimationPipeline.add_measurements
    original_process = online.process_pipeline_measurements
    pipelines, trace = [], []
    reached_time = 0.0
    assumed_sigma = None

    def initialize(pipeline, *args, **kwargs):
        nonlocal assumed_sigma
        original_init(pipeline, *args, **kwargs)
        baseline = float(pipeline.ekf.sigma_doppler)
        assumed_sigma = math.sqrt(baseline**2+sigma**2) if policy == "variance_aware" else baseline
        pipeline.ekf.sigma_doppler = assumed_sigma
        pipelines.append(pipeline)

    def add(pipeline, sensor, times, values):
        return original_add(pipeline, sensor, times, noise.apply(sensor, times, values))

    def process(pipeline, simulation_time, *args, **kwargs):
        nonlocal reached_time
        previous = len(pipeline.estimated_times)
        reached_time = float(simulation_time)
        try:
            return original_process(pipeline, simulation_time, *args, **kwargs)
        finally:
            for index in range(previous, len(pipeline.estimated_times)):
                trace.append(dict(available_s=float(simulation_time),
                                  state_time_s=float(pipeline.estimated_times[index]),
                                  state=pipeline.estimated_states[index].copy()))

    start = time.perf_counter()
    # Nested with main.run_online's timing instrumentation; restored on failure.
    with patch.object(online.OnlineEstimationPipeline, "__init__", initialize), \
         patch.object(online.OnlineEstimationPipeline, "add_measurements", add), \
         patch.object(online, "process_pipeline_measurements", process):
        try:
            result = main.run_online(case, "estimate", online)
        except Exception as exc:
            result = dict(status="error", safety_satisfied=False,
                          error=f"{type(exc).__name__}: {exc}",
                          wall_runtime_s=time.perf_counter()-start)
    pipeline = pipelines[0] if pipelines else None
    result.update(initialization_success=bool(pipeline and pipeline.initialized),
                  first_estimate_available_s=trace[0]["available_s"] if trace else None,
                  assumed_sigma_hz=assumed_sigma, checkpoint_scores=checkpoint_metrics(case, trace, reached_time),
                  **noise.metrics())
    return main.clean(result)


def trial_id(case_id, method, sigma=None, policy=None, realization=None):
    return json.dumps([case_id, method, sigma, policy, realization], separators=(",", ":"))


def trials(cases, realizations, seed, policies=POLICIES):
    for case in cases:
        yield case, "static", None, None, None
        yield case, "oracle", None, None, None
        yield case, "estimate", 0.0, "baseline", 0
        conditions = [(sigma, policy, r) for sigma in SIGMAS for r in range(realizations) for policy in policies]
        # Prespecified randomized order avoids consistently timing high noise last.
        rng = np.random.default_rng(np.random.SeedSequence([seed, *stable_words(case["encounter_id"]), 999]))
        for index in rng.permutation(len(conditions)):
            sigma, policy, realization = conditions[index]
            yield case, "estimate", sigma, policy, realization


def read_completed_parent(path):
    """The source experiment may be appending: ignore its unfinished last line."""
    if not path.exists():
        return []
    data = path.read_bytes()
    return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n") and line.strip()]


def fingerprint():
    hashes = main.source_hashes()
    hashes[str(Path(__file__).resolve().relative_to(main.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(source_hashes=hashes, environment=json.loads(json.dumps(main.environment())))


def write_json(path, value):
    path.write_text(json.dumps(main.clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(main.clean(v), allow_nan=False)+"\n" for v in values), encoding="utf-8")


def initialize_study(args):
    current = fingerprint()
    policies = POLICIES if args.filter_policy == "both" else (args.filter_policy,)
    configuration = dict(version=2, noise_seed=args.seed, realizations=args.realizations,
                         sizes=list(SWARM_SIZES), speed_mps=3.0, sigmas_hz=list(SIGMAS),
                         policies=list(policies), checkpoints_s=list(CHECKPOINTS),
                         forecast_horizon_s=FORECAST_HORIZON, bootstrap_samples=BOOTSTRAPS)
    # Record the actual EKF default, not an inferred sensor calibration.
    from estimation.ekf import ExtendedKalmanFilter
    configuration["base_sigma_hz"] = float(ExtendedKalmanFilter().sigma_doppler)
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != configuration:
            raise ValueError("configuration differs: resume with the same --filter-policy, --seed and --realizations; older two-size studies require a new output directory")
        if manifest["fingerprint"] != current:
            raise ValueError("source or environment changed; use a new output directory")
        for name in ("encounters.jsonl", "reference_snapshot.jsonl"):
            digest = hashlib.sha256((args.output/name).read_bytes()).hexdigest()
            if digest != manifest["input_hashes"][name]:
                raise ValueError(f"Saved input changed: {name}")
        return main.load_jsonl(args.output / "encounters.jsonl"), manifest
    if args.output.exists():
        raise ValueError("output exists; choose a fresh directory or use --resume")
    source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    # The noise runner is not a dependency of the source main evaluation. Older
    # main manifests predate its move into experiments/; compare actual shared
    # implementation files without requiring the new study's own source there.
    def shared_hashes(hashes):
        return {path:digest for path,digest in hashes.items()
                if Path(path.replace("\\","/")).as_posix() != "experiments/noise_robustness.py"}
    if shared_hashes(source_manifest["source_hashes"]) != shared_hashes(main.source_hashes()):
        raise ValueError("Source study implementation differs from current implementation")
    if source_manifest["environment"] != current["environment"]:
        raise ValueError("Source study environment differs from the current environment")
    parent_file = args.source / "encounters.jsonl"
    if hashlib.sha256(parent_file.read_bytes()).hexdigest() != source_manifest["encounters_sha256"]:
        raise ValueError("Source encounters do not match the source manifest")
    cases = [c for c in main.load_jsonl(parent_file) if c["swarm_size"] in SWARM_SIZES and c["speed_mps"] == 3.0]
    if not cases:
        raise ValueError("No 4/6/8/10-UAV, 3 m/s encounters in source study")
    groups = defaultdict(int)
    for c in cases:
        distances, spacing, _ = main.stationary_geometry(np.asarray(c["state"]), np.asarray(c["uav_positions"]), c["duration_s"])
        if main.classify(distances) != c["category"] or spacing < main.UAV_CLEARANCE:
            raise ValueError(f"Invalid source encounter: {c['encounter_id']}")
        groups[(c["swarm_size"], c["category"])] += 1
    if set(groups) != {(n, category) for n in SWARM_SIZES for category in main.CLASSES} or len(set(groups.values())) != 1:
        raise ValueError("Source encounters must be balanced across size and category")
    selected = {c["encounter_id"] for c in cases}
    references = [r for r in read_completed_parent(args.source / "results.jsonl")
                  if r["encounter_id"] in selected and r["method"] in ("static", "oracle")]
    keys = [(r["encounter_id"], r["method"]) for r in references]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate parent reference results")
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "encounters.jsonl", cases)
    write_jsonl(args.output / "reference_snapshot.jsonl", references)
    for relative in current["source_hashes"]:
        destination = args.output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(main.ROOT / relative, destination)
    manifest = dict(config=configuration, fingerprint=current, source=str(args.source.resolve()),
                    source_manifest=source_manifest, encounters=len(cases),
                    planned_trials=len(cases)*(3+len(SIGMAS)*len(policies)*args.realizations),
                    input_hashes={name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
                                  for name in ("encounters.jsonl", "reference_snapshot.jsonl")})
    write_json(args.output / "manifest.json", manifest)
    return cases, manifest


def bootstrap(values, label, bounds=(0.0, 1.0)):
    """Resample encounter means, never individual correlated noise repetitions."""
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return None, None, None
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    if np.ptp(values) < 1e-12:
        # A naive bootstrap gives [1,1] for perfect observed success. Use a
        # bounded-mean fallback instead of suggesting perfect known reliability.
        left, right = bounds
        if mean == right:
            return mean, left+(right-left)*0.025**(1/len(values)), right
        if mean == left:
            return mean, left, right-(right-left)*0.025**(1/len(values))
        radius = (right-left)*math.sqrt(math.log(40)/(2*len(values)))
        return mean, max(left,mean-radius), min(right,mean+radius)
    rng = np.random.default_rng(stable_words(label))
    samples = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return mean, float(low), float(high)


def clustered_metric(rows, getter):
    clusters = defaultdict(list)
    for row in rows:
        value = getter(row)
        if value is not None:
            clusters[row["encounter_id"]].append(float(value))
    return [float(np.mean(values)) for values in clusters.values()]


def write_csv(path, rows):
    import csv
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cases = {c["encounter_id"]: c for c in main.load_jsonl(output / "encounters.jsonl")}
    rows = main.load_jsonl(output / "results.jsonl")
    if not rows:
        print("No completed trials to summarize.")
        return
    if len({r["trial_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trial IDs")
    clean_controls = {r["encounter_id"]: r for r in rows if r["method"] == "estimate" and r["sigma_hz"] == 0}
    groups = defaultdict(list)
    for row in rows:
        case = cases[row["encounter_id"]]
        groups[(case["swarm_size"], case["speed_mps"], case["category"],
                row["method"], row["sigma_hz"], row["filter_policy"])].append(row)
    summaries, checkpoints = [], []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        n, speed, category, method, sigma, policy = key
        safe, low, high = bootstrap(clustered_metric(group, lambda r: bool(r.get("safety_satisfied"))), str(key)+"safe")
        pairable = [r for r in group if r["encounter_id"] in clean_controls]
        change, change_low, change_high = bootstrap(clustered_metric(
            pairable, lambda r: int(bool(r.get("safety_satisfied"))) -
            int(bool(clean_controls[r["encounter_id"]].get("safety_satisfied")))), str(key)+"change", bounds=(-1,1))
        if method == "estimate" and sigma == 0:
            change, change_low, change_high = 0.0, 0.0, 0.0  # Same trial compared with itself.
        common_safe = [r for r in pairable if r.get("safety_satisfied") and
                       clean_controls[r["encounter_id"]].get("safety_satisfied")]
        displacement_delta = clustered_metric(common_safe, lambda r: r["path_length_m"]-
                                               clean_controls[r["encounter_id"]]["path_length_m"])
        errors = sum(r["status"] == "error" for r in group)
        assumed = sorted({r["assumed_sigma_hz"] for r in group if r.get("assumed_sigma_hz") is not None})
        summary = dict(swarm_size=n, speed_mps=speed, category=category, method=method,
                       sigma_hz=sigma, radial_velocity_sigma_mps=(sigma*343/(2*10000) if sigma is not None else None),
                       filter_policy=policy, assumed_sigma_hz=assumed[0] if len(assumed)==1 else None,
                       trials=len(group), encounters=len({r["encounter_id"] for r in group}),
                       safety_rate=safe, safety_ci_low=low, safety_ci_high=high,
                       safety_change_vs_clean=change, safety_change_ci_low=change_low, safety_change_ci_high=change_high,
                       observed_drop_at_least_5pp=(change <= -0.05 if change is not None and method=="estimate" and sigma else None),
                       errors=errors, unknown_collision_outcomes=errors,
                       projectile_collisions=sum(r.get("projectile_collision") is True for r in group),
                       uav_collisions=sum(r.get("uav_collision") is True for r in group),
                       initialization_failure_rate=(float(np.mean(clustered_metric(group,
                           lambda r: not r.get("initialization_success", False)))) if method=="estimate" else None),
                       common_safe_trials=len(common_safe),
                       common_safe_path_change_mean_m=float(np.mean(displacement_delta)) if displacement_delta else None)
        condition_encounters = sum(c["swarm_size"]==n and c["category"]==category for c in cases.values())
        summary["planned_trials"] = condition_encounters * (
            manifest["config"]["realizations"] if method=="estimate" and sigma else 1)
        summary["condition_complete"] = len(group) == summary["planned_trials"]
        for field in ("position_rmse_m", "velocity_rmse_mps", "first_estimate_available_s", "first_command_sim_s",
                      "initialization_runtime_s", "planning_total_s", "estimation_total_s", "wall_runtime_s"):
            values = [r[field] for r in group if r.get(field) is not None]
            summary[field+"_median"] = float(np.median(values)) if values else None
            summary[field+"_p95"] = float(np.percentile(values,95)) if values else None
        screens = [r for r in group if r.get("partial_timing_screen_met") is not None]
        summary["partial_screen_trials"] = len(screens)
        summary["partial_screen_passes"] = sum(r["partial_timing_screen_met"] for r in screens)
        summaries.append(summary)
        if method != "estimate":
            continue
        for checkpoint in CHECKPOINTS:
            valid = []
            for row in group:
                score = next(s for s in row["checkpoint_scores"] if s["time_s"] == checkpoint)
                if score["status"] == "valid":
                    valid.append(dict(encounter_id=row["encounter_id"], **score))
            checkpoints.append(dict(swarm_size=n, speed_mps=speed, category=category, sigma_hz=sigma,
                                    filter_policy=policy, time_s=checkpoint, horizon_s=FORECAST_HORIZON,
                                    trials=len(group), valid_estimates=len(valid), availability_rate=len(valid)/len(group),
                                    position_rmse_m=math.sqrt(np.mean([s["position_error_m"]**2 for s in valid])) if valid else None,
                                    velocity_rmse_mps=math.sqrt(np.mean([s["velocity_error_mps"]**2 for s in valid])) if valid else None,
                                    forecast_rmse_m=math.sqrt(np.mean([s["forecast_error_m"]**2 for s in valid])) if valid else None))
    # Compare filter policies on exactly matching encounter/noise realizations.
    paired_policies = defaultdict(dict)
    for row in rows:
        if row["method"] == "estimate" and row["sigma_hz"] > 0:
            paired_policies[(row["encounter_id"],row["sigma_hz"],row["noise_realization"])][row["filter_policy"]] = row
    policy_groups = defaultdict(list)
    for (encounter_id,sigma,realization), pair in paired_policies.items():
        if all(policy in pair for policy in POLICIES):
            case = cases[encounter_id]
            policy_groups[(case["swarm_size"],case["category"],sigma)].append(dict(
                encounter_id=encounter_id, delta=int(bool(pair["variance_aware"].get("safety_satisfied")))-
                int(bool(pair["fixed"].get("safety_satisfied")))))
    policy_comparisons = []
    for key, pairs in sorted(policy_groups.items()):
        delta, low, high = bootstrap(clustered_metric(pairs,lambda r:r["delta"]),str(key)+"policies",bounds=(-1,1))
        policy_comparisons.append(dict(swarm_size=key[0],category=key[1],sigma_hz=key[2],matched_trial_pairs=len(pairs),
                                       safety_change_aware_minus_fixed=delta,ci_low=low,ci_high=high))
    write_csv(output / "policy_comparison.csv", policy_comparisons)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "checkpoint_summary.csv", checkpoints)
    flat = [{**{k: cases[r["encounter_id"]][k] for k in ("swarm_size", "speed_mps", "category", "geometry_id")},
             **{k: v for k, v in r.items() if k not in ("checkpoint_scores", "planning_events")}} for r in rows]
    write_csv(output / "metrics.csv", flat)
    score_rows = [{**{k: cases[r["encounter_id"]][k] for k in ("swarm_size","speed_mps","category")},
                   "trial_id": r["trial_id"], "encounter_id": r["encounter_id"],
                   "sigma_hz": r["sigma_hz"], "filter_policy": r["filter_policy"],
                   "noise_realization": r["noise_realization"], **score}
                  for r in rows for score in r.get("checkpoint_scores", [])]
    write_csv(output / "checkpoint_metrics.csv", score_rows)
    lines = ["# Doppler measurement-noise robustness", "",
             f"Completed {len(rows)} / {manifest['planned_trials']} planned trials.",
             "Additional Gaussian noise is in Hz; zero means no ADDED noise, not perfect measurements.",
             "Intervals resample encounters, keeping noise repetitions together; constant outcomes use bounded-mean fallbacks.",
             "Errors count as safety failures; collision outcomes on errors remain unknown.",
             "Fixed-time error metrics are conditional on an available estimate; read availability too.",
             "Timing screens exclude acoustic processing and networking; compute delays are not simulated.", "",
             "| UAVs | Class | Method/policy | Added sigma (Hz) | Trials | Geometries | Safe | Errors |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['swarm_size']} | {s['category']} | {s['method']}/{s['filter_policy'] or 'reference'} | "
                     f"{s['sigma_hz'] if s['sigma_hz'] is not None else 'N/A'} | {s['trials']} | "
                     f"{s['encounters']} | {s['safety_rate']:.1%} | {s['errors']} |")
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot_results(output, summaries, checkpoints, manifest["config"]["sizes"], manifest["config"]["policies"])
    print(f"Wrote report.md, summary.csv, metrics.csv, checkpoint tables and plots in {output}", flush=True)


def plot_results(output, summaries, checkpoints, sizes=SWARM_SIZES, policies=POLICIES):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mode in ("safety", "prediction"):
        fig, axes = plt.subplots(len(sizes), 4, figsize=(16, 3.5*len(sizes)), squeeze=False)
        for i, n in enumerate(sizes):
            for j, category in enumerate(main.CLASSES):
                ax = axes[i,j]
                if mode == "safety":
                    subset = [s for s in summaries if s["swarm_size"]==n and s["category"]==category and s["method"]=="estimate"]
                else:
                    subset = [s for s in checkpoints if s["swarm_size"]==n and s["category"]==category and s["time_s"]==0.6]
                for policy in policies:
                    series = sorted([s for s in subset if s["filter_policy"] in (policy,"baseline")], key=lambda s:s["sigma_hz"])
                    field = "safety_rate" if mode=="safety" else "forecast_rmse_m"
                    series = [s for s in series if s[field] is not None]
                    x, y = [s["sigma_hz"] for s in series], [s[field] for s in series]
                    ax.plot(x,y,marker="o",label=policy)
                    if mode=="safety":
                        ax.fill_between(x, [s["safety_ci_low"] if s["safety_ci_low"] is not None else s[field] for s in series],
                                        [s["safety_ci_high"] if s["safety_ci_high"] is not None else s[field] for s in series], alpha=0.15)
                if mode=="safety":
                    for method, style in (("oracle","--"),("static",":")):
                        ref = next((s for s in summaries if s["swarm_size"]==n and s["category"]==category and s["method"]==method), None)
                        if ref:
                            ax.axhline(ref["safety_rate"], linestyle=style, color="gray", label=method)
                    ax.set_ylim(-0.02,1.02)
                ax.set_title(f"N={n}: {category.replace('_',' ')}")
                ax.set_xlabel("Added Doppler sigma (Hz)")
                ax.set_ylabel("Safety success" if mode=="safety" else "0.2 s forecast RMSE (m)")
                ax.grid(alpha=0.25)
        axes[0,0].legend(fontsize=8)
        fig.suptitle("Safety vs measurement noise" if mode=="safety" else "Prediction at t=0.6 s (available estimates only)")
        fig.tight_layout()
        fig.savefig(output/f"{mode}_vs_noise.png", dpi=160)
        plt.close(fig)


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/main_02"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument("--realizations", type=int, default=5)
    parser.add_argument("--filter-policy", choices=(*POLICIES,"both"), default="fixed",
                        help="EKF noise policy to evaluate (default: fixed)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.realizations < 1:
        parser.error("seed must be nonnegative and realizations positive")
    if args.resume and args.summarize:
        parser.error("choose --resume or --summarize")
    if args.summarize:
        if not (args.output/"manifest.json").exists():
            parser.error("no noise experiment manifest in output directory")
        summarize(args.output)
        return
    from experiments import online_avoidance as online
    try:
        cases, manifest = initialize_study(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    references = {(r["encounter_id"],r["method"]):r for r in main.load_jsonl(args.output/"reference_snapshot.jsonl")}
    previous = main.load_jsonl(args.output/"results.jsonl")
    done = {r["trial_id"] for r in previous}
    if len(done) != len(previous):
        parser.error("duplicate completed trial IDs")
    print(f"{len(cases)} encounters; {manifest['planned_trials']} trials, {len(done)} saved. "
          "Existing oracle/static results are reused when available.", flush=True)
    try:
        with (args.output/"results.jsonl").open("a",encoding="utf-8") as file:
            for case, method, sigma, policy, realization in trials(cases,args.realizations,args.seed,manifest["config"]["policies"]):
                key = trial_id(case["encounter_id"],method,sigma,policy,realization)
                if key in done:
                    continue
                origin = "noise_study"
                start = time.perf_counter()
                if method == "estimate":
                    result = run_estimate(case,sigma,policy,realization,args.seed,online)
                elif (case["encounter_id"],method) in references:
                    result = {k:v for k,v in references[(case["encounter_id"],method)].items()
                              if k not in ("encounter_id","method")}
                    origin = "parent_snapshot"
                else:
                    try:
                        result = main.static_result(case) if method=="static" else main.run_online(case,method,online)
                    except Exception as exc:
                        result = dict(status="error",safety_satisfied=False,error=f"{type(exc).__name__}: {exc}",
                                      wall_runtime_s=time.perf_counter()-start)
                row = main.clean(dict(trial_id=key,encounter_id=case["encounter_id"],method=method,
                                      sigma_hz=sigma,filter_policy=policy,noise_realization=realization,
                                      origin=origin,**result))
                file.write(json.dumps(row,allow_nan=False)+"\n")
                file.flush()
                done.add(key)
                print(f"[{len(done)}/{manifest['planned_trials']}] {case['encounter_id']} {method} "
                      f"sigma={sigma} policy={policy} realization={realization}: "
                      f"safe={row['safety_satisfied']} status={row['status']}",flush=True)
    finally:
        summarize(args.output)


if __name__ == "__main__":
    main_cli()
