"""Supplementary Fibonacci-sphere formation evaluation across size and speed.

Run:     python -m experiments.fibonacci_evaluation --output results/fibonacci_03
Resume:  python -m experiments.fibonacci_evaluation --output results/fibonacci_03 --resume
Analyze: python -m experiments.fibonacci_evaluation --output results/fibonacci_03 --summarize

Design and limitations are documented in FIBONACCI_EXPERIMENT.md.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


from experiments.evaluation_common import (
    ROOT, CLASSES, METHODS, PROJECTILE_CONTACT, PROJECTILE_CLEARANCE,
    UAV_CONTACT, UAV_CLEARANCE, MANEUVER, fibonacci_formation as formation,
    stationary_geometry, classify, clean, source_hashes, environment, load_jsonl,
    static_result, classification_metrics, run_dopplerdodge, run_online, wilson,
)

SIZES = (4, 6, 8, 10)
SPEEDS = (1.0, 3.0, 5.0)
WARNING = 1.2  # Time to selected target point, not first safety entry.


def encounters(seed, repetitions):
    """Rejection sample geometry; verify the requested class for all three speeds.

    A draw shares target, direction and formation across speeds. Conditioning
    on actual swarm clearance gives balanced class coverage, not a natural
    deployment distribution.
    """
    for n in SIZES:
        for class_index, category in enumerate(CLASSES):
            for repetition in range(repetitions):
                rng = np.random.default_rng(np.random.SeedSequence([seed, n, class_index, repetition]))
                for attempt in range(10000):
                    rotation, triangular = np.linalg.qr(rng.normal(size=(3, 3)))
                    rotation *= np.sign(np.diag(triangular))
                    if np.linalg.det(rotation) < 0:
                        rotation[:, 0] *= -1
                    positions = formation(n) @ rotation
                    target_index = int(rng.integers(n))
                    target = positions[target_index].copy()
                    direction = rng.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    extra_distance = 0.0
                    if category == "multiple_threats":
                        second = int(rng.choice([i for i in range(n) if i != target_index]))
                        segment = positions[second] - target
                        extra_distance = float(np.linalg.norm(segment))
                        direction = segment / extra_distance
                    else:
                        perpendicular = rng.normal(size=3)
                        perpendicular -= direction * (perpendicular @ direction)
                        perpendicular /= np.linalg.norm(perpendicular)
                        low, high = {"direct_threat": (0.0, 0.05),
                                     "grazing_threat": (0.075, 0.095),
                                     "safe_miss": (0.12, 0.40)}[category]
                        target += perpendicular * rng.uniform(low, high)
                    candidates = []
                    for speed in SPEEDS:
                        velocity = direction * speed
                        state = np.r_[target - WARNING * velocity, velocity]
                        duration = WARNING + extra_distance / speed + 0.4
                        distances, min_uav, entry = stationary_geometry(state, positions, duration)
                        initially_safe = np.min(np.linalg.norm(state[:3]-positions, axis=1)) > PROJECTILE_CLEARANCE
                        if classify(distances) != category or not initially_safe or min_uav < UAV_CLEARANCE:
                            break
                        candidates.append(dict(
                            encounter_id=f"n{n}_{category}_{repetition:03d}_v{speed:g}",
                            geometry_id=f"n{n}_{category}_{repetition:03d}",
                            swarm_size=n, category=category, repetition=repetition,
                            speed_mps=speed, target_time_s=WARNING, duration_s=duration,
                            state=state.tolist(), uav_positions=positions.tolist(),
                            static_min_projectile_m=float(distances.min()),
                            static_min_uav_m=min_uav, first_safety_entry_s=entry,
                            static_threatened_uavs=int(np.sum(distances < PROJECTILE_CLEARANCE)),
                        ))
                    if len(candidates) == len(SPEEDS):
                        yield from candidates
                        break
                else:
                    raise RuntimeError(f"Cannot generate {n}/{category}/{repetition}")


def summarize(output):
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    study_methods = manifest["config"]["methods"]
    cases = {c["encounter_id"]: c for c in load_jsonl(output / "encounters.jsonl")}
    rows = load_jsonl(output / "results.jsonl")
    if not rows:
        print("No completed trials yet.")
        return
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["encounter_id"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate trial: {key}")
        seen.add(key)
        c = cases[row["encounter_id"]]
        groups[(c["swarm_size"], c["speed_mps"], c["category"], row["method"])].append(row)
    summaries = []
    for (n, speed, category, method), group in sorted(groups.items()):
        safe = sum(r.get("safety_satisfied") is True for r in group)
        low, high = wilson(safe, len(group))
        movement = [r["path_length_m"] for r in group if r.get("safety_satisfied") is True]
        times = [r["wall_runtime_s"] for r in group if r.get("status") == "ok"]
        screens = [r["partial_timing_screen_met"] for r in group
                   if r.get("partial_timing_screen_met") is not None]
        collision_scored = [r for r in group if r.get("projectile_collision") is not None and r.get("uav_collision") is not None]
        free = sum(r.get("status")=="ok" and not r["projectile_collision"] and not r["uav_collision"] for r in collision_scored)
        classified = [r for r in group if r.get("true_positives") is not None]
        tp = sum(r["true_positives"] for r in classified)
        fp = sum(r["false_positives"] for r in classified)
        positives = sum(r["threatened_uavs"] for r in classified)
        negatives = sum(r["nonthreatened_uavs"] for r in classified)
        summary = dict(swarm_size=n, speed_mps=speed, category=category, method=method,
                              trials=len(group), safety_successes=safe, safety_rate=safe/len(group),
                              safety_margin_success_rate=safe/len(group),collision_free_success_rate=free/len(group),
                              collision_evaluable_trials=len(collision_scored),
                              projectile_collision_rate=(sum(r["projectile_collision"] for r in collision_scored)/len(collision_scored) if collision_scored else None),
                              uav_collision_rate=(sum(r["uav_collision"] for r in collision_scored)/len(collision_scored) if collision_scored else None),
                              classification_evaluable_trials=len(classified),
                              hit_recall=tp/positives if positives else None,
                              false_positive_rate=fp/negatives if negatives else None,
                              safety_ci_low=low, safety_ci_high=high,
                              projectile_collisions=sum(r.get("projectile_collision") is True for r in group),
                              uav_collisions=sum(r.get("uav_collision") is True for r in group),
                              errors=sum(r["status"] == "error" for r in group),
                              partial_screen_trials=len(screens), partial_screen_passes=sum(screens),
                              safe_path_length_median_m=statistics.median(movement) if movement else None,
                              wall_runtime_median_s=statistics.median(times) if times else None)
        for field in ("min_projectile_m","min_uav_m","path_length_m","number_moved",
                      "detection_latency_s","avoidance_response_time_s","detection_compute_s",
                      "avoidance_response_compute_s","coordinated_plan_success","reactive_uavs_moved"):
            values = [r[field] for r in group if r.get(field) is not None]
            summary[field+"_observations"] = len(values)
            summary[field+"_mean"] = statistics.mean(values) if values else None
            summary[field+"_median"] = statistics.median(values) if values else None
            summary[field+"_min"] = min(values) if values else None
            summary[field+"_p95"] = float(np.percentile(values,95)) if values else None
        summaries.append(summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)
    flat_rows = [{**{k: cases[r["encounter_id"]][k] for k in
                    ("swarm_size", "speed_mps", "category", "first_safety_entry_s")},
                  **{k: v for k, v in r.items() if k != "planning_events"}} for r in rows]
    fields = list(dict.fromkeys(k for r in flat_rows for k in r))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = ["# Supplementary Fibonacci-sphere formation evaluation", "", f"Completed {len(rows)} / {len(cases)*len(study_methods)} method trials.",
             "Errors count as safety failures; collision counts exclude unknown/error outcomes.",
             "Collision-free success uses all trials; collision rates use collision-evaluable trials.",
             "HIT/FPR score first decisions against full-encounter stationary-swarm truth; coverage is in summary.csv.",
             "Detection/response are simulation times, not measured wall-clock controller latency.",
             "Unavailable metrics on imported older runs are not reconstructed or counted as zero.",
             "Timing screens exclude acoustic processing and are not real-time safety claims.", "",
             "| UAVs | Speed | Class | Method | Trials | Collision free | Projectile collision rate | UAV collision rate | Safe | Safety 95% interval | Closest projectile/UAV (m) | Errors |",
             "|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for s in summaries:
        rates = ["N/A" if s[k] is None else f"{s[k]:.1%}" for k in ("projectile_collision_rate","uav_collision_rate")]
        minima = ["N/A" if s[k] is None else f"{s[k]:.4f}" for k in ("min_projectile_m_min","min_uav_m_min")]
        lines.append(f"| {s['swarm_size']} | {s['speed_mps']:g} | {s['category']} | {s['method']} | "
                     f"{s['trials']} | {s['collision_free_success_rate']:.1%} | {rates[0]} | {rates[1]} | {s['safety_rate']:.1%} | "
                     f"{s['safety_ci_low']:.1%}–{s['safety_ci_high']:.1%} | {' / '.join(minima)} | {s['errors']} |")
    lines += ["", "Movement and first-decision metrics (N/A means unavailable):", "",
              "| UAVs | Speed | Class | Method | Mean path (m) | Mean UAVs moved | HIT recall | FPR | Mean detection (s) | Mean response (s) |",
              "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        metrics = [s[k] for k in ("path_length_m_mean","number_moved_mean","hit_recall","false_positive_rate",
                                 "detection_latency_s_mean","avoidance_response_time_s_mean")]
        lines.append(f"| {s['swarm_size']} | {s['speed_mps']:g} | {s['category']} | {s['method']} | "+
                     " | ".join("N/A" if v is None else f"{v:.4f}" for v in metrics)+" |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'report.md'}, metrics.csv and summary.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--reuse-results",type=Path,help="Import compatible saved methods into a new study")
    parser.add_argument("--dodge-threshold",type=float,help="Use a preselected threshold instead of calibration")
    parser.add_argument("--dodge-calibration-count",type=int,default=50)
    parser.add_argument("--dodge-calibration-seed",type=int,default=42)
    args = parser.parse_args()
    if args.repetitions < 1 or args.seed < 0:
        parser.error("repetitions must be positive and seed nonnegative")
    if args.dodge_calibration_count < 1 or args.dodge_calibration_seed < 0:
        parser.error("calibration count must be positive and seed nonnegative")
    if args.dodge_threshold is not None and (not math.isfinite(args.dodge_threshold) or args.dodge_threshold < 0):
        parser.error("Doppler Dodge threshold must be finite and nonnegative")
    if args.resume and args.reuse_results:
        parser.error("--reuse-results creates a new study; use only --resume afterward")
    if args.summarize:
        if not (args.output / "manifest.json").exists():
            parser.error("no experiment manifest in output directory")
        summarize(args.output)
        return
    # Load simulation dependencies before creating output, not once per encounter.
    from experiments import online_avoidance as online
    from baselines import dopplerdodge as baseline
    from baselines import dopplerdodge_coordinated as coordinated_baseline
    if not np.allclose([online.PROJECTILE_SAFETY_DISTANCE, online.UAV_SAFETY_DISTANCE,
                        online.UAV_RADIUS+online.PROJECTILE_RADIUS, 2*online.UAV_RADIUS,
                        online.MANEUVER_DURATION],
                       [PROJECTILE_CLEARANCE, UAV_CLEARANCE, PROJECTILE_CONTACT, UAV_CONTACT, MANEUVER]):
        parser.error("implementation safety/maneuver constants differ from the study protocol")
    config = dict(version=3, seed=args.seed, repetitions=args.repetitions, sizes=SIZES,
                  speeds=SPEEDS, classes=CLASSES, methods=METHODS, rms_radius_m=1.0,
                  target_time_s=WARNING, uplink_s=0.0, downlink_s=0.0, added_noise=False,
                  dopplerdodge=dict(requested_threshold=args.dodge_threshold,
                                   calibration_count=args.dodge_calibration_count,
                                   calibration_seed=args.dodge_calibration_seed))
    config = json.loads(json.dumps(config))
    hashes = source_hashes()
    env = json.loads(json.dumps(environment()))
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != config or manifest["source_hashes"] != hashes or manifest["environment"] != env:
            parser.error("configuration, source, or environment changed; for an older study use a NEW output with --reuse-results OLD_DIRECTORY")
        dataset = load_jsonl(args.output / "encounters.jsonl")
        actual_hash = hashlib.sha256((args.output / "encounters.jsonl").read_bytes()).hexdigest()
        if actual_hash != manifest["encounters_sha256"]:
            parser.error("encounter dataset changed")
    else:
        if args.output.exists():
            parser.error("output exists; use --resume or a new directory")
        imported = []
        provenance = None
        if args.reuse_results:
            parent = json.loads((args.reuse_results/"manifest.json").read_text(encoding="utf-8"))
            expected = {k:v for k,v in config.items() if k not in ("version","methods","dopplerdodge")}
            if any(parent["config"].get(k) != v for k,v in expected.items()):
                parser.error("source encounter protocol differs; use matching seed/repetitions")
            for path,digest in parent["source_hashes"].items():
                normalized = Path(path.replace("\\","/")).as_posix()
                if normalized in ("experiments/fibonacci_evaluation.py",):
                    continue  # Reporting/instrumentation adapter changed, not the controller.
                if normalized == "baselines/dopplerdodge_coordinated.py" and "dopplerdodge_coordinated" not in parent["config"]["methods"]:
                    continue  # Previously unused baseline API was extended for the new method.
                if hashes.get(path) != digest:
                    parser.error(f"cannot reuse: implementation changed: {path}")
            if parent["environment"] != env:
                parser.error("cannot reuse results from a different environment")
            encounter_path = args.reuse_results/"encounters.jsonl"
            if hashlib.sha256(encounter_path.read_bytes()).hexdigest() != parent["encounters_sha256"]:
                parser.error("source encounter dataset changed")
            dataset = load_jsonl(encounter_path)
            valid_ids = {c["encounter_id"] for c in dataset}
            imported = [r.copy() for r in load_jsonl(args.reuse_results/"results.jsonl")
                        if r["method"] in METHODS]
            if any(r["encounter_id"] not in valid_ids for r in imported):
                parser.error("source results reference unknown encounters")
            if len({(r["encounter_id"],r["method"]) for r in imported}) != len(imported):
                parser.error("duplicate source trials")
            for row in imported:
                row.setdefault("metric_schema_version",1)
                row["origin"] = "imported"
            provenance = dict(directory=str(args.reuse_results.resolve()),manifest=parent,
                              results_sha256=hashlib.sha256((args.reuse_results/"results.jsonl").read_bytes()).hexdigest())
        else:
            dataset = list(encounters(args.seed, args.repetitions))
        saved_threshold = parent.get("dopplerdodge_threshold") if args.reuse_results else None
        if saved_threshold is not None:
            if args.dodge_threshold is not None and args.dodge_threshold != saved_threshold:
                parser.error("cannot reuse baseline results with a different frozen threshold")
            threshold = saved_threshold
            calibration_details = parent.get("calibration_details","Frozen threshold inherited from source study")
        elif args.dodge_threshold is None:
            calibration_log = io.StringIO()
            print("Calibrating Doppler Dodge on separate trajectories...",flush=True)
            with contextlib.redirect_stdout(calibration_log):
                threshold = baseline.calibrate_threshold(args.dodge_calibration_count,args.dodge_calibration_seed)
            calibration_details = calibration_log.getvalue()
        else:
            threshold, calibration_details = args.dodge_threshold, "User-supplied frozen threshold"
        print(f"Frozen Doppler Dodge threshold: {threshold:.6f}",flush=True)
        args.output.mkdir(parents=True)
        dataset_text = "".join(json.dumps(clean(c), allow_nan=False)+"\n" for c in dataset)
        (args.output / "encounters.jsonl").write_text(dataset_text, encoding="utf-8")
        manifest = dict(config=config, source_hashes=hashes, environment=env,
                        encounters_sha256=hashlib.sha256((args.output / "encounters.jsonl").read_bytes()).hexdigest(),
                        dopplerdodge_threshold=threshold,calibration_details=calibration_details,
                        imported_results=provenance)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if imported:
            (args.output/"results.jsonl").write_text("".join(json.dumps(clean(r),allow_nan=False)+"\n" for r in imported),encoding="utf-8")
    previous = load_jsonl(args.output / "results.jsonl")
    done = {(r["encounter_id"], r["method"]) for r in previous}
    if len(done) != len(previous):
        parser.error("duplicate trials in results file")
    total = len(dataset)*len(METHODS)
    print(f"{len(dataset)} encounters, {total} method trials; {len(done)} already saved", flush=True)
    try:
        with (args.output / "results.jsonl").open("a", encoding="utf-8") as file:
            for case in dataset:
                # Rotate expensive methods so no baseline is always timed last.
                rotation = case["repetition"] % (len(METHODS)-1)
                costly = METHODS[1:]
                methods = ("static",)+costly[rotation:]+costly[:rotation]
                for method in methods:
                    key = (case["encounter_id"], method)
                    if key in done:
                        continue
                    start = time.perf_counter()
                    try:
                        if method == "static":
                            result = static_result(case)
                        elif method in ("dopplerdodge","dopplerdodge_coordinated"):
                            seed_words = [args.seed,*np.frombuffer(hashlib.sha256(case["encounter_id"].encode()).digest()[:16],dtype="<u4")]
                            dodge_seed = int(np.random.SeedSequence(seed_words).generate_state(1)[0])
                            if method == "dopplerdodge_coordinated":
                                result = run_dopplerdodge(case,manifest["dopplerdodge_threshold"],dodge_seed,coordinated_baseline,coordinated=True)
                            else:
                                result = run_dopplerdodge(case,manifest["dopplerdodge_threshold"],dodge_seed,baseline)
                        else:
                            result = run_online(case, method, online)
                    except Exception as exc:
                        result = dict(status="error", safety_satisfied=False, error=f"{type(exc).__name__}: {exc}",
                                      wall_runtime_s=time.perf_counter()-start)
                    row = clean(dict(encounter_id=case["encounter_id"], method=method, **result))
                    file.write(json.dumps(row, allow_nan=False)+"\n")
                    file.flush()
                    done.add(key)
                    print(f"[{len(done)}/{total}] {key[0]} {method}: "
                          f"safe={row['safety_satisfied']} status={row['status']}", flush=True)
    finally:
        summarize(args.output)


if __name__ == "__main__":
    main()
