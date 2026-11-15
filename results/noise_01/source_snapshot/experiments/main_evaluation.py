"""Paper experiment 1: paired closed-loop safety across swarm size and speed.

Run:     python -m experiments.main_evaluation --output results/main_01
Resume:  python -m experiments.main_evaluation --output results/main_01 --resume
Analyze: python -m experiments.main_evaluation --output results/main_01 --summarize

Design and limitations are documented in RESEARCH_EXPERIMENTS.md.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIZES = (4, 6, 8, 10)
SPEEDS = (1.0, 3.0, 5.0)
CLASSES = ("direct_threat", "grazing_threat", "multiple_threats", "safe_miss")
METHODS = ("static", "oracle", "estimate")
PROJECTILE_CONTACT = 0.07
PROJECTILE_CLEARANCE = 0.10
UAV_CONTACT = 0.10
UAV_CLEARANCE = 0.15
WARNING = 1.2  # Time to the selected target point, NOT first safety entry.
MANEUVER = 0.15


def formation(n):
    """Centered Fibonacci sphere, normalized to 1 m RMS radius."""
    z = 1 - 2 * (np.arange(n) + 0.5) / n
    angle = np.arange(n) * np.pi * (3 - np.sqrt(5))
    p = np.column_stack((np.sqrt(1-z*z)*np.cos(angle),
                         np.sqrt(1-z*z)*np.sin(angle), z))
    p -= p.mean(axis=0)
    return p / np.sqrt(np.mean(np.sum(p*p, axis=1)))


def stationary_geometry(state, positions, duration):
    relative = state[:3] - positions
    v = state[3:]
    a = float(v @ v)
    closest = np.clip(-(relative @ v) / a, 0, duration)
    distances = np.linalg.norm(relative + closest[:, None]*v, axis=1)
    entry = []
    for r in relative:
        b = float(2 * r @ v)
        c = float(r @ r) - PROJECTILE_CLEARANCE**2
        discriminant = b*b - 4*a*c
        if c <= 0:
            entry.append(0.0)
        elif discriminant >= 0:
            t = (-b - math.sqrt(discriminant)) / (2*a)
            if 0 <= t <= duration:
                entry.append(t)
    pair_distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    min_uav = float(pair_distances[np.triu_indices(len(positions), 1)].min())
    return distances, min_uav, min(entry) if entry else None


def classify(distances):
    threats = int(np.sum(distances < PROJECTILE_CLEARANCE))
    if threats >= 2:
        return "multiple_threats"
    if np.any(distances < PROJECTILE_CONTACT):
        return "direct_threat"
    return "grazing_threat" if threats else "safe_miss"


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


def clean(value):
    """Strict JSON: unavailable/nonfinite metrics are null, never NaN."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def source_hashes():
    paths = [p for directory in ("experiments", "simulation", "estimation", "planning")
             for p in (ROOT / directory).glob("*.py")]
    paths += [ROOT / "requirements.txt"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def environment():
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                packages=sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def static_result(case):
    projectile, uav = case["static_min_projectile_m"], case["static_min_uav_m"]
    return dict(status="ok", projectile_collision=projectile < PROJECTILE_CONTACT-1e-6,
                uav_collision=uav < UAV_CONTACT-1e-6,
                safety_satisfied=projectile >= PROJECTILE_CLEARANCE-1e-6 and uav >= UAV_CLEARANCE-1e-6,
                min_projectile_m=projectile, min_uav_m=uav, path_length_m=0.0,
                first_command_sim_s=None, initialization_runtime_s=None,
                initialization_success=None, first_estimate_sim_s=None,
                planning_attempts=0, successful_plans=0, position_rmse_m=None,
                velocity_rmse_mps=None, planning_total_s=0.0, planning_p95_s=None,
                estimation_total_s=0.0, acoustic_simulation_total_s=0.0,
                prefix_estimation_s=None, prefix_planning_s=None,
                measured_prefix_completion_screen_s=None, partial_timing_screen_met=None,
                planning_events=[], wall_runtime_s=0.0, error="")


def run_online(case, method, online):
    """Instrument in memory; no implementation files are edited.

    Prefix counts only estimation/planning through the first successful command.
    Sensor generation/extraction is timed separately because advance() mixes
    synthetic echo generation with Doppler extraction.
    """
    events = []
    prefix = None
    estimation_total = 0.0
    planning_total = 0.0
    acoustic_total = 0.0
    original_process = online.process_pipeline_measurements
    original_plan = online.plan_avoidance
    original_advance = online.MovingStreamingAcousticSensor.advance

    def process(pipeline, simulation_time, *args, **kwargs):
        nonlocal estimation_total
        start = time.perf_counter()
        try:
            return original_process(pipeline, simulation_time, *args, **kwargs)
        finally:
            estimation_total += time.perf_counter() - start

    def plan(*args, **kwargs):
        nonlocal planning_total, prefix
        start = time.perf_counter()
        result = original_plan(*args, **kwargs)
        elapsed = time.perf_counter() - start
        planning_total += elapsed
        events.append(dict(runtime_s=elapsed, success=bool(result.success)))
        if result.success and prefix is None:
            prefix = (estimation_total, planning_total)
        return result

    def advance(sensor, *args, **kwargs):
        nonlocal acoustic_total
        start = time.perf_counter()
        try:
            return original_advance(sensor, *args, **kwargs)
        finally:
            acoustic_total += time.perf_counter() - start

    start = time.perf_counter()
    with patch.object(online, "process_pipeline_measurements", process), \
         patch.object(online, "plan_avoidance", plan), \
         patch.object(online.MovingStreamingAcousticSensor, "advance", advance), \
         contextlib.redirect_stdout(io.StringIO()):
        result = online.run_online_avoidance(
            projectile_state=np.asarray(case["state"]), uav_positions=np.asarray(case["uav_positions"]),
            duration=case["duration_s"], state_source=method, verbose=False, gui=False,
        )
    wall = time.perf_counter() - start
    completion = (result.first_command_time + sum(prefix) + MANEUVER
                  if prefix is not None and result.first_command_time is not None else None)
    entry = case["first_safety_entry_s"]
    partial_screen = None if entry is None else bool(completion is not None and completion <= entry)
    return dict(status="ok",
                projectile_collision=result.minimum_projectile_separation < PROJECTILE_CONTACT-1e-6,
                uav_collision=result.minimum_uav_separation < UAV_CONTACT-1e-6,
                safety_satisfied=bool(result.safety_margin_satisfied),
                min_projectile_m=result.minimum_projectile_separation,
                min_uav_m=result.minimum_uav_separation, path_length_m=result.total_uav_path_length,
                first_command_sim_s=result.first_command_time,
                initialization_runtime_s=result.initialization_runtime,
                initialization_success=bool(result.initialized) if method == "estimate" else None,
                first_estimate_sim_s=result.first_estimate_time,
                planning_attempts=result.planning_attempts, successful_plans=result.successful_plans,
                position_rmse_m=online.rmse(result.position_errors) if method == "estimate" else None,
                velocity_rmse_mps=online.rmse(result.velocity_errors) if method == "estimate" else None,
                planning_total_s=planning_total,
                planning_p95_s=float(np.percentile([e["runtime_s"] for e in events], 95)) if events else None,
                estimation_total_s=estimation_total, acoustic_simulation_total_s=acoustic_total,
                prefix_estimation_s=prefix[0] if prefix else None,
                prefix_planning_s=prefix[1] if prefix else None,
                measured_prefix_completion_screen_s=completion, partial_timing_screen_met=partial_screen,
                planning_events=events, wall_runtime_s=wall, error="")


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return center-half, center+half


def summarize(output):
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
        summaries.append(dict(swarm_size=n, speed_mps=speed, category=category, method=method,
                              trials=len(group), safety_successes=safe, safety_rate=safe/len(group),
                              safety_ci_low=low, safety_ci_high=high,
                              projectile_collisions=sum(r.get("projectile_collision") is True for r in group),
                              uav_collisions=sum(r.get("uav_collision") is True for r in group),
                              errors=sum(r["status"] == "error" for r in group),
                              partial_screen_trials=len(screens), partial_screen_passes=sum(screens),
                              safe_path_length_median_m=statistics.median(movement) if movement else None,
                              wall_runtime_median_s=statistics.median(times) if times else None))
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
    lines = ["# Main evaluation", "", f"Completed {len(rows)} / {len(cases)*len(METHODS)} method trials.",
             "Errors count as safety failures; collision counts exclude unknown/error outcomes.",
             "Timing screens exclude acoustic processing and are not real-time safety claims.", "",
             "| UAVs | Speed | Class | Method | Trials | Safe | 95% Wilson interval | Errors |",
             "|---:|---:|---|---|---:|---:|---|---:|"]
    for s in summaries:
        lines.append(f"| {s['swarm_size']} | {s['speed_mps']:g} | {s['category']} | {s['method']} | "
                     f"{s['trials']} | {s['safety_rate']:.1%} | "
                     f"{s['safety_ci_low']:.1%}–{s['safety_ci_high']:.1%} | {s['errors']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'report.md'}, metrics.csv and summary.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.seed < 0:
        parser.error("repetitions must be positive and seed nonnegative")
    if args.summarize:
        if not (args.output / "manifest.json").exists():
            parser.error("no experiment manifest in output directory")
        summarize(args.output)
        return
    # Load simulation dependencies before creating output, not once per encounter.
    from experiments import online_avoidance as online
    if not np.allclose([online.PROJECTILE_SAFETY_DISTANCE, online.UAV_SAFETY_DISTANCE,
                        online.UAV_RADIUS+online.PROJECTILE_RADIUS, 2*online.UAV_RADIUS,
                        online.MANEUVER_DURATION],
                       [PROJECTILE_CLEARANCE, UAV_CLEARANCE, PROJECTILE_CONTACT, UAV_CONTACT, MANEUVER]):
        parser.error("implementation safety/maneuver constants differ from the study protocol")
    config = dict(version=1, seed=args.seed, repetitions=args.repetitions, sizes=SIZES,
                  speeds=SPEEDS, classes=CLASSES, methods=METHODS, rms_radius_m=1.0,
                  target_time_s=WARNING, uplink_s=0.0, downlink_s=0.0, added_noise=False)
    config = json.loads(json.dumps(config))
    hashes = source_hashes()
    env = json.loads(json.dumps(environment()))
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != config or manifest["source_hashes"] != hashes or manifest["environment"] != env:
            parser.error("configuration, source, or environment changed; use a new output directory")
        dataset = load_jsonl(args.output / "encounters.jsonl")
        actual_hash = hashlib.sha256((args.output / "encounters.jsonl").read_bytes()).hexdigest()
        if actual_hash != manifest["encounters_sha256"]:
            parser.error("encounter dataset changed")
    else:
        if args.output.exists():
            parser.error("output exists; use --resume or a new directory")
        dataset = list(encounters(args.seed, args.repetitions))
        args.output.mkdir(parents=True)
        dataset_text = "".join(json.dumps(clean(c), allow_nan=False)+"\n" for c in dataset)
        (args.output / "encounters.jsonl").write_text(dataset_text, encoding="utf-8")
        manifest = dict(config=config, source_hashes=hashes, environment=env,
                        encounters_sha256=hashlib.sha256((args.output / "encounters.jsonl").read_bytes()).hexdigest())
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    previous = load_jsonl(args.output / "results.jsonl")
    done = {(r["encounter_id"], r["method"]) for r in previous}
    if len(done) != len(previous):
        parser.error("duplicate trials in results file")
    print(f"{len(dataset)} encounters, {len(dataset)*3} method trials; {len(done)} already saved", flush=True)
    try:
        with (args.output / "results.jsonl").open("a", encoding="utf-8") as file:
            for case in dataset:
                # Alternate costly method order to reduce systematic warmup/order bias.
                methods = METHODS if case["repetition"] % 2 == 0 else ("static", "estimate", "oracle")
                for method in methods:
                    key = (case["encounter_id"], method)
                    if key in done:
                        continue
                    start = time.perf_counter()
                    try:
                        result = static_result(case) if method == "static" else run_online(case, method, online)
                    except Exception as exc:
                        result = dict(status="error", safety_satisfied=False, error=f"{type(exc).__name__}: {exc}",
                                      wall_runtime_s=time.perf_counter()-start)
                    row = clean(dict(encounter_id=case["encounter_id"], method=method, **result))
                    file.write(json.dumps(row, allow_nan=False)+"\n")
                    file.flush()
                    done.add(key)
                    print(f"[{len(done)}/{len(dataset)*3}] {key[0]} {method}: "
                          f"safe={row['safety_satisfied']} status={row['status']}", flush=True)
    finally:
        summarize(args.output)


if __name__ == "__main__":
    main()
