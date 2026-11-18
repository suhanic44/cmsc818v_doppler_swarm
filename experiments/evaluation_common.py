"""Shared geometry, controller instrumentation, metrics and study bookkeeping.

Evaluation entry points import this module independently of each other.
"""

import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("direct_threat", "grazing_threat", "multiple_threats", "safe_miss")
METHODS = ("static", "oracle", "estimate", "dopplerdodge", "dopplerdodge_coordinated")
PROJECTILE_CONTACT = 0.07
PROJECTILE_CLEARANCE = 0.10
UAV_CONTACT = 0.10
UAV_CLEARANCE = 0.15
MANEUVER = 0.15


def fibonacci_formation(n):
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
    paths = [p for directory in ("experiments", "simulation", "estimation", "planning", "baselines")
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
    return dict(status="ok", collision_free=projectile >= PROJECTILE_CONTACT-1e-6 and uav >= UAV_CONTACT-1e-6,
                number_moved=0, per_uav_path_lengths_m=[0.0]*case["swarm_size"],
                hit_recall=None, false_positive_rate=None, true_positives=None, false_positives=None,
                threatened_uavs=None, nonthreatened_uavs=None, first_predictions=None,
                detection_latency_s=None, avoidance_response_time_s=None, first_motion_sim_s=None,
                detection_compute_s=None,avoidance_response_compute_s=None,
                metric_schema_version=2,
                projectile_collision=projectile < PROJECTILE_CONTACT-1e-6,
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


def classification_metrics(case, predictions):
    """First decisions vs full-horizon, stationary counterfactual truth.

    Truth is used only here, never supplied to either detector.
    """
    if predictions is None:
        return dict(hit_recall=None, false_positive_rate=None, true_positives=None,
                    false_positives=None, threatened_uavs=None, nonthreatened_uavs=None,
                    first_predictions=None)
    distances, _, _ = stationary_geometry(np.asarray(case["state"]),
                                          np.asarray(case["uav_positions"]),case["duration_s"])
    truth = distances < PROJECTILE_CLEARANCE
    predicted = np.asarray(predictions,dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError("One HIT/MISS prediction per UAV is required")
    tp, fp = int(np.sum(predicted & truth)), int(np.sum(predicted & ~truth))
    positives, negatives = int(truth.sum()), int((~truth).sum())
    return dict(hit_recall=tp/positives if positives else None,
                false_positive_rate=fp/negatives if negatives else None,
                true_positives=tp,false_positives=fp,threatened_uavs=positives,nonthreatened_uavs=negatives,
                first_predictions=predicted.tolist())


def run_dopplerdodge(case, threshold, seed, baseline, coordinated=False):
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        run = baseline.run_coordinated_baseline if coordinated else baseline.run_baseline
        outcome = run(
            scenario=case["encounter_id"],threshold=threshold,seed=seed,gui=False,
            projectile_state=np.asarray(case["state"]),uav_positions=np.asarray(case["uav_positions"]),
            duration=case["duration_s"],
        )
        if coordinated:
            result,plan_success,reactive_moves = outcome
        else:
            result = outcome
    # Use the experiment's common tolerance, not baseline-specific comparisons.
    projectile, uav = result.min_projectile_distance, result.min_uav_distance
    row = static_result(case)
    completion = (result.first_response_time+result.detection_compute_s+result.response_compute_s+MANEUVER
                  if result.first_response_time is not None else None)
    row.update(collision_free=projectile >= PROJECTILE_CONTACT-1e-6 and uav >= UAV_CONTACT-1e-6,
               projectile_collision=projectile < PROJECTILE_CONTACT-1e-6,uav_collision=uav < UAV_CONTACT-1e-6,
               safety_satisfied=projectile >= PROJECTILE_CLEARANCE-1e-6 and uav >= UAV_CLEARANCE-1e-6,
               min_projectile_m=projectile,min_uav_m=uav,path_length_m=result.total_path_length,
               number_moved=result.number_moved,per_uav_path_lengths_m=list(result.path_lengths),
               first_motion_sim_s=result.first_motion_time,
               detection_latency_s=result.decision_latency,first_command_sim_s=result.first_response_time,
               detection_compute_s=result.detection_compute_s,avoidance_response_compute_s=result.response_compute_s,
               avoidance_response_time_s=result.response_delay if result.first_response_time is not None else None,
               estimation_total_s=result.detection_compute_s,planning_total_s=result.response_compute_s,
               acoustic_simulation_total_s=result.acoustic_generation_s,
               prefix_estimation_s=result.detection_compute_s,prefix_planning_s=result.response_compute_s,
               measured_prefix_completion_screen_s=completion,
               partial_timing_screen_met=(None if case["first_safety_entry_s"] is None else
                   bool(completion is not None and completion <= case["first_safety_entry_s"])),
               wall_runtime_s=time.perf_counter()-start,baseline_seed=seed,dodge_threshold=threshold,
               **classification_metrics(case,[p=="hit" for p in result.predictions]))
    if coordinated:
        row.update(coordinated_plan_success=bool(plan_success),reactive_uavs_moved=int(reactive_moves),
                   planning_attempts=1,successful_plans=int(plan_success),
                   planning_p95_s=result.response_compute_s,
                   planning_events=[dict(runtime_s=result.response_compute_s,success=bool(plan_success))])
    return row


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
    original_history_add = online.UAVMotionHistory.add
    import planning.avoidance_planner as planner_module
    original_predict = planner_module.predict_collisions
    current_time = 0.0
    detection_time = None
    first_predictions = None
    previous_positions = None
    path_lengths = np.zeros(case["swarm_size"])
    first_motion = None
    first_detection_compute = None

    def history_add(history, simulation_time, positions, velocities):
        nonlocal current_time, previous_positions, first_motion
        current_time = float(simulation_time)
        if previous_positions is not None:
            increments = np.linalg.norm(np.asarray(positions)-previous_positions,axis=1)
            path_lengths[:] += increments
            if first_motion is None and np.any(increments > 1e-9):
                first_motion = current_time
        previous_positions = np.asarray(positions).copy()
        return original_history_add(history,simulation_time,positions,velocities)

    def predict(*args, **kwargs):
        nonlocal first_predictions, detection_time, first_detection_compute
        start = time.perf_counter()
        predictions = original_predict(*args,**kwargs)
        if first_predictions is None:
            first_predictions = [p.threatened for p in predictions]
            detection_time = current_time
            first_detection_compute = estimation_total+time.perf_counter()-start
        return predictions

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
         patch.object(online.UAVMotionHistory,"add",history_add), \
         patch.object(planner_module,"predict_collisions",predict), \
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
                metric_schema_version=2,collision_free=bool(result.collision_free),
                number_moved=int(np.count_nonzero(path_lengths > 1e-6)),
                per_uav_path_lengths_m=path_lengths.tolist(),first_motion_sim_s=first_motion,
                detection_latency_s=detection_time,
                detection_compute_s=first_detection_compute,
                avoidance_response_compute_s=prefix[1] if prefix else None,
                avoidance_response_time_s=(result.first_command_time-detection_time
                    if result.first_command_time is not None and detection_time is not None else None),
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
                planning_events=events, wall_runtime_s=wall, error="",
                **classification_metrics(case,first_predictions))


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return center-half, center+half
