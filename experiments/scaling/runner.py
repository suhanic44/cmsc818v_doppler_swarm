"""Focused, exploratory size/spacing study. See SCALING_EXPERIMENT.md.

python -m experiments.scaling --output results/scaling_01
"""

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments import evaluation_common as main


SIZES = (10, 25, 50, 100)
RADII = (1.0, 2.0, 4.0)
CATEGORIES = ("direct_threat", "multiple_threats", "safe_miss")
METHODS = ("static", "estimate", "dopplerdodge", "dopplerdodge_coordinated")


def hash_words(text):
    return list(np.frombuffer(hashlib.sha256(text.encode()).digest()[:16], dtype="<u4"))


def spacing_metrics(positions):
    pairs = np.linalg.norm(positions[:,None]-positions[None,:],axis=2)
    np.fill_diagonal(pairs,np.inf)
    nearest = pairs.min(axis=1)
    return dict(initial_min_spacing_m=float(nearest.min()),
                initial_median_nearest_neighbor_m=float(np.median(nearest)))


def generate_encounters(sizes, radii, repetitions, seed, speed, warning):
    """Pair orientation/target/direction across radii; verify every swarm member.

    Radius changes starting distance from center and time spent crossing the
    formation. Each run covers exit from a sphere enclosing the entire swarm.
    """
    for n in sizes:
        for category_index, category in enumerate(CATEGORIES):
            for repetition in range(repetitions):
                rng = np.random.default_rng(np.random.SeedSequence([seed,n,category_index,repetition]))
                for attempt in range(10000):
                    rotation, triangular = np.linalg.qr(rng.normal(size=(3,3)))
                    rotation *= np.sign(np.diag(triangular))
                    if np.linalg.det(rotation)<0:
                        rotation[:,0]*=-1
                    unit_positions = main.fibonacci_formation(n) @ rotation
                    target_index = int(rng.integers(n))
                    direction = rng.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    if category == "multiple_threats":
                        second = int(rng.choice([i for i in range(n) if i!=target_index]))
                        direction = unit_positions[second]-unit_positions[target_index]
                        direction /= np.linalg.norm(direction)
                    elif np.dot(unit_positions[target_index],direction)>=0:
                        direction *= -1  # Approach a near-facing UAV from outside the formation.
                    perpendicular = rng.normal(size=3)
                    perpendicular -= direction*np.dot(perpendicular,direction)
                    perpendicular /= np.linalg.norm(perpendicular)
                    offset = (0.0 if category=="multiple_threats" else
                              rng.uniform(0,.05) if category=="direct_threat" else rng.uniform(.15,.50))
                    candidates = []
                    for radius in radii:
                        positions = unit_positions*radius
                        spacing = spacing_metrics(positions)
                        if spacing["initial_min_spacing_m"] < main.UAV_CLEARANCE:
                            raise ValueError(f"N={n}, R={radius}: initial spacing violates safety clearance")
                        target = positions[target_index]+offset*perpendicular
                        velocity = direction*speed
                        state = np.r_[target-warning*velocity,velocity]
                        enclosing_radius = float(np.max(np.linalg.norm(positions,axis=1)))+main.PROJECTILE_CLEARANCE
                        if np.linalg.norm(state[:3])<=enclosing_radius:
                            break
                        a = speed**2
                        b = 2*float(state[:3]@velocity)
                        c = float(state[:3]@state[:3])-enclosing_radius**2
                        discriminant = b*b-4*a*c
                        if discriminant<=0:
                            break
                        exit_time = (-b+math.sqrt(discriminant))/(2*a)
                        duration = exit_time+.2
                        if exit_time<warning or not math.isfinite(duration):
                            break
                        distances,min_uav,entry = main.stationary_geometry(state,positions,duration)
                        if main.classify(distances)!=category:
                            break
                        candidates.append(dict(
                            encounter_id=f"n{n}_r{radius:g}_{category}_{repetition:03d}",
                            geometry_id=f"n{n}_{category}_{repetition:03d}",
                            swarm_size=n,formation_radius_m=radius,category=category,repetition=repetition,
                            speed_mps=speed,target_time_s=warning,duration_s=duration,
                            state=state.tolist(),uav_positions=positions.tolist(),
                            static_min_projectile_m=float(distances.min()),static_min_uav_m=min_uav,
                            first_safety_entry_s=entry,static_threatened_uavs=int(np.sum(distances<main.PROJECTILE_CLEARANCE)),
                            shell_crowding_proxy_per_m2=n/(4*math.pi*radius**2),**spacing))
                    if len(candidates)==len(radii):
                        yield from candidates
                        break
                else:
                    raise ValueError(f"Unable to generate {n}/{category}/{repetition} across requested radii")


def _worker(connection, case, method, threshold, seed):
    """One isolated controller run; a timeout cannot leave its world in use."""
    start = time.perf_counter()
    try:
        from experiments import online_avoidance as online
        from baselines import dopplerdodge
        from baselines import dopplerdodge_coordinated
        if method=="estimate":
            result = main.run_online(case,method,online)
        elif method == "dopplerdodge":
            result = main.run_dopplerdodge(case,threshold,seed,dopplerdodge)
        elif method == "dopplerdodge_coordinated":
            result = main.run_dopplerdodge(case,threshold,seed,dopplerdodge_coordinated,coordinated=True)
        else:
            raise ValueError(f"Unknown controller: {method}")
        connection.send(main.clean(result))
    except Exception as exc:
        connection.send(dict(status="error",safety_satisfied=None,collision_free=None,
                             error=f"{type(exc).__name__}: {exc}",wall_runtime_s=time.perf_counter()-start))
    finally:
        connection.close()


def run_bounded(case, method, threshold, seed, timeout):
    context = mp.get_context("spawn")
    receiver,sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker,args=(sender,case,method,threshold,seed))
    start = time.perf_counter()
    process.start()
    sender.close()
    try:
        while True:
            remaining = timeout-(time.perf_counter()-start)
            if remaining<=0:
                return dict(status="timeout",safety_satisfied=None,collision_free=None,
                            error=f"Exceeded {timeout:g} s process runtime cap",process_elapsed_s=time.perf_counter()-start)
            if receiver.poll(min(remaining,.25)):
                try:
                    row = receiver.recv()
                except EOFError:
                    break
                row["process_elapsed_s"] = time.perf_counter()-start
                return row
            if not process.is_alive():
                break
        return dict(status="error",safety_satisfied=None,collision_free=None,
                    error=f"Controller process exited without a result (exitcode={process.exitcode})",
                    process_elapsed_s=time.perf_counter()-start)
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join()


def fingerprint():
    hashes = main.source_hashes()
    for path in Path(__file__).parent.glob("*.py"):
        hashes[str(path.relative_to(main.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(source_hashes=hashes,environment=json.loads(json.dumps(main.environment())))


def write_json(path, value):
    path.write_text(json.dumps(main.clean(value),indent=2,allow_nan=False),encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    cases = {c["encounter_id"]:c for c in main.load_jsonl(output/"encounters.jsonl")}
    rows = main.load_jsonl(output/"results.jsonl")
    if not rows:
        return
    if len({(r["encounter_id"],r["method"]) for r in rows})!=len(rows):
        raise ValueError("Duplicate trials in results file")
    groups = defaultdict(list)
    for r in rows:
        c=cases[r["encounter_id"]]
        groups[(c["swarm_size"],c["formation_radius_m"],c["category"],r["method"])].append(r)
    summaries=[]
    for (n,radius,category,method),group in sorted(groups.items()):
        observed=[r for r in group if r["status"]=="ok"]
        safe=sum(r.get("safety_satisfied") is True for r in observed)
        classified=[r for r in observed if r.get("true_positives") is not None]
        positives=sum(r["threatened_uavs"] for r in classified)
        negatives=sum(r["nonthreatened_uavs"] for r in classified)
        example=next(c for c in cases.values() if c["swarm_size"]==n and c["formation_radius_m"]==radius)
        summary=dict(swarm_size=n,formation_radius_m=radius,category=category,method=method,
                     trials=len(group),completed=len(observed),timeouts=sum(r["status"]=="timeout" for r in group),
                     errors=sum(r["status"]=="error" for r in group),
                     safe_completed=safe,observed_safety_rate=safe/len(observed) if observed else None,
                     budgeted_safety_success_rate=safe/len(group),
                     projectile_collision_rate=(sum(r["projectile_collision"] for r in observed)/len(observed) if observed else None),
                     uav_collision_rate=(sum(r["uav_collision"] for r in observed)/len(observed) if observed else None),
                     hit_recall=sum(r["true_positives"] for r in classified)/positives if positives else None,
                     false_positive_rate=sum(r["false_positives"] for r in classified)/negatives if negatives else None,
                     initial_min_spacing_m=example["initial_min_spacing_m"],
                     initial_median_nearest_neighbor_m=example["initial_median_nearest_neighbor_m"],
                     shell_crowding_proxy_per_m2=example["shell_crowding_proxy_per_m2"])
        for field in ("path_length_m","number_moved","min_projectile_m","min_uav_m",
                      "initialization_runtime_s","planning_total_s","wall_runtime_s","process_elapsed_s",
                      "coordinated_plan_success","reactive_uavs_moved"):
            values=[r[field] for r in observed if r.get(field) is not None]
            summary[field+"_mean"]=float(np.mean(values)) if values else None
            summary[field+"_max"]=max(values) if values else None
        summaries.append(summary)
    write_csv(output/"summary.csv",summaries)
    metadata=("swarm_size","formation_radius_m","category","geometry_id","duration_s",
              "initial_min_spacing_m","initial_median_nearest_neighbor_m","static_threatened_uavs")
    write_csv(output/"metrics.csv",[{**{k:cases[r["encounter_id"]][k] for k in metadata},
                                    **{k:v for k,v in r.items() if k not in ("planning_events","per_uav_path_lengths_m","first_predictions")}}
                                   for r in rows])
    paired=defaultdict(dict)
    for r in rows:
        paired[r["encounter_id"]][r["method"]]=r
    comparisons=[]
    for encounter_id,pair in paired.items():
        c=cases[encounter_id]
        for method_a,method_b in (("estimate","dopplerdodge"),("estimate","dopplerdodge_coordinated"),
                                   ("dopplerdodge_coordinated","dopplerdodge")):
            if method_a not in pair or method_b not in pair:
                continue
            a,b=pair[method_a],pair[method_b]
            common_safe=a.get("safety_satisfied") is True and b.get("safety_satisfied") is True
            comparisons.append(dict(encounter_id=encounter_id,**{k:c[k] for k in metadata},
                                    method_a=method_a,method_b=method_b,status_a=a["status"],status_b=b["status"],
                                    safe_a=a.get("safety_satisfied"),safe_b=b.get("safety_satisfied"),both_safe=common_safe,
                                    path_change_a_minus_b_m=(a["path_length_m"]-b["path_length_m"] if common_safe else None)))
    write_csv(output/"paired_comparison.csv",comparisons)
    lines=["# Exploratory swarm size and spacing study","",
           f"Completed {len(rows)} / {manifest['planned_trials']} method trials.",
           "Three geometries per condition at defaults: screening evidence, not a precise reliability estimate.",
           "Timeout/error means unknown simulation safety, not an observed collision.",
           "Budgeted success requires a safe returned result before the process runtime cap.",
           "The cap is a computation-budget limit, not simulated time-to-collision.","",
           "| UAVs | RMS radius (m) | Class | Method | Trials | Returned | Safe/returned | Timeouts | Mean path (m) |",
           "|---:|---:|---|---|---:|---:|---|---:|---:|"]
    for s in summaries:
        path="N/A" if s["path_length_m_mean"] is None else f"{s['path_length_m_mean']:.3f}"
        lines.append(f"| {s['swarm_size']} | {s['formation_radius_m']:g} | {s['category']} | {s['method']} | "
                     f"{s['trials']} | {s['completed']} | {s['safe_completed']}/{s['completed']} | {s['timeouts']} | {path} |")
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    plot_results(output,summaries,manifest["config"])
    print(f"Saved report.md, summary.csv, metrics.csv, paired_comparison.csv and scaling.png in {output}",flush=True)


def plot_results(output, summaries, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes,radii=config["sizes"],config["radii"]
    controllers=[method for method in config["methods"] if method!="static"]
    fig,axes=plt.subplots(len(CATEGORIES),len(controllers),figsize=(5.5*len(controllers),10),squeeze=False)
    for i,category in enumerate(CATEGORIES):
        for j,method in enumerate(controllers):
            ax=axes[i,j]
            data=np.full((len(sizes),len(radii)),np.nan)
            for s in summaries:
                if s["category"]==category and s["method"]==method:
                    row,col=sizes.index(s["swarm_size"]),radii.index(s["formation_radius_m"])
                    data[row,col]=s["budgeted_safety_success_rate"]
                    ax.text(col,row,f"{s['safe_completed']}/{s['trials']} safe\n{s['timeouts']} timeout",ha="center",va="center",fontsize=8)
            im=ax.imshow(data,vmin=0,vmax=1,cmap="YlGn",aspect="auto")
            ax.set_xticks(range(len(radii)),labels=radii)
            ax.set_yticks(range(len(sizes)),labels=sizes)
            ax.set_xlabel("RMS radius (m)")
            ax.set_ylabel("UAV count")
            ax.set_title(f"{method}: {category.replace('_',' ')}")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Safe result returned within process runtime cap",fraction=.025,pad=.03)
    fig.subplots_adjust(wspace=.32,hspace=.55,right=.87)
    fig.savefig(output/"scaling.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


def main_cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,default=Path("results/fibonacci_02"),help="Study supplying the frozen Doppler Dodge threshold")
    parser.add_argument("--dodge-threshold",type=float)
    parser.add_argument("--swarm-sizes",type=int,nargs="+",default=SIZES)
    parser.add_argument("--radii",type=float,nargs="+",default=RADII)
    parser.add_argument("--repetitions",type=int,default=3)
    parser.add_argument("--speed",type=float,default=3.0)
    parser.add_argument("--warning",type=float,default=1.2)
    parser.add_argument("--seed",type=int,default=57721)
    parser.add_argument("--trial-timeout",type=float,default=60.0)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--reuse-results",type=Path,help="Import existing trials into a new study and run missing methods")
    parser.add_argument("--summarize",action="store_true")
    args=parser.parse_args()
    if args.resume and args.reuse_results:
        parser.error("--reuse-results creates a new study; use only --resume afterward")
    if args.summarize:
        summarize(args.output)
        return
    if args.repetitions<1 or args.seed<0 or any(n<3 for n in args.swarm_sizes):
        parser.error("repetitions must be positive, seed nonnegative, sizes at least 3")
    if len(set(args.swarm_sizes))!=len(args.swarm_sizes) or len(set(args.radii))!=len(args.radii):
        parser.error("sizes and radii must not contain duplicates")
    if any(not math.isfinite(x) or x<=0 for x in [*args.radii,args.speed,args.warning,args.trial_timeout]):
        parser.error("radii, speed, warning and trial timeout must be finite and positive")
    if args.dodge_threshold is not None and (not math.isfinite(args.dodge_threshold) or args.dodge_threshold<0):
        parser.error("threshold must be finite and nonnegative")
    config=dict(version=2,sizes=list(args.swarm_sizes),radii=list(args.radii),categories=list(CATEGORIES),
                methods=list(METHODS),repetitions=args.repetitions,speed_mps=args.speed,
                warning_s=args.warning,seed=args.seed,trial_timeout_s=args.trial_timeout,
                requested_threshold=args.dodge_threshold,added_noise=False)
    current=fingerprint()
    if args.resume:
        manifest=json.loads((args.output/"manifest.json").read_text(encoding="utf-8"))
        if config!=manifest["config"] or current!=manifest["fingerprint"]:
            parser.error("configuration, source or environment changed; use a new output directory")
        if hashlib.sha256((args.output/"encounters.jsonl").read_bytes()).hexdigest()!=manifest["encounters_sha256"]:
            parser.error("saved encounter data changed")
        cases=main.load_jsonl(args.output/"encounters.jsonl")
    else:
        if args.output.exists():
            parser.error("output exists; use --resume or a fresh directory")
        imported=[]
        reuse_provenance=None
        if args.reuse_results:
            parent=json.loads((args.reuse_results/"manifest.json").read_text(encoding="utf-8"))
            expected={k:v for k,v in config.items() if k not in ("version","methods","requested_threshold")}
            if any(parent["config"].get(k)!=v for k,v in expected.items()):
                parser.error("source protocol differs; use matching size/radius/repetition/runtime settings")
            for path,digest in parent["fingerprint"]["source_hashes"].items():
                normalized=path.replace("\\","/")
                if normalized in ("experiments/fibonacci_evaluation.py","experiments/scaling/runner.py"):
                    continue
                if normalized=="baselines/dopplerdodge_coordinated.py" and "dopplerdodge_coordinated" not in parent["config"]["methods"]:
                    continue
                if current["source_hashes"].get(path)!=digest:
                    parser.error(f"cannot reuse: implementation changed: {path}")
            if parent["fingerprint"]["environment"]!=current["environment"]:
                parser.error("cannot reuse results from a different environment")
            old_cases=args.reuse_results/"encounters.jsonl"
            if hashlib.sha256(old_cases.read_bytes()).hexdigest()!=parent["encounters_sha256"]:
                parser.error("source encounter data changed")
            cases=main.load_jsonl(old_cases)
            imported=[r.copy() for r in main.load_jsonl(args.reuse_results/"results.jsonl") if r["method"] in METHODS]
            valid_ids={c["encounter_id"] for c in cases}
            if any(r["encounter_id"] not in valid_ids for r in imported):
                parser.error("source results reference unknown encounters")
            if len({(r["encounter_id"],r["method"]) for r in imported})!=len(imported):
                parser.error("duplicate source trials")
            threshold=parent["dopplerdodge_threshold"]
            if args.dodge_threshold is not None and args.dodge_threshold!=threshold:
                parser.error("cannot reuse baseline results with a different frozen threshold")
            calibration_provenance=parent["calibration_provenance"]
            reuse_provenance=dict(directory=str(args.reuse_results.resolve()),manifest=parent,
                                  results_sha256=hashlib.sha256((args.reuse_results/"results.jsonl").read_bytes()).hexdigest())
            for row in imported:
                row["origin"]="imported"
        elif args.dodge_threshold is None:
            reference=json.loads((args.reference/"manifest.json").read_text(encoding="utf-8"))
            threshold=reference.get("dopplerdodge_threshold")
            if threshold is None:
                parser.error("reference has no frozen Doppler Dodge threshold; supply --dodge-threshold")
            baseline_path=str(Path("baselines")/"dopplerdodge.py")
            if reference["source_hashes"].get(baseline_path)!=current["source_hashes"].get(baseline_path):
                parser.error("reference baseline implementation changed; supply an independently calibrated threshold")
            calibration_provenance=dict(directory=str(args.reference.resolve()),manifest=reference)
        else:
            threshold=args.dodge_threshold
            calibration_provenance="User-supplied frozen threshold"
        if not args.reuse_results:
            try:
                cases=list(generate_encounters(args.swarm_sizes,args.radii,args.repetitions,args.seed,args.speed,args.warning))
            except ValueError as exc:
                parser.error(str(exc))
        args.output.mkdir(parents=True)
        (args.output/"encounters.jsonl").write_text("".join(json.dumps(main.clean(c),allow_nan=False)+"\n" for c in cases),encoding="utf-8")
        manifest=dict(config=config,fingerprint=current,dopplerdodge_threshold=threshold,
                      calibration_provenance=calibration_provenance,planned_trials=len(cases)*len(METHODS),
                      imported_results=reuse_provenance,
                      encounters_sha256=hashlib.sha256((args.output/"encounters.jsonl").read_bytes()).hexdigest())
        write_json(args.output/"manifest.json",manifest)
        if imported:
            (args.output/"results.jsonl").write_text("".join(json.dumps(main.clean(r),allow_nan=False)+"\n" for r in imported),encoding="utf-8")
        with zipfile.ZipFile(args.output/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in current["source_hashes"]:
                archive.write(main.ROOT/relative,arcname=relative.replace("\\","/"))
    previous=main.load_jsonl(args.output/"results.jsonl")
    done={(r["encounter_id"],r["method"]) for r in previous}
    if len(done)!=len(previous):
        parser.error("duplicate saved trials")
    print(f"{len(cases)} encounters; {manifest['planned_trials']} method trials; runtime cap {args.trial_timeout:g} s per controller",flush=True)
    try:
        with (args.output/"results.jsonl").open("a",encoding="utf-8") as file:
            for case in cases:
                costly=METHODS[1:]
                rotation=case["repetition"]%len(costly)
                order=("static",)+costly[rotation:]+costly[:rotation]
                action_seed=int(np.random.SeedSequence([args.seed,*hash_words(case["geometry_id"])]).generate_state(1)[0])
                for method in order:
                    if (case["encounter_id"],method) in done:
                        continue
                    print(f"Starting {case['encounter_id']} {method}",flush=True)
                    result=(main.static_result(case) if method=="static" else
                            run_bounded(case,method,manifest["dopplerdodge_threshold"],action_seed,args.trial_timeout))
                    row=main.clean(dict(encounter_id=case["encounter_id"],method=method,**result))
                    file.write(json.dumps(row,allow_nan=False)+"\n")
                    file.flush()
                    done.add((case["encounter_id"],method))
                    print(f"[{len(done)}/{manifest['planned_trials']}] status={row['status']} safe={row.get('safety_satisfied')}",flush=True)
    finally:
        summarize(args.output)
