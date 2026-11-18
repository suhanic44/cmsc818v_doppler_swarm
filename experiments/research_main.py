"""Spacing-controlled main research evaluation. See MAIN_RESEARCH_PROTOCOL.md.

python -m experiments.research_main --output results/research_main_01
Append --resume to continue or --summarize to rebuild tables and figures.
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

from experiments import evaluation_common as shared


SIZES = (4, 6, 8, 10, 16, 25)
SPEEDS = (1.0, 3.0, 5.0)
METHODS = shared.METHODS
CATEGORIES = shared.CLASSES


def words(label):
    return list(np.frombuffer(hashlib.sha256(label.encode()).digest()[:16], dtype="<u4"))


def formation(rng, count, spacing, minimum):
    """Volumetric sequential-inhibition samples, with controlled median NN.

    Uniform-ball proposals are accepted subject to an exclusion distance. The
    resulting point process is not independent uniform sampling or a shell.
    """
    exclusion = .7 / count**(1/3)
    for _ in range(100):
        accepted = []
        for _ in range(10000):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            candidate = direction * rng.random()**(1/3)
            if not accepted or np.min(np.linalg.norm(np.asarray(accepted)-candidate,axis=1)) >= exclusion:
                accepted.append(candidate)
                if len(accepted)==count:
                    break
        if len(accepted)!=count:
            continue
        positions=np.asarray(accepted)
        positions -= positions.mean(axis=0)
        distances=np.linalg.norm(positions[:,None]-positions[None,:],axis=2)
        np.fill_diagonal(distances,np.inf)
        positions *= spacing/np.median(distances.min(axis=1))
        if spacing_statistics(positions)["initial_min_spacing_m"] >= minimum:
            return positions
    raise ValueError("Unable to generate requested spacing; choose a smaller minimum/median ratio")


def spacing_statistics(positions):
    distances=np.linalg.norm(positions[:,None]-positions[None,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    return dict(initial_min_spacing_m=float(distances.min()),
                initial_median_nearest_neighbor_m=float(np.median(distances.min(axis=1))),
                formation_rms_radius_m=float(np.sqrt(np.mean(np.sum(positions**2,axis=1)))),
                formation_max_radius_m=float(np.max(np.linalg.norm(positions,axis=1))))


def encounters(config, seed=None, repetitions=None):
    seed=config["seed"] if seed is None else seed
    repetitions=config["repetitions"] if repetitions is None else repetitions
    for count in config["sizes"]:
        for category_index,category in enumerate(CATEGORIES):
            for repetition in range(repetitions):
                rng=np.random.default_rng(np.random.SeedSequence([seed,count,category_index,repetition]))
                positions=formation(rng,count,config["spacing_m"],config["min_spacing_m"])
                stats=spacing_statistics(positions)
                for _ in range(10000):
                    direction=rng.normal(size=3)
                    direction/=np.linalg.norm(direction)
                    first=int(np.argmin(positions@direction))
                    target=positions[first].copy()
                    if category=="multiple_threats":
                        # Draw an opposite-side UAV, then verify the entire ray.
                        candidates=np.argsort(positions@direction)[-max(1,count//3):]
                        second=int(rng.choice(candidates[candidates!=first]))
                        direction=positions[second]-target
                        direction/=np.linalg.norm(direction)
                    perpendicular=rng.normal(size=3)
                    perpendicular-=direction*np.dot(perpendicular,direction)
                    perpendicular/=np.linalg.norm(perpendicular)
                    offset=(0.0 if category=="multiple_threats" else rng.uniform(*{
                        "direct_threat":(0,.05),"grazing_threat":(.075,.095),"safe_miss":(.15,.4)}[category]))
                    anchor=target+offset*perpendicular
                    relative=positions-anchor
                    projections=relative@direction
                    line_distances=np.linalg.norm(relative-projections[:,None]*direction,axis=1)
                    if shared.classify(line_distances)!=category:
                        continue
                    threats=line_distances<shared.PROJECTILE_CLEARANCE
                    if np.any(threats):
                        event=float(np.min(projections[threats]-np.sqrt(
                            shared.PROJECTILE_CLEARANCE**2-line_distances[threats]**2)))
                    else:
                        event=0.0  # Nominal passage of the selected front-side target for misses.
                    cases=[]
                    for speed in config["speeds"]:
                        start=event-speed*config["warning_s"]
                        if start >= float(projections.min())-shared.PROJECTILE_CLEARANCE:
                            break  # Begin outside the formation's incoming support plane.
                        end=float(projections.max())+.45+shared.PROJECTILE_CLEARANCE+.2*speed
                        state=np.r_[anchor+start*direction,speed*direction]
                        duration=(end-start)/speed
                        distances,uav_distance,entry=shared.stationary_geometry(state,positions,duration)
                        if shared.classify(distances)!=category:
                            break
                        cases.append(dict(encounter_id=f"s{seed}_n{count}_{category}_{repetition:03d}_v{speed:g}",
                                          geometry_id=f"s{seed}_n{count}_{category}_{repetition:03d}",
                                          swarm_size=count,speed_mps=speed,category=category,repetition=repetition,
                                          duration_s=duration,nominal_warning_s=config["warning_s"],
                                          first_safety_entry_s=entry,state=state.tolist(),uav_positions=positions.tolist(),
                                          static_min_projectile_m=float(distances.min()),static_min_uav_m=uav_distance,
                                          static_threatened_uavs=int(threats.sum()),**stats))
                    if len(cases)==len(config["speeds"]):
                        yield from cases
                        break
                else:
                    raise ValueError(f"Unable to construct {count}/{category}/{repetition}; no partial study written")


def calibrate(config):
    """Independent, stratified training bank. Never optimize against evaluation outcomes."""
    from baselines import dopplerdodge as detector
    from simulation.acoustic_sensor import SAMPLE_RATE
    from experiments.online_avoidance import INIT_WINDOW
    training=list(encounters(config,config["calibration_seed"],config["calibration_repetitions"]))
    audio_times=np.arange(0,INIT_WINDOW,1/SAMPLE_RATE)
    physics_times=np.arange(0,INIT_WINDOW+1e-9,1/240)
    scores,labels,records=[],[],[]
    for case in training:
        state=np.asarray(case["state"])
        positions=np.asarray(case["uav_positions"])
        distances,_,_=shared.stationary_geometry(state,positions,case["duration_s"])
        path=state[:3]+physics_times[:,None]*state[3:]
        for index,position in enumerate(positions):
            echo=detector.generate_dodge_echo(audio_times,physics_times,path,position)
            try:
                score=detector.analyze_echo(echo).score
            except ValueError as exc:
                if "No reflected signal" not in str(exc) and "three consecutive active" not in str(exc):
                    raise
                score=float("inf")  # Native streaming detector has no HIT evidence.
            truth=bool(distances[index]<shared.PROJECTILE_CLEARANCE)
            scores.append(score)
            labels.append(truth)
            records.append(dict(encounter_id=case["encounter_id"],uav_index=index,score=score,threatened=truth))
    scores=np.asarray(scores)
    labels=np.asarray(labels,dtype=bool)
    threshold=detector.select_hit_threshold(scores,labels)
    hits=scores<=threshold
    recall=float(hits[labels].mean())
    specificity=float((~hits[~labels]).mean())
    return threshold,training,records,dict(policy="minimum_threshold_covering_all_training_hits",
                                       carrier_frequency_hz=detector.CARRIER_FREQUENCY,
                                       startup_guard_buffers=1,
                                       balanced_accuracy=(recall+specificity)/2,hit_recall=recall,
                                       false_positive_rate=float((scores[~labels]<=threshold).mean()),
                                       samples=len(scores),threatened_samples=int(labels.sum()),
                                       nonthreatened_samples=int((~labels).sum()),
                                       no_signal_samples=int(np.sum(~np.isfinite(scores))))


def worker(connection,case,method,threshold,seed):
    try:
        from experiments import online_avoidance
        from baselines import dopplerdodge,dopplerdodge_coordinated
        if method in ("estimate","oracle"):
            row=shared.run_online(case,method,online_avoidance)
        elif method=="dopplerdodge":
            row=shared.run_dopplerdodge(case,threshold,seed,dopplerdodge)
        elif method=="dopplerdodge_coordinated":
            row=shared.run_dopplerdodge(case,threshold,seed,dopplerdodge_coordinated,coordinated=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        connection.send(shared.clean(row))
    except Exception as exc:
        connection.send(dict(status="error",safety_satisfied=None,collision_free=None,error=f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def run_bounded(case,method,threshold,seed,timeout):
    context=mp.get_context("spawn")
    receive,send=context.Pipe(duplex=False)
    process=context.Process(target=worker,args=(send,case,method,threshold,seed))
    start=time.perf_counter()
    process.start()
    send.close()
    try:
        while True:
            remaining=timeout-(time.perf_counter()-start)
            if remaining<=0:
                return dict(status="timeout",safety_satisfied=None,collision_free=None,
                            error=f"Process runtime exceeded {timeout:g} s",process_elapsed_s=time.perf_counter()-start)
            if receive.poll(min(.25,remaining)):
                try:
                    row=receive.recv()
                except EOFError:
                    break
                row["process_elapsed_s"]=time.perf_counter()-start
                return row
            if not process.is_alive():
                break
        return dict(status="error",safety_satisfied=None,collision_free=None,
                    error=f"Process exited without result: {process.exitcode}",process_elapsed_s=time.perf_counter()-start)
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
        process.join()


def write_json(path,value):
    path.write_text(json.dumps(shared.clean(value),indent=2,allow_nan=False),encoding="utf-8")


def write_jsonl(path,rows):
    path.write_text("".join(json.dumps(shared.clean(r),allow_nan=False)+"\n" for r in rows),encoding="utf-8")


def write_csv(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fingerprint():
    return dict(source_hashes=shared.source_hashes(),environment=json.loads(json.dumps(shared.environment())))


def prepare_dodge_refresh(source, output, current):
    """Fork saved encounters and unaffected rows, with explicit per-method provenance."""
    if output.exists():
        raise ValueError("output exists; use a new directory (or --resume after starting the refresh)")
    parent=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in parent["input_hashes"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"source input changed: {name}")
    allowed={"experiments/research_main.py","baselines/dopplerdodge.py",
             "baselines/dopplerdodge_coordinated.py","simulation/acoustic_sensor.py"}
    before={k.replace("\\","/"):v for k,v in parent["fingerprint"]["source_hashes"].items()}
    after={k.replace("\\","/"):v for k,v in current["source_hashes"].items()}
    unrelated=[k for k in before.keys()|after.keys() if before.get(k)!=after.get(k) and k not in allowed]
    if unrelated:
        raise ValueError(f"unrelated source changed; cannot refresh only Dodge: {sorted(unrelated)}")
    cases=shared.load_jsonl(source/"encounters.jsonl")
    if not cases or len({c["encounter_id"] for c in cases})!=len(cases):
        raise ValueError("source must contain a nonempty unique encounter bank")
    rows=shared.load_jsonl(source/"results.jsonl")
    case_ids={c["encounter_id"] for c in cases}
    keys=[(r["encounter_id"],r["method"]) for r in rows]
    if len(keys)!=len(set(keys)) or any(c not in case_ids or m not in METHODS for c,m in keys):
        raise ValueError("source contains duplicate or unknown trials")
    rerun={"dopplerdodge","dopplerdodge_coordinated"}
    retained=[r for r in rows if r["method"] not in rerun]
    expected={(c,m) for c in case_ids for m in METHODS if m not in rerun}
    if {(r["encounter_id"],r["method"]) for r in retained}!=expected:
        raise ValueError("source must contain all static/oracle/estimate rows, including recorded timeouts/errors")
    config=dict(parent["config"])
    config["requested_threshold"]=None  # Never inherit a threshold from the old detector.
    print("Recalibrating updated Dodge detectors on the independent training bank...",flush=True)
    threshold,training,scores,calibration=calibrate(config)
    output.mkdir(parents=True)
    # Preserve the exact evaluation bank bytes; copied result dictionaries are unmodified.
    (output/"encounters.jsonl").write_bytes((source/"encounters.jsonl").read_bytes())
    write_jsonl(output/"calibration_encounters.jsonl",training)
    write_csv(output/"calibration_scores.csv",[shared.clean(r) for r in scores])
    write_jsonl(output/"results.jsonl",retained)
    write_json(output/"inherited_manifest.json",parent)
    if (source/"source.zip").exists():
        (output/"inherited_source.zip").write_bytes((source/"source.zip").read_bytes())
    origins=parent.get("method_provenance",{m:parent["fingerprint"] for m in METHODS})
    manifest=dict(config=config,fingerprint=current,dopplerdodge_threshold=threshold,calibration=calibration,
                  planned_trials=len(cases)*len(METHODS),
                  reuse_results=str(source.resolve()),refreshed_methods=sorted(rerun),
                  inherited_results_sha256=hashlib.sha256((source/"results.jsonl").read_bytes()).hexdigest(),
                  method_provenance={m:current if m in rerun else origins[m] for m in METHODS},
                  input_hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                for name in ("encounters.jsonl","calibration_encounters.jsonl")})
    write_json(output/"manifest.json",manifest)
    with zipfile.ZipFile(output/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in current["source_hashes"]:
            archive.write(shared.ROOT/relative,arcname=relative.replace("\\","/"))
    return manifest,cases


def summarize(output):
    manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    cases={c["encounter_id"]:c for c in shared.load_jsonl(output/"encounters.jsonl")}
    rows=shared.load_jsonl(output/"results.jsonl")
    if not rows:
        print("No completed trials to summarize.")
        return
    if len({(r["encounter_id"],r["method"]) for r in rows})!=len(rows):
        raise ValueError("Duplicate saved trials")
    groups=defaultdict(list)
    for r in rows:
        case=cases[r["encounter_id"]]
        groups[(case["swarm_size"],case["speed_mps"],case["category"],r["method"])].append(r)
    summaries=[]
    for (n,speed,category,method),group in sorted(groups.items()):
        observed=[r for r in group if r["status"]=="ok"]
        safe=sum(r.get("safety_satisfied") is True for r in observed)
        free=sum(r.get("collision_free") is True for r in observed)
        low,high=shared.wilson(safe,len(group))
        classified=[r for r in observed if r.get("true_positives") is not None]
        positives=sum(r["threatened_uavs"] for r in classified)
        negatives=sum(r["nonthreatened_uavs"] for r in classified)
        summary=dict(swarm_size=n,speed_mps=speed,category=category,method=method,trials=len(group),
                     planned_trials=manifest["config"]["repetitions"],completed=len(observed),
                     timeouts=sum(r["status"]=="timeout" for r in group),errors=sum(r["status"]=="error" for r in group),
                     budgeted_safety_success_rate=safe/len(group),budgeted_safety_ci_low=low,budgeted_safety_ci_high=high,
                     observed_safety_rate=safe/len(observed) if observed else None,collision_free_success_rate=free/len(group),
                     projectile_collision_rate=sum(r["projectile_collision"] for r in observed)/len(observed) if observed else None,
                     uav_collision_rate=sum(r["uav_collision"] for r in observed)/len(observed) if observed else None,
                     classification_trials=len(classified),hit_recall=sum(r["true_positives"] for r in classified)/positives if positives else None,
                     false_positive_rate=sum(r["false_positives"] for r in classified)/negatives if negatives else None)
        fields=("path_length_m","path_length_per_uav_m","number_moved","fraction_moved","min_projectile_m","min_uav_m",
                "position_rmse_m","velocity_rmse_mps","initialization_runtime_s","detection_latency_s",
                "initialization_success","first_estimate_sim_s","avoidance_response_time_s","first_command_sim_s","planning_total_s","planning_p95_s",
                "estimation_total_s","process_elapsed_s","partial_timing_screen_met","coordinated_plan_success","reactive_uavs_moved")
        for field in fields:
            # Boolean indicators are numeric 0/1 outcomes for aggregation.
            # NumPy's interpolated percentile cannot subtract boolean values.
            values=[float(r[field]) for r in observed if r.get(field) is not None]
            summary[field+"_observations"]=len(values)
            summary[field+"_mean"]=float(np.mean(values)) if values else None
            summary[field+"_median"]=float(np.median(values)) if values else None
            summary[field+"_p95"]=float(np.percentile(values,95)) if values else None
        summaries.append(summary)
    write_csv(output/"summary.csv",summaries)
    metadata=("swarm_size","speed_mps","category","geometry_id","duration_s","first_safety_entry_s",
              "static_threatened_uavs","initial_min_spacing_m","initial_median_nearest_neighbor_m",
              "formation_rms_radius_m","formation_max_radius_m")
    write_csv(output/"metrics.csv",[{**{k:cases[r["encounter_id"]][k] for k in metadata},
                                    **{k:v for k,v in r.items() if k not in ("planning_events","first_predictions","per_uav_path_lengths_m")}}
                                   for r in rows])
    paired=defaultdict(dict)
    for r in rows:
        paired[r["encounter_id"]][r["method"]]=r
    comparisons=[]
    for encounter_id,pair in paired.items():
        for a,b in (("estimate","oracle"),("estimate","dopplerdodge"),
                    ("estimate","dopplerdodge_coordinated"),("dopplerdodge_coordinated","dopplerdodge")):
            if a not in pair or b not in pair:
                continue
            first,second=pair[a],pair[b]
            both_safe=first.get("safety_satisfied") is True and second.get("safety_satisfied") is True
            comparisons.append(dict(encounter_id=encounter_id,**{k:cases[encounter_id][k] for k in metadata},
                                    method_a=a,method_b=b,status_a=first["status"],status_b=second["status"],
                                    safe_a=first.get("safety_satisfied"),safe_b=second.get("safety_satisfied"),both_safe=both_safe,
                                    common_safe_path_change_a_minus_b_m=(first["path_length_m"]-second["path_length_m"] if both_safe else None)))
    write_csv(output/"paired_comparison.csv",comparisons)
    lines=["# Spacing-controlled main research evaluation","",f"Saved {len(rows)} / {manifest['planned_trials']} method trials.",
           "Timeouts/errors are unknown physical outcomes, not observed collisions.",
           "Budgeted success counts safe returned results among all attempts; observed safety uses completed runs only.",
           "The process runtime cap is not the projectile deadline. Compute delays are not applied to motion.",
           "Movement comparisons condition on common safe outcomes; oracle is a privileged reference, not an optimum.","",
           "| UAVs | Speed | Category | Method | Attempts | Returned | Timeouts | Budgeted safe | Observed safe | Mean path (m) |",
           "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        rate="N/A" if s["observed_safety_rate"] is None else f"{s['observed_safety_rate']:.1%}"
        path="N/A" if s["path_length_m_mean"] is None else f"{s['path_length_m_mean']:.3f}"
        lines.append(f"| {s['swarm_size']} | {s['speed_mps']:g} | {s['category']} | {s['method']} | {s['trials']} | "
                     f"{s['completed']} | {s['timeouts']} | {s['budgeted_safety_success_rate']:.1%} | {rate} | {path} |")
    if manifest.get("reuse_results"):
        lines[2:2]=[f"Static/oracle/estimate rows inherited from {manifest['reuse_results']}; only both Dodge variants refreshed.",
                    "This report combines method versions; see method_provenance and inherited_manifest.json.",""]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    plot_results(output,summaries,manifest["config"])
    print(f"Saved report, summary/metrics/paired CSVs and safety_vs_speed.png in {output}",flush=True)


def plot_results(output,summaries,config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(CATEGORIES),len(config["sizes"]),figsize=(3.6*len(config["sizes"]),12),squeeze=False)
    for i,category in enumerate(CATEGORIES):
        for j,n in enumerate(config["sizes"]):
            ax=axes[i,j]
            for method in METHODS:
                series=sorted([s for s in summaries if s["category"]==category and s["swarm_size"]==n and s["method"]==method],key=lambda s:s["speed_mps"])
                x=[s["speed_mps"] for s in series]
                y=[s["budgeted_safety_success_rate"] for s in series]
                ax.plot(x,y,marker="o",label=method)
            ax.set(title=f"N={n}: {category.replace('_',' ')}",xlabel="Projectile speed (m/s)",ylabel="Safe returned / attempted",ylim=(-.02,1.02))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Safety within process runtime budget; inspect timeout counts separately")
    fig.tight_layout()
    fig.savefig(output/"safety_vs_speed.png",dpi=150)
    plt.close(fig)


def main_cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--swarm-sizes",type=int,nargs="+",default=SIZES)
    parser.add_argument("--speeds",type=float,nargs="+",default=SPEEDS)
    parser.add_argument("--repetitions",type=int,default=50)
    parser.add_argument("--spacing",type=float,default=.6,help="Median nearest-neighbor distance in metres")
    parser.add_argument("--min-spacing",type=float,default=.3)
    parser.add_argument("--warning",type=float,default=1.2)
    parser.add_argument("--trial-timeout",type=float,default=60)
    parser.add_argument("--seed",type=int,default=8675309)
    parser.add_argument("--calibration-seed",type=int,default=2026)
    parser.add_argument("--calibration-repetitions",type=int,default=2)
    parser.add_argument("--dodge-threshold",type=float)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--reuse-results",type=Path,
                        help="Inherit research_main encounters/settings and static/oracle/estimate rows; recalibrate and rerun both Dodge variants in a new output")
    parser.add_argument("--summarize",action="store_true")
    args=parser.parse_args()
    if args.summarize:
        summarize(args.output)
        return
    if args.seed<0 or args.calibration_seed<0 or args.seed==args.calibration_seed or args.repetitions<1 or args.calibration_repetitions<1:
        parser.error("seeds must be distinct and nonnegative; repetition counts must be positive")
    if any(n<4 for n in args.swarm_sizes) or len(set(args.swarm_sizes))!=len(args.swarm_sizes):
        parser.error("sizes must be distinct integers at least 4")
    if len(set(args.speeds))!=len(args.speeds) or any(not math.isfinite(x) or x<=0 for x in [*args.speeds,args.spacing,args.min_spacing,args.warning,args.trial_timeout]):
        parser.error("speeds must be distinct; speeds/spacing/warning/runtime must be finite and positive")
    if args.min_spacing<shared.UAV_CLEARANCE or args.min_spacing>args.spacing:
        parser.error("minimum spacing must be at least UAV safety clearance and no greater than median spacing")
    if args.dodge_threshold is not None and (not math.isfinite(args.dodge_threshold) or args.dodge_threshold<0):
        parser.error("threshold must be finite and nonnegative")
    config=dict(version=1,sizes=list(args.swarm_sizes),speeds=list(args.speeds),methods=list(METHODS),
                categories=list(CATEGORIES),repetitions=args.repetitions,spacing_m=args.spacing,min_spacing_m=args.min_spacing,
                warning_s=args.warning,trial_timeout_s=args.trial_timeout,seed=args.seed,calibration_seed=args.calibration_seed,
                calibration_repetitions=args.calibration_repetitions,requested_threshold=args.dodge_threshold,added_noise=False)
    current=fingerprint()
    if args.reuse_results:
        if args.resume or args.dodge_threshold is not None:
            parser.error("--reuse-results requires a new output and fresh calibration; do not combine with --resume/--dodge-threshold")
        try:
            manifest,cases=prepare_dodge_refresh(args.reuse_results,args.output,current)
        except ValueError as exc:
            parser.error(str(exc))
        config=manifest["config"]
    elif args.resume:
        manifest=json.loads((args.output/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("reuse_results"):
            config=manifest["config"]  # A refresh inherits all encounter/runtime settings.
        if (manifest["config"]!=config
                or manifest["fingerprint"]["environment"]!=current["environment"]
                or manifest["fingerprint"]["source_hashes"]!=current["source_hashes"]):
            parser.error("source/configuration/environment changed; use original settings or a new output directory")
        for name,digest in manifest["input_hashes"].items():
            if hashlib.sha256((args.output/name).read_bytes()).hexdigest()!=digest:
                parser.error(f"input changed: {name}")
        cases=shared.load_jsonl(args.output/"encounters.jsonl")
    else:
        if args.output.exists():
            parser.error("output exists; use --resume or a new directory")
        print("Generating and class-checking complete evaluation encounter bank...",flush=True)
        cases=list(encounters(config))
        if args.dodge_threshold is None:
            print("Calibrating both Dodge detectors on a separate randomized training bank...",flush=True)
            threshold,training,scores,calibration=calibrate(config)
        else:
            threshold,training,scores,calibration=args.dodge_threshold,[],[],"User-supplied frozen threshold"
        args.output.mkdir(parents=True)
        write_jsonl(args.output/"encounters.jsonl",cases)
        write_jsonl(args.output/"calibration_encounters.jsonl",training)
        write_csv(args.output/"calibration_scores.csv",[shared.clean(r) for r in scores])
        manifest=dict(config=config,fingerprint=current,dopplerdodge_threshold=threshold,calibration=calibration,
                      planned_trials=len(cases)*len(METHODS),
                      input_hashes={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest()
                                    for name in ("encounters.jsonl","calibration_encounters.jsonl")})
        write_json(args.output/"manifest.json",manifest)
        with zipfile.ZipFile(args.output/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in current["source_hashes"]:
                archive.write(shared.ROOT/relative,arcname=relative.replace("\\","/"))
    previous=shared.load_jsonl(args.output/"results.jsonl")
    done={(r["encounter_id"],r["method"]) for r in previous}
    if len(done)!=len(previous):
        parser.error("duplicate saved trials")
    print(f"{len(cases)} encounters; {manifest['planned_trials']} method trials, {len(done)} saved; "
          f"Dodge threshold={manifest['dopplerdodge_threshold']:.6f}",flush=True)
    try:
        with (args.output/"results.jsonl").open("a",encoding="utf-8") as file:
            for case in cases:
                costly=METHODS[1:]
                rotation=case["repetition"]%len(costly)
                order=("static",)+costly[rotation:]+costly[:rotation]
                seed=int(np.random.SeedSequence([config["seed"],*words(case["encounter_id"])]).generate_state(1)[0])
                for method in order:
                    if (case["encounter_id"],method) in done:
                        continue
                    print(f"Starting {case['encounter_id']} {method}",flush=True)
                    result=(shared.static_result(case) if method=="static" else
                            run_bounded(case,method,manifest["dopplerdodge_threshold"],seed,config["trial_timeout_s"]))
                    row=shared.clean(dict(encounter_id=case["encounter_id"],method=method,action_seed=seed,**result))
                    if row.get("path_length_m") is not None:
                        row["path_length_per_uav_m"]=row["path_length_m"]/case["swarm_size"]
                    if row.get("number_moved") is not None:
                        row["fraction_moved"]=row["number_moved"]/case["swarm_size"]
                    file.write(json.dumps(row,allow_nan=False)+"\n")
                    file.flush()
                    done.add((case["encounter_id"],method))
                    print(f"[{len(done)}/{manifest['planned_trials']}] {method} status={row['status']} safe={row.get('safety_satisfied')}",flush=True)
    finally:
        summarize(args.output)


if __name__=="__main__":
    main_cli()
