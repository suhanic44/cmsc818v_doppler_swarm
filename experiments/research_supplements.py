"""Write a proposed supplementary design and trial budget; runs no simulations."""

import argparse
import json
from pathlib import Path


FACTORS = {
    "crowding": ("median_nn_m", [0.4, 0.6, 0.9]),
    "geometry": ("family", ["planar_disk", "volumetric_ball", "spherical_shell"]),
    "audio_noise": ("snr_db", [None, 30, 20, 10]),
    "warning": ("warning_s", [0.6, 0.9, 1.2, 1.8]),
    "execution_delay": ("delay_s", [0, 0.05, 0.10, 0.20]),
    "acceleration": ("transverse_acceleration_mps2", [0, 0.5, 1.0]),
    "dropout": ("observation_loss_probability", [0, 0.1, 0.3]),
    "localization": ("coordinate_sd_m", [0, 0.02, 0.05, 0.10]),
}


def build_plan(studies, sizes, speeds, repetitions):
    per_level = len(sizes) * len(speeds) * 4 * repetitions
    sweeps = []
    for name in studies:
        factor, levels = FACTORS[name]
        encounters = per_level * len(levels)
        sweeps.append(dict(study=name, factor=factor, levels=levels,
                           encounters=encounters, method_trials=encounters * 5,
                           controller_processes=encounters * 4,
                           runner_ready=False))
    processes = sum(s["controller_processes"] for s in sweeps)
    return dict(status="draft_design_only", protocol="SUPPLEMENTARY_RESEARCH_PROTOCOL.md",
                sizes=sizes, speeds_mps=speeds, categories=4, methods=5,
                repetitions=repetitions, proposed_evaluation_seed=910001,
                threshold_policy="inherit frozen nominal main-study threshold",
                clean_audio_snr_representation="null means no added waveform noise",
                sweeps=sweeps, total_method_trials=sum(s["method_trials"] for s in sweeps),
                total_controller_processes=processes,
                controller_cap_hours_at_60s=processes / 60,
                budget_note="Counts controls separately; excludes calibration, preparation and reporting.")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--studies", choices=tuple(FACTORS), nargs="+", default=list(FACTORS))
    parser.add_argument("--swarm-sizes", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--speeds", type=float, nargs="+", default=[3.0])
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    import math
    if args.repetitions < 1 or any(n < 4 for n in args.swarm_sizes):
        parser.error("repetitions must be positive and swarm sizes at least 4")
    if any(not math.isfinite(v) or v <= 0 for v in args.speeds):
        parser.error("speeds must be finite and positive")
    if any(len(values) != len(set(values)) for values in
           (args.studies, args.swarm_sizes, args.speeds)):
        parser.error("studies, swarm sizes and speeds must each be distinct")
    plan = build_plan(args.studies, args.swarm_sizes, args.speeds, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(plan, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Draft only: {plan['total_method_trials']:,} method trials; "
          f"{plan['controller_cap_hours_at_60s']:g} controller cap-hours. "
          f"Saved {args.output}")


if __name__ == "__main__":
    main_cli()
