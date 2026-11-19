
"""Reproduce coordinated DopplerDodge failures without modifying evaluation data."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

DEFAULT_IDS = [
    "s8675309_n6_direct_threat_016_v1",
    "s8675309_n6_direct_threat_016_v3",
    "s8675309_n6_direct_threat_016_v5",
]
MARKER = "DEBUG_RESULT_JSON="


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_case(results_dir, encounter_id):
    cases = {
        c["encounter_id"]: c
        for c in load_jsonl(results_dir / "encounters.jsonl")
    }
    rows = {
        r["encounter_id"]: r
        for r in load_jsonl(results_dir / "results.jsonl")
        if r["method"] == "dopplerdodge_coordinated"
    }
    if encounter_id not in cases or encounter_id not in rows:
        raise ValueError(
            f"Encounter or saved result missing: {encounter_id}"
        )
    return cases[encounter_id], rows[encounter_id]


def child(args):
    from experiments import evaluation_common as shared
    from experiments import research_main

    case, saved = get_case(args.results, args.encounter_id)
    manifest = json.loads(
        (args.results / "manifest.json").read_text(encoding="utf-8")
    )
    threshold = float(manifest["dopplerdodge_threshold"])
    seed = int(saved["action_seed"])

    print(
        f"Mode={args.child} ID={args.encounter_id} "
        f"seed={seed} threshold={threshold}",
        flush=True,
    )

    try:
        if args.child == "worker":
            outcome = research_main.run_bounded(
                case, "dopplerdodge_coordinated",
                threshold, seed, args.timeout
            )
        else:
            from baselines import dopplerdodge_coordinated
            outcome = shared.run_dopplerdodge(
                case, threshold, seed,
                dopplerdodge_coordinated, coordinated=True
            )

        summary = {
            k: outcome.get(k)
            for k in (
                "status", "error", "safety_satisfied",
                "path_length_m", "planning_total_s",
                "process_elapsed_s"
            )
        }
        print(
            MARKER + json.dumps(shared.clean(summary)),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)


def run_one(args, encounter_id, mode, output):
    cmd = [
        sys.executable, "-u", "-m",
        "experiments.debug_six_uav_errors",
        "--results", str(args.results.resolve()),
        "--child", mode,
        "--encounter-id", encounter_id,
        "--timeout", str(args.timeout),
    ]

    try:
        run = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.timeout + 20,
        )
        stdout, stderr, code = (
            run.stdout, run.stderr, run.returncode
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr += "\nDiagnostic subprocess exceeded outer timeout.\n"
        code = "outer-timeout"

    name = f"{encounter_id}_{mode}"
    (output / f"{name}.stdout.txt").write_text(
        stdout, encoding="utf-8"
    )
    (output / f"{name}.stderr.txt").write_text(
        stderr, encoding="utf-8"
    )

    result = None
    for line in stdout.splitlines():
        if line.startswith(MARKER):
            try:
                result = json.loads(line[len(MARKER):])
            except json.JSONDecodeError:
                pass

    status = result.get("status") if result else "no result"
    error = result.get("error") if result else "see stderr"
    print(
        f"  {mode}: exit={code}, status={status}, error={error}",
        flush=True,
    )
    return code, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=Path("results/research_main_01"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/research_main_01/six_uav_debug"),
    )
    parser.add_argument(
        "--ids", nargs="+", default=DEFAULT_IDS
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0
    )
    parser.add_argument(
        "--child", choices=("worker", "direct"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--encounter-id", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.child:
        if not args.encounter_id:
            parser.error(
                "--encounter-id is required with --child"
            )
        child(args)
        return

    args.output.mkdir(parents=True, exist_ok=True)

    report = [
        "# Six-UAV coordinated DopplerDodge reproduction",
        "",
        "Uses original saved encounters, action seeds, "
        "and calibrated threshold.",
        "Original evaluation files are not modified.",
        "",
        "| Encounter | Original status | "
        "Worker exit/status | Direct exit/status |",
        "|---|---|---|---|",
    ]

    for encounter_id in args.ids:
        _, saved = get_case(args.results, encounter_id)
        print(
            f"Checking {encounter_id} "
            f"(original: {saved['status']})",
            flush=True,
        )

        w_code, w_result = run_one(
            args, encounter_id, "worker", args.output
        )
        d_code, d_result = run_one(
            args, encounter_id, "direct", args.output
        )

        w_status = (
            w_result.get("status", "no result")
            if w_result else "no result"
        )
        d_status = (
            d_result.get("status", "no result")
            if d_result else "no result"
        )

        report.append(
            f"| `{encounter_id}` | {saved['status']} | "
            f"{w_code} / {w_status} | "
            f"{d_code} / {d_status} |"
        )

    report += [
        "",
        "Inspect .stderr.txt for tracebacks "
        "and .stdout.txt for worker results.",
        "A successful rerun does not explain the original exit; "
        "the failure may be intermittent.",
    ]

    target = args.output / "report.md"
    target.write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"\nSaved report: {target}")


if __name__ == "__main__":
    main()
