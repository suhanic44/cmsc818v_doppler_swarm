
"""Read-only diagnostics for the CMSC818V main evaluation.

Uses metrics.csv written by experiments.research_main.
Does not rerun simulations or modify existing results.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def number(row, key):
    value = row.get(key, "")
    if value is None or str(value).strip().lower() in (
        "", "none", "null", "nan"
    ):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def flag(row, key):
    value = str(row.get(key, "")).strip().lower()
    if value in ("true", "1", "1.0"):
        return True
    if value in ("false", "0", "0.0"):
        return False
    return None


def median(rows, key):
    values = [number(row, key) for row in rows]
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def percentile(rows, key, p=0.95):
    values = sorted(
        v for row in rows
        if (v := number(row, key)) is not None
    )
    if not values:
        return None
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def fraction(count, total):
    if not total:
        return "0/0 (N/A)"
    return f"{count}/{total} ({100 * count / total:.1f}%)"


def count_flag(rows, key, expected=True):
    return sum(flag(row, key) is expected for row in rows)


def status_ok(row):
    return row.get("status") == "ok"


def analyze_four(rows):
    target = [
        r for r in rows
        if int(float(r["swarm_size"])) == 4
        and r["method"] == "estimate"
    ]

    completed = [r for r in target if status_ok(r)]
    safe = [
        r for r in completed
        if flag(r, "safety_satisfied") is True
    ]
    unsafe = [
        r for r in completed
        if flag(r, "safety_satisfied") is False
    ]

    lines = [
        "## 1. Four-UAV Doppler Swarm failures",
        "",
        f"- Attempts: **{len(target)}**",
        f"- Returned: **{len(completed)}**",
        f"- Safe: **{len(safe)}**",
        f"- Unsafe: **{len(unsafe)}**",
        f"- Physical-contact failures: "
        f"**{count_flag(completed, 'collision_free', False)}**",
        "",
        "### Failure distribution",
        "",
        "| Speed (m/s) | Encounter | Safe | Unsafe | Other |",
        "|---:|---|---:|---:|---:|",
    ]

    groups = defaultdict(list)
    for row in target:
        groups[(float(row["speed_mps"]), row["category"])].append(row)

    for (speed, category), group in sorted(groups.items()):
        ok = [r for r in group if status_ok(r)]
        lines.append(
            f"| {speed:g} | {category} | "
            f"{count_flag(ok, 'safety_satisfied')} | "
            f"{count_flag(ok, 'safety_satisfied', False)} | "
            f"{len(group) - len(ok)} |"
        )

    lines += [
        "",
        "### Successful vs. unsuccessful trials",
        "",
        "| Diagnostic | Safe | Unsafe |",
        "|---|---:|---:|",
    ]

    def no_command(rs):
        return fraction(
            sum(number(r, "first_command_sim_s") is None for r in rs),
            len(rs),
        )

    def late_command(rs):
        eligible = [
            r for r in rs
            if number(r, "first_command_sim_s") is not None
            and number(r, "first_safety_entry_s") is not None
        ]
        late = sum(
            number(r, "first_command_sim_s")
            > number(r, "first_safety_entry_s")
            for r in eligible
        )
        return fraction(late, len(eligible))

    diagnostics = [
        (
            "Initialization succeeded",
            lambda rs: fraction(
                count_flag(rs, "initialization_success"), len(rs)
            ),
        ),
        (
            "No successful planner call",
            lambda rs: fraction(
                sum(
                    (number(r, "successful_plans") or 0) == 0
                    for r in rs
                ),
                len(rs),
            ),
        ),
        ("No first avoidance command", no_command),
        ("Command after first safety entry", late_command),
        (
            "Median position RMSE (m)",
            lambda rs: fmt(median(rs, "position_rmse_m")),
        ),
        (
            "Median velocity RMSE (m/s)",
            lambda rs: fmt(median(rs, "velocity_rmse_mps")),
        ),
        (
            "Median first estimate time (s)",
            lambda rs: fmt(median(rs, "first_estimate_sim_s")),
        ),
        (
            "Median first command time (s)",
            lambda rs: fmt(median(rs, "first_command_sim_s")),
        ),
        (
            "Median planning attempts",
            lambda rs: fmt(median(rs, "planning_attempts"), 1),
        ),
        (
            "Median successful plans",
            lambda rs: fmt(median(rs, "successful_plans"), 1),
        ),
        (
            "Median total UAV movement (m)",
            lambda rs: fmt(median(rs, "path_length_m")),
        ),
        (
            "Median minimum projectile distance (m)",
            lambda rs: fmt(median(rs, "min_projectile_m")),
        ),
        (
            "Projectile physical collisions",
            lambda rs: fraction(
                count_flag(rs, "projectile_collision"), len(rs)
            ),
        ),
        (
            "UAV-to-UAV physical collisions",
            lambda rs: fraction(
                count_flag(rs, "uav_collision"), len(rs)
            ),
        ),
    ]

    for label, calc in diagnostics:
        lines.append(
            f"| {label} | {calc(safe)} | {calc(unsafe)} |"
        )

    lines += [
        "",
        "### Sample failed encounters",
        "",
        "| Encounter ID | Speed | Category | Initialized | "
        "Position RMSE | Successful/attempted plans |",
        "|---|---:|---|---|---:|---:|",
    ]

    for r in unsafe[:12]:
        lines.append(
            f"| `{r.get('encounter_id', '?')}` | "
            f"{r['speed_mps']} | {r['category']} | "
            f"{flag(r, 'initialization_success')} | "
            f"{fmt(number(r, 'position_rmse_m'))} | "
            f"{fmt(number(r, 'successful_plans'), 0)}/"
            f"{fmt(number(r, 'planning_attempts'), 0)} |"
        )

    lines += [
        "",
        "**Interpretation:** Compare successful and failed trials "
        "to identify associations with estimation error, initialization, "
        "planning, or late commands. These associations do not by "
        "themselves establish the cause of a failure.",
        "",
    ]

    return lines


def analyze_twentyfive(rows):
    target = [
        r for r in rows
        if r["method"] == "dopplerdodge_coordinated"
    ]

    lines = [
        "## 2. Coordinated DopplerDodge timeouts",
        "",
        "The process timeout is a wall-clock computation limit, "
        "not a simulated collision deadline. Timed-out trials "
        "have unknown physical safety outcomes.",
        "",
        "### Runtime scaling",
        "",
        "| UAVs | Attempts | Returned | Timeouts | Errors | "
        "Median runtime (s) | P95 runtime (s) | "
        "Median planning time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    groups = defaultdict(list)

    for r in target:
        groups[int(float(r["swarm_size"]))].append(r)

    for n, group in sorted(groups.items()):
        ok = [r for r in group if status_ok(r)]

        lines.append(
            f"| {n} | {len(group)} | {len(ok)} | "
            f"{sum(r['status'] == 'timeout' for r in group)} | "
            f"{sum(r['status'] == 'error' for r in group)} | "
            f"{fmt(median(ok, 'process_elapsed_s'))} | "
            f"{fmt(percentile(ok, 'process_elapsed_s'))} | "
            f"{fmt(median(ok, 'planning_total_s'))} |"
        )

    large = groups.get(25, [])
    returned = [r for r in large if status_ok(r)]
    timeouts = [
        r for r in large if r.get("status") == "timeout"
    ]

    lines += [
        "",
        "### 25-UAV outcomes",
        "",
        f"- Total attempts: **{len(large)}**",
        f"- Returned: **{len(returned)}**",
        f"- Timeouts: **{len(timeouts)}**",
        f"- Errors: **{sum(r.get('status') == 'error' for r in large)}**",
        f"- Safety among returned trials: "
        f"**{fraction(count_flag(returned, 'safety_satisfied'), len(returned))}**",
        f"- Successful coordinated plans among returned trials: "
        f"**{fraction(count_flag(returned, 'coordinated_plan_success'), len(returned))}**",
        f"- Median elapsed time of timed-out attempts: "
        f"**{fmt(median(timeouts, 'process_elapsed_s'))} s**",
        "",
        "### Timeouts by speed and encounter type",
        "",
        "| Speed (m/s) | Encounter | Timeouts | "
        "Safety among returned |",
        "|---:|---|---:|---:|",
    ]

    buckets = defaultdict(list)

    for r in large:
        buckets[(float(r["speed_mps"]), r["category"])].append(r)

    for (speed, category), group in sorted(buckets.items()):
        ok = [r for r in group if status_ok(r)]
        timed_out = sum(
            r.get("status") == "timeout" for r in group
        )

        lines.append(
            f"| {speed:g} | {category} | "
            f"{fraction(timed_out, len(group))} | "
            f"{fraction(count_flag(ok, 'safety_satisfied'), len(ok))} |"
        )

    lines += [
        "",
        "### Sample timed-out encounter IDs",
        "",
        ", ".join(
            f"`{r.get('encounter_id', '?')}`"
            for r in timeouts[:12]
        ) or "None",
        "",
        "**Interpretation:** Increasing runtime and timeout frequency "
        "with swarm size supports a computational-scaling explanation. "
        "However, terminated workers do not return detailed internal "
        "planner metrics. The saved results cannot definitively "
        "identify the exact operation responsible for each timeout.",
        "",
    ]

    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/research_main_01"),
        help="Directory containing metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path",
    )

    args = parser.parse_args()
    source = args.results / "metrics.csv"

    if not source.is_file():
        parser.error(
            f"Missing {source}. Run from the repository root "
            "and check --results."
        )

    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "swarm_size",
        "speed_mps",
        "category",
        "method",
        "status",
        "safety_satisfied",
        "process_elapsed_s",
    }

    available = set(rows[0]) if rows else set()

    if not rows or not required.issubset(available):
        parser.error(
            f"Unexpected metrics.csv schema. "
            f"Missing: {sorted(required - available)}"
        )

    report = [
        "# Main evaluation diagnostics",
        "",
        f"Source: `{source}` ({len(rows)} method-trial records).",
        "",
        "Read-only analysis; no simulations rerun.",
        "",
    ]

    report += analyze_four(rows)
    report += analyze_twentyfive(rows)

    report += [
        "## Before using these findings in the paper",
        "",
        "1. Distinguish safety-margin failures from physical collisions.",
        "2. Report computational timeouts separately from collisions.",
        "3. Treat observed associations as possible explanations.",
        "4. Compare movement on common successful encounters.",
        "",
    ]

    output = args.output or args.results / "diagnostics.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")

    print(f"Analyzed {len(rows)} records.")
    print(f"Diagnostic report saved to: {output}")


if __name__ == "__main__":
    main()
