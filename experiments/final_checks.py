
"""Audit six-UAV coordinated-baseline errors and safety-matched movement pairs."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = (
    ("estimate", "oracle"),
    ("estimate", "dopplerdodge"),
    ("estimate", "dopplerdodge_coordinated"),
    ("dopplerdodge_coordinated", "dopplerdodge"),
)
THREATS = {"direct_threat", "grazing_threat", "multiple_threats"}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    value = str(value or "").strip().lower()
    if value in ("true", "1", "1.0"):
        return True
    if value in ("false", "0", "0.0"):
        return False
    return None


def numeric(value):
    if value is None or str(value).strip().lower() in (
        "", "none", "null", "nan"
    ):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentage(n, d):
    return f"{100*n/d:.1f}%" if d else "N/A"


def six_uav_errors(metrics):
    subset = [
        r for r in metrics
        if r["method"] == "dopplerdodge_coordinated"
        and int(float(r["swarm_size"])) == 6
    ]
    errors = [r for r in subset if r["status"] == "error"]
    by_message = Counter(
        (r.get("error") or "<empty error message>").strip()
        for r in errors
    )
    by_condition = Counter(
        (r["speed_mps"], r["category"]) for r in errors
    )

    lines = [
        "## 1. Six-UAV coordinated-baseline errors", "",
        f"Attempts: **{len(subset)}**; errors: **{len(errors)}**; "
        f"timeouts: **{sum(r['status'] == 'timeout' for r in subset)}**.",
        "",
        "### Exact recorded error messages", "",
        "| Count | Error message |",
        "|---:|---|",
    ]

    for msg, count in by_message.most_common():
        lines.append(
            f"| {count} | `{msg.replace('|', chr(92)+'|')}` |"
        )
    if not errors:
        lines.append("| 0 | None |")

    lines += [
        "", "### Error distribution", "",
        "| Speed (m/s) | Category | Errors |",
        "|---:|---|---:|",
    ]
    for (speed, category), count in sorted(
        by_condition.items(),
        key=lambda x: (float(x[0][0]), x[0][1]),
    ):
        lines.append(f"| {speed} | {category} | {count} |")
    if not by_condition:
        lines.append("| — | — | 0 |")

    lines += ["", "### Example encounter IDs", ""]
    lines += [
        f"- `{r['encounter_id']}`: `{r.get('error') or '<empty>'}`"
        for r in errors[:10]
    ]
    if not errors:
        lines.append("No errors recorded.")

    lines += [
        "",
        "**Interpretation:** These are recorded worker errors, "
        "not observed collisions. Use the exact messages to "
        "decide whether follow-up is warranted. This report "
        "cannot diagnose unlogged stack traces.",
        "",
    ]
    return lines


def movement_audit(metrics, paired):
    index = {}
    duplicates = []

    for row in metrics:
        key = (row["encounter_id"], row["method"])
        if key in index:
            duplicates.append(key)
        index[key] = row

    paired_index = {}
    duplicate_pairs = []

    for row in paired:
        key = (
            row["encounter_id"],
            row["method_a"],
            row["method_b"],
        )
        if key in paired_index:
            duplicate_pairs.append(key)
        paired_index[key] = row

    problems = []

    def issue(message):
        problems.append(message)

    expected_keys = set()
    groups = defaultdict(list)

    for (encounter, method), a in index.items():
        for aa, bb in PAIRS:
            if method != aa or (encounter, bb) not in index:
                continue

            key = (encounter, aa, bb)
            expected_keys.add(key)
            b = index[(encounter, bb)]
            p = paired_index.get(key)

            if p is None:
                issue(f"Missing paired row: {key}")
                continue

            safe_a = (
                a["status"] == "ok"
                and boolean(a.get("safety_satisfied")) is True
            )
            safe_b = (
                b["status"] == "ok"
                and boolean(b.get("safety_satisfied")) is True
            )
            both = safe_a and safe_b

            if (
                p.get("status_a") != a["status"]
                or p.get("status_b") != b["status"]
            ):
                issue(f"Status mismatch: {key}")

            if (
                boolean(p.get("safe_a"))
                != boolean(a.get("safety_satisfied"))
                or boolean(p.get("safe_b"))
                != boolean(b.get("safety_satisfied"))
            ):
                issue(f"Safety flag mismatch: {key}")

            if boolean(p.get("both_safe")) != both:
                issue(f"both_safe mismatch: {key}")

            delta = numeric(
                p.get("common_safe_path_change_a_minus_b_m")
            )

            if both:
                pa = numeric(a.get("path_length_m"))
                pb = numeric(b.get("path_length_m"))

                if pa is None or pb is None:
                    issue(f"Missing path length: {key}")
                    continue

                if delta is None or abs(delta - (pa - pb)) > 1e-6:
                    issue(f"Movement delta mismatch: {key}")

                n = int(float(a["swarm_size"]))
                category = a["category"]

                groups[(aa, bb, n, "all")].append((pa, pb))
                groups[(aa, bb, "all", "all")].append((pa, pb))

                if category in THREATS:
                    groups[(aa, bb, n, "threats")].append(
                        (pa, pb)
                    )
                    groups[(aa, bb, "all", "threats")].append(
                        (pa, pb)
                    )

            elif delta is not None:
                issue(
                    f"Movement delta present without both safe: {key}"
                )

    extras = set(paired_index) - expected_keys
    for key in sorted(extras):
        issue(f"Unexpected paired row: {key}")

    if duplicates:
        issue(
            f"Duplicate metrics keys: {len(duplicates)} "
            f"(first: {duplicates[0]})"
        )

    if duplicate_pairs:
        issue(
            f"Duplicate paired keys: {len(duplicate_pairs)} "
            f"(first: {duplicate_pairs[0]})"
        )

    lines = [
        "## 2. Paired movement comparison audit", "",
        f"Metrics records: **{len(metrics)}**; "
        f"paired rows: **{len(paired)}**; "
        f"expected pair keys: **{len(expected_keys)}**.",
        "",
        f"Duplicate metrics keys: **{len(duplicates)}**; "
        f"duplicate pair keys: **{len(duplicate_pairs)}**; "
        f"unexpected pair keys: **{len(extras)}**.",
        "",
        f"Audit issues (up to 30 shown): **{len(problems)}**.",
        "",
    ]

    if problems:
        lines.extend(
            f"- {problem}" for problem in problems[:30]
        )
    else:
        lines.append(
            "**PASS:** Saved pair flags and movement differences "
            "agree with raw metrics."
        )

    lines += [
        "",
        "### Safety-matched movement: common-safe encounters only",
        "",
        "The percent reduction compares mean movement across "
        "the same common-safe encounters: "
        "100 × (1 − mean A / mean B). "
        "It is not the average of per-trial reductions.",
        "",
        "| A | B | UAVs | Subset | Common safe | "
        "Mean A (m) | Mean B (m) | Mean A − B (m) | "
        "Reduction vs B | A moved less |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    counts = sorted({
        int(float(r["swarm_size"])) for r in metrics
    })

    for aa, bb in PAIRS:
        for n in [*counts, "all"]:
            for scope in ("all", "threats"):
                values = groups.get(
                    (aa, bb, n, scope), []
                )
                if not values:
                    continue

                ma = sum(x for x, _ in values) / len(values)
                mb = sum(y for _, y in values) / len(values)

                reduction = (
                    f"{100*(1-ma/mb):.1f}%"
                    if mb > 1e-12 else "N/A"
                )
                wins = sum(
                    x < y - 1e-9 for x, y in values
                )

                lines.append(
                    f"| {aa} | {bb} | {n} | {scope} | "
                    f"{len(values)} | {ma:.4f} | {mb:.4f} | "
                    f"{ma-mb:+.4f} | {reduction} | "
                    f"{percentage(wins, len(values))} |"
                )

    lines += [
        "",
        "**Important:** These are conditional movement "
        "comparisons, not claims of equal safety. "
        "Report overall safety and timeout rates separately. "
        "'Threats' excludes safe_miss.",
        "",
    ]

    return lines, not problems


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/research_main_01"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    metrics = read_csv(
        args.results / "metrics.csv"
    )
    paired = read_csv(
        args.results / "paired_comparison.csv"
    )

    if not metrics or not paired:
        parser.error("Input CSVs must not be empty")

    report = [
        "# Final experimental checks", "",
        "Read-only checks of the main research evaluation.",
        "",
    ]

    report += six_uav_errors(metrics)
    audit, passed = movement_audit(metrics, paired)
    report += audit

    output = (
        args.output
        or args.results / "final_checks.md"
    )
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        f"Read {len(metrics)} metrics and "
        f"{len(paired)} paired comparisons."
    )
    print(
        f"Movement audit: "
        f"{'PASS' if passed else 'ISSUES FOUND'}"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
