#!/usr/bin/env python3
"""Publication figures from the frozen CMSC818V main evaluation.

Usage:
  python experiments/plot_research_figures.py --results results/research_main_01

Outputs: figure_01_safety, figure_02_movement, figure_04_runtime
         as PDF, SVG, PNG (300 dpi), plus source-summary CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METHODS = {
    "estimate": ("Doppler Swarm (ours)", "#1FAE63"),
    "dopplerdodge": ("DopplerDodge", "#D84B4B"),
    "dopplerdodge_coordinated": ("Informed DopplerDodge", "#2878C8"),
}
RNG_SEED = 818


def configure_style():
    sns.set_theme(context="paper", style="white", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D8DDE3",
        "axes.labelcolor": "#344054",
        "text.color": "#263445",
        "xtick.color": "#667085",
        "ytick.color": "#667085",
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "grid.color": "#E9EDF1",
        "grid.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def style_axes(ax):
    # Only a few quiet horizontal guides; no minor grid, markers, or heavy spines.
    sns.despine(ax=ax, top=True, right=True, left=True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#E9EDF1", linewidth=0.75)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="both", length=0, pad=7)


def new_figure(title):
    fig, ax = plt.subplots(figsize=(8.6, 4.65))
    fig.suptitle(title, x=0.105, y=0.967, ha="left",
                 fontsize=15, fontweight="semibold", color="#263445")
    fig.subplots_adjust(left=0.105, right=0.975, top=0.745, bottom=0.19)
    return fig, ax


def top_legend(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left",
               bbox_to_anchor=(0.105, 0.882), ncol=ncol,
               frameon=False, columnspacing=1.8, handlelength=2.5,
               handletextpad=0.6, borderaxespad=0)


def note(fig, value):
    fig.text(0.105, 0.055, value, fontsize=8.3, color="#7C8796")


def save(fig, output: Path, name: str):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Saved {name}.pdf / .svg / .png")


def wilson_interval(success: int, total: int, z=1.9599639845):
    if total == 0:
        return np.nan, np.nan
    p = success / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return center-half, center+half


def bootstrap_ci(values, stat, rng, n_boot=1500):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    if len(a) == 1:
        return a[0], a[0]
    # Batch to limit peak memory for large groups.
    out = []
    for _ in range(0, n_boot, 150):
        batch = min(150, n_boot - len(out))
        samples = a[rng.integers(0, len(a), size=(batch, len(a)))]
        out.extend(stat(samples, axis=1).tolist())
    return np.percentile(out, [2.5, 97.5])


def draw_series(ax, table, sizes, metric, low, high, show_band=True):
    positions = np.arange(len(sizes))
    for method, (label, color) in METHODS.items():
        sub = table[table["method"] == method].set_index("swarm_size").reindex(sizes)
        y = sub[metric].to_numpy(dtype=float)
        lo = sub[low].to_numpy(dtype=float)
        hi = sub[high].to_numpy(dtype=float)
        valid = np.isfinite(y)
        if not valid.any():
            print(f"WARNING: no {metric} data for {method}")
            continue
        x = positions[valid]
        # Restrained confidence ribbons, rather than busy point-by-point error bars.
        if show_band:
            ax.fill_between(x, lo[valid], hi[valid], color=color, alpha=0.10,
                            linewidth=0, zorder=2)
        ax.plot(x, y[valid], color=color,
                linewidth=2.8 if method == "estimate" else 2.15,
                solid_capstyle="round", label=label, zorder=3)
    ax.set_xticks(positions, [str(n) for n in sizes])
    ax.set_xlim(-0.13, len(sizes) - 0.87)


def figure_safety(df, sizes, out):
    rows = []
    for n in sizes:
        for method in METHODS:
            g = df[(df.swarm_size == n) & (df.method == method)]
            total = len(g)
            safe = int(g["safe"].sum())
            lo, hi = wilson_interval(safe, total)
            rows.append(dict(swarm_size=n, method=method, attempts=total,
                             safe=safe, completed=int(g["complete"].sum()),
                             timeouts=int(g["status"].eq("timeout").sum()),
                             rate=100*safe/total if total else np.nan,
                             ci_low=100*lo, ci_high=100*hi))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "figure_01_safety_data.csv", index=False)
    fig, ax = new_figure("Avoidance safety by swarm size")
    draw_series(ax, summary, sizes, "rate", "ci_low", "ci_high", show_band=False)
    ax.axhline(100, color="#B7BEC8", ls=(0, (4, 3)), lw=1.35,
               label="Oracle", zorder=1)
    ax.set(xlabel="UAVs", ylabel="Safety (%)")
    ax.set_ylim(50, 103)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    style_axes(ax)
    top_legend(fig, ax, ncol=4)
    note(fig, "Budgeted safety · Safe completed trials / all attempts · Timeouts are not collisions")
    save(fig, out, "figure_01_safety")


def figure_movement(df, sizes, out):
    # One identical subset of encounters for all three curves; exclude safe misses.
    subset = df[df.method.isin(METHODS) & df.category.ne("safe_miss") & df["safe"]].copy()
    subset["path_length_m"] = pd.to_numeric(subset["path_length_m"], errors="coerce")
    subset = subset.dropna(subset=["path_length_m"])
    counts = subset.groupby("encounter_id")["method"].nunique()
    valid_ids = counts[counts == len(METHODS)].index
    common = subset[subset.encounter_id.isin(valid_ids)]
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for n in sizes:
        for method in METHODS:
            g = common[(common.swarm_size == n) & (common.method == method)]
            a = g.path_length_m.to_numpy(dtype=float)
            lo, hi = bootstrap_ci(a, np.mean, rng)
            rows.append(dict(swarm_size=n, method=method, paired_encounters=len(a),
                             mean_m=np.mean(a) if len(a) else np.nan,
                             ci_low=lo, ci_high=hi))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "figure_02_movement_data.csv", index=False)
    fig, ax = new_figure("Swarm displacement on common-safe threats")
    draw_series(ax, summary, sizes, "mean_m", "ci_low", "ci_high")
    ax.set(xlabel="UAVs", ylabel="Displacement (m)")
    ax.set_yscale("log")
    ax.set_ylim(0.085, 4.0)
    ax.set_yticks([0.1, 0.3, 1.0, 3.0], labels=["0.1", "0.3", "1", "3"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    style_axes(ax)
    top_legend(fig, ax)
    note(fig, "Mean total displacement · Same threat encounters safe for all three methods · 95% bootstrap CI")
    save(fig, out, "figure_02_movement")


def figure_runtime(df, sizes, out):
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for n in sizes:
        for method in METHODS:
            attempts = df[(df.swarm_size == n) & (df.method == method)]
            completed = attempts[attempts.complete]
            a = pd.to_numeric(completed["planning_total_s"], errors="coerce").dropna().to_numpy(float)
            lo, hi = bootstrap_ci(a, np.median, rng)
            rows.append(dict(swarm_size=n, method=method, attempts=len(attempts),
                             completed=len(completed), timing_samples=len(a),
                             timeouts=int(attempts.status.eq("timeout").sum()),
                             median_s=np.median(a) if len(a) else np.nan,
                             ci_low=lo, ci_high=hi))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "figure_04_runtime_data.csv", index=False)
    fig, ax = new_figure("Planning computation by swarm size")
    draw_series(ax, summary, sizes, "median_s", "ci_low", "ci_high")
    ax.set(xlabel="UAVs", ylabel="Planning time (s)")
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 100)
    ax.set_yticks([1e-4, 1e-2, 1, 100],
                  labels=["0.0001", "0.01", "1", "100"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    style_axes(ax)
    top_legend(fig, ax)
    note(fig, "Median cumulative planning time · Completed trials only · 95% bootstrap CI")
    timeout_text = "; ".join(
        f"{METHODS[m][0]}: {int(g.timeouts.sum())} timeouts"
        for m, g in summary.groupby("method", sort=False) if g.timeouts.sum()
    )
    print("Runtime timeout summary:", timeout_text or "none")
    save(fig, out, "figure_04_runtime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/research_main_01"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    csv_path = args.results / "metrics.csv"
    if not csv_path.is_file():
        parser.error(f"Not found: {csv_path}")
    out = args.output or args.results.parent / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=False)
    required = {"encounter_id", "swarm_size", "category", "method", "status",
                "safety_satisfied", "path_length_m", "planning_total_s"}
    missing = required - set(df.columns)
    if missing:
        parser.error(f"Missing columns in {csv_path}: {sorted(missing)}")
    df["complete"] = df.status.eq("ok")
    df["safe"] = df.complete & df.safety_satisfied.astype(str).str.lower().eq("true")
    df["swarm_size"] = pd.to_numeric(df.swarm_size, errors="raise").astype(int)
    sizes = sorted(df.swarm_size.unique())
    print(f"Loaded {len(df):,} method trials, sizes={sizes}")
    configure_style()
    figure_safety(df, sizes, out)
    figure_movement(df, sizes, out)
    figure_runtime(df, sizes, out)
    print(f"All figures and plotted data saved to {out}")


if __name__ == "__main__":
    main()
