#!/usr/bin/env python3
"""Publication-quality Figure 4: trajectory-estimation RMSE vs swarm size.

Reads the *existing* main-evaluation metrics.csv; never changes trial results.
Plots Doppler Swarm ('estimate') only, using all completed trials regardless
of whether avoidance succeeded. Each encounter contributes one RMSE value.

Usage (from repo root):
    python plot_estimation_accuracy.py --results results/research_main_01

Outputs in results/paper_figures/:
    figure_03_estimation_accuracy.pdf / .svg / .png
    figure_03_estimation_accuracy_data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import numpy as np
import pandas as pd
import seaborn as sns

GREEN = "#1FAE63"  # Doppler Swarm (ours), matches existing figures
INK = "#263445"
MUTED = "#667085"
GRID = "#E9EDF1"
METRICS = (
    ("position_rmse_m", "Position RMSE", "Position error (m)"),
    ("velocity_rmse_mps", "Velocity RMSE", "Velocity error (m/s)"),
)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator,
                        draws: int = 2000) -> tuple[float, float]:
    """Nonparametric percentile CI, resampling independent encounter trials."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    medians = np.empty(draws)
    for start in range(0, draws, 100):
        batch = min(100, draws - start)
        sample = values[rng.integers(len(values), size=(batch, len(values)))]
        medians[start:start + batch] = np.median(sample, axis=1)
    return tuple(float(v) for v in np.percentile(medians, [2.5, 97.5]))


def set_style() -> None:
    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#D8DDE3", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def summarize(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for size in sorted(df["swarm_size"].unique()):
        group = df[df["swarm_size"] == size]
        completed = group[group["status"] == "ok"]
        for metric, _, _ in METRICS:
            numeric = pd.to_numeric(completed[metric], errors="coerce")
            valid = numeric[np.isfinite(numeric) & (numeric >= 0)].to_numpy(float)
            low, high = bootstrap_median_ci(valid, rng)
            rows.append({
                "swarm_size": int(size), "metric": metric,
                "attempted_trials": len(group),
                "completed_trials": len(completed),
                "valid_rmse_trials": len(valid),
                "missing_rmse_in_completed": len(completed) - len(valid),
                "median_rmse": float(np.median(valid)) if len(valid) else np.nan,
                "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, output: Path) -> None:
    sizes = sorted(summary["swarm_size"].unique())
    positions = np.arange(len(sizes), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.8), sharex=True)
    fig.suptitle("Trajectory estimation accuracy by swarm size",
                 x=0.075, y=0.97, ha="left", fontsize=16,
                 fontweight="semibold", color=INK)
    fig.text(0.075, 0.885, "Doppler Swarm (ours)  ·  Median RMSE across completed encounters",
             color=MUTED, fontsize=9.5)

    for ax, (metric, title, ylabel), panel in zip(axes, METRICS, "ab"):
        d = summary[summary["metric"] == metric].set_index("swarm_size").reindex(sizes)
        med = d["median_rmse"].to_numpy(float)
        valid = np.isfinite(med)
        if not np.any(valid):
            raise ValueError(f"No valid completed-trial values for {metric}")

        ax.plot(positions[valid], med[valid], color=GREEN, linewidth=3.0,
                solid_capstyle="round", zorder=3)
        ax.set_title(f"{panel}   {title}", loc="left", pad=12, color=INK)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("UAVs", labelpad=9)
        ax.set_xticks(positions, [str(s) for s in sizes])
        ax.set_xlim(-0.2, len(sizes) - 0.8)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", length=0, pad=7)
        sns.despine(ax=ax, left=True, top=True, right=True)

    fig.subplots_adjust(left=0.075, right=0.98, top=0.77, bottom=0.22, wspace=0.27)
    fig.text(0.075, 0.075,
             "All completed trials, including unsafe outcomes  ·  Median RMSE",
             color="#7C8796", fontsize=8.6)
    for ext in ("pdf", "svg", "png"):
        file = output / f"figure_03_estimation_accuracy.{ext}"
        fig.savefig(file, dpi=300, bbox_inches="tight", pad_inches=0.18)
        print(f"Saved {file}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/research_main_01"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    csv = args.results / "metrics.csv"
    if not csv.is_file():
        parser.error(f"Cannot find {csv}")
    df = pd.read_csv(csv, low_memory=False)
    required = {"method", "swarm_size", "status", "encounter_id"} | {m[0] for m in METRICS}
    missing = required - set(df.columns)
    if missing:
        parser.error(f"Missing expected columns in {csv}: {sorted(missing)}")
    ours = df[df["method"] == "estimate"].copy()
    if ours.empty:
        parser.error("No Doppler Swarm ('estimate') rows found")
    if ours["encounter_id"].duplicated().any():
        parser.error("Duplicate Doppler Swarm encounter IDs; expected one trial per encounter")
    ours["swarm_size"] = pd.to_numeric(ours["swarm_size"], errors="raise").astype(int)
    out = args.output or args.results.parent / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(818)
    summary = summarize(ours, rng)
    summary_path = out / "figure_03_estimation_accuracy_data.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Loaded {len(ours):,} Doppler Swarm trials. Saved {summary_path}")
    print(summary[["swarm_size", "metric", "valid_rmse_trials", "median_rmse", "ci95_low", "ci95_high"]].to_string(index=False))
    if (summary["missing_rmse_in_completed"] > 0).any():
        print("WARNING: Some completed trials lack finite RMSE; counts are in the summary CSV.")
    if (ours["status"] != "ok").any():
        print("WARNING: Non-completed trials excluded from RMSE because no usable estimate is available.")
    set_style()
    plot(summary, out)


if __name__ == "__main__":
    main()
