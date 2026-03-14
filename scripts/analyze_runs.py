# analyze_runs.py
#
# Usage:
#   python analyze_runs.py
#
# Set the USER SETTINGS block below before running.
# Two study modes:
#   STUDY = "shots"    -> Study 1 plots (shot sweep, qubit lines, runtime bar)
#   STUDY = "segments" -> Study 2 plots (segment sensitivity, supplemental)

from __future__ import annotations

from pathlib import Path

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================================
# USER SETTINGS
# =============================================================================

# Path to CSV produced by sweep_study1_shots.sh or sweep_study2_segments.sh
CSV_FILE = os.path.join(os.getcwd(), "runs/study1_shots_20260304T070527Z.csv")

# Output directory for plots + summary CSV
OUT_DIR = os.path.join(os.getcwd(), "analysis_out/study1")

# Which study to produce plots for: "shots" or "segments"
STUDY = "shots"

# Optional: filter to a specific run_id string. None = use all rows.
RUN_ID_FILTER = None

# For the runtime bar chart (Study 1): which shot count to use as the "fixed N*" reference.
# Set this to your chosen elbow point after reviewing the shot sweep plots.
RUNTIME_FIXED_SHOTS = 50000

# Plot style
USE_LOGY = False  # log-scale y axis on relative error plots
FIG_DPI = 150
SAVE_DPI = 300

# Color palette: one color per qubit count (6, 8, 10) — consistent across all figures
QUBIT_COLORS = {6: "#CC0000", 8: "#1a6faf", 10: "#2ca02c"}

# For segment study: which alpha to show on the segment sensitivity plot
SEGMENT_PLOT_ALPHA = 0.99

# =============================================================================
# Helpers
# =============================================================================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def q(x: pd.Series, p: float) -> float:
    return float(np.nanquantile(x.to_numpy(dtype=float), p))


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": SAVE_DPI,
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
        }
    )


def load_and_clean(csv_path: Path, run_id: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if run_id is not None:
        df = df[df["run_id"].astype(str) == str(run_id)].copy()

    for col in ["total_shots_budget", "n_index_qubits", "n_segments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    for col in ["mean_rel_err", "var_rel_err", "tvar_rel_err", "runtime_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["total_shots_budget", "n_index_qubits", "alpha"])

    df["total_shots_budget"] = df["total_shots_budget"].astype(int)
    df["n_index_qubits"] = df["n_index_qubits"].astype(int)
    if "n_segments" in df.columns:
        df["n_segments"] = df["n_segments"].astype(int)

    return df


def summarize_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over (n_index_qubits, total_shots_budget, alpha)."""
    gcols = ["n_index_qubits", "total_shots_budget", "alpha"]
    metrics = [
        m
        for m in [
            "mean_rel_err",
            "var_rel_err",
            "tvar_rel_err",
            "runtime_s",
            "var_bisection_iters",
        ]
        if m in df.columns
    ]

    rows = []
    for keys, g in df.groupby(gcols):
        row = dict(zip(gcols, keys))
        row["n"] = len(g)
        for m in metrics:
            col = g[m].dropna()
            row[f"{m}_median"] = col.median()
            row[f"{m}_p25"] = col.quantile(0.25)
            row[f"{m}_p75"] = col.quantile(0.75)
            row[f"{m}_mean"] = col.mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values(gcols).reset_index(drop=True)


def summarize_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over (n_segments, alpha) — for Study 2."""
    gcols = ["n_segments", "alpha"]
    metrics = [
        m for m in ["mean_rel_err", "var_rel_err", "tvar_rel_err"] if m in df.columns
    ]

    rows = []
    for keys, g in df.groupby(gcols):
        row = dict(zip(gcols, keys))
        row["n"] = len(g)
        for m in metrics:
            col = g[m].dropna()
            row[f"{m}_median"] = col.median()
            row[f"{m}_p25"] = col.quantile(0.25)
            row[f"{m}_p75"] = col.quantile(0.75)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(gcols).reset_index(drop=True)


# =============================================================================
# Study 1 plots
# =============================================================================


def _shot_sweep_ax(
    ax, df_sum: pd.DataFrame, metric: str, alpha: float, logy: bool
) -> None:
    """
    Draw one axes panel: median relative error vs shots for one metric + alpha.
    Lines = qubit counts, shaded band = IQR (p25–p75).
    """
    med = f"{metric}_median"
    p25 = f"{metric}_p25"
    p75 = f"{metric}_p75"

    d = df_sum[df_sum["alpha"] == alpha].copy()
    qubit_counts = sorted(d["n_index_qubits"].unique())

    for nq in qubit_counts:
        dn = d[d["n_index_qubits"] == nq].sort_values("total_shots_budget")
        color = QUBIT_COLORS.get(nq, None)
        x = dn["total_shots_budget"].to_numpy()
        y = dn[med].to_numpy()
        y25 = dn[p25].to_numpy()
        y75 = dn[p75].to_numpy()

        ax.plot(
            x,
            y,
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=f"{nq} qubits",
        )
        ax.fill_between(x, y25, y75, alpha=0.15, color=color)

    ax.set_xlabel("Total shots budget")
    ax.set_ylabel("Relative error")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    if logy:
        ax.set_yscale("log")
    ax.legend(title="Qubit count", loc="upper right", framealpha=0.85)


def plot_study1_metric(
    df_sum: pd.DataFrame, metric: str, title: str, out_dir: Path, logy: bool
) -> None:
    """
    One figure per metric (mean / VaR / TVaR).
    Two panels side by side: alpha=0.95 and alpha=0.99.
    (For mean_rel_err the two panels will be identical since mean has no alpha;
     we still facet for visual consistency — or you can pass alphas=[0.95] only.)
    """
    alphas = sorted(df_sum["alpha"].unique())

    # Mean doesn't vary by alpha — only show one panel
    if metric == "mean_rel_err":
        alphas = [alphas[0]]

    ncols = len(alphas)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, a in zip(axes, alphas):
        _shot_sweep_ax(ax, df_sum, metric, a, logy)
        suffix = "" if metric == "mean_rel_err" else f"  (α={a})"
        ax.set_title(f"{title}{suffix}")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = f"{metric}_vs_shots.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_runtime_bar(df: pd.DataFrame, out_dir: Path, fixed_shots: int) -> None:
    """
    Bar chart: median runtime ± IQR per qubit count at a fixed shot budget.
    """
    if "runtime_s" not in df.columns:
        print("  Skipping runtime bar chart (no runtime_s column).")
        return

    # Use one alpha only (runtime is the same for both rows of same rep)
    d = df[
        (df["total_shots_budget"] == fixed_shots) & (df["alpha"] == df["alpha"].min())
    ].copy()

    if d.empty:
        print(f"  Skipping runtime bar chart — no rows found for shots={fixed_shots}.")
        return

    qubit_counts = sorted(d["n_index_qubits"].unique())
    medians = []
    p25s = []
    p75s = []

    for nq in qubit_counts:
        col = d[d["n_index_qubits"] == nq]["runtime_s"].dropna()
        medians.append(col.median())
        p25s.append(col.quantile(0.25))
        p75s.append(col.quantile(0.75))

    medians = np.array(medians)
    err_lo = medians - np.array(p25s)
    err_hi = np.array(p75s) - medians

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    colors = [QUBIT_COLORS.get(nq, "#555555") for nq in qubit_counts]
    bars = ax.bar(
        [str(nq) for nq in qubit_counts],
        medians,
        yerr=[err_lo, err_hi],
        capsize=6,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )

    ax.set_xlabel("Qubit count (n)")
    ax.set_ylabel("Wall-clock time per run (s)")
    ax.set_title(f"Runtime vs. qubit count\n(shots = {fixed_shots:,}, max segments)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f s"))

    # Annotate bars with median value
    for bar, med in zip(bars, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(err_hi) * 0.05,
            f"{med:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "runtime_vs_qubits.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved runtime_vs_qubits.png")


# =============================================================================
# Study 2 plots
# =============================================================================


def plot_segment_sensitivity(
    df_sum: pd.DataFrame, out_dir: Path, alpha: float, logy: bool
) -> None:
    """
    Segment sensitivity plot (Study 2 / supplemental).
    x = n_segments (log2 scale), y = relative error.
    Three lines: mean_rel_err, var_rel_err, tvar_rel_err.
    """
    d = df_sum[df_sum["alpha"] == alpha].copy().sort_values("n_segments")
    max_segs = d["n_segments"].max()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    line_styles = {
        "mean_rel_err": ("Mean", "#CC0000", "o", "-"),
        "var_rel_err": (f"VaR (α={alpha})", "#1a6faf", "s", "--"),
        "tvar_rel_err": (f"TVaR (α={alpha})", "#2ca02c", "^", "-."),
    }

    for metric, (label, color, marker, ls) in line_styles.items():
        med = f"{metric}_median"
        p25 = f"{metric}_p25"
        p75 = f"{metric}_p75"
        if med not in d.columns:
            continue
        x = d["n_segments"].to_numpy()
        y = d[med].to_numpy()
        y25 = d[p25].to_numpy()
        y75 = d[p75].to_numpy()

        ax.plot(
            x,
            y,
            marker=marker,
            markersize=5,
            linewidth=2,
            color=color,
            linestyle=ls,
            label=label,
        )
        ax.fill_between(x, y25, y75, alpha=0.12, color=color)

    # Mark maximum viable segments with a dashed vertical line
    ax.axvline(
        max_segs,
        color="#888888",
        linewidth=1.2,
        linestyle=":",
        label=f"Max segments (2^n = {max_segs})",
    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("Number of encoding segments")
    ax.set_ylabel("Relative error")
    ax.set_title(f"Segment sensitivity  (n=8 qubits, α={alpha})")
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_dir / f"segment_sensitivity_alpha_{alpha}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved segment_sensitivity_alpha_{alpha}.png")


# =============================================================================
# Main
# =============================================================================

set_plot_style()

csv_path = Path(CSV_FILE)
out_dir = Path(OUT_DIR)
ensure_dir(out_dir)

df = load_and_clean(csv_path, RUN_ID_FILTER)
if df.empty:
    raise SystemExit("No rows found after filtering. Check CSV_FILE / RUN_ID_FILTER.")

print(f"Loaded {len(df)} rows from {csv_path.name}")
print(f"Study mode: {STUDY}")

if STUDY == "shots":
    df_sum = summarize_shots(df)
    df_sum.to_csv(out_dir / "summary_shots.csv", index=False)
    print(f"Summary -> {out_dir / 'summary_shots.csv'}")

    # One figure per metric, two panels (alpha=0.95 and alpha=0.99)
    plot_study1_metric(df_sum, "mean_rel_err", "Mean relative error", out_dir, USE_LOGY)
    plot_study1_metric(df_sum, "var_rel_err", "VaR relative error", out_dir, USE_LOGY)
    plot_study1_metric(df_sum, "tvar_rel_err", "TVaR relative error", out_dir, USE_LOGY)

    # Runtime bar chart at fixed N*
    plot_runtime_bar(df, out_dir, fixed_shots=RUNTIME_FIXED_SHOTS)

elif STUDY == "segments":
    df_sum = summarize_segments(df)
    df_sum.to_csv(out_dir / "summary_segments.csv", index=False)
    print(f"Summary -> {out_dir / 'summary_segments.csv'}")

    plot_segment_sensitivity(df_sum, out_dir, alpha=SEGMENT_PLOT_ALPHA, logy=USE_LOGY)

else:
    raise SystemExit(f"Unknown STUDY value: {STUDY!r}. Must be 'shots' or 'segments'.")

print(f"\nAll outputs -> {out_dir.resolve()}")
