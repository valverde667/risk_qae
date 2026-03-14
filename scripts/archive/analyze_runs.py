# scripts/analyze_runs.py
#
# Usage:
#   python scripts/analyze_runs.py
#
# Edit the "USER SETTINGS" block below to point to your CSV and tweak plot behavior.

from __future__ import annotations

from pathlib import Path

import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS (edit these)
# =============================================================================

# Input CSV produced by scripts/run_one.py.
CSV_FILE = os.path.join(os.getcwd(), "runs/hail_data_run.csv")

# Where to write plots + summary CSV.
OUT_DIR = os.path.join(os.getcwd(), "analysis_out/grid_run")

# Optional filter: set to None to analyze all rows, or a specific run_id string.
RUN_ID_FILTER = None  # e.g. "grid_20260223T120000Z"

# Plot options
USE_LOGY = False  # log-scale relative-error plots (good for heavy tails)
FONT_SCALE = 1.35  # seaborn scaling
FIG_DPI = 160
SAVE_DPI = 300

# Which metrics to plot
PLOT_MEAN = True
PLOT_VAR = True
PLOT_TVAR = True
PLOT_TVAR_BUDGET_BREAKDOWN = True  # requires per-component TVaR logging columns

# =============================================================================
# Helpers
# =============================================================================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def q(x: pd.Series, p: float) -> float:
    return float(np.nanquantile(x.to_numpy(dtype=float), p))


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": SAVE_DPI,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def load_and_clean(csv_path: Path, run_id: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if run_id is not None:
        df = df[df["run_id"].astype(str) == str(run_id)].copy()

    # Basic typing
    df["total_shots_budget"] = pd.to_numeric(
        df["total_shots_budget"], errors="coerce"
    ).astype("Int64")
    df["n_index_qubits"] = pd.to_numeric(df["n_index_qubits"], errors="coerce").astype(
        "Int64"
    )
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    # Convert metrics to numeric
    for col in ["mean_rel_err", "var_rel_err", "tvar_rel_err"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows missing key group fields
    df = df.dropna(subset=["total_shots_budget", "n_index_qubits", "alpha"])

    # Convert Int64 -> int for grouping/plotting convenience
    df["total_shots_budget"] = df["total_shots_budget"].astype(int)
    df["n_index_qubits"] = df["n_index_qubits"].astype(int)

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary per (n_index_qubits, total_shots_budget, alpha):
      - median
      - p25/p75 (IQR)
      - p10/p90
      - n runs
    """
    group_cols = ["n_index_qubits", "total_shots_budget", "alpha"]

    def agg_metric(metric: str) -> pd.DataFrame:
        g = df.groupby(group_cols)[metric]
        out = g.agg(n="count", median="median", mean="mean").reset_index()
        out[f"{metric}_p25"] = g.apply(lambda s: q(s, 0.25)).to_numpy()
        out[f"{metric}_p75"] = g.apply(lambda s: q(s, 0.75)).to_numpy()
        out[f"{metric}_p10"] = g.apply(lambda s: q(s, 0.10)).to_numpy()
        out[f"{metric}_p90"] = g.apply(lambda s: q(s, 0.90)).to_numpy()
        out = out.rename(
            columns={
                "median": f"{metric}_median",
                "mean": f"{metric}_mean",
                "n": f"{metric}_n",
            }
        )
        return out

    metrics = [
        m for m in ["mean_rel_err", "var_rel_err", "tvar_rel_err"] if m in df.columns
    ]
    if not metrics:
        raise ValueError(
            "No rel_err columns found in CSV (expected mean_rel_err/var_rel_err/tvar_rel_err)."
        )

    pieces = [agg_metric(m) for m in metrics]

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=group_cols, how="outer")

    return out.sort_values(group_cols).reset_index(drop=True)


def plot_mean_rel_err_by_alpha(
    df_sum: pd.DataFrame, out_dir: Path, *, alpha: float, logy: bool
) -> None:
    metric = "mean_rel_err"
    med = f"{metric}_median"
    p25 = f"{metric}_p25"
    p75 = f"{metric}_p75"

    cols_needed = ["n_index_qubits", "total_shots_budget", "alpha", med, p25, p75]
    d = df_sum[cols_needed].dropna(subset=[med]).copy()
    d = d[d["alpha"] == float(alpha)].copy()

    fig, ax = plt.subplots(figsize=(7.8, 4.2))

    sns.lineplot(
        data=d,
        x="total_shots_budget",
        y=med,
        hue="n_index_qubits",
        marker="o",
        ax=ax,
    )

    for nq in sorted(d["n_index_qubits"].unique()):
        dn = d[d["n_index_qubits"] == nq].sort_values("total_shots_budget")
        ax.fill_between(
            dn["total_shots_budget"].to_numpy(),
            dn[p25].to_numpy(),
            dn[p75].to_numpy(),
            alpha=0.18,
        )

    ax.set_title(f"mean_rel_err vs total shots (alpha={alpha})")
    ax.set_xlabel("Total shots budget")
    ax.set_ylabel("Relative error")
    if logy:
        ax.set_yscale("log")
    ax.legend(title="n_index_qubits", loc="best")

    fig.tight_layout()
    fig.savefig(
        out_dir / f"mean_rel_err_vs_shots_alpha_{alpha}.png", bbox_inches="tight"
    )
    plt.close(fig)


def plot_metric_with_bands(
    df_sum: pd.DataFrame,
    metric: str,
    out_dir: Path,
    *,
    facet_by_alpha: bool,
    logy: bool,
) -> None:
    """
    Line plot of median(metric) vs shots with IQR band (p25–p75).
    Hue = n_index_qubits.
    """
    ensure_dir(out_dir)

    med = f"{metric}_median"
    p25 = f"{metric}_p25"
    p75 = f"{metric}_p75"

    cols_needed = ["n_index_qubits", "total_shots_budget", "alpha", med, p25, p75]
    d = df_sum[cols_needed].dropna(subset=[med]).copy()

    if facet_by_alpha:
        alphas = sorted(d["alpha"].unique())
        fig, axes = plt.subplots(
            len(alphas), 1, figsize=(7.8, 3.4 * len(alphas)), sharex=True
        )
        if len(alphas) == 1:
            axes = [axes]

        for ax, a in zip(axes, alphas):
            da = d[d["alpha"] == a].copy()

            sns.lineplot(
                data=da,
                x="total_shots_budget",
                y=med,
                hue="n_index_qubits",
                marker="o",
                ax=ax,
            )

            # IQR bands per n_index_qubits
            for nq in sorted(da["n_index_qubits"].unique()):
                dn = da[da["n_index_qubits"] == nq].sort_values("total_shots_budget")
                ax.fill_between(
                    dn["total_shots_budget"].to_numpy(),
                    dn[p25].to_numpy(),
                    dn[p75].to_numpy(),
                    alpha=0.18,
                )

            ax.set_title(f"{metric} vs total shots (alpha={a})")
            ax.set_xlabel("Total shots budget")
            ax.set_ylabel("Relative error")
            if logy:
                ax.set_yscale("log")
            ax.legend(title="n_index_qubits", loc="best")

        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_vs_shots_facet_alpha.png", bbox_inches="tight")
        plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(7.8, 4.2))

        sns.lineplot(
            data=d,
            x="total_shots_budget",
            y=med,
            hue="n_index_qubits",
            style="alpha",
            marker="o",
            ax=ax,
        )

        for (nq, a), dn in d.groupby(["n_index_qubits", "alpha"]):
            dn = dn.sort_values("total_shots_budget")
            ax.fill_between(
                dn["total_shots_budget"].to_numpy(),
                dn[p25].to_numpy(),
                dn[p75].to_numpy(),
                alpha=0.15,
            )

        ax.set_title(f"{metric} vs total shots")
        ax.set_xlabel("Total shots budget")
        ax.set_ylabel("Relative error")
        if logy:
            ax.set_yscale("log")
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_vs_shots.png", bbox_inches="tight")
        plt.close(fig)


def plot_tvar_budget_breakdown(df: pd.DataFrame, out_dir: Path, *, logy: bool) -> None:
    """
    Plot median TVaR shots spent in tail_prob vs tail_component, grouped by alpha and qubits.
    Requires your per-component TVaR columns to exist in the CSV.
    """
    required = [
        "total_shots_budget",
        "n_index_qubits",
        "alpha",
        "tvar_tail_prob_shots_used",
        "tvar_tail_component_shots_used",
    ]
    if not all(c in df.columns for c in required):
        print(
            "Skipping TVaR budget breakdown plot (missing per-component TVaR shot columns)."
        )
        return

    d = df.copy()
    d["tvar_tail_prob_shots_used"] = pd.to_numeric(
        d["tvar_tail_prob_shots_used"], errors="coerce"
    )
    d["tvar_tail_component_shots_used"] = pd.to_numeric(
        d["tvar_tail_component_shots_used"], errors="coerce"
    )
    d = d.dropna(subset=["tvar_tail_prob_shots_used", "tvar_tail_component_shots_used"])

    gcols = ["n_index_qubits", "total_shots_budget", "alpha"]
    s = (
        d.groupby(gcols)
        .agg(
            tail_prob_shots_median=("tvar_tail_prob_shots_used", "median"),
            tail_comp_shots_median=("tvar_tail_component_shots_used", "median"),
        )
        .reset_index()
    )

    s_long = s.melt(
        id_vars=gcols,
        value_vars=["tail_prob_shots_median", "tail_comp_shots_median"],
        var_name="component",
        value_name="shots_used_median",
    )
    s_long["component"] = s_long["component"].map(
        {
            "tail_prob_shots_median": "tail_prob",
            "tail_comp_shots_median": "tail_component",
        }
    )

    alphas = sorted(s_long["alpha"].unique())
    fig, axes = plt.subplots(
        len(alphas), 1, figsize=(7.8, 3.4 * len(alphas)), sharex=True
    )
    if len(alphas) == 1:
        axes = [axes]

    for ax, a in zip(axes, alphas):
        da = s_long[s_long["alpha"] == a].copy()
        sns.lineplot(
            data=da,
            x="total_shots_budget",
            y="shots_used_median",
            hue="n_index_qubits",
            style="component",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"TVaR shot breakdown (alpha={a})")
        ax.set_xlabel("Total shots budget")
        ax.set_ylabel("Median shots used")
        if logy:
            ax.set_yscale("log")
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "tvar_tail_shot_breakdown.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# RUN (no main; executes when you run the file)
# =============================================================================

set_plot_style()

csv_path = Path(CSV_FILE)
out_dir = Path(OUT_DIR)
ensure_dir(out_dir)

df = load_and_clean(csv_path, RUN_ID_FILTER)
if df.empty:
    raise SystemExit("No rows found after filtering. Check CSV_FILE / RUN_ID_FILTER.")

df_sum = summarize(df)
df_sum.to_csv(out_dir / "summary_by_shots_qubits_alpha.csv", index=False)

if PLOT_MEAN and "mean_rel_err_median" in df_sum.columns:
    # plot_metric_with_bands(df_sum, "mean_rel_err", out_dir, facet_by_alpha=False, logy=USE_LOGY)
    plot_mean_rel_err_by_alpha(df_sum, out_dir, alpha=0.95, logy=USE_LOGY)
    plot_mean_rel_err_by_alpha(df_sum, out_dir, alpha=0.99, logy=USE_LOGY)


if PLOT_VAR and "var_rel_err_median" in df_sum.columns:
    plot_metric_with_bands(
        df_sum, "var_rel_err", out_dir, facet_by_alpha=True, logy=USE_LOGY
    )

if PLOT_TVAR and "tvar_rel_err_median" in df_sum.columns:
    plot_metric_with_bands(
        df_sum, "tvar_rel_err", out_dir, facet_by_alpha=True, logy=USE_LOGY
    )

if PLOT_TVAR_BUDGET_BREAKDOWN:
    plot_tvar_budget_breakdown(df, out_dir, logy=False)

print(f"Saved summary + plots to: {out_dir.resolve()}")
