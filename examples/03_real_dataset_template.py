"""Example 03 — Real dataset template (fill in later).

This script is meant as a *drop-in template* for a real dataset.
For now it contains:
- a minimal CLI skeleton
- a data loading placeholder (CSV)
- standard pre-flight checks
- consistent reporting + JSON export block
- classical-on-discretized baselines for quick validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from risk_qae import RiskQAEConfig, estimate_risk_measures
from risk_qae.config import (
    AEConfig,
    BackendConfig,
    BudgetConfig,
    DiscretizationConfig,
    ValueEncodingConfig,
)

# NEW: explicit discretization + classical references
from risk_qae.discretization.histogram import discretize_samples
from risk_qae.metrics.classical import (
    classical_mean,
    classical_scaled_mean_amplitude,
    classical_tvar,
    classical_tvar_index_tail,
    classical_var,
    summarize_samples,
)


def _rel_err(est: float, ref: float) -> float:
    denom = max(abs(ref), 1e-12)
    return abs(est - ref) / denom


def load_losses_csv(path: Path, column: str) -> np.ndarray:
    """Placeholder CSV loader.

    Replace with pandas if preferred. For a lightweight template, this assumes:
    - header row present
    - numeric column
    """
    import csv

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader:
            v = row.get(column, None)
            if v is None or v == "":
                continue
            vals.append(float(v))
    return np.asarray(vals, dtype=float)


def main() -> None:
    # Example of using cmd line arguments. But, we will hard code for now.
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--path", type=str, required=False, help="Path to CSV file")
    # ap.add_argument("--column", type=str, required=False, default="loss", help="Column name holding losses")
    # ap.add_argument("--out", type=str, required=False, default="risk_qae_result.json", help="Output JSON filename")
    # args = ap.parse_args()

    # if args.path is None:
    #     raise SystemExit("Provide --path to your CSV (this is a template).")

    # path = Path(args.path).expanduser().resolve()
    # samples = load_losses_csv(path, args.column)
    samples = np.load("mc_losses_gamma_model.npy")  # Hard code

    # --- pre-checks ---
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise SystemExit("No valid numeric samples found.")
    if np.any(samples < 0):
        print(
            "Warning: negative values found. If these are gains, transform to losses first."
        )

    # --- config (tune later) ---
    cfg = RiskQAEConfig(
        backend=BackendConfig(provider="aer"),
        discretization=DiscretizationConfig(n_index_qubits=8),
        budget=BudgetConfig(
            total_shots=20_000, shots_per_call=2_000, max_circuit_calls=100
        ),
        value_encoding=ValueEncodingConfig(method="piecewise_prefix", n_segments=16),
        ae=AEConfig(
            method="grover_mle",
            grover_mle_grid_size=2000,
            grover_powers=(0, 1, 2, 4, 8),
            # epsilon_target=None,
            # alpha_confidence=0.05,
        ),
    )

    alphas = (0.95, 0.99)

    # -------------------------------------------------------------------------
    # Classical-on-discretized diagnostics (explicit discretization)
    # -------------------------------------------------------------------------
    dist = discretize_samples(samples, cfg.discretization)
    ssum = summarize_samples(samples, upper_q=0.999, xmax=float(dist.bounds[1]))

    cmean = classical_mean(dist).mean
    csm = classical_scaled_mean_amplitude(dist)
    xmin, xmax = float(csm.xmin), float(csm.xmax)
    mean_recon = xmin + (xmax - xmin) * float(csm.a_mean)

    classical_per_alpha = {}
    for a in alphas:
        cvar = classical_var(dist, a)
        ctvar_val = classical_tvar(dist, a)
        ctvar_idx = classical_tvar_index_tail(dist, a)
        classical_per_alpha[a] = {
            "var": cvar,
            "tvar_value_tail": ctvar_val,
            "tvar_index_tail": ctvar_idx,
        }

    # -------------------------------------------------------------------------
    # Quantum run
    # -------------------------------------------------------------------------
    res = estimate_risk_measures({"samples": samples}, alphas=list(alphas), config=cfg)

    # --- report ---
    print("=== Example 03: Real dataset template ===")
    # print("path:", str(path))
    print("n_samples:", int(samples.size))
    print()

    print("=== Discretization / sample sanity ===")
    print("n_bins:", int(dist.pmf.size))
    print("bounds:", dist.bounds)
    print("samples summary:", ssum)
    print()

    print("=== Classical (ON DISCRETIZED DIST) ===")
    print("Mean:", cmean)
    print("Scaled mean amplitude a_mean:", csm.a_mean)
    print("Mean recon from a_mean:", mean_recon)
    for a in alphas:
        cvar = classical_per_alpha[a]["var"]
        ctvar_val = classical_per_alpha[a]["tvar_value_tail"]
        ctvar_idx = classical_per_alpha[a]["tvar_index_tail"]
        print(
            f"VaR({a}):",
            cvar.var,
            " bracket:",
            cvar.bracket,
            " idx:",
            cvar.var_bin_index,
        )
        print(
            f"TVaR({a}) value-tail:", ctvar_val.tvar, " tail_prob:", ctvar_val.tail_prob
        )
        print(
            f"TVaR({a}) index-tail:", ctvar_idx.tvar, " tail_prob:", ctvar_idx.tail_prob
        )
    print()

    print("=== Quantum ===")
    print("Mean:", res.mean.mean, "CI:", res.mean.mean_ci)
    for a in alphas:
        print(f"VaR({a}):", res.var[a].var, " bracket:", res.var[a].bracket)
        print(f"TVaR({a}):", res.tvar[a].tvar)
        if res.tvar[a].tvar < res.var[a].bracket[0]:
            print("  WARNING: TVaR fell below VaR bracket lower edge.")
            print("    TVaR:", res.tvar[a].tvar)
            print("    VaR bracket:", res.var[a].bracket)
            print(
                "    This indicates an inconsistency in tail definition/scaling and/or early-stop units."
            )
    print()

    print("=== Relative error vs classical-on-discretized ===")
    print("mean rel_err:", _rel_err(res.mean.mean, cmean))
    for a in alphas:
        cvar = classical_per_alpha[a]["var"].var
        ctvar = classical_per_alpha[a]["tvar_value_tail"].tvar
        print(f"var({a})  rel_err:", _rel_err(res.var[a].var, cvar))
        print(f"tvar({a}) rel_err:", _rel_err(res.tvar[a].tvar, ctvar))
    print()

    # --- export (handy for sharing / regression tests) ---
    out = {
        # "path": str(path),
        "n_samples": int(samples.size),
        "diagnostics": {
            "discretization": {
                "n_bins": int(dist.pmf.size),
                "bounds": tuple(map(float, dist.bounds)),
            },
            "samples_summary": {
                "mean_raw": ssum.mean_raw,
                "mean_winsor": ssum.mean_winsor,
                "winsor_bounds": tuple(map(float, ssum.winsor_bounds)),
                "mass_above_xmax": (
                    None
                    if ssum.mass_above_xmax is None
                    else float(ssum.mass_above_xmax)
                ),
            },
            "classical_on_discretized": {
                "mean": cmean,
                "a_mean": float(csm.a_mean),
                "mean_recon_from_a_mean": float(mean_recon),
                "per_alpha": {
                    str(a): {
                        "var": classical_per_alpha[a]["var"].__dict__,
                        "tvar_value_tail": classical_per_alpha[a][
                            "tvar_value_tail"
                        ].__dict__,
                        "tvar_index_tail": classical_per_alpha[a][
                            "tvar_index_tail"
                        ].__dict__,
                    }
                    for a in alphas
                },
            },
        },
        "result": {
            "mean": {
                "mean": res.mean.mean,
                "ci": res.mean.mean_ci,
                "shots": res.mean.shots_used,
                "circuits": res.mean.circuits_run,
            },
            "var": {str(a): res.var[a].__dict__ for a in res.var.keys()},
            "tvar": {str(a): res.tvar[a].__dict__ for a in res.tvar.keys()},
            "total_shots_used": res.total_shots_used,
            "total_circuits_run": res.total_circuits_run,
            "discretization_summary": res.discretization_summary,
            "config_frozen": res.config_frozen,
        },
    }
    # Path(args.out).write_text(json.dumps(out, indent=2))
    # print("Wrote:", args.out)


if __name__ == "__main__":
    main()
