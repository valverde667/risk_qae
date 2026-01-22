"""Example 03 — Real dataset template (fill in later).

This script is meant as a *drop-in template* for a real dataset.
For now it contains:
- a minimal CLI skeleton
- a data loading placeholder (CSV)
- standard pre-flight checks
- a consistent reporting + JSON export block
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

    res = estimate_risk_measures({"samples": samples}, alphas=[0.95, 0.99], config=cfg)

    # --- report ---
    print("=== Example 03: Real dataset template ===")
    # print("path:", str(path))
    print("n_samples:", int(samples.size))
    print("Mean:", res.mean.mean, "CI:", res.mean.mean_ci)
    for a in (0.95, 0.99):
        print(f"VaR({a}):", res.var[a].var, " bracket:", res.var[a].bracket)
        print(f"TVaR({a}):", res.tvar[a].tvar)

    # --- export (handy for sharing / regression tests) ---
    out = {
        # "path": str(path),
        "n_samples": int(samples.size),
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
