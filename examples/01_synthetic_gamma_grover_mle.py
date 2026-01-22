"""Example 01 — Synthetic Gamma losses + Grover-MLE AE (QFT-free).

What this demonstrates
- synthetic, right-tailed losses (Gamma)
- low-qubit discretization (2**n bins)
- Grover-schedule + MLE amplitude estimation with optional early stopping
- prints key diagnostics + basic sanity checks
"""

from __future__ import annotations

import numpy as np

from risk_qae import RiskQAEConfig, estimate_risk_measures
from risk_qae.config import (
    AEConfig,
    BackendConfig,
    BudgetConfig,
    DiscretizationConfig,
    ValueEncodingConfig,
)


def main() -> None:
    # --- synthetic loss distribution (positive, right-tailed) ---
    rng = np.random.default_rng(0)
    samples = rng.gamma(shape=2.0, scale=100.0, size=50_000)

    # --- config: fast + stable defaults ---
    cfg = RiskQAEConfig(
        backend=BackendConfig(provider="aer"),
        discretization=DiscretizationConfig(n_index_qubits=6),  # 64 bins (fast)
        budget=BudgetConfig(
            total_shots=10_000,
            shots_per_call=1_000,
            max_circuit_calls=50,
            seed=123,
        ),
        value_encoding=ValueEncodingConfig(method="piecewise_prefix", n_segments=16),
        ae=AEConfig(
            method="grover_mle",
            grover_mle_grid_size=1200,  # bump to 2000+ for stability
            grover_powers=(0, 1, 2, 4, 8),  # omit to use auto-doubling schedule
            epsilon_target=25.0,  # stop when CI half-width <= 25 (loss units)
            alpha_confidence=0.05,  # 95% CI
        ),
    )

    res = estimate_risk_measures(
        {"samples": samples},
        alphas=[0.99],
        config=cfg,
    )

    print("=== Example 01: Gamma + Grover-MLE ===")
    print("Mean:", res.mean.mean)
    print("Mean CI:", res.mean.mean_ci)
    print("VaR(0.99):", res.var[0.99].var, " bracket:", res.var[0.99].bracket)
    print("TVaR(0.99):", res.tvar[0.99].tvar)
    print("Total shots used:", res.total_shots_used)
    print("Total circuits run:", res.total_circuits_run)
    print("Mean Grover powers:", res.mean.diagnostics.get("powers"))
    print("Mean early_stop:", res.mean.diagnostics.get("early_stop"))

    # --- sanity checks ---
    assert res.total_shots_used <= cfg.budget.total_shots
    assert res.mean.mean >= 0.0
    assert res.var[0.99].var >= 0.0
    # discretization-consistent TVaR check
    assert res.tvar[0.99].tvar >= res.var[0.99].bracket[0]


if __name__ == "__main__":
    main()
