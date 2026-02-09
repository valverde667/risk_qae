"""Example 01 — Synthetic Gamma losses + Grover-MLE AE (QFT-free).

What this demonstrates
- synthetic, right-tailed losses (Gamma)
- low-qubit discretization (2**n bins)
- Grover-schedule + MLE amplitude estimation with optional early stopping
- classical-on-discretized comparisons for fast validation
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

# NEW: discretization helper (functional wrapper)
from risk_qae.discretization.histogram import discretize_samples

# NEW: classical + debug helpers
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
            epsilon_target=25.0,  # NOTE: currently used across subproblems (see diagnostics)
            alpha_confidence=0.05,  # 95% CI
        ),
    )

    alpha = 0.99

    # -------------------------------------------------------------------------
    # Classical-on-discretized diagnostics (explicit discretization)
    # -------------------------------------------------------------------------
    dist = discretize_samples(samples, cfg.discretization)
    ssum = summarize_samples(samples, upper_q=0.999, xmax=float(dist.bounds[1]))

    cmean = classical_mean(dist).mean
    cvar = classical_var(dist, alpha)
    ctvar_val = classical_tvar(dist, alpha)
    ctvar_idx = classical_tvar_index_tail(dist, alpha)

    csm = classical_scaled_mean_amplitude(dist)
    xmin, xmax = float(csm.xmin), float(csm.xmax)
    mean_recon = xmin + (xmax - xmin) * float(csm.a_mean)

    print("=== Example 01: Gamma + Grover-MLE ===")
    print()
    print("=== Discretization / sample sanity ===")
    print("n_bins:", dist.pmf.size)
    print("bounds:", dist.bounds)
    print("samples summary:", ssum)
    print()

    print("=== Classical (ON DISCRETIZED DIST) ===")
    print("Mean:", cmean)
    print("Scaled mean amplitude a_mean:", csm.a_mean)
    print("Mean recon from a_mean:", mean_recon, f"(bounds=({xmin}, {xmax}))")
    print(
        f"VaR({alpha}):",
        cvar.var,
        " bracket:",
        cvar.bracket,
        " idx:",
        cvar.var_bin_index,
    )
    print(
        f"TVaR({alpha}) value-tail:", ctvar_val.tvar, " tail_prob:", ctvar_val.tail_prob
    )
    print(
        f"TVaR({alpha}) index-tail:", ctvar_idx.tvar, " tail_prob:", ctvar_idx.tail_prob
    )
    print()

    # -------------------------------------------------------------------------
    # Quantum run
    # -------------------------------------------------------------------------
    res = estimate_risk_measures(
        {"samples": samples},
        alphas=[alpha],
        config=cfg,
    )

    print("=== Quantum ===")
    print("Mean:", res.mean.mean)
    print("Mean CI:", res.mean.mean_ci)
    print(f"VaR({alpha}):", res.var[alpha].var, " bracket:", res.var[alpha].bracket)
    print(f"TVaR({alpha}):", res.tvar[alpha].tvar)
    print("Total shots used:", res.total_shots_used)
    print("Total circuits run:", res.total_circuits_run)
    print("Mean Grover powers:", res.mean.diagnostics.get("powers"))
    print("Mean early_stop:", res.mean.diagnostics.get("early_stop"))
    print()

    print("=== Relative error vs classical-on-discretized ===")
    print("mean rel_err:", _rel_err(res.mean.mean, cmean))
    print("var  rel_err:", _rel_err(res.var[alpha].var, cvar.var))
    print("tvar rel_err:", _rel_err(res.tvar[alpha].tvar, ctvar_val.tvar))
    print()

    # --- sanity checks ---
    assert res.total_shots_used <= cfg.budget.total_shots
    assert res.mean.mean >= 0.0
    assert res.var[alpha].var >= 0.0

    # Document known inconsistency (will be fixed during TVaR debugging)
    if res.tvar[alpha].tvar < res.var[alpha].bracket[0]:
        print("WARNING: TVaR fell below VaR bracket lower edge.")
        print("  TVaR:", res.tvar[alpha].tvar)
        print("  VaR bracket:", res.var[alpha].bracket)
        print(
            "  This indicates an inconsistency in tail definition/scaling and/or early-stop units."
        )
    else:
        assert res.tvar[alpha].tvar >= res.var[alpha].bracket[0]


if __name__ == "__main__":
    main()
