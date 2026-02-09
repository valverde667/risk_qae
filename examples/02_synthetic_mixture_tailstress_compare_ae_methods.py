"""Example 02 — Tail-stress mixture + compare AE modes.

We build a mixture distribution with a small-probability extreme-loss component,
then run the workflow twice:
  (A) budgeted_fixed_schedule  (shots-first sampling baseline)
  (B) grover_mle              (Grover powers + MLE; QFT-free)

This is useful for:
- demonstrating why Grover schedules exist
- surfacing diagnostic traces in VaR bisection and TVaR tails
- comparing against classical-on-discretized baselines
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


def make_tail_stress_samples(rng: np.random.Generator, n: int = 80_000) -> np.ndarray:
    """Mixture: mostly moderate gamma, rare extreme lognormal."""
    p_extreme = 0.02
    u = rng.random(n)
    base = rng.gamma(shape=2.2, scale=80.0, size=n)
    extreme = rng.lognormal(mean=7.3, sigma=0.55, size=n)  # very large tail
    samples = np.where(u < p_extreme, extreme, base)
    return samples.astype(float)


def run_one(samples: np.ndarray, *, method: str) -> None:
    # Common knobs
    common = dict(
        backend=BackendConfig(provider="aer"),
        discretization=DiscretizationConfig(n_index_qubits=7),  # 128 bins
        budget=BudgetConfig(
            total_shots=12_000, shots_per_call=1_000, max_circuit_calls=60, seed=7
        ),
        value_encoding=ValueEncodingConfig(method="piecewise_prefix", n_segments=16),
    )

    if method == "budgeted_fixed_schedule":
        ae = AEConfig(method="budgeted_fixed_schedule")
    elif method == "grover_mle":
        ae = AEConfig(
            method="grover_mle",
            grover_mle_grid_size=1500,
            grover_powers=(0, 1, 2, 4, 8),
            # keep early stop off here to compare fairly; turn on later if desired
            epsilon_target=None,
            alpha_confidence=0.05,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    cfg = RiskQAEConfig(**common, ae=ae)

    # -------------------------------------------------------------------------
    # Classical-on-discretized diagnostics (explicit discretization)
    # -------------------------------------------------------------------------
    dist = discretize_samples(samples, cfg.discretization)
    ssum = summarize_samples(samples, upper_q=0.999, xmax=float(dist.bounds[1]))

    cmean = classical_mean(dist).mean
    csm = classical_scaled_mean_amplitude(dist)
    xmin, xmax = float(csm.xmin), float(csm.xmax)
    mean_recon = xmin + (xmax - xmin) * float(csm.a_mean)

    print()
    print(f"=== {method} ===")
    print("--- Discretization / sample sanity ---")
    print("n_bins:", dist.pmf.size)
    print("bounds:", dist.bounds)
    print("samples summary:", ssum)
    print("--- Classical (ON DISCRETIZED DIST) ---")
    print("Mean:", cmean)
    print("Scaled mean amplitude a_mean:", csm.a_mean)
    print("Mean recon from a_mean:", mean_recon)
    print()

    # -------------------------------------------------------------------------
    # Quantum run
    # -------------------------------------------------------------------------
    res = estimate_risk_measures({"samples": samples}, alphas=[0.95, 0.99], config=cfg)

    print("--- Quantum ---")
    print("Mean:", res.mean.mean, "CI:", res.mean.mean_ci)

    print()
    print("--- Per-alpha comparisons (quantum vs classical-on-discretized) ---")
    for a in (0.95, 0.99):
        # Classical
        cvar = classical_var(dist, a)
        ctvar_val = classical_tvar(dist, a)
        ctvar_idx = classical_tvar_index_tail(dist, a)

        # Quantum
        vr = res.var[a]
        tr = res.tvar[a]

        print(f"[alpha={a}]")
        print(
            "  Classical VaR:",
            cvar.var,
            "bracket:",
            cvar.bracket,
            "idx:",
            cvar.var_bin_index,
        )
        print(
            "  Classical TVaR (value-tail):",
            ctvar_val.tvar,
            "tail_prob:",
            ctvar_val.tail_prob,
        )
        print(
            "  Classical TVaR (index-tail):",
            ctvar_idx.tvar,
            "tail_prob:",
            ctvar_idx.tail_prob,
        )

        print(
            "  Quantum   VaR:",
            vr.var,
            "bracket:",
            vr.bracket,
            "iters:",
            vr.diagnostics.get("iters"),
        )
        print("  Quantum  TVaR:", tr.tvar, "tail_prob:", tr.tail_prob)

        print("  RelErr VaR  (quant vs classical):", _rel_err(vr.var, cvar.var))
        print("  RelErr TVaR (quant vs classical):", _rel_err(tr.tvar, ctvar_val.tvar))

        # Document known inconsistency rather than failing the example
        if tr.tvar < vr.bracket[0]:
            print("  WARNING: TVaR fell below VaR bracket lower edge.")
            print("    TVaR:", tr.tvar)
            print("    VaR bracket:", vr.bracket)
            print(
                "    This indicates an inconsistency in tail definition/scaling and/or early-stop units."
            )
        print()

    print("Total shots used:", res.total_shots_used)
    print("Total circuits run:", res.total_circuits_run)

    # show Grover powers used (if any)
    print("Mean powers:", res.mean.diagnostics.get("powers"))


def main() -> None:
    rng = np.random.default_rng(1)
    samples = make_tail_stress_samples(rng)

    print("=== Example 02: Tail-stress mixture ===")
    print(
        "n_samples:",
        samples.size,
        "min/median/max:",
        float(samples.min()),
        float(np.median(samples)),
        float(samples.max()),
    )

    run_one(samples, method="budgeted_fixed_schedule")
    run_one(samples, method="grover_mle")


if __name__ == "__main__":
    main()
