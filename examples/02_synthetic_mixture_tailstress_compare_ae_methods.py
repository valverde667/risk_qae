"""Example 02 — Tail-stress mixture + compare AE modes.

We build a mixture distribution with a small-probability extreme-loss component,
then run the workflow twice:
  (A) budgeted_fixed_schedule  (shots-first sampling baseline)
  (B) grover_mle              (Grover powers + MLE; QFT-free)

This is useful for:
- demonstrating why Grover schedules exist
- surfacing diagnostic traces in VaR bisection and TVaR tails
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

    res = estimate_risk_measures({"samples": samples}, alphas=[0.95, 0.99], config=cfg)

    print()
    print(f"=== {method} ===")
    print("Mean:", res.mean.mean, "CI:", res.mean.mean_ci)
    for a in (0.95, 0.99):
        vr = res.var[a]
        tr = res.tvar[a]
        print(
            f"VaR({a}):",
            vr.var,
            " bracket:",
            vr.bracket,
            " iters:",
            vr.diagnostics.get("iters"),
        )
        print(f"TVaR({a}):", tr.tvar, " tail_prob:", tr.tail_prob)
        # discretization-consistent check
        assert tr.tvar >= vr.bracket[0]
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
