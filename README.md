# risk-qae

A research-oriented codebase for computing **mean**, **VaR**, and **TVaR** for a **1D loss distribution** using a
**shots-budgeted quantum execution pipeline** designed to stay compatible with **Qiskit primitives** and to run on
**Quantum Rings** (tensor-network simulation) as well as other Qiskit backends.

The project supports:
- **Data-driven discretization** of sample losses into a low-qubit PMF (uniform bins on `2**n`)
- **Classical reference metrics** (mean / VaR / TVaR) on the discretized distribution
- **Quantum circuit construction** (PMF state prep + indicators + value encoding)
- **Budgeted end-to-end execution** via Qiskit primitives (Sampler-style)
- **Configurable value encoding**, including a faster **piecewise prefix** encoder (recommended)
- **Grover-schedule / iterative AE (MLE)** execution mode with optional **early stopping** and confidence intervals

> Note on “QAE”: this repository currently implements a **QFT-free amplitude estimation** method
> (**Grover powers + maximum-likelihood fit**) in addition to a simple shots-first sampling baseline.
> The canonical **QFT/phase-estimation** QAE circuit is not implemented yet.

---

## Project stages

### Stage 1 (implemented)
- Discretize samples into a PMF on `2**n` uniform bins (low-qubit defaults).
- Data-driven bounds with percentile clipping for stable scaling.
- Classical reference implementations for mean / VaR / TVaR from the discretized distribution.

### Stage 2 (implemented)
- Circuit factories (Qiskit optional):
  - state preparation `A` from a PMF
  - index-threshold indicator circuits (<=k, >=k)
  - value-encoding circuits for mean/tail components
- Estimation-problem builders (provider-agnostic specs; optional conversion to Qiskit `EstimationProblem`).

### Stage 3 (implemented)
- Shots-budgeted execution runner using primitives (Sampler-style).
- VaR outer-loop via bisection on CDF queries.
- TVaR via tail probability + scaled tail component.

### Stage 4 (implemented)
- Configurable value encoding via `cfg.value_encoding`:
  - `method="piecewise_prefix"` with `n_segments` (recommended)
  - `method="naive_table"` (baseline; slower and larger circuits)
- Piecewise prefix value encoding to reduce circuit size and compile time.

### Stage 5 (implemented)
- **Grover-schedule / iterative AE (MLE)** runner:
  - builds Grover operator from `A` and the objective qubit
  - executes multiple Grover powers under a strict shot budget
  - fits amplitude via maximum likelihood and returns a CI (Wilks-style)
  - supports `epsilon_target` / `alpha_confidence` early stopping
- Runner selection via `cfg.ae.method`:
  - `budgeted_fixed_schedule` (shots-first sampling baseline)
  - `grover_mle` (Grover powers + MLE)
- Correct **whole-request accounting** for `total_shots_used` and `total_circuits_run`
  (including TVaR), plus diagnostics surfacing for VaR bisection and TVaR tail subcalls.

### Next (planned)
- Implement canonical **QFT/phase-estimation** QAE (optional), while keeping the same public API.
- Harden the **Quantum Rings backend adapter** and add an integration smoke test.

---

## Installation

Base install (interfaces; **no Qiskit dependency**):
```bash
pip install -e .
```

## Quick smoke test script to run.
``` python 
import numpy as np

from risk_qae import RiskQAEConfig, estimate_risk_measures
from risk_qae.config import (
    AEConfig,
    BackendConfig,
    DiscretizationConfig,
    BudgetConfig,
    ValueEncodingConfig,
)

# --- generate a synthetic loss distribution (positive, right-tailed) ---
rng = np.random.default_rng(0)
samples = rng.gamma(shape=2.0, scale=100.0, size=50_000)

# --- configure a "fast" run (64 bins) ---
cfg = RiskQAEConfig(
    backend=BackendConfig(provider="aer"),
    discretization=DiscretizationConfig(n_index_qubits=6),  # 2^6 = 64 bins
    budget=BudgetConfig(
        total_shots=10_000,
        shots_per_call=1_000,
        max_circuit_calls=50,
    ),
    # Use Grover schedule + MLE amplitude estimation
    ae=AEConfig(
        method="grover_mle",
        epsilon_target=25.0,
        alpha_confidence=0.05,
        grover_mle_grid_size=1200,
        grover_powers=(0, 1, 2, 4, 8),
    ),
    # (Recommended) keep the faster value encoding
    value_encoding=ValueEncodingConfig(method="piecewise_prefix", n_segments=16),
)

# --- run full workflow: mean + VaR + TVaR ---
res = estimate_risk_measures(
    {"samples": samples},
    alphas=[0.99],
    config=cfg,
)

# --- prints / diagnostics ---
print("Mean powers used:", res.mean.diagnostics.get("powers"))
print("Mean:", res.mean.mean)
print("VaR(0.99):", res.var[0.99].var)
print("TVaR(0.99):", res.tvar[0.99].tvar)
print("Total shots used:", res.total_shots_used)
print("Total circuits run:", res.total_circuits_run)

trace0 = res.var[0.99].diagnostics.get("bisection_trace", [])
first_step_diag = (trace0[0].get("ae_diag") if trace0 else {}) or {}

print()
print("VaR first-step powers:", first_step_diag.get("powers"))
print("VaR first-step early_stop:", first_step_diag.get("early_stop"))
print("TVaR tail_prob early_stop:", res.tvar[0.99].diagnostics.get("tail_prob_early_stop"))
print("TVaR tail_component early_stop:", res.tvar[0.99].diagnostics.get("tail_component_early_stop"))
print()
print("Mean early_stop:", res.mean.diagnostics.get("early_stop"))
print("VaR early_stop:", res.var[0.99].diagnostics.get("early_stop"))
print("TVaR early_stop:", res.tvar[0.99].diagnostics.get("early_stop"))

# --- basic sanity assertions (cheap + high value) ---
assert res.total_shots_used <= cfg.budget.total_shots
assert res.mean.mean >= 0.0
assert res.var[0.99].var >= 0.0

# IMPORTANT: with discretized VaR reported as the *upper bin edge*, TVaR>=VaR is not guaranteed.
# Use a discretization-consistent check instead:
assert res.tvar[0.99].tvar >= res.var[0.99].bracket[0]
```