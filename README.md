# risk-qae

A research-oriented codebase for computing **mean**, **VaR**, and **TVaR** for a **1D loss distribution** using a
**shots-budgeted quantum execution pipeline** designed to stay compatible with **Qiskit primitives** and to run on
**Quantum Rings** (tensor-network simulation) as well as other Qiskit backends.

The project supports:
- **Data-driven discretization** of sample losses into a low-qubit PMF (uniform bins on `2**n`)
- **Classical reference metrics** (mean / VaR / TVaR) on the discretized distribution
- **Quantum circuit construction** (state prep + indicators + value encoding)
- **Budgeted execution** using Qiskit primitives (Sampler-style) to estimate mean / VaR / TVaR end-to-end
- **Configurable value encoding**, including a faster **piecewise prefix** encoder (recommended)

> Note: The current “AE” execution is a **shots-first sampling estimator** (Monte Carlo amplitude estimation via a Sampler primitive).
> A Grover-schedule / iterative AE upgrade is planned, while keeping the same public API and budget controls.

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

### Next (planned)
- Replace the sampling estimator with a **Grover-schedule / iterative AE** implementation under the same `BudgetConfig`.
- Harden the **Quantum Rings backend adapter** and add an integration smoke test.

---

## Installation

Base install (Stage 1 + interfaces; **no Qiskit dependency**):
```bash
pip install -e .
```

## Quick Smoke Test
``` python 
import numpy as np

from risk_qae import RiskQAEConfig, estimate_risk_measures
from risk_qae.config import BackendConfig, BudgetConfig, DiscretizationConfig, ValueEncodingConfig

# --- synthetic loss distribution (positive, right-tailed) ---
rng = np.random.default_rng(0)
samples = rng.gamma(shape=2.0, scale=100.0, size=50_000)

cfg = RiskQAEConfig(
    backend=BackendConfig(provider="aer"),
    discretization=DiscretizationConfig(n_index_qubits=6),  # 64 bins (fast)
    budget=BudgetConfig(total_shots=10_000, shots_per_call=1_000, max_circuit_calls=50),
    value_encoding=ValueEncodingConfig(method="piecewise_prefix", n_segments=16),
)

res = estimate_risk_measures({"samples": samples}, alphas=[0.99], config=cfg)

print("Mean:", res.mean.mean)
print("VaR(0.99):", res.var[0.99].var)
print("TVaR(0.99):", res.tvar[0.99].tvar)
print("Total shots used:", res.total_shots_used)
print("Total circuits run:", res.total_circuits_run)

# Basic sanity checks
assert res.total_shots_used <= cfg.budget.total_shots
assert res.tvar[0.99].tvar >= res.var[0.99].bracket[0]  # discretization-consistent check
```