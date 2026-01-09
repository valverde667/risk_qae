# risk-qae

Compute mean, VaR, and TVaR for a 1D loss distribution. Scaffold currently includes discretization + classical reference metrics; quantum modules come next.
# risk-qae

A research-oriented codebase for computing **mean**, **VaR**, and **TVaR** for a **1D loss distribution** using
**Quantum Amplitude Estimation (QAE)**. The codebase is designed to stay compatible with **Qiskit primitives**
so it can run on **Quantum Rings** (tensor-network simulation) as well as other Qiskit backends.

## Project stages

### Stage 1 (implemented)
- Discretize samples into a PMF on `2**n` uniform bins (low-qubit defaults).
- Data-driven bounds with percentile clipping for stable scaling.
- Classical reference implementations for mean / VaR / TVaR from the discretized distribution.

### Stage 2 (implemented)
- Circuit factories (Qiskit optional):
  - state preparation `A` from a PMF
  - index-threshold indicator circuits (<=k, >=k)
  - value-encoding circuits (table-lookup controlled rotations)
- Estimation-problem builders (provider-agnostic specs; optional conversion to Qiskit `EstimationProblem`).

### Stage 3 (planned)
- Shots-budgeted amplitude-estimation execution using primitives (Sampler/Estimator)
- VaR outer-loop search that calls AE for CDF queries
- TVaR via tail probability + scaled tail component

## Installation

Base install (Stage 1 + imports for Stage 2 interfaces; **no Qiskit dependency**):
```bash
pip install -e .
```

Quantum extras (to actually build circuits and run QAE):
```bash
pip install -e ".[quantum]"
```

## Quick smoke test (Stage 1)

```bash
python - << 'PY'
import numpy as np
from risk_qae.discretization.histogram import HistogramDiscretizer
from risk_qae.metrics.classical import classical_mean, classical_var, classical_tvar

rng = np.random.default_rng(0)
samples = rng.gamma(shape=2.0, scale=100.0, size=50_000)

disc = HistogramDiscretizer()
dist = disc.fit({"samples": samples})  # defaults: n_index_qubits=8, clip=(0.1,99.9)

print("pmf sum:", float(dist.pmf.sum()))
print("bounds:", dist.bounds)

print("mean:", classical_mean(dist))
print("VaR(0.99):", classical_var(dist, 0.99))
print("TVaR(0.99):", classical_tvar(dist, 0.99))
PY
```
