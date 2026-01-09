"""risk_qae

A research-oriented codebase for risk measures (mean, VaR, TVaR) via a
shots-first "amplitude estimation" architecture compatible with Quantum Rings
through Qiskit Primitives.

- Stage 1: discretization + classical reference metrics
- Stage 2: circuit factories + estimation-problem builders (Qiskit optional)
- Stage 3: budgeted execution runner + VaR/TVaR orchestration (implemented)
"""

from .config import (
    AEConfig,
    BackendConfig,
    BoundsConfig,
    BudgetConfig,
    DiagnosticsConfig,
    DiscretizationConfig,
    RiskQAEConfig,
    VaRSearchConfig,
)
from .types import BackendHandle, CircuitArtifact, EstimationProblemSpec, LossData

# Public API (importable without Qiskit; execution requires quantum extras)
from .api import (
    MeanResult,
    RiskMeasuresResult,
    TVaRResult,
    VaRResult,
    estimate_mean,
    estimate_risk_measures,
    estimate_tvar,
    estimate_var,
)

__all__ = [
    "LossData",
    "CircuitArtifact",
    "EstimationProblemSpec",
    "BackendHandle",
    "BoundsConfig",
    "DiscretizationConfig",
    "AEConfig",
    "BudgetConfig",
    "VaRSearchConfig",
    "BackendConfig",
    "DiagnosticsConfig",
    "RiskQAEConfig",
    "MeanResult",
    "VaRResult",
    "TVaRResult",
    "RiskMeasuresResult",
    "estimate_mean",
    "estimate_var",
    "estimate_tvar",
    "estimate_risk_measures",
]
