"""risk_qae

A research-oriented codebase for risk measures (mean, VaR, TVaR) via Quantum Amplitude Estimation.

- Stage 1: discretization + classical reference metrics
- Stage 2: circuit factories + estimation-problem builders (Qiskit optional)
- Stage 3: budgeted amplitude-estimation execution (planned)
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
]
