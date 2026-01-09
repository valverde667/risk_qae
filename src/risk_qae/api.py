"""Public, user-facing API.

This module exposes the primary entry points promised by the API spec.
Importing this module does **not** require Qiskit. Quantum execution only
happens when you call the estimation functions and have installed the
appropriate extras.
"""

from __future__ import annotations

from .metrics.quantum import (
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
    "MeanResult",
    "VaRResult",
    "TVaRResult",
    "RiskMeasuresResult",
    "estimate_mean",
    "estimate_var",
    "estimate_tvar",
    "estimate_risk_measures",
]
