from .classical import ClassicalMeanResult, ClassicalTVaRResult, ClassicalVaRResult
from .classical import classical_mean, classical_tvar, classical_var

# Quantum-facing results/functions (require qiskit only at execution time)
from .quantum import MeanResult, RiskMeasuresResult, TVaRResult, VaRResult
from .quantum import estimate_mean, estimate_risk_measures, estimate_tvar, estimate_var

__all__ = [
    "ClassicalMeanResult",
    "ClassicalVaRResult",
    "ClassicalTVaRResult",
    "classical_mean",
    "classical_var",
    "classical_tvar",
    "MeanResult",
    "VaRResult",
    "TVaRResult",
    "RiskMeasuresResult",
    "estimate_mean",
    "estimate_var",
    "estimate_tvar",
    "estimate_risk_measures",
]
