"""Circuit construction utilities (Stage 2).

These modules are written to keep the core package importable without Qiskit.
Circuit builders will raise ImportError with an actionable message if Qiskit
(and, when relevant, qiskit-algorithms) is not installed.

Install quantum extras:
    pip install 'risk_qae[quantum]'
"""

from .stateprep import build_state_preparation
from .indicator import build_indicator_leq_index, build_indicator_geq_index
from .value_encoding import build_scaled_value_rotation
from .problems import (
    build_mean_problem,
    build_cdf_problem,
    build_tail_prob_problem,
    build_tail_scaled_component_problem,
    to_qiskit_estimation_problem,
)

__all__ = [
    "build_state_preparation",
    "build_indicator_leq_index",
    "build_indicator_geq_index",
    "build_scaled_value_rotation",
    "build_mean_problem",
    "build_cdf_problem",
    "build_tail_prob_problem",
    "build_tail_scaled_component_problem",
    "to_qiskit_estimation_problem",
]
