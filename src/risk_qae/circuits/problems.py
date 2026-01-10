from __future__ import annotations

from typing import Callable

import numpy as np

from ..discretization.histogram import DiscretizedDistribution
from ..types import EstimationProblemSpec
from .indicator import build_indicator_geq_index, build_indicator_leq_index
from .stateprep import build_state_preparation
from .value_encoding import (
    build_scaled_value_rotation,
    build_scaled_value_rotation_piecewise_prefix,
)
from ..config import ValueEncodingConfig


def _require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for circuit construction. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


def build_mean_problem(
    dist: DiscretizedDistribution,
    *,
    value_encoding: ValueEncodingConfig | None = None,
) -> EstimationProblemSpec:
    """Build an estimation problem for E[X] using value encoding.

    The objective probability is:
        a = E[g(X)] where g maps losses into [0,1] using dist.bounds.

    Post-processing returns:
        E[X] = x_min + (x_max - x_min) * a
    """
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    A = build_state_preparation(dist.pmf)
    ve = value_encoding or ValueEncodingConfig()

    if ve.method == "piecewise_prefix":
        V = build_scaled_value_rotation_piecewise_prefix(
            dist.n_index_qubits,
            dist.bin_values,
            dist.bounds,
            n_segments=ve.n_segments,
        )
    else:
        V = build_scaled_value_rotation(
            dist.n_index_qubits,
            dist.bin_values,
            dist.bounds,
            method=ve.method,
        )

    n = dist.n_index_qubits
    qc = QuantumCircuit(n + 1, name="mean_problem")
    qc.compose(A.circuit, qubits=qc.qubits[:n], inplace=True)
    qc.compose(V.circuit, qubits=qc.qubits, inplace=True)

    x_min, x_max = dist.bounds
    denom = float(x_max - x_min)

    def post(a: float) -> float:
        return float(x_min) + denom * float(a)

    return EstimationProblemSpec(
        state_preparation=qc,
        objective_qubits=(n,),
        grover_operator=None,
        post_processing=post,
        metadata={"type": "mean", "bounds": dist.bounds},
    )


def build_cdf_problem(dist: DiscretizedDistribution, k: int) -> EstimationProblemSpec:
    """Build an estimation problem for CDF(k) = P(I <= k)."""
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    n = dist.n_index_qubits
    A = build_state_preparation(dist.pmf)
    Ind = build_indicator_leq_index(n, k)

    qc = QuantumCircuit(n + 1, name=f"cdf_leq_{k}")
    qc.compose(A.circuit, qubits=qc.qubits[:n], inplace=True)
    qc.compose(Ind.circuit, qubits=qc.qubits, inplace=True)

    return EstimationProblemSpec(
        state_preparation=qc,
        objective_qubits=(n,),
        metadata={"type": "cdf", "k": k},
    )


def build_tail_prob_problem(
    dist: DiscretizedDistribution, k: int
) -> EstimationProblemSpec:
    """Build an estimation problem for tail probability P(I >= k)."""
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    n = dist.n_index_qubits
    A = build_state_preparation(dist.pmf)
    Ind = build_indicator_geq_index(n, k)

    qc = QuantumCircuit(n + 1, name=f"tail_geq_{k}")
    qc.compose(A.circuit, qubits=qc.qubits[:n], inplace=True)
    qc.compose(Ind.circuit, qubits=qc.qubits, inplace=True)

    return EstimationProblemSpec(
        state_preparation=qc,
        objective_qubits=(n,),
        metadata={"type": "tail_prob", "k": k},
    )


def build_tail_scaled_component_problem(
    dist: DiscretizedDistribution,
    k: int,
    *,
    value_encoding: ValueEncodingConfig | None = None,
) -> EstimationProblemSpec:
    """Build an estimation problem for E[g(X) * 1_{I >= k}].

    This provides the 'scaled tail component' used to compute TVaR:

        E[X * 1_tail] = x_min * P(tail) + (x_max - x_min) * E[g(X)*1_tail]

    where (x_min, x_max) are the clipped bounds used for scaling.
    """
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    n = dist.n_index_qubits
    A = build_state_preparation(dist.pmf)

    # Zero out bins below k by setting value to x_min (=> g=0).
    x_min, _ = dist.bounds
    vals = np.array(dist.bin_values, dtype=float)
    vals[:k] = float(x_min)

    ve = value_encoding or ValueEncodingConfig()

    if ve.method == "piecewise_prefix":
        V = build_scaled_value_rotation_piecewise_prefix(
            n,
            vals,
            dist.bounds,
            n_segments=ve.n_segments,
        )
    else:
        V = build_scaled_value_rotation(
            n,
            vals,
            dist.bounds,
            method=ve.method,
        )

    qc = QuantumCircuit(n + 1, name=f"tail_scaled_component_{k}")
    qc.compose(A.circuit, qubits=qc.qubits[:n], inplace=True)
    qc.compose(V.circuit, qubits=qc.qubits, inplace=True)

    return EstimationProblemSpec(
        state_preparation=qc,
        objective_qubits=(n,),
        metadata={"type": "tail_scaled_component", "k": k, "bounds": dist.bounds},
    )


def to_qiskit_estimation_problem(spec: EstimationProblemSpec):
    """Convert an EstimationProblemSpec to qiskit_algorithms' EstimationProblem.

    This is an optional helper; it requires qiskit-algorithms to be installed.
    """
    try:
        from qiskit_algorithms import EstimationProblem  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "qiskit-algorithms is required for amplitude estimation. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc

    return EstimationProblem(
        state_preparation=spec.state_preparation,
        objective_qubits=list(spec.objective_qubits),
        grover_operator=spec.grover_operator,
        post_processing=spec.post_processing,
    )
