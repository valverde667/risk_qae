from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, TypedDict


class LossData(TypedDict, total=False):
    """Input container for a 1D loss distribution.

    Exactly one of the following inputs should be provided:
      - samples
      - (pmf and bin_values)
    """

    samples: Sequence[float]
    pmf: Sequence[float]
    bin_values: Sequence[float]
    metadata: dict[str, Any]


# ----------------------------
# Stage 2: quantum-facing types
# ----------------------------


@dataclass(frozen=True)
class CircuitArtifact:
    """A lightweight wrapper around a circuit and its semantic roles.

    `circuit` is intentionally typed as Any so the core package can be imported
    without requiring Qiskit. When Qiskit is installed, this will typically be
    a `qiskit.circuit.QuantumCircuit`.
    """

    circuit: Any
    objective_qubits: Sequence[int]
    ancilla_qubits: Sequence[int] = ()
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EstimationProblemSpec:
    """Provider-agnostic description of an amplitude-estimation problem."""

    state_preparation: Any
    objective_qubits: Sequence[int]
    grover_operator: Any | None = None
    post_processing: Callable[[float], float] | None = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BackendHandle:
    """Container for execution primitives (Sampler/Estimator) and backend metadata.

    This is intentionally minimal and loosely typed so it can represent:
      - Quantum Rings QrSamplerV2/QrEstimatorV2
      - Qiskit Aer primitives
      - other Qiskit-compatible providers
    """

    sampler: Any | None = None
    estimator: Any | None = None
    backend: Any | None = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]
    provider: str | None = None


class SupportsBuildBackend(Protocol):
    def get_backend(self, **kwargs: Any) -> BackendHandle: ...
