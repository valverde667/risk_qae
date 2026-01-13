from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import CircuitArtifact


def _require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for circuit construction. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


def build_state_preparation(
    pmf: Sequence[float],
    *,
    method: str = "state_preparation",
) -> CircuitArtifact:
    """Build the state-preparation operator A for a discrete PMF.

    Goal:
        A|0...0> = sum_i sqrt(p_i) |i>

    Parameters
    ----------
    pmf:
        Probabilities over N=2**n computational basis states.
    method:
        - "state_preparation": uses qiskit.circuit.library.StatePreparation on amplitude vector sqrt(pmf)
          (unitary/invertible; required for Grover-based AE)

    Returns
    -------
    CircuitArtifact
        circuit: QuantumCircuit with n qubits
        objective_qubits: empty for pure state-prep (objective defined by downstream problems)

    Notes
    -----
    For Grover-based amplitude estimation, A must be unitary so A.inverse() exists.
    Qiskit's Initialize can include reset operations (non-unitary) and is not invertible.
    """
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    p = np.asarray(pmf, dtype=float)
    if p.ndim != 1:
        raise ValueError("pmf must be a 1D sequence.")
    if np.any(p < 0):
        raise ValueError("pmf must be non-negative.")
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("pmf must have a positive, finite sum.")
    p = p / s

    amps = np.sqrt(p).astype(complex)
    n = int(np.log2(len(amps)))
    if 2**n != len(amps):
        raise ValueError("pmf length must be a power of two (N = 2**n_index_qubits).")

    qc = QuantumCircuit(n, name="A_stateprep")
    prep = StatePreparation(amps)
    qc.append(prep, qc.qubits)

    # Aer can't assemble opaque "state_preparation" instructions; decompose to basis gates.
    qc = qc.decompose(reps=10)

    return CircuitArtifact(
        circuit=qc,
        objective_qubits=(),
        ancilla_qubits=(),
        metadata={"method": method, "n": n},
    )
