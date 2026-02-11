# src/risk_qae/circuits/grover.py

from __future__ import annotations

from typing import Sequence


def _require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for Grover operator construction. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


def _mcz(qc, controls: Sequence[int], target: int) -> None:
    """Multi-controlled Z via H - MCX - H (no ancilla)."""
    qc.h(target)
    # Qiskit will decompose this as needed; Aer knows the resulting ops.
    qc.mcx(list(controls), target, ancilla_qubits=None, mode="noancilla")
    qc.h(target)


def _phase_flip_on_all_zeros(qc, qubits: Sequence[int]) -> None:
    """
    S0 = I - 2|0...0><0...0| on given qubits (up to global phase).
    Implemented as X...X, then phase flip on |11..1>, then X...X.
    """
    qs = list(qubits)
    n = len(qs)
    if n == 0:
        return
    if n == 1:
        qc.z(qs[0])
        return

    for q in qs:
        qc.x(q)

    target = qs[-1]
    controls = qs[:-1]
    _mcz(qc, controls, target)

    for q in qs:
        qc.x(q)


def _phase_flip_on_objective_one(qc, objective_qubits: Sequence[int]) -> None:
    """
    Sf = I - 2*Π_good where "good" means objective qubit(s)=1.

    - 1 objective qubit: Z on it
    - multiple: flip phase only when ALL objective qubits are |1>
    """
    objs = [int(q) for q in objective_qubits]
    if len(objs) == 0:
        raise ValueError("objective_qubits must be non-empty.")
    if len(objs) == 1:
        qc.z(objs[0])
        return

    target = objs[-1]
    controls = objs[:-1]
    _mcz(qc, controls, target)


def build_grover_operator(state_preparation, objective_qubits: Sequence[int]):
    """
    Build Grover operator for amplitude amplification:

        Q = - A S0 A† Sf

    We implement the circuit (right-to-left application):
        Sf -> A† -> S0 -> A

    Global phase is irrelevant for sampling, so we ignore the leading '-'.
    """
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    if state_preparation is None:
        raise ValueError("state_preparation must be provided.")
    if len(objective_qubits) < 1:
        raise ValueError("objective_qubits must be non-empty.")

    A = state_preparation
    try:
        A_dag = A.inverse()
    except Exception as exc:
        raise ValueError(
            "state_preparation is not invertible. Ensure it has no measurements/resets "
            "and is unitary so Grover amplification can be constructed."
        ) from exc

    n_qubits = int(A.num_qubits)
    qc = QuantumCircuit(n_qubits, name="grover_Q")
    qargs = list(range(n_qubits))

    # Sf: phase flip on good states (objective=1)
    _phase_flip_on_objective_one(qc, objective_qubits)

    # A† (inline circuit, avoids opaque *_dg gate names)
    qc.compose(A_dag, qubits=qargs, inplace=True)

    # S0: phase flip on |0...0> (across ALL qubits A acts on)
    _phase_flip_on_all_zeros(qc, qargs)

    # A (inline circuit)
    qc.compose(A, qubits=qargs, inplace=True)

    return qc
