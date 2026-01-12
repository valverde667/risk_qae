# Build a deliberately simple grover operator.

from __future__ import annotations

from typing import Iterable, Sequence


def _require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for Grover operator construction. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


def build_s_chi(n_qubits: int, objective_qubits: Sequence[int]):
    """Phase flip on 'good' states. For objective=1 marking, Z on objective qubit(s) suffices."""
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name="S_chi")
    for q in objective_qubits:
        qc.z(q)
    return qc


def build_s0(n_qubits: int):
    """Reflection about |0...0>: S0 = I - 2|0><0| (global phase ignored)."""
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name="S0")

    # For 1 qubit: S0 is (up to global phase) Z.
    if n_qubits == 1:
        qc.z(0)
        return qc

    qubits = list(range(n_qubits))
    last = qubits[-1]
    ctrls = qubits[:-1]

    # Flip |0...0> -> |1...1>
    for q in qubits:
        qc.x(q)

    # Apply phase flip on |1...1> via MCZ (H, MCX, H on last)
    qc.h(last)
    qc.mcx(ctrls, last, mode="noancilla")
    qc.h(last)

    # Unflip
    for q in qubits:
        qc.x(q)

    return qc


def build_grover_operator(A, objective_qubits: Sequence[int]):
    """Build Grover iterate Q = A S0 A^† S_chi (global phase ignored)."""
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    n_qubits = A.num_qubits
    S0 = build_s0(n_qubits)
    S_chi = build_s_chi(n_qubits, objective_qubits)

    qc = QuantumCircuit(n_qubits, name="Grover_Q")
    qc.compose(A, inplace=True)
    qc.compose(S0, inplace=True)
    qc.compose(A.inverse(), inplace=True)
    qc.compose(S_chi, inplace=True)
    return qc


def apply_power(qc, Q, power: int):
    """Append Q^power onto qc."""
    if power <= 0:
        return
    for _ in range(int(power)):
        qc.compose(Q, inplace=True)
