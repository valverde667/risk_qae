from __future__ import annotations

from typing import Iterable, Sequence

from ..types import CircuitArtifact


def _require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for circuit construction. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


def _indices_leq(k: int) -> Iterable[int]:
    return range(0, k + 1)


def _indices_geq(k: int, n_states: int) -> Iterable[int]:
    return range(k, n_states)


def _apply_mark_for_index(qc, index_qubits, flag_qubit, i: int) -> None:
    """Flip flag_qubit iff index_qubits encode integer i (little-endian)."""
    # Qiskit uses little-endian convention for integer value if we treat qubit 0 as LSB.
    bits = [(i >> b) & 1 for b in range(len(index_qubits))]
    # Convert 0-controls to 1-controls by X pre/post.
    zeros = [q for q, bit in zip(index_qubits, bits) if bit == 0]
    for q in zeros:
        qc.x(q)
    qc.mcx(index_qubits, flag_qubit)
    for q in zeros:
        qc.x(q)


def build_indicator_leq_index(
    n_index_qubits: int,
    k: int,
    *,
    method: str = "auto",
) -> CircuitArtifact:
    """Indicator circuit that flips a flag qubit when I <= k.

    Layout:
        index register: n_index_qubits qubits
        flag qubit: 1 qubit (objective), appended as the last qubit

    Parameters
    ----------
    n_index_qubits:
        Number of qubits in the index register.
    k:
        Threshold index (inclusive).
    method:
        - "auto": attempt to use Qiskit library integer comparator; fallback to naive enumeration.
        - "naive_enumeration": mark all basis states i<=k with multi-controlled-X

    Notes
    -----
    The naive enumeration method is O(2^n) gates and is intended for *low qubit counts*,
    which aligns with the project v1 defaults (e.g., n=6..10).
    """
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    if n_index_qubits <= 0:
        raise ValueError("n_index_qubits must be >= 1.")
    n_states = 2**n_index_qubits
    if not (0 <= k < n_states):
        raise ValueError(f"k must be in [0, {n_states-1}].")

    qc = QuantumCircuit(n_index_qubits + 1, name=f"ind_leq_{k}")
    index_qubits = list(qc.qubits[:n_index_qubits])
    flag = qc.qubits[n_index_qubits]

    if method in ("auto", "library"):
        # Best-effort: use Qiskit's arithmetic comparator if present.
        try:
            from qiskit.circuit.library.arithmetic import IntegerComparator  # type: ignore

            comp = IntegerComparator(
                num_state_qubits=n_index_qubits, value=k + 1, geq=False
            )  # geq False => < value
            # IntegerComparator typically uses an ancilla output qubit; we map it to our flag.
            # If signature differs, we'll fall back.
            qc.compose(comp, qubits=qc.qubits, inplace=True)
            return CircuitArtifact(
                circuit=qc,
                objective_qubits=(n_index_qubits,),
                ancilla_qubits=(),
                metadata={"k": k, "relation": "<=", "method": "library"},
            )
        except Exception:
            if method == "library":
                raise

    # Fallback: naive enumeration
    for i in _indices_leq(k):
        _apply_mark_for_index(qc, index_qubits, flag, i)

    return CircuitArtifact(
        circuit=qc,
        objective_qubits=(n_index_qubits,),
        ancilla_qubits=(),
        metadata={"k": k, "relation": "<=", "method": "naive_enumeration"},
    )


def build_indicator_geq_index(
    n_index_qubits: int,
    k: int,
    *,
    method: str = "auto",
) -> CircuitArtifact:
    """Indicator circuit that flips a flag qubit when I >= k."""
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit

    if n_index_qubits <= 0:
        raise ValueError("n_index_qubits must be >= 1.")
    n_states = 2**n_index_qubits
    if not (0 <= k < n_states):
        raise ValueError(f"k must be in [0, {n_states-1}].")

    qc = QuantumCircuit(n_index_qubits + 1, name=f"ind_geq_{k}")
    index_qubits = list(qc.qubits[:n_index_qubits])
    flag = qc.qubits[n_index_qubits]

    # For now, reuse naive enumeration (good enough for low n).
    # Library comparator support can be added later (v2).
    if method not in ("auto", "naive_enumeration"):
        raise ValueError(
            "Only method='auto' or 'naive_enumeration' is supported in v0.1."
        )

    for i in _indices_geq(k, n_states):
        _apply_mark_for_index(qc, index_qubits, flag, i)

    return CircuitArtifact(
        circuit=qc,
        objective_qubits=(n_index_qubits,),
        ancilla_qubits=(),
        metadata={"k": k, "relation": ">=", "method": "naive_enumeration"},
    )
