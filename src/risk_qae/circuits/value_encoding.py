from __future__ import annotations

import math
from typing import Sequence, Tuple

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


def _theta_from_prob(p: float) -> float:
    # Want P(|1>) = p after RY(theta) on |0>
    # sin(theta/2)^2 = p -> theta = 2*arcsin(sqrt(p))
    p = min(max(float(p), 0.0), 1.0)
    return 2.0 * float(np.arcsin(np.sqrt(p)))


def build_scaled_value_rotation(
    n_index_qubits: int,
    bin_values: Sequence[float],
    bounds: tuple[float, float],
    *,
    method: str = "naive_table",
) -> CircuitArtifact:
    """Build a value-encoding circuit that maps X into an objective qubit probability.

    For each basis state |i> (index register), apply a controlled RY on the objective qubit
    such that:

        P(objective=1 | i) = g(x_i),

    where g(x) = (x - x_min)/(x_max - x_min) clipped to [0,1].

    Layout:
        index register: n_index_qubits
        objective qubit: 1 qubit, appended last

    Notes
    -----
    The v0.1 implementation uses a naive table-lookup: one multi-controlled rotation per bin.
    This is acceptable for low qubit counts (e.g., 2**8=256 bins) and is friendly to tensor-network
    simulators where depth may be less problematic than state-vector size.

    Future versions can swap this out for a structured qROM-based loader.
    """
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import RYGate
    from qiskit.circuit.library import MCMT

    if n_index_qubits <= 0:
        raise ValueError("n_index_qubits must be >= 1.")
    n_states = 2**n_index_qubits
    if len(bin_values) != n_states:
        raise ValueError(
            f"bin_values must have length {n_states} (got {len(bin_values)})."
        )

    x_min, x_max = bounds
    denom = float(x_max - x_min)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("bounds must satisfy x_max > x_min and be finite.")

    xs = np.asarray(bin_values, dtype=float)
    g = (xs - float(x_min)) / denom
    g = np.clip(g, 0.0, 1.0)

    qc = QuantumCircuit(n_index_qubits + 1, name="val_encode")
    index_qubits = list(qc.qubits[:n_index_qubits])
    obj = qc.qubits[n_index_qubits]

    def apply_mc_ry(theta: float, i: int) -> None:
        bits = [(i >> b) & 1 for b in range(n_index_qubits)]
        zeros = [q for q, bit in zip(index_qubits, bits) if bit == 0]
        for q in zeros:
            qc.x(q)

        # Try native multi-controlled RY if available
        if hasattr(qc, "mcry"):
            qc.mcry(theta, index_qubits, obj, None, mode="noancilla")
        else:
            # Generic: MCMT(RYGate(theta), num_ctrl_qubits, 1)
            gate = MCMT(
                RYGate(theta), num_ctrl_qubits=n_index_qubits, num_target_qubits=1
            )
            qc.append(gate, index_qubits + [obj])

        for q in zeros:
            qc.x(q)

    if method != "naive_table":
        raise ValueError("Only method='naive_table' is supported in v0.1.")

    for i in range(n_states):
        theta = _theta_from_prob(float(g[i]))
        if theta == 0.0:
            continue
        apply_mc_ry(theta, i)

    return CircuitArtifact(
        circuit=qc,
        objective_qubits=(n_index_qubits,),
        ancilla_qubits=(),
        metadata={
            "method": method,
            "bounds": bounds,
            "n_index_qubits": n_index_qubits,
        },
    )


def build_scaled_value_rotation_piecewise_prefix(
    n_index_qubits: int,
    bin_values: Sequence[float],
    bounds: tuple[float, float],
    *,
    n_segments: int = 16,
) -> CircuitArtifact:
    """
    Piecewise value encoding using prefix controls on the MSBs.

    This replaces the O(2^n) naive table lookup with O(n_segments * log2(n_segments))
    multi-controlled rotations, where n_segments is a power of two.

    Each segment corresponds to a fixed prefix on the m = log2(n_segments) most
    significant index bits, and uses the average g(x) over that segment.

    Returns a CircuitArtifact consistent with build_scaled_value_rotation().
    """
    _require_qiskit()
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import RYGate, MCMT

    if n_index_qubits <= 0:
        raise ValueError("n_index_qubits must be >= 1.")
    n_states = 2**n_index_qubits
    if len(bin_values) != n_states:
        raise ValueError(
            f"bin_values must have length {n_states} (got {len(bin_values)})."
        )

    if n_segments < 1 or n_segments > n_states or (n_segments & (n_segments - 1)) != 0:
        raise ValueError("n_segments must be a power of two and <= 2**n_index_qubits.")

    x_min, x_max = bounds
    denom = float(x_max - x_min)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("bounds must satisfy x_max > x_min and be finite.")

    m = int(math.log2(n_segments))  # number of MSB control bits
    seg_size = 2 ** (n_index_qubits - m)

    xs = np.asarray(bin_values, dtype=float)
    g = (xs - float(x_min)) / denom
    g = np.clip(g, 0.0, 1.0)

    qc = QuantumCircuit(n_index_qubits + 1, name="val_encode_piecewise")
    index_qubits = list(qc.qubits[:n_index_qubits])
    obj = qc.qubits[n_index_qubits]

    # Controls are MSBs: highest-index qubits in Qiskit's ordering
    controls = [index_qubits[n_index_qubits - 1 - j] for j in range(m)]

    def apply_prefix_controlled_ry(theta: float, prefix: int) -> None:
        # pattern bits for prefix, MSB-first over m bits
        bits = [(prefix >> (m - 1 - j)) & 1 for j in range(m)]
        zeros = [q for q, bit in zip(controls, bits) if bit == 0]
        for q in zeros:
            qc.x(q)

        # Apply multi-controlled RY with m controls
        if hasattr(qc, "mcry"):
            qc.mcry(theta, controls, obj, None, mode="noancilla")
        else:
            gate = MCMT(RYGate(theta), num_ctrl_qubits=m, num_target_qubits=1)
            qc.append(gate, controls + [obj])

        for q in zeros:
            qc.x(q)

    # For each prefix segment, use average probability over indices in the segment
    for s in range(n_segments):
        start = s * seg_size
        end = start + seg_size
        p = float(np.mean(g[start:end]))
        theta = _theta_from_prob(p)
        if theta == 0.0:
            continue
        apply_prefix_controlled_ry(theta, s)

    return CircuitArtifact(
        circuit=qc,
        objective_qubits=(n_index_qubits,),
        ancilla_qubits=(),
        metadata={
            "method": "piecewise_prefix",
            "n_segments": int(n_segments),
            "bounds": bounds,
            "n_index_qubits": n_index_qubits,
        },
    )
