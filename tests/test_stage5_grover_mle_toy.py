import numpy as np

import pytest

from risk_qae.ae.grover_mle import GroverMLEAERunner
from risk_qae.config import BudgetConfig
from risk_qae.types import BackendHandle, EstimationProblemSpec

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")


def test_grover_mle_recovers_toy_amplitude():
    from qiskit.circuit import QuantumCircuit
    from qiskit_aer.primitives import SamplerV2

    # Toy: A prepares objective with P(1)=a by RY(2*theta) on qubit 0
    a_true = 0.2
    theta = float(np.arcsin(np.sqrt(a_true)))

    A = QuantumCircuit(1, name="A_toy")
    A.ry(2.0 * theta, 0)

    spec = EstimationProblemSpec(state_preparation=A, objective_qubits=(0,))

    sampler = SamplerV2()
    bh = BackendHandle(sampler=sampler, estimator=None, backend=None)

    runner = GroverMLEAERunner(grid_size=1200, powers=(0, 1, 2, 4))
    res = runner.run(
        spec,
        budget=BudgetConfig(total_shots=2000, shots_per_call=500, max_circuit_calls=10),
        backend=bh,
    )

    # Loose tolerance (shots + grid + MLE approximation)
    assert abs(res.estimate - a_true) < 0.08
