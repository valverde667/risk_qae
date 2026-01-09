import pytest

import risk_qae
from risk_qae import RiskQAEConfig
from risk_qae.circuits import (
    build_state_preparation,
    build_indicator_leq_index,
    build_scaled_value_rotation,
)


def test_budgeted_ae_runner_requires_quantum_extras():
    # Calling the runner without Qiskit installed should raise a clear ImportError.
    from risk_qae.ae.budgeted_ae import BudgetedAERunner
    from risk_qae.types import BackendHandle, EstimationProblemSpec
    from risk_qae.config import BudgetConfig

    runner = BudgetedAERunner()
    problem = EstimationProblemSpec(state_preparation=None, objective_qubits=(0,))
    with pytest.raises(ImportError):
        runner.run(
            problem, budget=BudgetConfig(total_shots=10), backend=BackendHandle()
        )


def test_stage2_imports_ok():
    cfg = RiskQAEConfig()
    assert cfg.discretization.n_index_qubits == 8
    assert cfg.bounds.clip_percentiles == (0.1, 99.9)


def test_circuit_builders_raise_without_qiskit():
    pmf = [1.0, 0.0]
    with pytest.raises(ImportError):
        build_state_preparation(pmf)

    with pytest.raises(ImportError):
        build_indicator_leq_index(1, 0)

    with pytest.raises(ImportError):
        build_scaled_value_rotation(1, [0.0, 1.0], (0.0, 1.0))
