# tests/test_stage2_imports.py
import importlib.util

import pytest


def _has_qiskit() -> bool:
    return importlib.util.find_spec("qiskit") is not None


def test_budgeted_ae_runner_behavior_depends_on_quantum_extras():
    """
    If Qiskit is NOT installed, calling runner.run should raise ImportError.
    If Qiskit IS installed, runner.run should get past the runtime check and then
    fail with ValueError because BackendHandle has no sampler (as constructed here).
    """
    from risk_qae.ae.budgeted_ae import BudgetedAERunner
    from risk_qae.types import BackendHandle, EstimationProblemSpec
    from risk_qae.config import BudgetConfig

    runner = BudgetedAERunner()
    problem = EstimationProblemSpec(state_preparation=None, objective_qubits=(0,))

    if not _has_qiskit():
        with pytest.raises(ImportError):
            runner.run(
                problem, budget=BudgetConfig(total_shots=10), backend=BackendHandle()
            )
    else:
        with pytest.raises(ValueError, match="BackendHandle\\.sampler is required"):
            runner.run(
                problem, budget=BudgetConfig(total_shots=10), backend=BackendHandle()
            )


def test_circuit_builders_raise_only_without_qiskit():
    """
    If Qiskit is NOT installed, circuit builders should raise ImportError.
    If Qiskit IS installed, they should return a QuantumCircuit (or similar artifact).
    """
    pmf = [1.0, 0.0]

    from risk_qae.circuits.stateprep import build_state_preparation

    if not _has_qiskit():
        with pytest.raises(ImportError):
            build_state_preparation(pmf)
    else:
        qc = build_state_preparation(pmf)
        # Avoid being too strict about types; just ensure we got "something circuit-like".
        assert qc is not None
