from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import BudgetConfig
from ..types import BackendHandle, EstimationProblemSpec
from .results import AEResult


def _require_quantum_runtime() -> None:
    try:
        import qiskit  # noqa: F401
        import qiskit_algorithms  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Amplitude estimation requires Qiskit + qiskit-algorithms. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


@dataclass
class BudgetedAERunner:
    """Shots-first amplitude estimation runner (Stage 3 will implement).

    In v0.1 Stage 2 we only define the public interface so downstream modules
    (VaR search, TVaR orchestration) can be built against it.

    Planned v0.1 implementation:
      - "budgeted_fixed_schedule": deterministic schedule of Grover powers
      - consumes BudgetConfig.total_shots with per-call shot caps
      - executes using primitives in BackendHandle (Sampler/Estimator)
    """

    def run(
        self,
        problem: EstimationProblemSpec,
        *,
        budget: BudgetConfig,
        backend: BackendHandle,
    ) -> AEResult:
        _require_quantum_runtime()
        raise NotImplementedError(
            "BudgetedAERunner is defined in Stage 2 but implemented in Stage 3. "
            "Next step: wire qiskit-algorithms amplitude estimators to primitives and "
            "implement a fixed-schedule, shots-budgeted estimator."
        )
