from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

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
        """Estimate the objective amplitude under a strict shots budget.

        v0.1 implementation (Phase 3):
        - Uses the SamplerV2-style primitive provided in BackendHandle.
        - Measures ONLY the objective qubit into a 1-bit classical register.
        - Consumes BudgetConfig.total_shots using repeated primitive calls capped by
          BudgetConfig.shots_per_call.

        Notes
        -----
        This estimator is "shots-first" and intentionally simple. It produces an
        amplitude estimate by direct sampling (Monte Carlo). It does NOT yet apply
        Grover-powered amplitude amplification schedules.

        The returned `AEResult.estimate` is post-processed if
        `problem.post_processing` is provided.
        """

        _require_quantum_runtime()
        if backend.sampler is None:
            raise ValueError(
                "BackendHandle.sampler is required for execution. "
                "Use risk_qae.backends.get_backend(...) or provide a BackendHandle."
            )

        try:
            from qiskit import QuantumCircuit  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Qiskit is required for execution. Install with: pip install 'risk_qae[quantum]'"
            ) from exc

        obj = list(problem.objective_qubits)
        if len(obj) != 1:
            raise NotImplementedError(
                "v0.1 BudgetedAERunner supports exactly one objective qubit. "
                f"Got objective_qubits={problem.objective_qubits!r}."
            )
        obj_q = int(obj[0])

        # Build a measurement wrapper circuit that measures ONLY the objective qubit.
        base = problem.state_preparation
        if not isinstance(base, QuantumCircuit):
            raise TypeError(
                "problem.state_preparation must be a qiskit.QuantumCircuit in v0.1. "
                "(This keeps the executor simple; we can generalize later.)"
            )

        qc = QuantumCircuit(
            base.num_qubits, 1, name=getattr(base, "name", "ae_measure")
        )
        qc.compose(base, inplace=True)
        qc.measure(obj_q, 0)

        total_shots = int(budget.total_shots)
        if total_shots <= 0:
            raise ValueError("BudgetConfig.total_shots must be positive")

        per_call = int(budget.shots_per_call)
        if per_call <= 0:
            raise ValueError("BudgetConfig.shots_per_call must be positive")

        max_calls = int(budget.max_circuit_calls)
        if max_calls <= 0:
            raise ValueError("BudgetConfig.max_circuit_calls must be positive")

        shots_remaining = total_shots
        calls = 0
        ones = 0
        shots_used = 0

        while shots_remaining > 0:
            if calls >= max_calls:
                break
            s = min(per_call, shots_remaining)

            job = backend.sampler.run([qc], shots=int(s))
            pub_result = job.result()[0]
            counts = _get_counts(pub_result)

            # counts keys are bitstrings; we measure a single bit, so use '0'/'1'.
            ones += int(counts.get("1", 0))
            shots_used += int(s)
            shots_remaining -= int(s)
            calls += 1

        if shots_used <= 0:
            raise RuntimeError(
                "No shots were executed. Check backend configuration and budget limits."
            )

        p = ones / shots_used
        ci = _wilson_ci(ones, shots_used)

        post = problem.post_processing
        if post is not None:
            est = float(post(p))
            ci_post = (float(post(ci[0])), float(post(ci[1])))
            ci_post = (min(ci_post), max(ci_post))
        else:
            est = float(p)
            ci_post = ci

        return AEResult(
            estimate=est,
            ci=ci_post,
            shots_used=shots_used,
            circuits_run=calls,
            diagnostics={
                "raw_amplitude": float(p),
                "raw_ci": ci,
                "objective_qubit": obj_q,
                "shots_requested": total_shots,
                "shots_used": shots_used,
                "calls": calls,
                "budget": {
                    "total_shots": total_shots,
                    "shots_per_call": per_call,
                    "max_circuit_calls": max_calls,
                },
                "problem_metadata": dict(problem.metadata or {}),
                "backend_metadata": dict(backend.metadata or {}),
            },
        )


def _get_counts(pub_result: Any) -> dict[str, int]:
    """Extract counts from a SamplerV2 pub result across implementations.

    Sampler V2 results differ slightly across providers. We try a few common
    access patterns.
    """
    # Preferred in many SamplerV2 implementations
    if hasattr(pub_result, "join_data"):
        try:
            return dict(pub_result.join_data().get_counts())
        except Exception:
            pass

    if hasattr(pub_result, "get_counts"):
        try:
            return dict(pub_result.get_counts())
        except Exception:
            pass

    # Fall back to DataBin style access
    data = getattr(pub_result, "data", None)
    if data is not None:
        for attr in ("meas", "m"):  # 'meas' is the common default register name
            reg = getattr(data, attr, None)
            if reg is None:
                continue
            if hasattr(reg, "get_counts"):
                try:
                    return dict(reg.get_counts())
                except Exception:
                    pass

    raise TypeError(
        "Unable to extract counts from sampler result. "
        "This provider may use an unsupported result format."
    )


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (float(lo), float(hi))
