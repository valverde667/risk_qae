from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import BudgetConfig
from ..types import BackendHandle, EstimationProblemSpec
from .results import AEResult
from ..circuits.grover import build_grover_operator


def _require_quantum_runtime() -> None:
    """Import guard for quantum execution dependencies."""
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for Grover-based amplitude estimation. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


def _get_counts(pub_result: Any) -> dict[str, int]:
    """
    Extract counts from a SamplerV2-style pub_result.

    We keep this minimal and consistent with budgeted_ae.py behavior:
    - keys are bitstrings for the measured classical register
    """
    # Common patterns across primitives versions:
    # - pub_result.data.meas.get_counts()
    # - pub_result.data.c.get_counts()
    # - pub_result.get_counts()
    for path in (
        ("data", "meas", "get_counts"),
        ("data", "c", "get_counts"),
        ("get_counts",),
    ):
        obj = pub_result
        ok = True
        for p in path:
            if callable(getattr(obj, p, None)):
                obj = getattr(obj, p)()
            else:
                obj = getattr(obj, p, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, dict):
            return {str(k): int(v) for k, v in obj.items()}
    raise TypeError(
        "Unable to extract counts from sampler result; unsupported primitives format."
    )


def _measure_objective_only(qc, objective_qubit: int):
    """
    Return a copy of qc that measures ONLY the objective qubit into a 1-bit classical register.
    """
    _require_quantum_runtime()
    from qiskit.circuit import QuantumCircuit

    out = QuantumCircuit(qc.num_qubits, 1, name=f"{qc.name}_meas_obj")
    out.compose(qc, inplace=True)
    out.measure(objective_qubit, 0)
    return out


def _log_likelihood_theta(theta: float, data: list[tuple[int, int, int]]) -> float:
    """
    Log-likelihood for theta given (k, ones, shots) observations, where
    p_k(theta) = sin^2((2k+1) theta).

    data: list of (k, ones, shots)
    """
    ll = 0.0
    eps = 1e-12
    for k, ones, shots in data:
        t = (2 * k + 1) * theta
        p = math.sin(t) ** 2
        p = min(max(p, eps), 1.0 - eps)
        ll += ones * math.log(p) + (shots - ones) * math.log(1.0 - p)
    return ll


def _mle_theta_grid(
    data: list[tuple[int, int, int]],
    *,
    grid_size: int = 2000,
) -> tuple[float, tuple[float, float]]:
    """
    MLE estimate of theta over a grid on [0, pi/2], plus a crude CI from a
    log-likelihood drop (Wilks approx).

    Returns: (theta_hat, (theta_lo, theta_hi))
    """
    # Grid in theta (not amplitude). theta in [0, pi/2].
    thetas = np.linspace(0.0, 0.5 * math.pi, int(grid_size), dtype=float)
    lls = np.array(
        [_log_likelihood_theta(float(th), data) for th in thetas], dtype=float
    )

    j = int(np.argmax(lls))
    theta_hat = float(thetas[j])
    ll_max = float(lls[j])

    # Wilks: 2*(ll_max - ll(theta)) ~ chi2_1. Use ~3.84 for 95% CI.
    # This is an approximation; good enough for diagnostics in v0.1.
    cutoff = ll_max - 0.5 * 3.84
    mask = lls >= cutoff
    if not np.any(mask):
        return theta_hat, (theta_hat, theta_hat)

    idx = np.where(mask)[0]
    theta_lo = float(thetas[int(idx[0])])
    theta_hi = float(thetas[int(idx[-1])])
    return theta_hat, (theta_lo, theta_hi)


@dataclass
class GroverMLEAERunner:
    """
    Grover-schedule amplitude estimation via maximum likelihood over a theta grid.

    - Builds Grover operator Q from the provided state_preparation circuit A and objective qubit.
    - Executes circuits Q^k A |0>, measuring ONLY the objective qubit.
    - Collects counts for several k values under a strict shots budget.
    - Fits theta (and thus amplitude a = sin^2(theta)) by MLE.

    This is a practical “iterative AE” baseline that uses Grover powers without requiring
    full IQAE interval logic yet.
    """

    grid_size: int = 2000
    powers: Sequence[int] | None = None  # if None, auto schedule

    def run(
        self,
        problem: EstimationProblemSpec,
        *,
        budget: BudgetConfig,
        backend: BackendHandle,
    ) -> AEResult:
        _require_quantum_runtime()
        if backend.sampler is None:
            raise ValueError(
                "BackendHandle.sampler is required for execution. "
                "Use risk_qae.backends.get_backend(...) or provide a BackendHandle."
            )

        # Build A and Grover operator Q
        A = problem.state_preparation
        if A is None:
            raise ValueError(
                "EstimationProblemSpec.state_preparation must be provided."
            )
        if len(problem.objective_qubits) != 1:
            raise ValueError(
                "GroverMLEAERunner supports exactly one objective qubit in v0.1."
            )
        obj = int(problem.objective_qubits[0])

        Q = build_grover_operator(A, (obj,))

        # Decide powers schedule
        total_shots = int(budget.total_shots)
        per_call = int(budget.shots_per_call)
        max_calls = int(budget.max_circuit_calls)

        if total_shots <= 0:
            raise ValueError("BudgetConfig.total_shots must be >= 1.")
        if per_call <= 0:
            raise ValueError("BudgetConfig.shots_per_call must be >= 1.")
        if max_calls <= 0:
            raise ValueError("BudgetConfig.max_circuit_calls must be >= 1.")

        if self.powers is None:
            # Simple doubling schedule starting at k=0.
            # We'll cap the length by available calls.
            # Example: 0,1,2,4,8,...
            powers: list[int] = [0]
            k = 1
            while len(powers) < max_calls:
                powers.append(k)
                k *= 2
        else:
            powers = [int(x) for x in self.powers]
            if any(k < 0 for k in powers):
                raise ValueError("Grover powers must be >= 0.")

        # Execute under strict shots budget
        from qiskit.circuit import QuantumCircuit

        shots_remaining = total_shots
        calls = 0
        data: list[tuple[int, int, int]] = []
        circuits_run = 0
        shots_used = 0

        diagnostics: dict[str, Any] = {"powers": [], "per_power": []}

        for k in powers:
            if shots_remaining <= 0 or calls >= max_calls:
                break

            s = min(per_call, shots_remaining)
            if s <= 0:
                break

            # Build circuit: start with A|0>, then apply Q^k (Q already includes A and A† internally),
            # but the standard Grover AE circuit is Q^k applied after A:
            #   |psi_k> = (Q)^k A |0>
            #
            # We can implement this as:
            #   qc = A; then append Q k times.
            qc = QuantumCircuit(A.num_qubits, name=f"grover_k_{k}")
            qc.compose(A, inplace=True)
            for _ in range(int(k)):
                qc.compose(Q, inplace=True)

            qc_m = _measure_objective_only(qc, obj)

            job = backend.sampler.run([qc_m], shots=int(s))
            pub_result = job.result()[0]
            counts = _get_counts(pub_result)
            ones = int(counts.get("1", 0))
            data.append((int(k), ones, int(s)))

            diagnostics["powers"].append(int(k))
            diagnostics["per_power"].append(
                {
                    "k": int(k),
                    "shots": int(s),
                    "ones": int(ones),
                    "p_hat": float(ones) / float(s),
                }
            )

            shots_used += int(s)
            shots_remaining -= int(s)
            calls += 1
            circuits_run += 1

        if len(data) == 0:
            raise RuntimeError(
                "No Grover experiments were executed (budget/calls too small)."
            )

        theta_hat, (theta_lo, theta_hi) = _mle_theta_grid(
            data, grid_size=self.grid_size
        )

        # Convert to amplitude a = sin^2(theta)
        a_hat = float(math.sin(theta_hat) ** 2)
        a_lo = float(math.sin(theta_lo) ** 2)
        a_hi = float(math.sin(theta_hi) ** 2)

        # Post-processing (e.g., map E[g(X)] -> E[X])
        estimate = a_hat
        ci = (a_lo, a_hi)
        if problem.post_processing is not None:
            estimate = float(problem.post_processing(a_hat))
            ci = (
                float(problem.post_processing(a_lo)),
                float(problem.post_processing(a_hi)),
            )

        return AEResult(
            estimate=float(estimate),
            ci=ci,
            shots_used=int(shots_used),
            circuits_run=int(circuits_run),
            diagnostics=diagnostics,
        )
