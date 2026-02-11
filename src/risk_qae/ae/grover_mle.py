from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from statistics import NormalDist

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
    """Return a copy of qc that measures ONLY the objective qubit into a 1-bit classical register."""
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
    alpha_confidence: float = 0.05,
    theta_center: float | None = None,
    theta_halfwidth: float | None = None,
) -> tuple[float, tuple[float, float]]:
    """
    MLE fit for theta in a = sin^2(theta), theta in [0, pi/2].

    Each experiment is (k, ones, shots) where the measured probability is:
        p_k(theta) = sin^2((2k+1) * theta)

    Returns:
        theta_hat, (theta_lo, theta_hi) where (lo,hi) is an approximate
        (1 - alpha_confidence) confidence interval via a Wilks likelihood-ratio cutoff.

    NOTE: If theta_center/theta_halfwidth are provided, restrict the search grid to
    [theta_center-theta_halfwidth, theta_center+theta_halfwidth] intersect [0, pi/2].
    This avoids selecting a wrong MLE lobe when large Grover powers are used.
    """
    import numpy as np
    from statistics import NormalDist

    if len(data) == 0:
        raise ValueError("MLE data must be non-empty.")

    # Theta grid on [0, pi/2], optionally restricted to a window around theta_center
    lo, hi = 0.0, 0.5 * np.pi
    if theta_center is not None and theta_halfwidth is not None:
        lo = max(0.0, float(theta_center) - float(theta_halfwidth))
        hi = min(0.5 * np.pi, float(theta_center) + float(theta_halfwidth))
        if hi <= lo:
            lo, hi = 0.0, 0.5 * np.pi  # fallback

    thetas = np.linspace(lo, hi, int(grid_size), dtype=float)

    # Accumulate log-likelihood on the grid
    lls = np.zeros_like(thetas)
    eps = 1e-12

    for k, ones, shots in data:
        k = int(k)
        ones = int(ones)
        shots = int(shots)
        if shots <= 0:
            continue
        if ones < 0 or ones > shots:
            raise ValueError("Invalid (ones, shots) pair in MLE data.")

        angle = (2.0 * k + 1.0) * thetas
        p = np.sin(angle) ** 2
        p = np.clip(p, eps, 1.0 - eps)

        lls += ones * np.log(p) + (shots - ones) * np.log(1.0 - p)

    # MLE
    idx = int(np.argmax(lls))
    theta_hat = float(thetas[idx])
    ll_max = float(lls[idx])

    # Wilks: 2*(ll_max - ll(theta)) ~ chi2_1.
    p_cov = 1.0 - float(alpha_confidence)
    p_cov = min(max(p_cov, 1e-12), 1.0 - 1e-12)
    z = NormalDist().inv_cdf((p_cov + 1.0) / 2.0)
    chi2_q = float(z * z)

    cutoff = ll_max - 0.5 * chi2_q

    mask = lls >= cutoff
    if not np.any(mask):
        return theta_hat, (theta_hat, theta_hat)

    left = idx
    while left > 0 and mask[left - 1]:
        left -= 1
    right = idx
    while right < (len(mask) - 1) and mask[right + 1]:
        right += 1

    theta_lo = float(thetas[left])
    theta_hi = float(thetas[right])
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
        epsilon_target: float | None = None,
        alpha_confidence: float = 0.05,
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

        # Scale epsilon_target from loss units to probability units (if bounds exist)
        x_min, x_max = problem.metadata.get("bounds", (0.0, 1.0))
        denom = max(float(x_max) - float(x_min), 1e-12)

        scaled_epsilon = None
        if epsilon_target is not None:
            scaled_epsilon = float(epsilon_target) / denom

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
            powers: list[int] = [0]
            k = 1
            while len(powers) < max_calls:
                powers.append(k)
                k *= 2
        else:
            powers = [int(x) for x in self.powers]
            if any(k < 0 for k in powers):
                raise ValueError("Grover powers must be >= 0.")

        # theta window settings to prevent wrong-lobe MLE for large powers
        k_max = int(max(powers)) if len(powers) else 0
        theta_halfwidth = 0.225 * math.pi / (2 * k_max + 1)  # conservative, lobe-safe
        theta_center: float | None = None  # set from k=0 measurement

        # Execute under strict shots budget
        from qiskit.circuit import QuantumCircuit

        shots_remaining = total_shots
        calls = 0
        data: list[tuple[int, int, int]] = []
        circuits_run = 0
        shots_used = 0

        diagnostics: dict[str, Any] = {"powers": [], "per_power": []}

        estimate: float | None = None
        ci: tuple[float, float] | None = None

        for k in powers:
            if shots_remaining <= 0 or calls >= max_calls:
                break

            s = min(per_call, shots_remaining)
            if s <= 0:
                break

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

            shots_used += int(s)
            shots_remaining -= int(s)
            calls += 1
            circuits_run += 1

            diagnostics["powers"].append(int(k))
            diagnostics["per_power"].append(
                {
                    "k": int(k),
                    "shots": int(s),
                    "ones": int(ones),
                    "p_hat": float(ones) / float(s),
                }
            )

            # lock theta center using k=0 empirical amplitude (once)
            if int(k) == 0 and theta_center is None:
                a0_hat = float(ones) / float(s)
                a0_hat = min(max(a0_hat, 1e-12), 1.0 - 1e-12)
                theta_center = math.asin(math.sqrt(a0_hat))
                diagnostics["theta_center_from_k0"] = float(theta_center)
                diagnostics["theta_halfwidth"] = float(theta_halfwidth)

            theta_hat, (theta_lo, theta_hi) = _mle_theta_grid(
                data,
                grid_size=self.grid_size,
                alpha_confidence=alpha_confidence,
                theta_center=theta_center,
                theta_halfwidth=theta_halfwidth,
            )

            a_hat = float(math.sin(theta_hat) ** 2)
            a_lo = float(math.sin(theta_lo) ** 2)
            a_hi = float(math.sin(theta_hi) ** 2)

            # Early-stop in AMPLITUDE space (compare to scaled epsilon)
            current_amplitude_half_width = 0.5 * abs(a_hi - a_lo)

            estimate = a_hat
            ci = (a_lo, a_hi)
            if problem.post_processing is not None:
                estimate = float(problem.post_processing(a_hat))
                ci = (
                    float(problem.post_processing(a_lo)),
                    float(problem.post_processing(a_hi)),
                )

            if (
                scaled_epsilon is not None
                and current_amplitude_half_width <= scaled_epsilon
            ):
                diagnostics["early_stop"] = {
                    "reason": "epsilon_target_met",
                    "epsilon_target": float(epsilon_target),
                    "scaled_epsilon": float(scaled_epsilon),
                    "half_width_loss": 0.5 * abs(ci[1] - ci[0]),
                    "half_width_amplitude": float(current_amplitude_half_width),
                    "alpha_confidence": float(alpha_confidence),
                    "shots_used": int(shots_used),
                    "circuits_run": int(circuits_run),
                    "calls": int(calls),
                }
                break

        if len(data) == 0 or estimate is None or ci is None:
            raise RuntimeError(
                "No Grover experiments were executed (budget/calls too small)."
            )

        return AEResult(
            estimate=float(estimate),
            ci=ci,
            shots_used=int(shots_used),
            circuits_run=int(circuits_run),
            diagnostics=diagnostics,
        )
