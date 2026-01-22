from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any
import hashlib

from qiskit import transpile

import math

from ..config import BudgetConfig
from ..types import BackendHandle, EstimationProblemSpec
from .results import AEResult


def _compile_for_backend(qc, backend_handle, opt_level: int = 1):
    # Decompose high-level instructions like StatePreparation
    qc2 = qc.decompose(reps=10)

    # If the BackendHandle has an AerSimulator (or other target), transpile to it
    target = getattr(backend_handle, "backend", None)
    if target is not None:
        return transpile(qc2, backend=target, optimization_level=opt_level)

    # Fallback: generic basis
    return transpile(
        qc2, basis_gates=["rz", "sx", "x", "cx"], optimization_level=opt_level
    )


def _require_quantum_runtime() -> None:
    try:
        import qiskit  # noqa: F401
        import qiskit_algorithms  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Amplitude estimation requires Qiskit + qiskit-algorithms. Install with: "
            "pip install 'risk_qae[quantum]'"
        ) from exc


def _stable_circuit_text(qc) -> str:
    """Best-effort stable textual form for caching."""
    try:
        from qiskit.qasm3 import dumps  # type: ignore

        return dumps(qc)
    except Exception:
        # Fallback: QASM2 (may be unavailable in some future qiskit versions)
        try:
            return qc.qasm()
        except Exception:
            # Last resort: repr (not ideal but avoids crashing)
            return repr(qc)


def _compile_for_backend(
    qc, backend_handle, *, decompose_reps: int = 3, opt_level: int = 0
):
    # Decompose high-level instructions (StatePreparation, etc.)
    qc2 = qc.decompose(reps=decompose_reps)

    target = getattr(backend_handle, "backend", None)
    if target is not None:
        return transpile(qc2, backend=target, optimization_level=opt_level)

    return transpile(
        qc2, basis_gates=["rz", "sx", "x", "cx"], optimization_level=opt_level
    )


def _compile_cache_key(
    qc, backend_handle, *, decompose_reps: int, opt_level: int
) -> str:
    txt = _stable_circuit_text(qc)
    backend_id = str(type(getattr(backend_handle, "backend", None)).__name__)
    payload = f"{backend_id}|reps={decompose_reps}|opt={opt_level}|{txt}".encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass
class BudgetedAERunner:
    """Shots-first amplitude estimation runner.

    v0.1 behavior:
      - Uses BackendHandle.sampler (SamplerV2-style primitive).
      - Measures ONLY the objective qubit into a 1-bit classical register.
      - Consumes BudgetConfig.total_shots using repeated primitive calls capped by
        BudgetConfig.shots_per_call.
      - Optionally early-stops if a post-processed CI half-width meets epsilon_target.

    Notes
    -----
    This estimator is "shots-first" and intentionally simple. It produces an
    amplitude estimate by direct sampling (Monte Carlo). It does NOT apply
    Grover-powered amplitude amplification schedules.
    """

    decompose_reps: int = 3
    transpile_optimization_level: int = 0
    _compile_cache: dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        problem: EstimationProblemSpec,
        *,
        budget: BudgetConfig,
        backend: BackendHandle,
        epsilon_target: float | None = None,
        alpha_confidence: float = 0.05,
    ) -> AEResult:
        """Estimate the objective amplitude under a strict shots budget.

        Parameters
        ----------
        problem:
            Estimation problem spec containing state_preparation and objective qubit.
        budget:
            Shot budget controls.
        backend:
            Backend handle containing a Sampler-like primitive.
        epsilon_target:
            Optional early-stop threshold on the half-width of the (post-processed) CI.
        alpha_confidence:
            Two-sided CI confidence level is (1 - alpha_confidence). Default 0.05 => 95% CI.
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
                "Qiskit is required for execution. Install with: "
                "pip install 'risk_qae[quantum]'"
            ) from exc

        obj = list(problem.objective_qubits)
        if len(obj) != 1:
            raise NotImplementedError(
                "v0.1 BudgetedAERunner supports exactly one objective qubit. "
                f"Got objective_qubits={problem.objective_qubits!r}."
            )
        obj_q = int(obj[0])

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

        # Convert alpha_confidence -> z for two-sided Wilson CI
        p_conf = 1.0 - float(alpha_confidence)
        p_conf = min(max(p_conf, 1e-12), 1.0 - 1e-12)
        z = float(NormalDist().inv_cdf(0.5 + 0.5 * p_conf))

        shots_remaining = total_shots
        calls = 0
        ones = 0
        shots_used = 0
        early_stop: dict[str, Any] | None = None

        # compile once + cache
        key = _compile_cache_key(
            qc,
            backend,
            decompose_reps=self.decompose_reps,
            opt_level=self.transpile_optimization_level,
        )

        qc_compiled = self._compile_cache.get(key)
        if qc_compiled is None:
            qc_compiled = _compile_for_backend(
                qc,
                backend,
                decompose_reps=self.decompose_reps,
                opt_level=self.transpile_optimization_level,
            )
            self._compile_cache[key] = qc_compiled

        while shots_remaining > 0:
            if calls >= max_calls:
                break

            s = min(per_call, shots_remaining)
            if s <= 0:
                break

            job = backend.sampler.run([qc_compiled], shots=int(s))
            pub_result = job.result()[0]
            counts = _get_counts(pub_result)

            ones += int(counts.get("1", 0))
            shots_used += int(s)
            shots_remaining -= int(s)
            calls += 1

            # Optional early stop on post-processed CI half-width
            if epsilon_target is not None and shots_used > 0:
                p_hat = ones / shots_used
                ci_raw = _wilson_ci(ones, shots_used, z=z)

                post = problem.post_processing
                if post is not None:
                    ci_tmp = (float(post(ci_raw[0])), float(post(ci_raw[1])))
                    ci_tmp = (min(ci_tmp), max(ci_tmp))
                else:
                    ci_tmp = ci_raw

                half_width = 0.5 * abs(ci_tmp[1] - ci_tmp[0])
                if half_width <= float(epsilon_target):
                    early_stop = {
                        "reason": "epsilon_target_met",
                        "epsilon_target": float(epsilon_target),
                        "half_width": float(half_width),
                        "alpha_confidence": float(alpha_confidence),
                        "shots_used": int(shots_used),
                        "circuits_run": int(calls),
                        "calls": int(calls),
                    }
                    break

        if shots_used <= 0:
            raise RuntimeError(
                "No shots were executed. Check backend configuration and budget limits."
            )

        p = ones / shots_used
        ci = _wilson_ci(ones, shots_used, z=z)

        post = problem.post_processing
        if post is not None:
            est = float(post(p))
            ci_post = (float(post(ci[0])), float(post(ci[1])))
            ci_post = (min(ci_post), max(ci_post))
        else:
            est = float(p)
            ci_post = ci

        diagnostics: dict[str, Any] = {
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
        }
        if early_stop is not None:
            diagnostics["early_stop"] = early_stop

        return AEResult(
            estimate=float(est),
            ci=ci_post,
            shots_used=int(shots_used),
            circuits_run=int(calls),
            diagnostics=diagnostics,
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
