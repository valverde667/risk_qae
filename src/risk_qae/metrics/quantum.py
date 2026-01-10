from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ae.budgeted_ae import BudgetedAERunner
from ..backends.factory import get_backend
from ..config import RiskQAEConfig
from ..discretization.histogram import HistogramDiscretizer
from ..types import BackendHandle, LossData
from ..circuits.problems import (
    build_cdf_problem,
    build_mean_problem,
    build_tail_prob_problem,
    build_tail_scaled_component_problem,
)


@dataclass(frozen=True)
class MeanResult:
    mean: float
    mean_ci: tuple[float, float] | None
    shots_used: int
    circuits_run: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class VaRResult:
    alpha: float
    var: float
    var_bin_index: int
    bracket: tuple[float, float]
    shots_used: int
    circuits_run: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class TVaRResult:
    alpha: float
    tvar: float
    var_used: float
    tail_prob: float
    tail_mean_numerator: float
    shots_used: int
    circuits_run: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RiskMeasuresResult:
    mean: MeanResult
    var: dict[float, VaRResult]
    tvar: dict[float, TVaRResult]
    total_shots_used: int
    total_circuits_run: int
    discretization_summary: dict[str, Any]
    config_frozen: dict[str, Any]


def estimate_mean(
    data: LossData,
    *,
    config: RiskQAEConfig | None = None,
    backend: BackendHandle | None = None,
) -> MeanResult:
    """Estimate E[X] for a loss distribution using shots-budgeted execution."""

    cfg = config or RiskQAEConfig()
    dist = _discretize(data, cfg)
    bh = backend or get_backend(cfg.backend)

    runner = BudgetedAERunner()
    problem = build_mean_problem(dist, value_encoding=cfg.value_encoding)
    res = runner.run(problem, budget=cfg.budget, backend=bh)

    return MeanResult(
        mean=float(res.estimate),
        mean_ci=res.ci,
        shots_used=int(res.shots_used),
        circuits_run=int(res.circuits_run),
        diagnostics=dict(res.diagnostics or {}),
    )


def estimate_var(
    data: LossData,
    alpha: float,
    *,
    config: RiskQAEConfig | None = None,
    backend: BackendHandle | None = None,
) -> VaRResult:
    """Estimate VaR_alpha for a loss distribution via bisection on the discretized index."""

    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1)")

    cfg = config or RiskQAEConfig()
    dist = _discretize(data, cfg)
    bh = backend or get_backend(cfg.backend)

    # Allocate a slice of the total budget to VaR bisection.
    total = int(cfg.budget.total_shots)
    var_budget_shots = max(
        1, int(total * float(cfg.var_search.shots_fraction_of_total))
    )
    per_call = int(cfg.budget.shots_per_call)
    max_calls = int(cfg.budget.max_circuit_calls)

    # Reserve shots per bisection step.
    max_iters = int(cfg.var_search.max_iters)
    shots_per_step = max(1, var_budget_shots // max(1, max_iters))

    runner = BudgetedAERunner()
    n = dist.pmf.size

    lo = 0
    hi = n - 1
    iters = 0
    shots_used = 0
    circuits_run = 0
    trace: list[dict[str, Any]] = []

    # Bisection on the bin index k such that P(I <= k) >= alpha.
    while (hi - lo) > int(cfg.var_search.target_bin_resolution) and iters < max_iters:
        mid = (lo + hi) // 2
        b = cfg.budget
        step_budget = type(b)(
            total_shots=shots_per_step,
            shots_per_call=min(per_call, shots_per_step),
            allocation=b.allocation,
            max_circuit_calls=max_calls,
            seed=b.seed,
        )
        prob_spec = build_cdf_problem(dist, mid)
        ae = runner.run(prob_spec, budget=step_budget, backend=bh)
        cdf_est = float(ae.estimate)

        shots_used += int(ae.shots_used)
        circuits_run += int(ae.circuits_run)
        trace.append(
            {"k": mid, "cdf_est": cdf_est, "ci": ae.ci, "shots": ae.shots_used}
        )

        if cdf_est >= alpha:
            hi = mid
        else:
            lo = mid + 1
        iters += 1

    k_star = max(0, min(hi, n - 1))
    var_t = float(dist.bin_edges[k_star + 1])  # conservative: upper edge
    bracket = (float(dist.bin_edges[k_star]), float(dist.bin_edges[k_star + 1]))

    return VaRResult(
        alpha=float(alpha),
        var=var_t,
        var_bin_index=int(k_star),
        bracket=bracket,
        shots_used=shots_used,
        circuits_run=circuits_run,
        diagnostics={"bisection_trace": trace, "iters": iters},
    )


def estimate_tvar(
    data: LossData,
    alpha: float,
    *,
    config: RiskQAEConfig | None = None,
    backend: BackendHandle | None = None,
    var_result: VaRResult | None = None,
) -> TVaRResult:
    """Estimate TVaR_alpha = E[X | X >= VaR_alpha] for a loss distribution."""

    cfg = config or RiskQAEConfig()
    dist = _discretize(data, cfg)
    bh = backend or get_backend(cfg.backend)

    # If VaR isn't provided, compute it first under this config/budget.
    vr = var_result or estimate_var(data, alpha, config=cfg, backend=bh)
    k = int(vr.var_bin_index)

    # Remaining budget after (optional) VaR.
    total = int(cfg.budget.total_shots)
    # If a VaRResult is provided, assume it came from the same overall budget and
    # subtract its actual usage so TVaR respects the shots-first intent.
    remain = max(1, total - int(vr.shots_used))
    # Split remaining shots between tail prob and tail component.
    shots_each = max(1, remain // 2)

    b = cfg.budget
    step_budget = type(b)(
        total_shots=shots_each,
        shots_per_call=min(int(b.shots_per_call), shots_each),
        allocation=b.allocation,
        max_circuit_calls=int(b.max_circuit_calls),
        seed=b.seed,
    )
    runner = BudgetedAERunner()

    tail_prob_spec = build_tail_prob_problem(dist, k)
    tail_prob_ae = runner.run(tail_prob_spec, budget=step_budget, backend=bh)
    tail_prob = float(tail_prob_ae.estimate)

    tail_comp_spec = build_tail_scaled_component_problem(dist, k)
    tail_comp_ae = runner.run(tail_comp_spec, budget=step_budget, backend=bh)
    tail_scaled_component = float(tail_comp_ae.estimate)

    x_min, x_max = dist.bounds
    denom = float(x_max - x_min)
    tail_mean_numerator = float(x_min) * tail_prob + denom * tail_scaled_component

    if tail_prob <= 0.0:
        tvar = float(np.max(dist.bin_values))
    else:
        tvar = tail_mean_numerator / tail_prob

    shots_used = (
        int(vr.shots_used) + int(tail_prob_ae.shots_used) + int(tail_comp_ae.shots_used)
    )
    circuits_run = (
        int(vr.circuits_run)
        + int(tail_prob_ae.circuits_run)
        + int(tail_comp_ae.circuits_run)
    )

    return TVaRResult(
        alpha=float(alpha),
        tvar=float(tvar),
        var_used=float(vr.var),
        tail_prob=float(tail_prob),
        tail_mean_numerator=float(tail_mean_numerator),
        shots_used=shots_used,
        circuits_run=circuits_run,
        diagnostics={
            "tail_prob_raw": dict(tail_prob_ae.diagnostics or {}),
            "tail_component_raw": dict(tail_comp_ae.diagnostics or {}),
            "var": dict(vr.diagnostics or {}),
        },
    )


def estimate_risk_measures(
    data: LossData,
    *,
    alphas: list[float] | None = None,
    config: RiskQAEConfig | None = None,
    backend: BackendHandle | None = None,
) -> RiskMeasuresResult:
    """Estimate mean + (VaR, TVaR) for multiple alphas."""

    cfg = config or RiskQAEConfig()
    dist = _discretize(data, cfg)
    bh = backend or get_backend(cfg.backend)

    # Split the *single* total budget across mean and each alpha, so the function
    # respects a shots-first "whole request" budget.
    total = int(cfg.budget.total_shots)
    alpha_list = alphas or [0.95, 0.99]
    n_alpha = max(1, len(alpha_list))

    # Heuristic split: mean gets 20% (min 1k), the rest split evenly across alphas.
    mean_shots = max(1_000, int(0.2 * total))
    mean_shots = min(mean_shots, max(1, total - n_alpha))
    remaining = max(1, total - mean_shots)
    shots_per_alpha = max(1, remaining // n_alpha)

    # Run mean under its allocated budget.
    b0 = cfg.budget
    mean_cfg = RiskQAEConfig(
        discretization=cfg.discretization,
        bounds=cfg.bounds,
        ae=cfg.ae,
        budget=type(b0)(
            total_shots=mean_shots,
            shots_per_call=min(int(b0.shots_per_call), mean_shots),
            allocation=b0.allocation,
            max_circuit_calls=int(b0.max_circuit_calls),
            seed=b0.seed,
        ),
        var_search=cfg.var_search,
        backend=cfg.backend,
        diagnostics=cfg.diagnostics,
    )

    runner = BudgetedAERunner()
    mean_problem = build_mean_problem(dist)
    mean_ae = runner.run(mean_problem, budget=mean_cfg.budget, backend=bh)
    mean_res = MeanResult(
        mean=float(mean_ae.estimate),
        mean_ci=mean_ae.ci,
        shots_used=int(mean_ae.shots_used),
        circuits_run=int(mean_ae.circuits_run),
        diagnostics=dict(mean_ae.diagnostics or {}),
    )

    var_res: dict[float, VaRResult] = {}
    tvar_res: dict[float, TVaRResult] = {}

    # Run each alpha under its own allocated budget.
    for a in alpha_list:
        alpha_cfg = RiskQAEConfig(
            discretization=cfg.discretization,
            bounds=cfg.bounds,
            ae=cfg.ae,
            budget=type(b0)(
                total_shots=shots_per_alpha,
                shots_per_call=min(int(b0.shots_per_call), shots_per_alpha),
                allocation=b0.allocation,
                max_circuit_calls=int(b0.max_circuit_calls),
                seed=b0.seed,
            ),
            var_search=cfg.var_search,
            backend=cfg.backend,
            diagnostics=cfg.diagnostics,
        )

        # Use the already-discretized distribution to avoid re-binning.
        vr = _estimate_var_from_dist(dist, float(a), alpha_cfg, bh)
        tr = _estimate_tvar_from_dist(dist, float(a), alpha_cfg, bh, vr)
        var_res[float(a)] = vr
        tvar_res[float(a)] = tr

    total_shots_used = mean_res.shots_used + sum(v.shots_used for v in var_res.values())
    total_circuits = mean_res.circuits_run + sum(
        v.circuits_run for v in var_res.values()
    )

    return RiskMeasuresResult(
        mean=mean_res,
        var=var_res,
        tvar=tvar_res,
        total_shots_used=int(total_shots_used),
        total_circuits_run=int(total_circuits),
        discretization_summary={
            "n_index_qubits": int(dist.n_index_qubits),
            "n_bins": int(dist.pmf.size),
            "bounds": tuple(map(float, dist.bounds)),
            "clip_percentiles": tuple(map(float, cfg.bounds.clip_percentiles)),
        },
        config_frozen=_freeze_config(cfg),
    )


def _estimate_var_from_dist(
    dist, alpha: float, cfg: RiskQAEConfig, bh: BackendHandle
) -> VaRResult:
    """Internal helper: VaR from an already-discretized distribution."""
    # Reuse estimate_var's logic but avoid a second discretization pass.
    data: LossData = {"pmf": dist.pmf.tolist(), "bin_values": dist.bin_values.tolist()}
    return estimate_var(data, alpha, config=cfg, backend=bh)


def _estimate_tvar_from_dist(
    dist, alpha: float, cfg: RiskQAEConfig, bh: BackendHandle, vr: VaRResult
) -> TVaRResult:
    """Internal helper: TVaR from an already-discretized distribution."""
    data: LossData = {"pmf": dist.pmf.tolist(), "bin_values": dist.bin_values.tolist()}
    return estimate_tvar(data, alpha, config=cfg, backend=bh, var_result=vr)


def _discretize(data: LossData, cfg: RiskQAEConfig):
    disc = HistogramDiscretizer()
    return disc.fit(data, dcfg=cfg.discretization, bcfg=cfg.bounds)


def _freeze_config(cfg: RiskQAEConfig) -> dict[str, Any]:
    """Convert nested dataclasses into plain dicts."""
    return {
        "discretization": cfg.discretization.__dict__,
        "bounds": cfg.bounds.__dict__,
        "ae": cfg.ae.__dict__,
        "budget": cfg.budget.__dict__,
        "var_search": cfg.var_search.__dict__,
        "backend": cfg.backend.__dict__,
        "diagnostics": cfg.diagnostics.__dict__,
    }
