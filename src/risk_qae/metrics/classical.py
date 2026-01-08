from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..discretization.histogram import DiscretizedDistribution


@dataclass(frozen=True)
class ClassicalMeanResult:
    mean: float


@dataclass(frozen=True)
class ClassicalVaRResult:
    alpha: float
    var: float
    var_bin_index: int
    bracket: tuple[float, float]


@dataclass(frozen=True)
class ClassicalTVaRResult:
    alpha: float
    tvar: float
    var_used: float
    tail_prob: float
    tail_mean_numerator: float


def classical_mean(dist: DiscretizedDistribution) -> ClassicalMeanResult:
    mean = float(np.sum(dist.pmf * dist.bin_values))
    return ClassicalMeanResult(mean=mean)


def classical_var(dist: DiscretizedDistribution, alpha: float) -> ClassicalVaRResult:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    cdf = np.cumsum(dist.pmf)
    idx = int(np.searchsorted(cdf, alpha, side="left"))
    idx = max(0, min(idx, dist.pmf.size - 1))

    # Choose the upper edge of the bin as a conservative threshold ensuring P(X<=t) >= alpha
    t = float(dist.bin_edges[idx + 1])
    bracket = (float(dist.bin_edges[idx]), float(dist.bin_edges[idx + 1]))
    return ClassicalVaRResult(alpha=alpha, var=t, var_bin_index=idx, bracket=bracket)


def classical_tvar(dist: DiscretizedDistribution, alpha: float) -> ClassicalTVaRResult:
    var_res = classical_var(dist, alpha)
    var_t = var_res.var

    # Tail event uses representative values; consistent with discretization resolution.
    mask = dist.bin_values >= var_t
    tail_prob = float(np.sum(dist.pmf[mask]))
    tail_num = float(np.sum(dist.pmf[mask] * dist.bin_values[mask]))

    if tail_prob <= 0:
        # If the tail is empty under the representative mapping, fall back to the max bin value.
        tvar = float(np.max(dist.bin_values))
    else:
        tvar = tail_num / tail_prob

    return ClassicalTVaRResult(
        alpha=alpha,
        tvar=tvar,
        var_used=var_t,
        tail_prob=tail_prob,
        tail_mean_numerator=tail_num,
    )
