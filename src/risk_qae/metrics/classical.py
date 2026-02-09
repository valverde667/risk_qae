from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..discretization.histogram import DiscretizedDistribution


# =============================================================================
# Canonical classical metrics on DiscretizedDistribution (stable API)
# =============================================================================


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


# =============================================================================
# Debug / validation utilities (sample-side + circuit-aligned scalings)
# =============================================================================


@dataclass(frozen=True)
class ClassicalSampleSummary:
    """
    Sample-side sanity checks (useful when your dist was made via clipping/winsorization).
    """

    mean_raw: float
    mean_winsor: float
    winsor_bounds: tuple[float, float]
    mass_above_xmax: Optional[float]


def summarize_samples(
    samples: Sequence[float] | np.ndarray,
    *,
    lower_q: float = 0.0,
    upper_q: float = 0.999,
    xmax: float | None = None,
) -> ClassicalSampleSummary:
    """
    Returns raw mean, winsorized mean (by quantile clipping), and optionally mass above xmax.

    - winsor_bounds are the (lo, hi) quantile values used for clipping.
    - mass_above_xmax is None if xmax is not provided.
    """
    if not (0.0 <= lower_q < 1.0) or not (0.0 < upper_q <= 1.0) or lower_q >= upper_q:
        raise ValueError("Require 0 <= lower_q < upper_q <= 1")

    x = np.asarray(samples, dtype=float)

    mean_raw = float(np.mean(x))
    lo = float(np.quantile(x, lower_q))
    hi = float(np.quantile(x, upper_q))
    mean_winsor = float(np.mean(np.clip(x, lo, hi)))

    mass = None if xmax is None else float(np.mean(x > float(xmax)))

    return ClassicalSampleSummary(
        mean_raw=mean_raw,
        mean_winsor=mean_winsor,
        winsor_bounds=(lo, hi),
        mass_above_xmax=mass,
    )


@dataclass(frozen=True)
class ClassicalScaledMeanResult:
    """
    a_mean is the classical value your mean AE problem should be estimating if the
    value encoding is f(x) = (x-xmin)/(xmax-xmin) clipped to [0,1].
    """

    a_mean: float
    xmin: float
    xmax: float


def classical_scaled_mean_amplitude(
    dist: DiscretizedDistribution,
) -> ClassicalScaledMeanResult:
    xmin, xmax = map(float, dist.bounds)
    rng = xmax - xmin
    if rng <= 0:
        raise ValueError(f"Invalid bounds: {dist.bounds}")

    frac = (dist.bin_values - xmin) / rng
    frac = np.clip(frac, 0.0, 1.0)
    a_mean = float(np.sum(dist.pmf * frac))

    return ClassicalScaledMeanResult(a_mean=a_mean, xmin=xmin, xmax=xmax)


def classical_tvar_index_tail(
    dist: DiscretizedDistribution, alpha: float
) -> ClassicalTVaRResult:
    """
    Debug TVaR variant that defines the tail by bin index rather than value comparison.

    This is often closer to how indicator circuits are implemented (tail = {i >= i*}),
    and can help diagnose inconsistencies due to using VaR as an *upper edge* while
    tail membership is checked against bin_values.
    """
    var_res = classical_var(dist, alpha)
    idx = var_res.var_bin_index

    mask = np.arange(dist.pmf.size) >= idx
    tail_prob = float(np.sum(dist.pmf[mask]))
    tail_num = float(np.sum(dist.pmf[mask] * dist.bin_values[mask]))

    if tail_prob <= 0:
        tvar = float(np.max(dist.bin_values))
    else:
        tvar = tail_num / tail_prob

    return ClassicalTVaRResult(
        alpha=alpha,
        tvar=tvar,
        var_used=var_res.var,
        tail_prob=tail_prob,
        tail_mean_numerator=tail_num,
    )
