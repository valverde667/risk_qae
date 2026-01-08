from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BoundsConfig, DiscretizationConfig
from ..types import LossData
from ..utils.validation import validate_loss_data


@dataclass(frozen=True)
class DiscretizedDistribution:
    n_index_qubits: int
    pmf: np.ndarray
    bin_edges: np.ndarray
    bin_values: np.ndarray
    bounds: tuple[float, float]
    scale_shift: float
    scale_factor: float


class HistogramDiscretizer:
    def fit(
        self,
        data: LossData,
        dcfg: DiscretizationConfig = DiscretizationConfig(),
        bcfg: BoundsConfig = BoundsConfig(),
    ) -> DiscretizedDistribution:
        """Discretize input into a PMF on N=2**n uniform bins.

        - If `samples` is provided, bins are constructed on clipped percentile bounds.
        - If `pmf` and `bin_values` are provided, the PMF is normalized and validated.
        """
        validate_loss_data(data)

        if data.get("samples") is None:
            pmf = np.asarray(data["pmf"], dtype=float)
            bin_values = np.asarray(data["bin_values"], dtype=float)
            pmf = pmf / float(pmf.sum())

            n = int(np.log2(pmf.size))
            if 2**n != pmf.size:
                raise ValueError("pmf length must be a power of 2 for index-register mapping")

            x_min = float(np.min(bin_values))
            x_max = float(np.max(bin_values))
            denom = max(x_max - x_min, bcfg.epsilon)
            edges = np.linspace(x_min, x_max, pmf.size + 1, dtype=float)
            return DiscretizedDistribution(
                n_index_qubits=n,
                pmf=pmf,
                bin_edges=edges,
                bin_values=bin_values,
                bounds=(x_min, x_max),
                scale_shift=x_min,
                scale_factor=1.0 / denom,
            )

        samples = np.asarray(data["samples"], dtype=float)

        n = int(dcfg.n_index_qubits)
        if n < 1:
            raise ValueError("n_index_qubits must be >= 1")
        N = 2**n

        p_low, p_high = bcfg.clip_percentiles
        x_min = float(np.percentile(samples, p_low))
        x_max = float(np.percentile(samples, p_high))
        if not np.isfinite([x_min, x_max]).all():
            raise ValueError("percentile clipping produced non-finite bounds")
        if (x_max - x_min) < bcfg.epsilon:
            x_min = float(np.min(samples))
            x_max = float(np.max(samples))
        if (x_max - x_min) < bcfg.epsilon:
            x_min = float(samples[0])
            x_max = x_min + 1.0

        clipped = np.clip(samples, x_min, x_max)
        counts, edges = np.histogram(clipped, bins=N, range=(x_min, x_max))
        total = float(np.sum(counts))
        if total <= 0:
            raise ValueError("histogram has zero total count")
        pmf = counts.astype(float) / total

        if dcfg.value_repr == "lower_edge":
            bin_values = edges[:-1]
        else:
            bin_values = 0.5 * (edges[:-1] + edges[1:])

        denom = max(x_max - x_min, bcfg.epsilon)
        return DiscretizedDistribution(
            n_index_qubits=n,
            pmf=pmf,
            bin_edges=edges,
            bin_values=bin_values,
            bounds=(x_min, x_max),
            scale_shift=x_min,
            scale_factor=1.0 / denom,
        )
