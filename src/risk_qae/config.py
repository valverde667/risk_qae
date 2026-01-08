from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscretizationConfig:
    n_index_qubits: int = 8
    method: str = "uniform_bins"
    value_repr: str = "midpoint"


@dataclass(frozen=True)
class BoundsConfig:
    strategy: str = "data_driven_clipped"
    clip_percentiles: tuple[float, float] = (0.1, 99.9)
    epsilon: float = 1e-12


@dataclass(frozen=True)
class RiskQAEConfig:
    discretization: DiscretizationConfig = DiscretizationConfig()
    bounds: BoundsConfig = BoundsConfig()
