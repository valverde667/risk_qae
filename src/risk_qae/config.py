from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class DiscretizationConfig:
    """How to map samples into a discrete distribution over 2**n bins."""

    n_index_qubits: int = 8
    method: str = "uniform_bins"
    value_repr: str = "midpoint"


@dataclass(frozen=True)
class BoundsConfig:
    """Data-driven clipping and scaling bounds for value encoding."""

    strategy: str = "data_driven_clipped"
    clip_percentiles: tuple[float, float] = (0.1, 99.9)
    epsilon: float = 1e-12


from typing import Sequence  # add near top (or just use tuple[...] below)


@dataclass(frozen=True)
class AEConfig:
    """Amplitude estimation algorithm selection.

    method:
      - "budgeted_fixed_schedule" (current v0.1): direct sampling (BudgetedAERunner)
      - "grover_mle": Grover schedule + MLE fit (GroverMLEAERunner)
    """

    method: str = "budgeted_fixed_schedule"  # v0.1 default

    # Grover-MLE options (used only when method == "grover_mle")
    grover_mle_grid_size: int = 2000
    grover_powers: tuple[int, ...] | None = (
        None  # e.g. (0,1,2,4); None => auto schedule
    )


@dataclass(frozen=True)
class BudgetConfig:
    """Shot-budget controls (v0.1 focuses on shots-first UX)."""

    total_shots: int = 50_000
    shots_per_call: int = 2_000
    allocation: str = "flat"
    max_circuit_calls: int = 200
    seed: int | None = None


@dataclass(frozen=True)
class VaRSearchConfig:
    """Outer-loop search configuration for VaR."""

    method: str = "bisection_index"
    target_bin_resolution: int = 1
    max_iters: int = 32
    shots_fraction_of_total: float = 0.7


@dataclass(frozen=True)
class BackendConfig:
    """Backend selection and transpilation knobs."""

    provider: str = "quantum_rings"  # or "aer"
    qr_workspace: str | None = None
    qr_backend_name: str | None = None
    transpile_optimization_level: int = 1


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Controls for returning optional diagnostics."""

    return_circuits: bool = False
    return_histograms: bool = True
    return_runtime_breakdown: bool = True


@dataclass
class ValueEncodingConfig:
    method: str = "piecewise_prefix"  # or "table"
    n_segments: int = 16  # must be power of two


@dataclass(frozen=True)
class RiskQAEConfig:
    """Top-level configuration container."""

    discretization: DiscretizationConfig = DiscretizationConfig()
    bounds: BoundsConfig = BoundsConfig()
    ae: AEConfig = AEConfig()
    budget: BudgetConfig = BudgetConfig()
    var_search: VaRSearchConfig = VaRSearchConfig()
    backend: BackendConfig = BackendConfig()
    value_encoding: ValueEncodingConfig = field(default_factory=ValueEncodingConfig)
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
