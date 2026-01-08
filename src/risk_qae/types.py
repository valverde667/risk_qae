from __future__ import annotations

from typing import Any, Sequence, TypedDict


class LossData(TypedDict, total=False):
    """Input container for a 1D loss distribution."""

    samples: Sequence[float]
    pmf: Sequence[float]
    bin_values: Sequence[float]
    metadata: dict[str, Any]
