from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AEResult:
    """Result from an amplitude-estimation run."""

    estimate: float
    ci: tuple[float, float] | None
    shots_used: int
    circuits_run: int
    diagnostics: Mapping[str, Any] = None  # type: ignore[assignment]
