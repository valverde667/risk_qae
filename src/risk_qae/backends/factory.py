from __future__ import annotations

from ..config import BackendConfig
from ..types import BackendHandle


def get_backend(cfg: BackendConfig, **kwargs) -> BackendHandle:
    """Create a BackendHandle based on configuration.

    Parameters
    ----------
    cfg:
        BackendConfig selecting provider and provider-specific settings.

    Returns
    -------
    BackendHandle
        Minimal handle containing primitives/backend objects.

    Notes
    -----
    In v0.1 Stage 2 this is a thin dispatcher. Provider adapters will be
    fleshed out in later stages once we add execution support.
    """
    provider = (cfg.provider or "").lower()
    if provider == "quantum_rings":
        from .quantum_rings import get_quantum_rings_backend

        return get_quantum_rings_backend(cfg, **kwargs)

    if provider == "aer":
        from .aer import get_aer_backend

        return get_aer_backend(cfg, **kwargs)

    raise ValueError(f"Unknown backend provider: {cfg.provider!r}")
