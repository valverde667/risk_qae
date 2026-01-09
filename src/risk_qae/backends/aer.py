from __future__ import annotations

from ..config import BackendConfig
from ..types import BackendHandle


def get_aer_backend(cfg: BackendConfig, **kwargs) -> BackendHandle:
    """Return a Qiskit Aer-based BackendHandle (stub).

    This will be implemented when we wire amplitude estimation execution.
    """
    try:
        from qiskit_aer.primitives import SamplerV2, EstimatorV2  # type: ignore
        from qiskit_aer import Aer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit Aer is not installed. Install with: pip install 'risk_qae[aer]'"
        ) from exc

    backend = Aer.get_backend("aer_simulator")
    sampler = SamplerV2(backend=backend)
    estimator = EstimatorV2(backend=backend)
    return BackendHandle(
        sampler=sampler,
        estimator=estimator,
        backend=backend,
        metadata={"provider": "aer", "backend_name": getattr(backend, "name", None)},
    )
