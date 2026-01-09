from __future__ import annotations

from ..config import BackendConfig
from ..types import BackendHandle


def get_quantum_rings_backend(cfg: BackendConfig, **kwargs) -> BackendHandle:
    """Return a Quantum Rings BackendHandle (stub).

    This adapter is intentionally thin and will be expanded when we implement
    execution in Stage 3. It is expected to construct primitives like
    QrSamplerV2 / QrEstimatorV2 and return them in a BackendHandle.

    Quantum Rings' Qiskit Toolkit package name and import paths may vary by
    installation; we keep imports local and provide a clear error message.
    """
    try:
        # These names are documented by Quantum Rings; exact modules may differ by version.
        from quantumrings.toolkit.qiskit import QrRuntimeService, QrSamplerV2, QrEstimatorV2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Quantum Rings Qiskit Toolkit is not installed or import paths differ. "
            "Install the Quantum Rings toolkit and ensure it is on your PYTHONPATH."
        ) from exc

    workspace = cfg.qr_workspace or kwargs.get("qr_workspace")
    backend_name = cfg.qr_backend_name or kwargs.get("qr_backend_name")
    service = (
        QrRuntimeService(workspace=workspace)
        if workspace is not None
        else QrRuntimeService()
    )
    backend = service.backend(backend_name) if backend_name else service.backend()

    sampler = QrSamplerV2(backend=backend)
    estimator = QrEstimatorV2(backend=backend)
    return BackendHandle(
        sampler=sampler,
        estimator=estimator,
        backend=backend,
        metadata={
            "provider": "quantum_rings",
            "workspace": workspace,
            "backend_name": backend_name,
        },
    )
