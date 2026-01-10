from __future__ import annotations

from ..config import BackendConfig
from ..types import BackendHandle

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2


def get_aer_backend(cfg: BackendConfig, **kwargs) -> BackendHandle:
    aer_backend = AerSimulator()
    sampler = AerSamplerV2(
        default_shots=kwargs.get("default_shots", 1024),
        seed=kwargs.get("seed", None),
        options=kwargs.get("options", None),
    )
    estimator = AerEstimatorV2(options=kwargs.get("options", None))
    return BackendHandle(sampler=sampler, estimator=estimator, backend=None)
