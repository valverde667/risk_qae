"""Backend adapters.

The core package is backend-agnostic and works with Qiskit primitives.

Backends are optional and may require additional dependencies.
"""

from ..config import BackendConfig
from ..types import BackendHandle

from .factory import get_backend

__all__ = ["BackendConfig", "BackendHandle", "get_backend"]
