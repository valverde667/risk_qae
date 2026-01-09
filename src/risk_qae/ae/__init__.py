"""Amplitude estimation interfaces (Stage 2).

Stage 2 defines the runner interface. Stage 3 will implement budgeted AE.
"""

from .results import AEResult
from .budgeted_ae import BudgetedAERunner

__all__ = ["AEResult", "BudgetedAERunner"]
