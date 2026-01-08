from __future__ import annotations

import numpy as np

from ..types import LossData


class LossDataError(ValueError):
    pass


def validate_loss_data(data: LossData) -> None:
    has_samples = data.get("samples") is not None
    has_pmf = data.get("pmf") is not None
    has_bin_values = data.get("bin_values") is not None

    if has_samples and (has_pmf or has_bin_values):
        raise LossDataError("Provide either `samples` or (`pmf` and `bin_values`), not both.")

    if has_samples:
        samples = np.asarray(data["samples"], dtype=float)
        if samples.size == 0:
            raise LossDataError("`samples` is empty.")
        if not np.isfinite(samples).all():
            raise LossDataError("`samples` contains NaN or infinite values.")
        return
    if has_pmf and has_bin_values:
        pmf = np.asarray(data["pmf"], dtype=float)
        bin_values = np.asarray(data["bin_values"], dtype=float)
        if pmf.size == 0:
            raise LossDataError("`pmf` is empty.")
        if pmf.shape != bin_values.shape:
            raise LossDataError("`pmf` and `bin_values` must have the same length.")
        if (pmf < 0).any():
            raise LossDataError("`pmf` must be non-negative.")
        if not np.isfinite(pmf).all() or not np.isfinite(bin_values).all():
            raise LossDataError("`pmf`/`bin_values` contains NaN or infinite values.")
        if float(pmf.sum()) <= 0:
            raise LossDataError("`pmf` must sum to > 0.")
        return

    raise LossDataError("Provide either `samples` or (`pmf` and `bin_values`).")
