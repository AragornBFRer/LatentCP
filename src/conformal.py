"""Split-conformal quantile computation."""
from __future__ import annotations

import numpy as np


def split_conformal(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("Need at least one calibration score")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = max(1, min(k, n))
    # use partition for O(n)
    idx = k - 1
    qhat = np.partition(scores, idx)[idx]
    return float(qhat)
