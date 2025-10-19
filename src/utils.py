"""Utility helpers for reproducible experiments."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Return a NumPy Generator seeded by ``seed`` (defaults to entropy)."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed NumPy's global RNG and return a Generator for local use."""
    g = rng_from_seed(seed)
    np.random.seed(int(seed))
    return g


def ensure_dir(path: str | Path) -> None:
    """Create parent directory for ``path`` if needed."""
    target = Path(path)
    if target.suffix:
        target = target.parent
    target.mkdir(parents=True, exist_ok=True)


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    max_a = np.max(a, axis=axis, keepdims=True)
    shifted = np.exp(a - max_a)
    s = np.log(np.sum(shifted, axis=axis, keepdims=True))
    out = max_a + s
    if axis is None:
        return out.squeeze()
    return np.squeeze(out, axis=axis)


def simplex_vertices(k: int, dim: int, scale: float, *, allow_smaller_dim: bool = False) -> np.ndarray:
    """Place ``k`` points on a centered simplex embedded in ``dim`` dimensions."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k == 1:
        return np.zeros((1, dim))
    if dim < k - 1 and not allow_smaller_dim:
        raise ValueError("Need dim >= k-1 to embed simplex without distortion")
    base = np.eye(k, dtype=float) - 1.0 / k
    verts = np.zeros((k, dim), dtype=float)
    cols = min(dim, base.shape[1])
    verts[:, :cols] = base[:, :cols]
    if dim > base.shape[1]:
        # remaining coordinates stay at zero (already set)
        pass
    delta_current = np.linalg.norm(verts[0] - verts[1])
    if not math.isfinite(delta_current) or delta_current == 0.0:
        raise ValueError("Simplex construction failed")
    if scale == 0.0:
        return np.zeros_like(verts)
    factor = scale / delta_current
    return verts * factor


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer labels of shape (n,)."""
    n = labels.shape[0]
    out = np.zeros((n, num_classes), dtype=float)
    out[np.arange(n), labels] = 1.0
    return out


def center_columns(arr: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return column-centered copy of ``arr`` with optional weights."""
    if weights is None:
        mean = arr.mean(axis=0, keepdims=True)
    else:
        wsum = np.sum(weights)
        if wsum == 0.0:
            return arr.copy()
        mean = np.sum(arr * weights[:, None], axis=0, keepdims=True) / wsum
    return arr - mean
