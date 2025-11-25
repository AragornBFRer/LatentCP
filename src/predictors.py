"""Lightweight regression baselines for latent conformal experiments."""
from __future__ import annotations

import numpy as np


def _augment(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


def _solve_linear(design: np.ndarray, target: np.ndarray, ridge_alpha: float) -> np.ndarray:
    gram = design.T @ design
    rhs = design.T @ target
    if ridge_alpha > 0.0 and design.shape[1] > 1:
        penalty = ridge_alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0  # leave the intercept unpenalized
        gram = gram + penalty
    try:
        sol = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(gram) @ rhs
    return sol


class Predictor:
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def variant_name(self) -> str:
        return self.__class__.__name__


class IgnoreZPredictor(Predictor):
    def __init__(self, ridge_alpha: float = 0.0) -> None:
        self.ridge_alpha = ridge_alpha
        self.coef: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "IgnoreZPredictor":
        design = _augment(X)
        self.coef = _solve_linear(design, Y, self.ridge_alpha)
        return self

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("Model not fitted")
        design = _augment(X)
        return design @ self.coef


