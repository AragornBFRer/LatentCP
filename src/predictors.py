"""Regression models using oracle or EM-imputed cluster labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import one_hot


def _augment(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


class Predictor:
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def variant_name(self) -> str:
        return self.__class__.__name__


class IgnoreZPredictor(Predictor):
    def __init__(self) -> None:
        self.theta: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "IgnoreZPredictor":
        design = _augment(X)
        sol, *_ = np.linalg.lstsq(design, Y, rcond=None)
        self.theta = sol
        return self

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("Model not fitted")
        design = _augment(X)
        return design @ self.theta


class OracleZPredictor(Predictor):
    def __init__(self) -> None:
        self.theta: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, *, Z_true: np.ndarray, **kwargs) -> "OracleZPredictor":
        Z = Z_true.astype(int)
        K = int(Z.max()) + 1
        design = _augment(X)
        d = design.shape[1]
        if K > 1:
            dummies = one_hot(Z, K)[:, 1:]
            design = np.hstack([design, dummies])
        sol, *_ = np.linalg.lstsq(design, Y, rcond=None)
        self.theta = sol[:d]
        self.beta = np.zeros(K)
        if K > 1:
            self.beta[1:] = sol[d:]
        return self

    def predict_mean(self, X: np.ndarray, *, Z_true: np.ndarray, **kwargs) -> np.ndarray:
        if self.theta is None or self.beta is None:
            raise RuntimeError("Model not fitted")
        design = _augment(X)
        base = design @ self.theta
        adj = self.beta[Z_true.astype(int)]
        return base + adj


class EMHardPredictor(Predictor):
    def __init__(self) -> None:
        self.theta: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, *, zhat: np.ndarray, **kwargs) -> "EMHardPredictor":
        labels = zhat.astype(int)
        K = int(labels.max()) + 1
        design = _augment(X)
        d = design.shape[1]
        if K > 1:
            dummies = one_hot(labels, K)[:, 1:]
            design = np.hstack([design, dummies])
        sol, *_ = np.linalg.lstsq(design, Y, rcond=None)
        self.theta = sol[:d]
        self.beta = np.zeros(K)
        if K > 1:
            self.beta[1:] = sol[d:]
        return self

    def predict_mean(self, X: np.ndarray, *, zhat: np.ndarray, **kwargs) -> np.ndarray:
        if self.theta is None or self.beta is None:
            raise RuntimeError("Model not fitted")
        design = _augment(X)
        base = design @ self.theta
        adj = self.beta[zhat.astype(int)]
        return base + adj


@dataclass
class SoftParams:
    theta: np.ndarray
    beta: np.ndarray
    theta_per_class: np.ndarray | None


class EMSoftPredictor(Predictor):
    def __init__(self, *, alt_iters: int = 10, tol: float = 1e-8, class_specific_slopes: bool = False) -> None:
        self.alt_iters = alt_iters
        self.tol = tol
        self.class_specific_slopes = class_specific_slopes
        self.params: SoftParams | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, *, tau: np.ndarray, **kwargs) -> "EMSoftPredictor":
        K = tau.shape[1]
        design = _augment(X)
        # initial solution ignoring Z
        sol, *_ = np.linalg.lstsq(design, Y, rcond=None)
        theta = sol
        beta = np.zeros(K)
        theta_k = None

        weights = tau.sum(axis=1)
        weights = np.clip(weights, 1e-8, None)
        target = Y.copy()

        for _ in range(self.alt_iters):
            residual = target - design @ theta
            denom = tau.sum(axis=0) + 1e-12
            beta = (tau * residual[:, None]).sum(axis=0) / denom
            beta -= np.average(beta, weights=denom)
            target = Y - tau @ beta
            W = np.sqrt(weights)[:, None]
            lhs = W * design
            rhs = target * np.sqrt(weights)
            theta_new, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            if np.linalg.norm(theta_new - theta) <= self.tol * (1.0 + np.linalg.norm(theta)):
                theta = theta_new
                break
            theta = theta_new

        if self.class_specific_slopes:
            theta_k = np.zeros((K, design.shape[1]))
            for k in range(K):
                w = np.sqrt(np.clip(tau[:, k], 0.0, None))
                if w.sum() <= 1e-8:
                    theta_k[k] = theta
                    continue
                lhs = design * w[:, None]
                rhs = Y * w
                sol_k, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
                theta_k[k] = sol_k

        self.params = SoftParams(theta=theta, beta=beta, theta_per_class=theta_k)
        return self

    def predict_mean(self, X: np.ndarray, *, tau: np.ndarray, **kwargs) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model not fitted")
        design = _augment(X)
        if self.class_specific_slopes and self.params.theta_per_class is not None:
            preds = design @ self.params.theta_per_class.T
            return np.sum(tau * preds, axis=1)
        base = design @ self.params.theta
        adjust = tau @ self.params.beta
        return base + adjust
