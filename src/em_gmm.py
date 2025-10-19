"""Expectation-maximisation for Gaussian mixture models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .utils import logsumexp


@dataclass
class GMMParams:
    pi: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    cov_type: str
    converged: bool
    n_iter: int
    log_likelihood: float

    def as_dict(self) -> dict:
        return {
            "pi": self.pi,
            "means": self.means,
            "covs": self.covs,
            "cov_type": self.cov_type,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "log_likelihood": self.log_likelihood,
        }


def _init_means(data: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    if n < K:
        raise ValueError("Not enough samples to initialise means")
    idx = rng.choice(n, size=K, replace=False)
    means = data[idx].copy()
    for _ in range(5):
        dists = np.sum((data[:, None, :] - means[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                means[k] = data[mask].mean(axis=0)
            else:
                means[k] = data[rng.integers(0, n)]
    return means


def _estimate_log_gaussian_full(data: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = data.shape[1]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-6 * np.eye(d)
        chol = np.linalg.cholesky(cov)
    solve = np.linalg.solve(chol, (data - mean).T)
    quad = np.sum(solve ** 2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


def _estimate_log_gaussian_diag(data: np.ndarray, mean: np.ndarray, diag_cov: np.ndarray) -> np.ndarray:
    prec = 1.0 / diag_cov
    quad = np.sum(((data - mean) ** 2) * prec, axis=1)
    log_det = np.sum(np.log(diag_cov))
    d = data.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


def _estimate_log_prob(
    data: np.ndarray,
    pi: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    cov_type: str,
) -> np.ndarray:
    n, d = data.shape
    K = means.shape[0]
    log_prob = np.empty((n, K))
    for k in range(K):
        if cov_type == "full":
            log_pdf = _estimate_log_gaussian_full(data, means[k], covs[k])
        else:
            log_pdf = _estimate_log_gaussian_diag(data, means[k], covs[k])
        log_prob[:, k] = np.log(pi[k] + 1e-16) + log_pdf
    return log_prob


def _m_step(
    data: np.ndarray,
    tau: np.ndarray,
    cov_type: str,
    reg_covar: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = data.shape
    K = tau.shape[1]
    Nk = tau.sum(axis=0) + 1e-16
    pi = Nk / n
    means = (tau.T @ data) / Nk[:, None]
    covs = []
    for k in range(K):
        diff = data - means[k]
        weights = tau[:, k][:, None]
        if cov_type == "full":
            cov = (weights * diff).T @ diff / max(Nk[k], 1e-12)
            cov += reg_covar * np.eye(d)
        else:
            cov = np.sum(weights * (diff ** 2), axis=0) / max(Nk[k], 1e-12)
            cov += reg_covar
        covs.append(cov)
    covs = np.array(covs)
    return pi, means, covs


def fit_gmm_em(
    data: np.ndarray,
    K: int,
    *,
    cov_type: str = "full",
    max_iter: int = 200,
    tol: float = 1e-5,
    reg_covar: float = 1e-6,
    init: str = "kmeans",
    rng: np.random.Generator | None = None,
) -> GMMParams:
    data = np.asarray(data, dtype=float)
    n, d = data.shape
    if rng is None:
        rng = np.random.default_rng()

    if init.lower() == "kmeans":
        means = _init_means(data, K, rng)
    else:
        means = data[rng.choice(n, size=K, replace=False)]
    cov_init = np.cov(data.T) + reg_covar * np.eye(d)
    if cov_type == "diag":
        covs = np.tile(np.diag(cov_init), (K, 1))
    else:
        covs = np.tile(cov_init, (K, 1, 1))
    pi = np.full(K, 1.0 / K)

    log_prob = _estimate_log_prob(data, pi, means, covs, cov_type)
    prev_ll = np.sum(logsumexp(log_prob, axis=1))
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        log_resp = log_prob - logsumexp(log_prob, axis=1)[:, None]
        tau = np.exp(log_resp)
        pi, means, covs = _m_step(data, tau, cov_type, reg_covar)
        log_prob = _estimate_log_prob(data, pi, means, covs, cov_type)
        ll = np.sum(logsumexp(log_prob, axis=1))
        improvement = ll - prev_ll
        if improvement <= tol * (1.0 + abs(prev_ll)):
            converged = True
            n_iter = it
            break
        prev_ll = ll
        n_iter = it
    else:
        ll = prev_ll

    return GMMParams(
        pi=pi,
        means=means,
        covs=covs,
        cov_type=cov_type,
        converged=converged,
        n_iter=n_iter,
        log_likelihood=float(np.sum(logsumexp(log_prob, axis=1))),
    )


def gmm_responsibilities(data: np.ndarray, params: GMMParams) -> np.ndarray:
    log_prob = _estimate_log_prob(data, params.pi, params.means, params.covs, params.cov_type)
    log_resp = log_prob - logsumexp(log_prob, axis=1)[:, None]
    tau = np.exp(log_resp)
    tau /= tau.sum(axis=1, keepdims=True)
    return tau
