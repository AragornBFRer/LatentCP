"""Synthetic data generation for latent-cluster experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import DGPConfig, RunConfig
from .utils import simplex_vertices


@dataclass
class DatasetSplit:
    X: np.ndarray
    R: np.ndarray
    Y: np.ndarray
    Z: np.ndarray


def _build_beta(K: int, spread: float) -> np.ndarray:
    if spread == 0.0:
        return np.zeros(K)
    offsets = np.linspace(-(K - 1) / 2.0, (K - 1) / 2.0, K)
    offsets -= offsets.mean()
    return spread * offsets


def _make_covariance(dgp: DGPConfig, use_full_S: bool) -> np.ndarray:
    d_r = dgp.d_R
    d_x = dgp.d_X
    sigma2 = dgp.sigma_s ** 2
    if use_full_S:
        cov_rr = sigma2 * np.eye(d_r)
        cov_xx = sigma2 * np.eye(d_x)
        if dgp.cov_type_true == "full" and dgp.rho_rx != 0.0:
            scale = dgp.rho_rx * sigma2
            cov_rx = scale * np.eye(d_r, d_x)
        else:
            cov_rx = np.zeros((d_r, d_x))
        top = np.hstack([cov_rr, cov_rx])
        bottom = np.hstack([cov_rx.T, cov_xx])
        cov = np.vstack([top, bottom])
        if dgp.cov_type_true == "diag":
            cov = np.diag(np.diag(cov))
    else:
        cov = sigma2 * np.eye(d_r)
        if dgp.cov_type_true == "diag":
            cov = np.diag(np.diag(cov))
    cov += 1e-9 * np.eye(cov.shape[0])
    return cov


def generate_data(
    dgp_cfg: DGPConfig,
    run_cfg: RunConfig,
    n_train: int,
    n_cal: int,
    n_test: int,
    rng: np.random.Generator,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, Dict[str, np.ndarray]]:
    K = run_cfg.K
    d_r = dgp_cfg.d_R
    d_x = dgp_cfg.d_X
    total = n_train + n_cal + n_test

    means_r = simplex_vertices(K, d_r, run_cfg.delta, allow_smaller_dim=True)

    use_full_S = dgp_cfg.use_S.upper() == "RX"
    if use_full_S:
        means_full = np.zeros((K, d_r + d_x))
        means_full[:, :d_r] = means_r
        # keep X means at zero unless delta also controls them later
    else:
        means_full = None

    pi = np.full(K, 1.0 / K)
    z = rng.choice(K, size=total, p=pi)

    R = np.empty((total, d_r))
    X = np.empty((total, d_x))

    if use_full_S:
        cov = _make_covariance(dgp_cfg, True)
        for k in range(K):
            mask = z == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            R_and_X = rng.multivariate_normal(mean=means_full[k], cov=cov, size=n_k)
            R[mask] = R_and_X[:, :d_r]
            X[mask] = R_and_X[:, d_r:]
    else:
        cov_r = _make_covariance(dgp_cfg, False)
        for k in range(K):
            mask = z == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            R[mask] = rng.multivariate_normal(mean=means_r[k], cov=cov_r, size=n_k)
        # X independent standard normal
        X[:] = rng.normal(size=(total, d_x))

    theta = rng.normal(scale=1.0 / max(1, np.sqrt(d_x)), size=d_x)
    beta = _build_beta(K, run_cfg.beta_spread)
    noise = rng.normal(scale=dgp_cfg.sigma_y, size=total)

    linear = X @ theta
    cluster_shift = beta[z]
    Y = linear + cluster_shift + noise

    train_slice = slice(0, n_train)
    cal_slice = slice(n_train, n_train + n_cal)
    test_slice = slice(n_train + n_cal, total)

    train = DatasetSplit(X=X[train_slice], R=R[train_slice], Y=Y[train_slice], Z=z[train_slice])
    cal = DatasetSplit(X=X[cal_slice], R=R[cal_slice], Y=Y[cal_slice], Z=z[cal_slice])
    test = DatasetSplit(X=X[test_slice], R=R[test_slice], Y=Y[test_slice], Z=z[test_slice])

    info = {
        "pi": pi,
        "means_r": means_r,
        "theta": theta,
        "beta": beta,
    }
    if means_full is not None:
        info["means_full"] = means_full

    return train, cal, test, info
