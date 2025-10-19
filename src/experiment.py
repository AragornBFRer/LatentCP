"""Experiment orchestration for latent conformal simulations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ExperimentConfig, RunConfig, iter_run_configs, load_config
from .conformal import split_conformal
from .data_gen import DatasetSplit, generate_data
from .em_gmm import GMMParams, fit_gmm_em, gmm_responsibilities
from .metrics import avg_length, coverage, cross_entropy, hard_accuracy, mean_max_tau
from .predictors import EMHardPredictor, EMSoftPredictor, IgnoreZPredictor, OracleZPredictor
from .utils import ensure_dir, rng_from_seed


def _combo_seed(run_cfg: RunConfig) -> int:
    payload = f"{run_cfg.seed}|{run_cfg.K}|{run_cfg.delta}|{run_cfg.beta_spread}|{int(run_cfg.use_x_in_em)}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _build_em_features(split: DatasetSplit, use_x: bool) -> np.ndarray:
    if use_x:
        return np.hstack([split.R, split.X])
    return split.R


def _run_single(cfg: ExperimentConfig, run_cfg: RunConfig) -> Dict[str, float]:
    rng = rng_from_seed(_combo_seed(run_cfg))
    train, cal, test, true_params = generate_data(
        cfg.dgp_cfg, run_cfg, cfg.global_cfg.n_train, cfg.global_cfg.n_cal, cfg.global_cfg.n_test, rng
    )

    K_fit = cfg.em_cfg.K_fit or run_cfg.K
    S_train = _build_em_features(train, run_cfg.use_x_in_em)
    S_cal = _build_em_features(cal, run_cfg.use_x_in_em)
    S_test = _build_em_features(test, run_cfg.use_x_in_em)

    em_params = fit_gmm_em(
        S_train,
        K_fit,
        cov_type=cfg.em_cfg.cov_type_fit,
        max_iter=cfg.em_cfg.max_iter,
        tol=cfg.em_cfg.tol,
        reg_covar=cfg.em_cfg.reg_covar,
        init=cfg.em_cfg.init,
        rng=rng,
    )

    tau_train = gmm_responsibilities(S_train, em_params)
    tau_cal = gmm_responsibilities(S_cal, em_params)
    tau_test = gmm_responsibilities(S_test, em_params)

    zhat_train = tau_train.argmax(axis=1)
    zhat_cal = tau_cal.argmax(axis=1)
    zhat_test = tau_test.argmax(axis=1)

    oracle = OracleZPredictor().fit(train.X, train.Y, Z_true=train.Z)
    hard = EMHardPredictor().fit(train.X, train.Y, zhat=zhat_train)
    soft = EMSoftPredictor(
        alt_iters=cfg.model_cfg.soft_alt_iters,
        tol=cfg.model_cfg.soft_tol,
        class_specific_slopes=cfg.model_cfg.soft_class_specific_slopes,
    ).fit(train.X, train.Y, tau=tau_train)
    ignore = IgnoreZPredictor().fit(train.X, train.Y)

    scores = {
        "oracle": np.abs(cal.Y - oracle.predict_mean(cal.X, Z_true=cal.Z)),
        "hard": np.abs(cal.Y - hard.predict_mean(cal.X, zhat=zhat_cal)),
        "soft": np.abs(cal.Y - soft.predict_mean(cal.X, tau=tau_cal)),
        "ignore": np.abs(cal.Y - ignore.predict_mean(cal.X)),
    }

    qhat = {name: split_conformal(vals, cfg.global_cfg.alpha) for name, vals in scores.items()}

    preds_test = {
        "oracle": oracle.predict_mean(test.X, Z_true=test.Z),
        "hard": hard.predict_mean(test.X, zhat=zhat_test),
        "soft": soft.predict_mean(test.X, tau=tau_test),
        "ignore": ignore.predict_mean(test.X),
    }

    results = {}
    for name in preds_test:
        q = qhat[name]
        mu = preds_test[name]
        results[f"coverage_{name}"] = coverage(test.Y, mu, q)
        results[f"length_{name}"] = avg_length(q)

    results["len_gap_soft"] = results["length_soft"] - results["length_oracle"]
    results["len_gap_hard"] = results["length_hard"] - results["length_oracle"]
    results["len_gap_ignore"] = results["length_ignore"] - results["length_oracle"]

    results["acc_hard"] = hard_accuracy(zhat_test, test.Z)
    results["mean_max_tau"] = mean_max_tau(tau_test)
    results["cross_entropy"] = cross_entropy(tau_test, test.Z)

    results.update(
        {
            "seed": run_cfg.seed,
            "K": run_cfg.K,
            "delta": run_cfg.delta,
            "beta_spread": run_cfg.beta_spread,
            "use_x_in_em": run_cfg.use_x_in_em,
            "em_converged": em_params.converged,
            "em_iter": em_params.n_iter,
            "em_loglik": em_params.log_likelihood,
        }
    )

    return results


def run_experiment(cfg_path: str) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    rows: List[Dict[str, float]] = []
    for run_cfg in iter_run_configs(cfg):
        rows.append(_run_single(cfg, run_cfg))

    df = pd.DataFrame(rows)
    results_path = Path(cfg.io_cfg.results_csv)
    ensure_dir(results_path)
    if results_path.exists():
        prev = pd.read_csv(results_path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(results_path, index=False)
    df.attrs["results_path"] = str(results_path)
    return df
