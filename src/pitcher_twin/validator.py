"""Classifier-based realism validation for simulated pitches."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def temporal_train_holdout(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_cols = [column for column in ["game_date", "game_pk", "pitch_number"] if column in df.columns]
    ordered = df.sort_values(sort_cols).reset_index(drop=True)
    split = int(len(ordered) * train_fraction)
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def _auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _fit_logistic(x_train: np.ndarray, y_train: np.ndarray, l2: float = 1.0):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    xz = (x_train - mean) / std
    n_features = xz.shape[1]

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        w = params[:n_features]
        b = params[-1]
        logits = xz @ w + b
        loss = np.mean(np.logaddexp(0.0, logits) - y_train * logits)
        loss += 0.5 * l2 * np.dot(w, w) / len(y_train)
        probs = _sigmoid(logits)
        err = probs - y_train
        grad_w = (xz.T @ err) / len(y_train) + l2 * w / len(y_train)
        grad_b = float(err.mean())
        return float(loss), np.r_[grad_w, grad_b]

    result = minimize(
        fun=lambda p: loss_and_grad(p)[0],
        x0=np.zeros(n_features + 1),
        jac=lambda p: loss_and_grad(p)[1],
        method="L-BFGS-B",
        options={"maxiter": 300},
    )
    return result.x[:n_features], float(result.x[-1]), mean, std


def _stratified_classifier_split(
    n_per_class: int,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_per_class < 2:
        raise ValueError("Classifier two-sample test requires at least 2 rows per class.")

    rng = np.random.default_rng(random_state)
    real_idx = rng.permutation(n_per_class)
    fake_idx = rng.permutation(n_per_class) + n_per_class
    test_n = int(np.ceil(n_per_class * test_fraction))
    test_n = min(max(1, test_n), n_per_class - 1)
    train_idx = np.r_[real_idx[test_n:], fake_idx[test_n:]]
    test_idx = np.r_[real_idx[:test_n], fake_idx[:test_n]]
    return rng.permutation(train_idx), rng.permutation(test_idx)


def classifier_two_sample_test(
    real_holdout: pd.DataFrame,
    simulated: pd.DataFrame,
    feature_columns: list[str],
    test_fraction: float = 0.30,
    random_state: int = 42,
) -> dict[str, object]:
    n = min(len(real_holdout), len(simulated))
    real = real_holdout[feature_columns].dropna().head(n)
    fake = simulated[feature_columns].dropna().head(n)
    n = min(len(real), len(fake))
    real = real.head(n)
    fake = fake.head(n)
    x = pd.concat([real, fake], ignore_index=True).to_numpy(float)
    y = np.r_[np.ones(len(real)), np.zeros(len(fake))]
    train_idx, test_idx = _stratified_classifier_split(n, test_fraction, random_state)
    weights, bias, mean, std = _fit_logistic(x[train_idx], y[train_idx], l2=2.0)
    train_scores = ((x[train_idx] - mean) / std) @ weights + bias
    test_scores = ((x[test_idx] - mean) / std) @ weights + bias
    raw_auc = _auc_score(y[test_idx], test_scores)
    train_raw_auc = _auc_score(y[train_idx], train_scores)
    auc = max(raw_auc, 1.0 - raw_auc)
    train_auc = max(train_raw_auc, 1.0 - train_raw_auc)
    coefficients = np.abs(weights)
    leakage = sorted(
        [
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(feature_columns, coefficients, strict=True)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )
    return {
        "auc": auc,
        "raw_auc": raw_auc,
        "train_auc": train_auc,
        "train_raw_auc": train_raw_auc,
        "n_real": int(len(real)),
        "n_simulated": int(len(fake)),
        "classifier_split": {
            "strategy": "stratified_holdout",
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "test_fraction": float(test_fraction),
            "random_state": int(random_state),
        },
        "top_leakage_features": leakage[:5],
    }
