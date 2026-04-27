#!/usr/bin/env python3
"""Real-data viability smoke test for the Pitcher Twin project.

This script intentionally uses no mock rows. It reads an existing public
Statcast cache, picks pitcher/pitch-type pairs with enough real pitches, and
tests whether simple player-specific generators look more realistic than
random or league-average baselines.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


DEFAULT_DATA = (
    Path(__file__).resolve().parents[2]
    / "trajekt-scout"
    / "data"
    / "processed"
    / "latest_statcast.csv"
)


BASE_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "spin_axis_cos",
    "spin_axis_sin",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
]

EVALUATION_MIN_TRAIN = 100
EVALUATION_MIN_HOLDOUT = 50
TEMPORAL_SUCCESS_AUC = 0.60


@dataclass
class Candidate:
    pitcher_id: int
    pitcher_name: str
    pitch_type: str
    n: int
    games: int
    train_n: int
    holdout_n: int


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC using average ranks, avoiding sklearn dependency."""
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic_classifier(
    x_train: np.ndarray, y_train: np.ndarray, l2: float = 1.0
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
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
        probs = sigmoid(logits)
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
    params = result.x
    return params[:n_features], float(params[-1]), mean, std


def c2st_auc(
    real_holdout: np.ndarray,
    simulated: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Classifier two-sample test: lower AUC means harder to tell fake from real."""
    n = min(len(real_holdout), len(simulated))
    if n < 30:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "n": n}

    real = real_holdout[rng.choice(len(real_holdout), size=n, replace=False)]
    fake = simulated[rng.choice(len(simulated), size=n, replace=False)]
    aucs: list[float] = []
    raw_aucs: list[float] = []

    for _ in range(repeats):
        real_idx = rng.permutation(n)
        fake_idx = rng.permutation(n)
        split = max(10, n // 2)

        x_train = np.vstack([real[real_idx[:split]], fake[fake_idx[:split]]])
        y_train = np.r_[np.ones(split), np.zeros(split)]
        x_test = np.vstack([real[real_idx[split:]], fake[fake_idx[split:]]])
        y_test = np.r_[np.ones(n - split), np.zeros(n - split)]

        order = rng.permutation(len(y_train))
        x_train = x_train[order]
        y_train = y_train[order]

        w, b, mean, std = fit_logistic_classifier(x_train, y_train, l2=2.0)
        scores = ((x_test - mean) / std) @ w + b
        raw_auc = auc_score(y_test, scores)
        raw_aucs.append(raw_auc)
        aucs.append(max(raw_auc, 1.0 - raw_auc))

    return {
        "auc_mean": float(np.mean(aucs)),
        "raw_auc_mean": float(np.mean(raw_aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "n": n,
        "classifier_split": {
            "strategy": "repeated_balanced_holdout",
            "repeats": int(repeats),
            "train_rows_per_class": int(split),
            "test_rows_per_class": int(n - split),
        },
    }


def read_statcast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].notna()].copy()
    df["spin_axis_rad"] = np.deg2rad(pd.to_numeric(df["spin_axis"], errors="coerce"))
    df["spin_axis_cos"] = np.cos(df["spin_axis_rad"])
    df["spin_axis_sin"] = np.sin(df["spin_axis_rad"])

    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "pitcher_name" not in df.columns:
        df["pitcher_name"] = df.get("player_name", df["pitcher"].astype(str))

    required = ["pitcher", "pitcher_name", "pitch_type", "game_pk", "game_date", *BASE_FEATURES]
    clean = df.dropna(subset=required).copy()

    sort_cols = ["game_date"]
    if "game_pk" in clean.columns:
        sort_cols.append("game_pk")
    if "pitch_number" in clean.columns:
        sort_cols.append("pitch_number")
    clean = clean.sort_values(sort_cols)
    return clean


def find_candidates(df: pd.DataFrame, limit: int) -> list[Candidate]:
    rows: list[Candidate] = []
    grouped = (
        df.groupby(["pitcher", "pitcher_name", "pitch_type"], dropna=True)
        .agg(n=("pitch_type", "size"), games=("game_pk", "nunique"))
        .reset_index()
        .sort_values(["n", "games"], ascending=False)
    )
    for row in grouped.itertuples(index=False):
        subset = df[
            (df["pitcher"] == row.pitcher)
            & (df["pitch_type"] == row.pitch_type)
        ].sort_values("game_date")
        split = int(len(subset) * 0.7)
        rows.append(
            Candidate(
                pitcher_id=int(row.pitcher),
                pitcher_name=str(row.pitcher_name),
                pitch_type=str(row.pitch_type),
                n=int(row.n),
                games=int(row.games),
                train_n=split,
                holdout_n=len(subset) - split,
            )
        )
        if len(rows) >= limit:
            break
    return rows


def robust_cov(x: np.ndarray) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = np.eye(x.shape[1]) * float(cov)
    jitter = np.eye(x.shape[1]) * 1e-5
    return cov + jitter


def split_pitcher_rows(
    subset: pd.DataFrame,
    split_kind: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_kind == "temporal":
        subset = subset.sort_values("game_date")
        split = int(len(subset) * 0.7)
        return subset.iloc[:split], subset.iloc[split:]

    if split_kind == "random":
        indices = rng.permutation(len(subset))
        split = int(len(subset) * 0.7)
        train_idx = indices[:split]
        holdout_idx = indices[split:]
        return subset.iloc[train_idx], subset.iloc[holdout_idx]

    raise ValueError(f"Unknown split kind: {split_kind}")


def sample_models(
    df: pd.DataFrame,
    candidate: Candidate,
    n_samples: int,
    split_kind: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    subset = df[
        (df["pitcher"] == candidate.pitcher_id)
        & (df["pitch_type"] == candidate.pitch_type)
    ].sort_values("game_date")
    train, holdout = split_pitcher_rows(subset, split_kind, rng)
    x_train = train[BASE_FEATURES].to_numpy(float)
    x_holdout = holdout[BASE_FEATURES].to_numpy(float)

    n = min(n_samples, max(30, len(x_holdout) * 3))
    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    stds[stds < 1e-8] = 1.0

    league_mask = (df["pitch_type"] == candidate.pitch_type) & (
        df["pitcher"] != candidate.pitcher_id
    )
    if split_kind == "temporal":
        league_mask = league_mask & (df["game_date"] <= train["game_date"].max())

    league_pool = df[league_mask][BASE_FEATURES].to_numpy(float)
    if len(league_pool) < 50:
        league_pool = df[
            (df["pitch_type"] == candidate.pitch_type)
            & (df["pitcher"] != candidate.pitcher_id)
        ][BASE_FEATURES].to_numpy(float)

    models: dict[str, np.ndarray] = {}
    models["random_independent_noise"] = rng.normal(means, stds, size=(n, len(BASE_FEATURES)))
    if len(league_pool):
        models["league_same_pitch_empirical"] = league_pool[
            rng.choice(len(league_pool), size=n, replace=True)
        ]
    models["player_empirical_bootstrap"] = x_train[
        rng.choice(len(x_train), size=n, replace=True)
    ]
    models["player_multivariate_gaussian"] = rng.multivariate_normal(
        means,
        robust_cov(x_train),
        size=n,
        check_valid="ignore",
    )
    return x_holdout, models


def evaluate_candidate(
    df: pd.DataFrame,
    candidate: Candidate,
    repeats: int,
    n_samples: int,
    split_kind: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    real_holdout, models = sample_models(df, candidate, n_samples, split_kind, rng)
    results = {}
    for name, simulated in models.items():
        results[name] = c2st_auc(real_holdout, simulated, repeats=repeats, rng=rng)
    return {
        "pitcher": candidate.pitcher_name,
        "pitcher_id": candidate.pitcher_id,
        "pitch_type": candidate.pitch_type,
        "n_total": candidate.n,
        "games": candidate.games,
        "train_n": candidate.train_n,
        "holdout_n": candidate.holdout_n,
        "split": split_kind,
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    df = read_statcast(args.data)
    candidates = find_candidates(df, args.top)

    report = {
        "data_path": str(args.data),
        "rows_clean": int(len(df)),
        "date_min": str(df["game_date"].min().date()),
        "date_max": str(df["game_date"].max().date()),
        "feature_count": len(BASE_FEATURES),
        "split": args.split,
        "features": BASE_FEATURES,
        "validation_thresholds": {
            "candidate_thresholds": {
                "min_train": EVALUATION_MIN_TRAIN,
                "min_holdout": EVALUATION_MIN_HOLDOUT,
            },
            "temporal_success_auc": TEMPORAL_SUCCESS_AUC,
            "classifier_split": {
                "strategy": "repeated_balanced_holdout",
                "repeats": args.repeats,
                "minimum_rows_per_class": 30,
            },
        },
        "candidates": [candidate.__dict__ for candidate in candidates],
        "evaluations": [],
    }

    print("REAL STATCAST VIABILITY SMOKE TEST")
    print(f"Data: {args.data}")
    print(f"Rows after feature cleaning: {len(df):,}")
    print(f"Date range: {report['date_min']} to {report['date_max']}")
    print(f"Features: {len(BASE_FEATURES)} physical pitch features")
    print()
    print("Top real pitcher/pitch candidates in this cache:")
    for i, c in enumerate(candidates, start=1):
        print(
            f"{i:>2}. {c.pitcher_name:<24} {c.pitch_type:<3} "
            f"n={c.n:<4} games={c.games:<2} train={c.train_n:<3} holdout={c.holdout_n:<3}"
        )

    print()
    print(f"Split: {args.split}")
    print("C2ST AUC: 0.50 means classifier cannot tell simulated from real.")
    print("Higher means the generated data is visibly fake/different.")
    print()

    for c in candidates:
        if c.holdout_n < EVALUATION_MIN_HOLDOUT or c.train_n < EVALUATION_MIN_TRAIN:
            continue
        result = evaluate_candidate(
            df,
            c,
            repeats=args.repeats,
            n_samples=args.samples,
            split_kind=args.split,
            rng=rng,
        )
        report["evaluations"].append(result)
        print(f"{c.pitcher_name} {c.pitch_type} (real holdout n={c.holdout_n})")
        for model_name, metrics in result["models"].items():
            print(
                f"  {model_name:<30} "
                f"AUC={metrics['auc_mean']:.3f} +/- {metrics['auc_std']:.3f} "
                f"(n={metrics['n']})"
            )
        print()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
