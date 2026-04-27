"""Real-data generative pitch models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from pitcher_twin.features import FEATURE_GROUPS, build_feature_matrix

try:  # sklearn is optional for direct script execution in lean Python installs.
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - exercised by direct python3 smoke run.
    GaussianMixture = None
    StandardScaler = None


@dataclass
class GeneratorModel:
    model_name: str
    pitcher_name: str
    pitch_type: str
    feature_group: str
    feature_columns: list[str]
    payload: dict[str, object]


CONTEXT_CONDITIONING_FEATURES = [
    "balls",
    "strikes",
    "count_bucket_code",
    "inning",
    "pitcher_game_pitch_count",
    "batter_stand_code",
    "pitcher_score_diff",
]


def _rng(random_state: int | None = None) -> np.random.Generator:
    return np.random.default_rng(random_state)


def _feature_values(df: pd.DataFrame, feature_group: str) -> np.ndarray:
    return build_feature_matrix(df, feature_group).to_numpy(float)


def _sort_pitch_rows(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        column
        for column in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if column in df.columns
    ]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols).reset_index(drop=True)


def _recent_feature_values(
    df: pd.DataFrame,
    feature_group: str,
    min_rows: int = 10,
    fraction: float = 0.40,
) -> np.ndarray | None:
    ordered = _sort_pitch_rows(df)
    columns = FEATURE_GROUPS[feature_group]
    valid = ordered[columns].dropna().reset_index(drop=True)
    if len(valid) <= min_rows:
        return None
    recent_n = max(min_rows, int(np.ceil(len(valid) * fraction)))
    recent_n = min(recent_n, len(valid) - 1)
    return valid.tail(recent_n).to_numpy(float)


def _recent_game_window_values(
    df: pd.DataFrame,
    feature_group: str,
    max_games: int = 8,
    min_rows: int = 10,
) -> dict[str, object] | None:
    if "game_pk" not in df.columns:
        return None
    columns = FEATURE_GROUPS[feature_group]
    keep_columns = ["game_pk", *columns]
    if "game_date" in df.columns:
        keep_columns.insert(1, "game_date")
    frame = _sort_pitch_rows(df)[keep_columns].dropna(subset=columns)
    game_order = (
        frame[["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(game_order) < 2:
        return None
    game_window = min(max_games, len(game_order))
    recent_game_ids = set(game_order.tail(game_window)["game_pk"].tolist())
    recent_frame = frame[frame["game_pk"].isin(recent_game_ids)]
    if len(recent_frame) < min_rows:
        return None
    return {
        "pool": recent_frame[columns].to_numpy(float),
        "game_window": int(game_window),
        "game_count": int(len(game_order)),
        "source_row_count": int(len(recent_frame)),
    }


def _robust_cov(values: np.ndarray) -> np.ndarray:
    cov = np.cov(values, rowvar=False)
    if cov.ndim == 0:
        cov = np.eye(values.shape[1]) * float(cov)
    return cov + np.eye(values.shape[1]) * 1e-5


def _rank_uniform(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    uniforms = (ranks - 0.5) / len(values)
    return np.clip(uniforms, 1e-4, 1.0 - 1e-4)


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-4, None)
    repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    scale = np.sqrt(np.clip(np.diag(repaired), 1e-8, None))
    repaired = repaired / scale[:, None] / scale[None, :]
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _latent_correlation(latent: np.ndarray) -> np.ndarray:
    corr = np.eye(latent.shape[1])
    centered = latent - latent.mean(axis=0)
    std = centered.std(axis=0, ddof=1)
    active = std > 1e-8
    if int(active.sum()) >= 2:
        corr[np.ix_(active, active)] = np.corrcoef(latent[:, active], rowvar=False)
    return corr


def _gaussian_copula_residual_payload(residuals: np.ndarray) -> dict[str, object] | None:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or residuals.shape[0] < 20:
        return None

    latent_columns = [ndtri(_rank_uniform(residuals[:, index])) for index in range(residuals.shape[1])]
    latent = np.column_stack(latent_columns)
    corr = _latent_correlation(latent)
    return {
        "copula_kind": "gaussian_empirical_margins",
        "copula_corr": _nearest_correlation(corr),
        "residual_margins": np.sort(residuals, axis=0),
        "residual_count": int(len(residuals)),
    }


def fit_residual_gaussian_copula(residuals: np.ndarray) -> dict[str, object] | None:
    """Fit empirical Gaussian-copula residual margins for downstream residual layers."""
    return _gaussian_copula_residual_payload(residuals)


def sample_residual_gaussian_copula(
    payload: dict[str, object],
    n: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample residuals from a fitted empirical Gaussian-copula payload."""
    return _sample_copula_residuals(payload, n=n, rng=_rng(random_state))


def _context_training_payload(
    player_train: pd.DataFrame,
    feature_group: str,
) -> dict[str, object] | None:
    feature_columns = FEATURE_GROUPS[feature_group]
    context_columns = [
        column
        for column in CONTEXT_CONDITIONING_FEATURES
        if column in player_train.columns and column not in feature_columns
    ]
    if not context_columns:
        return None

    frame = _sort_pitch_rows(player_train)[feature_columns + context_columns].dropna()
    if len(frame) < 10:
        return None

    context_pool = frame[context_columns].to_numpy(float)
    feature_pool = frame[feature_columns].to_numpy(float)
    context_mean = context_pool.mean(axis=0)
    context_std = context_pool.std(axis=0)
    context_std[context_std < 1e-8] = 1.0
    return {
        "feature_pool": feature_pool,
        "context_pool": context_pool,
        "context_mean": context_mean,
        "context_std": context_std,
        "context_columns": context_columns,
        "source_row_count": int(len(frame)),
        "k_neighbors": int(min(max(10, len(frame) // 3), len(frame))),
        "bandwidth": 1.25,
    }


def _recent_weighted_game_drift_payload(
    player_train: pd.DataFrame,
    feature_group: str,
) -> dict[str, object] | None:
    """Estimate baseline + recent game/day offset + pitch-level residual noise."""
    if "game_pk" not in player_train.columns:
        return None

    feature_columns = FEATURE_GROUPS[feature_group]
    context_columns = [
        column
        for column in CONTEXT_CONDITIONING_FEATURES
        if column in player_train.columns and column not in feature_columns
    ]
    keep_columns = ["game_pk", *feature_columns, *context_columns]
    if "game_date" in player_train.columns:
        keep_columns.insert(1, "game_date")
    frame = _sort_pitch_rows(player_train)[keep_columns].dropna(subset=feature_columns)
    if len(frame) < 10:
        return None

    game_order = (
        frame[["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(game_order) < 2:
        return None

    feature_values = frame[feature_columns].to_numpy(float)
    baseline_mean = feature_values.mean(axis=0)
    game_means = []
    game_counts = []
    aligned_game_means = np.zeros_like(feature_values)
    for row_index, game_pk in enumerate(game_order["game_pk"].tolist()):
        mask = frame["game_pk"].to_numpy() == game_pk
        game_values = feature_values[mask]
        game_mean = game_values.mean(axis=0)
        game_means.append(game_mean)
        game_counts.append(int(mask.sum()))
        aligned_game_means[mask] = game_mean

    game_means_array = np.asarray(game_means)
    positions = np.arange(len(game_means_array), dtype=float)
    half_life_games = 1.35
    recency_weights = np.exp(-(positions.max() - positions) / half_life_games)
    recency_weights *= np.sqrt(np.asarray(game_counts, dtype=float))
    recency_weights = recency_weights / recency_weights.sum()
    predicted_game_mean = np.average(game_means_array, axis=0, weights=recency_weights)

    residuals = feature_values - aligned_game_means
    recent_values = frame[feature_columns].tail(max(10, int(len(frame) * 0.35))).to_numpy(float)
    residual_cov = _robust_cov(residuals)
    recent_cov = _robust_cov(recent_values)
    cov = 0.70 * residual_cov + 0.30 * recent_cov

    payload: dict[str, object] = {
        "baseline_mean": baseline_mean,
        "predicted_game_mean": predicted_game_mean,
        "cov": cov,
        "game_count": int(len(game_means_array)),
        "game_counts": game_counts,
        "game_weights": recency_weights,
        "source_row_count": int(len(frame)),
        "context_columns": [],
    }

    if context_columns and len(frame) >= 20:
        context_values = frame[context_columns].dropna()
        if len(context_values) == len(frame):
            context_array = context_values.to_numpy(float)
            context_mean = context_array.mean(axis=0)
            context_std = context_array.std(axis=0)
            context_std[context_std < 1e-8] = 1.0
            context_z = (context_array - context_mean) / context_std
            y = feature_values - aligned_game_means
            ridge = 5.0
            xtx = context_z.T @ context_z + np.eye(context_z.shape[1]) * ridge
            beta = np.linalg.solve(xtx, context_z.T @ y)
            context_residuals = y - context_z @ beta
            payload.update(
                {
                    "cov": 0.70 * _robust_cov(context_residuals) + 0.30 * recent_cov,
                    "context_columns": context_columns,
                    "context_mean": context_mean,
                    "context_std": context_std,
                    "context_beta": beta,
                }
            )
    return payload


def _recent_weighted_game_drift_copula_payload(
    player_train: pd.DataFrame,
    feature_group: str,
) -> dict[str, object] | None:
    payload = _recent_weighted_game_drift_payload(player_train, feature_group)
    if payload is None:
        return None

    feature_columns = FEATURE_GROUPS[feature_group]
    keep_columns = ["game_pk", *feature_columns]
    if "game_date" in player_train.columns:
        keep_columns.insert(1, "game_date")
    frame = _sort_pitch_rows(player_train)[keep_columns].dropna(subset=feature_columns)
    if len(frame) < 20:
        return None

    feature_values = frame[feature_columns].to_numpy(float)
    aligned_game_means = np.zeros_like(feature_values)
    for game_pk in frame["game_pk"].drop_duplicates().tolist():
        mask = frame["game_pk"].to_numpy() == game_pk
        aligned_game_means[mask] = feature_values[mask].mean(axis=0)

    residuals = feature_values - aligned_game_means
    copula_payload = _gaussian_copula_residual_payload(residuals)
    if copula_payload is None:
        return None

    return {
        **payload,
        **copula_payload,
    }


def fit_generator_suite(
    player_train: pd.DataFrame,
    league_df: pd.DataFrame,
    pitcher_name: str,
    pitch_type: str,
    feature_group: str = "physics_core",
    random_state: int = 42,
) -> dict[str, GeneratorModel]:
    """Fit baseline real-data generators for one player/pitch pair."""
    feature_columns = FEATURE_GROUPS[feature_group]
    x_train = _feature_values(player_train, feature_group)
    if len(x_train) < 10:
        raise ValueError("At least 10 player training rows are required.")

    league_pool_df = league_df[
        (league_df["pitch_type"] == pitch_type)
        & (league_df["pitcher"] != player_train["pitcher"].iloc[0])
    ]
    league_pool = _feature_values(league_pool_df, feature_group) if len(league_pool_df) else x_train
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    recent_values = _recent_feature_values(player_train, feature_group)
    recent_game_window = _recent_game_window_values(player_train, feature_group)

    suite = {
        "random_independent_noise": GeneratorModel(
            "random_independent_noise",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            {"mean": mean, "std": std, "source_row_count": int(len(x_train))},
        ),
        "league_same_pitch_empirical": GeneratorModel(
            "league_same_pitch_empirical",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            {"pool": league_pool, "source_row_count": int(len(league_pool))},
        ),
        "player_empirical_bootstrap": GeneratorModel(
            "player_empirical_bootstrap",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            {"pool": x_train, "source_row_count": int(len(x_train))},
        ),
        "player_multivariate_gaussian": GeneratorModel(
            "player_multivariate_gaussian",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            {"mean": mean, "cov": _robust_cov(x_train), "source_row_count": int(len(x_train))},
        ),
    }

    if recent_values is not None:
        recent_mean = recent_values.mean(axis=0)
        suite["player_recent_empirical_bootstrap"] = GeneratorModel(
            "player_recent_empirical_bootstrap",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            {"pool": recent_values, "source_row_count": int(len(recent_values))},
        )
        suite["player_recent_multivariate_gaussian"] = GeneratorModel(
            "player_recent_multivariate_gaussian",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            {
                "mean": recent_mean,
                "cov": _robust_cov(recent_values),
                "source_row_count": int(len(recent_values)),
            },
        )

    if recent_game_window is not None:
        suite["player_recent_game_window_empirical"] = GeneratorModel(
            "player_recent_game_window_empirical",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            recent_game_window,
        )

    context_payload = _context_training_payload(player_train, feature_group)
    if context_payload is not None:
        suite["player_context_weighted_gaussian"] = GeneratorModel(
            "player_context_weighted_gaussian",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            context_payload,
        )

    game_drift_payload = _recent_weighted_game_drift_payload(player_train, feature_group)
    if game_drift_payload is not None:
        suite["player_recent_weighted_game_drift_gaussian"] = GeneratorModel(
            "player_recent_weighted_game_drift_gaussian",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            game_drift_payload,
        )
    game_drift_copula_payload = _recent_weighted_game_drift_copula_payload(
        player_train,
        feature_group,
    )
    if game_drift_copula_payload is not None:
        suite["player_recent_weighted_game_drift_copula"] = GeneratorModel(
            "player_recent_weighted_game_drift_copula",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            game_drift_copula_payload,
        )

    if GaussianMixture is not None and StandardScaler is not None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        max_components = max(1, min(5, len(x_train) // 10))
        best_gmm = None
        best_bic = float("inf")
        for n_components in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                n_init=3,
                random_state=random_state,
            )
            gmm.fit(x_scaled)
            bic = gmm.bic(x_scaled)
            if bic < best_bic:
                best_bic = float(bic)
                best_gmm = gmm
        suite["player_gmm"] = GeneratorModel(
            "player_gmm",
            pitcher_name,
            pitch_type,
            feature_group,
            feature_columns,
            {"scaler": scaler, "gmm": best_gmm, "bic": best_bic},
        )
    return suite


def sample_generator(
    model: GeneratorModel,
    n: int,
    random_state: int = 42,
    context_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rng = _rng(random_state)
    payload = model.payload
    if model.model_name == "random_independent_noise":
        values = rng.normal(payload["mean"], payload["std"], size=(n, len(model.feature_columns)))
    elif model.model_name in {
        "league_same_pitch_empirical",
        "player_empirical_bootstrap",
        "player_recent_empirical_bootstrap",
        "player_recent_game_window_empirical",
    }:
        pool = payload["pool"]
        values = pool[rng.choice(len(pool), size=n, replace=True)]
    elif model.model_name in {
        "player_multivariate_gaussian",
        "player_recent_multivariate_gaussian",
    }:
        values = rng.multivariate_normal(
            payload["mean"],
            payload["cov"],
            size=n,
            check_valid="ignore",
        )
    elif model.model_name == "player_context_weighted_gaussian":
        values = _sample_context_weighted_gaussian(model, n=n, rng=rng, context_df=context_df)
    elif model.model_name == "player_recent_weighted_game_drift_gaussian":
        values = _sample_recent_weighted_game_drift_gaussian(
            model,
            n=n,
            rng=rng,
            context_df=context_df,
        )
    elif model.model_name == "player_recent_weighted_game_drift_copula":
        values = _sample_recent_weighted_game_drift_copula(
            model,
            n=n,
            rng=rng,
            context_df=context_df,
        )
    elif model.model_name == "player_gmm":
        scaled, _ = payload["gmm"].sample(n)
        values = payload["scaler"].inverse_transform(scaled)
    else:
        raise KeyError(f"Unknown generator model: {model.model_name}")

    return pd.DataFrame(values, columns=model.feature_columns)


def _context_effects_for_sampling(
    payload: dict[str, object],
    n: int,
    rng: np.random.Generator,
    context_df: pd.DataFrame | None,
) -> np.ndarray:
    context_columns = payload.get("context_columns", [])
    if (
        context_df is None
        or not context_columns
        or not set(context_columns).issubset(context_df.columns)
    ):
        return np.zeros((n, len(payload["predicted_game_mean"])))

    context_values = context_df[list(context_columns)].dropna().to_numpy(float)
    if len(context_values) == 0:
        return np.zeros((n, len(payload["predicted_game_mean"])))
    if len(context_values) < n:
        context_values = context_values[rng.choice(len(context_values), size=n, replace=True)]
    else:
        context_values = context_values[:n]
    context_z = (context_values - payload["context_mean"]) / payload["context_std"]
    return context_z @ payload["context_beta"]


def _sample_recent_weighted_game_drift_gaussian(
    model: GeneratorModel,
    n: int,
    rng: np.random.Generator,
    context_df: pd.DataFrame | None = None,
) -> np.ndarray:
    payload = model.payload
    base = payload["predicted_game_mean"]
    effects = _context_effects_for_sampling(payload, n, rng, context_df)
    noise = rng.multivariate_normal(
        np.zeros(len(model.feature_columns)),
        payload["cov"],
        size=n,
        check_valid="ignore",
    )
    return base + effects + noise


def _sample_copula_residuals(payload: dict[str, object], n: int, rng: np.random.Generator) -> np.ndarray:
    margins = payload["residual_margins"]
    corr = payload["copula_corr"]
    latent = rng.multivariate_normal(
        np.zeros(margins.shape[1]),
        corr,
        size=n,
        check_valid="ignore",
    )
    uniforms = np.clip(ndtr(latent), 1e-4, 1.0 - 1e-4)
    residuals = np.empty((n, margins.shape[1]), dtype=float)
    for column_index in range(margins.shape[1]):
        residuals[:, column_index] = np.quantile(
            margins[:, column_index],
            uniforms[:, column_index],
            method="linear",
        )
    return residuals


def _sample_recent_weighted_game_drift_copula(
    model: GeneratorModel,
    n: int,
    rng: np.random.Generator,
    context_df: pd.DataFrame | None = None,
) -> np.ndarray:
    payload = model.payload
    base = payload["predicted_game_mean"]
    effects = _context_effects_for_sampling(payload, n, rng, context_df)
    residuals = _sample_copula_residuals(payload, n, rng)
    return base + effects + residuals


def _sample_context_weighted_gaussian(
    model: GeneratorModel,
    n: int,
    rng: np.random.Generator,
    context_df: pd.DataFrame | None = None,
) -> np.ndarray:
    payload = model.payload
    feature_pool = payload["feature_pool"]
    context_pool = payload["context_pool"]
    context_mean = payload["context_mean"]
    context_std = payload["context_std"]
    context_columns = payload["context_columns"]
    k_neighbors = int(payload["k_neighbors"])
    bandwidth = float(payload["bandwidth"])

    if context_df is not None and set(context_columns).issubset(context_df.columns):
        target_context = context_df[list(context_columns)].dropna().to_numpy(float)
        if len(target_context) == 0:
            target_context = context_pool[rng.choice(len(context_pool), size=n, replace=True)]
        elif len(target_context) < n:
            target_context = target_context[rng.choice(len(target_context), size=n, replace=True)]
        else:
            target_context = target_context[:n]
    else:
        target_context = context_pool[rng.choice(len(context_pool), size=n, replace=True)]

    context_z = (context_pool - context_mean) / context_std
    values = []
    for target in target_context:
        target_z = (target - context_mean) / context_std
        distances = np.linalg.norm(context_z - target_z, axis=1)
        neighbor_idx = np.argsort(distances)[:k_neighbors]
        local_features = feature_pool[neighbor_idx]
        weights = np.exp(-0.5 * np.square(distances[neighbor_idx] / bandwidth))
        weights_sum = weights.sum()
        if weights_sum <= 0:
            weights = np.ones(len(neighbor_idx)) / len(neighbor_idx)
        else:
            weights = weights / weights_sum
        mean = np.average(local_features, axis=0, weights=weights)
        centered = local_features - mean
        cov = (centered * weights[:, None]).T @ centered
        cov += np.eye(local_features.shape[1]) * 1e-5
        values.append(rng.multivariate_normal(mean, cov, check_valid="ignore"))
    return np.asarray(values)
