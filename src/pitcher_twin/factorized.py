"""Factorized physics-residual pitch models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitcher_twin.features import (
    CONTEXT_FEATURES,
    FEATURE_GROUPS,
    PITCH_PHYSICS_FEATURES,
    TRAJECTORY_FEATURES,
)
from pitcher_twin.models import (
    GeneratorModel,
    fit_generator_suite,
    fit_residual_gaussian_copula,
    sample_generator,
    sample_residual_gaussian_copula,
)
from pitcher_twin.validator import classifier_two_sample_test

MOVEMENT_FLIGHT_COLUMNS = ["pfx_x", "pfx_z"]
COMMAND_COLUMNS = ["plate_x", "plate_z"]
FACTORIZED_PHYSICS_COLUMNS = (
    PITCH_PHYSICS_FEATURES + MOVEMENT_FLIGHT_COLUMNS + TRAJECTORY_FEATURES + COMMAND_COLUMNS
)
VALIDATION_LAYERS = [
    "command_representation",
    "movement_only",
    "release_only",
    "trajectory_only",
    "physics_core",
]


@dataclass
class ResidualLayer:
    name: str
    conditioning_columns: list[str]
    target_columns: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    beta: np.ndarray
    residual_cov: np.ndarray
    residual_copula: dict[str, object] | None
    source_row_count: int


@dataclass
class FactorizedPhysicsModel:
    model_name: str
    pitcher_name: str
    pitch_type: str
    release_model: GeneratorModel
    release_model_name: str
    release_variance_floor_columns: list[str]
    release_variance_floor_std: np.ndarray
    movement_layer: ResidualLayer
    trajectory_layer: ResidualLayer
    command_layer: ResidualLayer
    downstream_residual_columns: list[str]
    downstream_residual_cov: np.ndarray
    downstream_residual_copula: dict[str, object] | None
    downstream_residual_offset: np.ndarray
    context_columns: list[str]
    feature_columns: list[str]
    source_row_count: int


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Residual layer input contains non-finite values.")
    return values


def fit_residual_layer(
    frame: pd.DataFrame,
    *,
    name: str,
    conditioning_columns: list[str],
    target_columns: list[str],
    ridge: float = 10.0,
) -> ResidualLayer:
    keep = conditioning_columns + target_columns
    fit_frame = frame[keep].dropna().reset_index(drop=True)
    if len(fit_frame) < max(20, len(conditioning_columns) + 3):
        raise ValueError(f"Not enough rows to fit residual layer {name}.")

    x = _matrix(fit_frame, conditioning_columns)
    y = _matrix(fit_frame, target_columns)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    xz = (x - x_mean) / x_std
    y_mean = y.mean(axis=0)
    yc = y - y_mean
    xtx = xz.T @ xz + np.eye(xz.shape[1]) * ridge
    beta = np.linalg.solve(xtx, xz.T @ yc)
    pred = y_mean + xz @ beta
    residuals = y - pred
    residual_cov = np.cov(residuals, rowvar=False)
    if residual_cov.ndim == 0:
        residual_cov = np.eye(y.shape[1]) * float(residual_cov)
    residual_cov = residual_cov + np.eye(y.shape[1]) * 1e-5
    return ResidualLayer(
        name=name,
        conditioning_columns=conditioning_columns,
        target_columns=target_columns,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        beta=beta,
        residual_cov=residual_cov,
        residual_copula=fit_residual_gaussian_copula(residuals),
        source_row_count=int(len(fit_frame)),
    )


def _predict_layer_mean(layer: ResidualLayer, context: pd.DataFrame) -> np.ndarray:
    x = _matrix(context, layer.conditioning_columns)
    xz = (x - layer.x_mean) / layer.x_std
    return layer.y_mean + xz @ layer.beta


def _layer_residuals(layer: ResidualLayer, frame: pd.DataFrame) -> np.ndarray:
    mean = _predict_layer_mean(layer, frame)
    target = _matrix(frame, layer.target_columns)
    return target - mean


def _recent_game_residual_offset(source_frame: pd.DataFrame, residuals: np.ndarray) -> np.ndarray:
    if "game_pk" not in source_frame.columns or len(source_frame) != len(residuals):
        return np.zeros(residuals.shape[1])

    order_columns = ["game_pk"]
    if "game_date" in source_frame.columns:
        order_columns.insert(0, "game_date")
    game_order = source_frame[order_columns].drop_duplicates().reset_index(drop=True)
    if len(game_order) < 2:
        return np.zeros(residuals.shape[1])

    game_means = []
    game_counts = []
    game_values = source_frame["game_pk"].to_numpy()
    for game_pk in game_order["game_pk"].tolist():
        mask = game_values == game_pk
        game_means.append(residuals[mask].mean(axis=0))
        game_counts.append(int(mask.sum()))

    game_means_array = np.asarray(game_means)
    positions = np.arange(len(game_means_array), dtype=float)
    half_life_games = 1.35
    recency_weights = np.exp(-(positions.max() - positions) / half_life_games)
    recency_weights *= np.sqrt(np.asarray(game_counts, dtype=float))
    recency_weights = recency_weights / recency_weights.sum()
    return np.average(game_means_array, axis=0, weights=recency_weights)


def sample_residual_layer(
    layer: ResidualLayer,
    context: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    mean = _predict_layer_mean(layer, context)
    if layer.residual_copula is not None:
        residuals = sample_residual_gaussian_copula(
            layer.residual_copula,
            n=len(context),
            random_state=random_state,
        )
    else:
        rng = np.random.default_rng(random_state)
        residuals = rng.multivariate_normal(
            np.zeros(len(layer.target_columns)),
            layer.residual_cov,
            size=len(context),
            check_valid="ignore",
        )
    values = mean + residuals
    return pd.DataFrame(values, columns=layer.target_columns)


def _available_context_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in CONTEXT_FEATURES if column in frame.columns]


def _select_release_model(suite: dict[str, GeneratorModel]) -> tuple[str, GeneratorModel]:
    for name in (
        "player_recent_weighted_game_drift_copula",
        "player_recent_weighted_game_drift_gaussian",
        "player_context_weighted_gaussian",
        "player_recent_multivariate_gaussian",
        "player_multivariate_gaussian",
    ):
        if name in suite:
            return name, suite[name]
    if not suite:
        raise ValueError("No release generator models were fitted.")
    name = next(iter(suite))
    return name, suite[name]


def fit_factorized_physics_model(
    player_train: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    random_state: int = 42,
) -> FactorizedPhysicsModel:
    context_columns = _available_context_columns(player_train)
    keep = FACTORIZED_PHYSICS_COLUMNS + context_columns
    metadata_columns = [
        column
        for column in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if column in player_train.columns and column not in keep
    ]
    source_frame = player_train[metadata_columns + keep].dropna(subset=keep)
    if metadata_columns:
        source_frame = source_frame.sort_values(metadata_columns, kind="mergesort")
    source_frame = source_frame.reset_index(drop=True)
    frame = source_frame[keep].reset_index(drop=True)
    if len(frame) < 40:
        raise ValueError("At least 40 complete rows are required for factorized physics.")

    release_suite = fit_generator_suite(
        player_train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        feature_group="release_only",
        random_state=random_state,
    )
    release_name, release_model = _select_release_model(release_suite)
    movement_layer = fit_residual_layer(
        frame,
        name="movement",
        conditioning_columns=PITCH_PHYSICS_FEATURES + context_columns,
        target_columns=MOVEMENT_FLIGHT_COLUMNS,
        ridge=10.0,
    )
    trajectory_layer = fit_residual_layer(
        frame,
        name="trajectory",
        conditioning_columns=PITCH_PHYSICS_FEATURES + MOVEMENT_FLIGHT_COLUMNS + context_columns,
        target_columns=TRAJECTORY_FEATURES,
        ridge=10.0,
    )
    command_layer = fit_residual_layer(
        frame,
        name="command",
        conditioning_columns=(
            PITCH_PHYSICS_FEATURES
            + MOVEMENT_FLIGHT_COLUMNS
            + TRAJECTORY_FEATURES
            + context_columns
        ),
        target_columns=COMMAND_COLUMNS,
        ridge=10.0,
    )
    downstream_residuals = np.column_stack(
        [
            _layer_residuals(movement_layer, frame),
            _layer_residuals(trajectory_layer, frame),
            _layer_residuals(command_layer, frame),
        ]
    )
    downstream_residual_cov = np.cov(downstream_residuals, rowvar=False)
    if downstream_residual_cov.ndim == 0:
        downstream_residual_cov = np.eye(downstream_residuals.shape[1]) * float(
            downstream_residual_cov
        )
    downstream_residual_cov = (
        downstream_residual_cov + np.eye(downstream_residuals.shape[1]) * 1e-5
    )
    return FactorizedPhysicsModel(
        model_name="player_factorized_physics_residual",
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        release_model=release_model,
        release_model_name=release_name,
        release_variance_floor_columns=PITCH_PHYSICS_FEATURES,
        release_variance_floor_std=frame[PITCH_PHYSICS_FEATURES].std(ddof=1).to_numpy(
            float,
            copy=True,
        ),
        movement_layer=movement_layer,
        trajectory_layer=trajectory_layer,
        command_layer=command_layer,
        downstream_residual_columns=(
            MOVEMENT_FLIGHT_COLUMNS + TRAJECTORY_FEATURES + COMMAND_COLUMNS
        ),
        downstream_residual_cov=downstream_residual_cov,
        downstream_residual_copula=fit_residual_gaussian_copula(downstream_residuals),
        downstream_residual_offset=_recent_game_residual_offset(
            source_frame,
            downstream_residuals,
        ),
        context_columns=context_columns,
        feature_columns=FACTORIZED_PHYSICS_COLUMNS,
        source_row_count=int(len(frame)),
    )


def _context_for_sampling(
    model: FactorizedPhysicsModel,
    context_df: pd.DataFrame | None,
    n: int,
) -> pd.DataFrame:
    if context_df is None or not model.context_columns:
        return pd.DataFrame(index=range(n))
    context = context_df[model.context_columns].dropna().reset_index(drop=True)
    if context.empty:
        return pd.DataFrame(index=range(n))
    if len(context) >= n:
        return context.head(n).reset_index(drop=True)
    repeats = int(np.ceil(n / len(context)))
    return pd.concat([context] * repeats, ignore_index=True).head(n)


def _normalize_spin_axis_components(samples: pd.DataFrame) -> pd.DataFrame:
    if not {"spin_axis_cos", "spin_axis_sin"}.issubset(samples.columns):
        return samples
    result = samples.copy()
    norm = np.sqrt(result["spin_axis_cos"] ** 2 + result["spin_axis_sin"] ** 2).to_numpy(
        float,
        copy=True,
    )
    norm[norm < 1e-8] = 1.0
    result["spin_axis_cos"] = result["spin_axis_cos"] / norm
    result["spin_axis_sin"] = result["spin_axis_sin"] / norm
    return result


def _apply_release_variance_floor(
    samples: pd.DataFrame,
    model: FactorizedPhysicsModel,
) -> pd.DataFrame:
    result = samples.copy()
    for column, target_std in zip(
        model.release_variance_floor_columns,
        model.release_variance_floor_std,
        strict=True,
    ):
        if column not in result.columns or not np.isfinite(target_std) or target_std <= 0:
            continue
        current_std = float(result[column].std(ddof=1))
        if not np.isfinite(current_std) or current_std <= 1e-8 or current_std >= target_std:
            continue
        current_mean = float(result[column].mean())
        result[column] = (result[column] - current_mean) * (target_std / current_std) + current_mean
    return result


def sample_factorized_physics(
    model: FactorizedPhysicsModel,
    n: int,
    context_df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    context = _context_for_sampling(model, context_df, n)
    downstream_residuals = _sample_downstream_residuals(model, n=n, random_state=random_state + 1)
    movement_end = len(MOVEMENT_FLIGHT_COLUMNS)
    trajectory_end = movement_end + len(TRAJECTORY_FEATURES)

    release = sample_generator(
        model.release_model,
        n=n,
        random_state=random_state,
        context_df=context if not context.empty else None,
    ).reset_index(drop=True)
    release = _normalize_spin_axis_components(release)
    release = _apply_release_variance_floor(release, model)
    release = _normalize_spin_axis_components(release)
    stage = pd.concat([release, context], axis=1)
    movement = pd.DataFrame(
        _predict_layer_mean(model.movement_layer, stage) + downstream_residuals[:, :movement_end],
        columns=MOVEMENT_FLIGHT_COLUMNS,
    )
    stage = pd.concat([stage, movement], axis=1)
    trajectory = pd.DataFrame(
        _predict_layer_mean(model.trajectory_layer, stage)
        + downstream_residuals[:, movement_end:trajectory_end],
        columns=TRAJECTORY_FEATURES,
    )
    stage = pd.concat([stage, trajectory], axis=1)
    command = pd.DataFrame(
        _predict_layer_mean(model.command_layer, stage) + downstream_residuals[:, trajectory_end:],
        columns=COMMAND_COLUMNS,
    )
    output = pd.concat([release, movement, trajectory, command], axis=1)
    return output[model.feature_columns]


def _sample_downstream_residuals(
    model: FactorizedPhysicsModel,
    n: int,
    random_state: int,
) -> np.ndarray:
    if model.downstream_residual_copula is not None:
        residuals = sample_residual_gaussian_copula(
            model.downstream_residual_copula,
            n=n,
            random_state=random_state,
        )
    else:
        rng = np.random.default_rng(random_state)
        residuals = rng.multivariate_normal(
            np.zeros(len(model.downstream_residual_columns)),
            model.downstream_residual_cov,
            size=n,
            check_valid="ignore",
        )
    return residuals + model.downstream_residual_offset


def _fit_baseline_model(
    train: pd.DataFrame,
    league_df: pd.DataFrame,
    pitcher_name: str,
    pitch_type: str,
    feature_group: str,
    requested: str,
    random_state: int,
) -> GeneratorModel:
    suite = fit_generator_suite(
        train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        feature_group=feature_group,
        random_state=random_state,
    )
    if requested in suite:
        return suite[requested]
    _, fallback = _select_release_model(suite)
    return fallback


def validate_factorized_physics(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    n_samples: int = 300,
    random_state: int = 42,
) -> dict[str, object]:
    factorized = fit_factorized_physics_model(
        train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        random_state=random_state,
    )
    sample_count = max(n_samples, len(holdout))
    factorized_samples = sample_factorized_physics(
        factorized,
        n=sample_count,
        context_df=holdout,
        random_state=random_state + 10,
    )

    layer_results: dict[str, dict[str, object]] = {}
    for index, feature_group in enumerate(VALIDATION_LAYERS):
        columns = FEATURE_GROUPS[feature_group]
        factorized_metrics = classifier_two_sample_test(
            holdout,
            factorized_samples,
            columns,
            random_state=random_state + 100 + index,
        )
        gaussian = _fit_baseline_model(
            train,
            league_df,
            pitcher_name,
            pitch_type,
            feature_group,
            "player_recent_weighted_game_drift_gaussian",
            random_state,
        )
        copula = _fit_baseline_model(
            train,
            league_df,
            pitcher_name,
            pitch_type,
            feature_group,
            "player_recent_weighted_game_drift_copula",
            random_state,
        )
        gaussian_samples = sample_generator(
            gaussian,
            n=sample_count,
            random_state=random_state + 200 + index,
            context_df=holdout,
        )
        copula_samples = sample_generator(
            copula,
            n=sample_count,
            random_state=random_state + 300 + index,
            context_df=holdout,
        )
        gaussian_metrics = classifier_two_sample_test(
            holdout,
            gaussian_samples,
            columns,
            random_state=random_state + 400 + index,
        )
        copula_metrics = classifier_two_sample_test(
            holdout,
            copula_samples,
            columns,
            random_state=random_state + 500 + index,
        )
        layer_results[feature_group] = {
            "features": columns,
            "factorized_auc": float(factorized_metrics["auc"]),
            "game_drift_gaussian_auc": float(gaussian_metrics["auc"]),
            "game_drift_copula_auc": float(copula_metrics["auc"]),
            "factorized_top_leakage": factorized_metrics["top_leakage_features"],
        }

    return {
        "model_name": factorized.model_name,
        "pitcher_name": pitcher_name,
        "pitch_type": pitch_type,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "release_model": factorized.release_model_name,
        "layer_results": layer_results,
    }
