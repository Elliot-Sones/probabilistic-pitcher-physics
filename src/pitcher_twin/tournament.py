"""Model tournament experiments for pitch generator variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

from pitcher_twin.factorized import (
    VALIDATION_LAYERS,
    fit_factorized_physics_model,
    sample_factorized_physics,
)
from pitcher_twin.features import (
    CONTEXT_FEATURES,
    FEATURE_GROUPS,
    PITCH_PHYSICS_FEATURES,
    RECENT_STATE_FEATURES,
    add_recent_pitcher_state_features,
)
from pitcher_twin.models import GeneratorModel, fit_generator_suite
from pitcher_twin.validator import classifier_two_sample_test

DERIVED_FEATURE_COLUMNS = [
    "derived_spin_axis_angle",
    "derived_spin_axis_norm_error",
    "derived_release_xz_radius",
    "derived_movement_magnitude",
    "derived_movement_angle",
    "derived_velocity_adjusted_pfx_x",
    "derived_velocity_adjusted_pfx_z",
    "derived_plate_radius",
]

RELEASE_STATE_ANCHOR_COLUMNS = [
    "release_speed",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
]

PITCH_FAMILY_BY_PITCH_TYPE = {
    "FF": "rising_fastball",
    "FA": "rising_fastball",
    "SI": "sinker",
    "FT": "sinker",
    "FC": "cutter",
    "CH": "changeup",
    "FS": "changeup",
    "FO": "changeup",
    "SL": "breaking",
    "ST": "breaking",
    "SV": "breaking",
    "CU": "breaking",
    "KC": "breaking",
    "CS": "breaking",
}

PITCH_FAMILY_RELEASE_SPIN_SETTINGS = {
    "rising_fastball": {
        "physics_anchor_alpha": 0.70,
        "release_anchor_alpha": 0.35,
        "spin_residual_alpha": 0.05,
        "release_geometry_alpha": 0.35,
        "spin_recent_fraction": 0.45,
    },
    "sinker": {
        "physics_anchor_alpha": 0.48,
        "release_anchor_alpha": 0.42,
        "spin_residual_alpha": 0.08,
        "release_geometry_alpha": 0.50,
        "spin_recent_fraction": 0.55,
    },
    "changeup": {
        "physics_anchor_alpha": 0.42,
        "release_anchor_alpha": 0.38,
        "spin_residual_alpha": 0.12,
        "release_geometry_alpha": 0.45,
        "spin_recent_fraction": 0.60,
    },
    "breaking": {
        "physics_anchor_alpha": 0.38,
        "release_anchor_alpha": 0.35,
        "spin_residual_alpha": 0.15,
        "release_geometry_alpha": 0.40,
        "spin_recent_fraction": 0.60,
    },
    "cutter": {
        "physics_anchor_alpha": 0.50,
        "release_anchor_alpha": 0.38,
        "spin_residual_alpha": 0.10,
        "release_geometry_alpha": 0.42,
        "spin_recent_fraction": 0.55,
    },
    "unknown": {
        "physics_anchor_alpha": 0.50,
        "release_anchor_alpha": 0.35,
        "spin_residual_alpha": 0.08,
        "release_geometry_alpha": 0.35,
        "spin_recent_fraction": 0.50,
    },
}

CONDITIONAL_STATE_CANDIDATE_COLUMNS = [
    *CONTEXT_FEATURES,
    "previous_pitch_type_code",
    "previous_release_speed",
    "previous_release_spin_rate",
    "previous_pfx_x",
    "previous_pfx_z",
    "previous_plate_x",
    "previous_plate_z",
    "rolling_5_release_speed_mean",
    "rolling_5_release_spin_rate_mean",
    "rolling_5_pfx_x_mean",
    "rolling_5_pfx_z_mean",
    "rolling_5_plate_x_mean",
    "rolling_5_plate_z_mean",
    "rolling_10_release_speed_mean",
    "rolling_10_release_spin_rate_mean",
    "rolling_10_pfx_x_mean",
    "rolling_10_pfx_z_mean",
    "rolling_20_release_speed_mean",
    "rolling_20_release_spin_rate_mean",
    "rolling_20_pfx_x_mean",
    "rolling_20_pfx_z_mean",
]
@dataclass
class TournamentModel:
    model_name: str
    feature_columns: list[str]
    payload: dict[str, object]


def build_derived_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append geometry-style pitch features without replacing raw Statcast columns."""
    required = [
        "spin_axis_cos",
        "spin_axis_sin",
        "release_pos_x",
        "release_pos_z",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing derived feature input columns: {', '.join(missing)}")

    result = frame.copy()
    spin_cos = pd.to_numeric(result["spin_axis_cos"], errors="coerce")
    spin_sin = pd.to_numeric(result["spin_axis_sin"], errors="coerce")
    release_speed = pd.to_numeric(result["release_speed"], errors="coerce")
    safe_velocity = release_speed.where(release_speed.abs() > 1e-8, np.nan)
    result["derived_spin_axis_angle"] = np.arctan2(spin_sin, spin_cos)
    result["derived_spin_axis_norm_error"] = np.sqrt(spin_cos**2 + spin_sin**2) - 1.0
    result["derived_release_xz_radius"] = np.sqrt(
        pd.to_numeric(result["release_pos_x"], errors="coerce") ** 2
        + pd.to_numeric(result["release_pos_z"], errors="coerce") ** 2
    )
    result["derived_movement_magnitude"] = np.sqrt(
        pd.to_numeric(result["pfx_x"], errors="coerce") ** 2
        + pd.to_numeric(result["pfx_z"], errors="coerce") ** 2
    )
    result["derived_movement_angle"] = np.arctan2(
        pd.to_numeric(result["pfx_z"], errors="coerce"),
        pd.to_numeric(result["pfx_x"], errors="coerce"),
    )
    result["derived_velocity_adjusted_pfx_x"] = (
        pd.to_numeric(result["pfx_x"], errors="coerce") / safe_velocity
    )
    result["derived_velocity_adjusted_pfx_z"] = (
        pd.to_numeric(result["pfx_z"], errors="coerce") / safe_velocity
    )
    result["derived_plate_radius"] = np.sqrt(
        pd.to_numeric(result["plate_x"], errors="coerce") ** 2
        + pd.to_numeric(result["plate_z"], errors="coerce") ** 2
    )
    return result


def _feature_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    values = frame[columns].dropna().to_numpy(float)
    if len(values) < 10:
        raise ValueError("At least 10 complete rows are required.")
    return values


def _robust_cov(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    cov = np.cov(values, rowvar=False)
    if cov.ndim == 0:
        cov = np.eye(values.shape[1]) * float(cov)
    return cov + np.eye(values.shape[1]) * 1e-5


def fit_pca_latent_model(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    variance_threshold: float = 0.95,
    max_components: int = 8,
) -> TournamentModel:
    values = _feature_matrix(train, feature_columns)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (values - mean) / std
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    explained = singular_values**2
    explained_ratio = explained / explained.sum()
    component_count = int(np.searchsorted(np.cumsum(explained_ratio), variance_threshold) + 1)
    component_count = max(1, min(component_count, max_components, vt.shape[0]))
    components = vt[:component_count]
    latent = standardized @ components.T
    reconstructed = latent @ components
    residuals = standardized - reconstructed
    return TournamentModel(
        model_name="pca_latent_residual",
        feature_columns=feature_columns,
        payload={
            "mean": mean,
            "std": std,
            "components": components,
            "latent_mean": latent.mean(axis=0),
            "latent_cov": _robust_cov(latent),
            "residual_cov": _robust_cov(residuals),
            "component_count": component_count,
            "source_row_count": int(len(values)),
        },
    )


def fit_context_neighbor_model(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    context_columns: list[str],
    k_neighbors: int = 35,
) -> TournamentModel:
    keep = feature_columns + context_columns
    frame = train[keep].dropna().reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 complete rows are required.")
    context_values = frame[context_columns].to_numpy(float)
    context_mean = context_values.mean(axis=0)
    context_std = context_values.std(axis=0)
    context_std[context_std < 1e-8] = 1.0
    return TournamentModel(
        model_name="context_neighbor_residual",
        feature_columns=feature_columns,
        payload={
            "feature_pool": frame[feature_columns].to_numpy(float),
            "context_pool": context_values,
            "context_columns": context_columns,
            "context_mean": context_mean,
            "context_std": context_std,
            "k_neighbors": int(min(max(3, k_neighbors), len(frame))),
            "source_row_count": int(len(frame)),
        },
    )


def fit_derived_joint_gaussian_model(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> TournamentModel:
    enriched = build_derived_physics_features(train)
    joint_columns = feature_columns + [
        column for column in DERIVED_FEATURE_COLUMNS if column not in feature_columns
    ]
    values = _feature_matrix(enriched, joint_columns)
    return TournamentModel(
        model_name="derived_joint_gaussian",
        feature_columns=feature_columns,
        payload={
            "joint_columns": joint_columns,
            "mean": values.mean(axis=0),
            "cov": _robust_cov(values),
            "source_row_count": int(len(values)),
        },
    )


def fit_conditional_state_mixture_model(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    random_state: int = 42,
    max_states: int = 4,
    max_components: int = 3,
) -> TournamentModel:
    """Fit state-conditioned full-physics mixture distributions."""
    enriched = add_recent_pitcher_state_features(train)
    missing = [column for column in feature_columns if column not in enriched.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    frame = enriched.dropna(subset=feature_columns).reset_index(drop=True)
    if len(frame) < 40:
        raise ValueError("At least 40 complete rows are required.")

    state_columns = _available_conditional_state_columns(frame)
    if not state_columns:
        raise ValueError("No usable conditional state columns are available.")

    state_frame = frame[state_columns].apply(pd.to_numeric, errors="coerce")
    state_fill = state_frame.median(numeric_only=True).fillna(0.0)
    state_values = state_frame.fillna(state_fill).to_numpy(float)
    state_mean = state_values.mean(axis=0)
    state_std = state_values.std(axis=0)
    state_std[state_std < 1e-8] = 1.0
    state_z = (state_values - state_mean) / state_std

    state_count = int(min(max_states, max(2, len(frame) // 40)))
    state_count = min(state_count, len(frame))
    labels = KMeans(n_clusters=state_count, n_init=10, random_state=random_state).fit_predict(state_z)
    centers = np.vstack([state_z[labels == label].mean(axis=0) for label in range(state_count)])

    feature_values = frame[feature_columns].to_numpy(float)
    state_models = []
    state_weights = []
    for state_index in range(state_count):
        mask = labels == state_index
        values = feature_values[mask]
        state_weights.append(float(mask.mean()))
        state_models.append(
            _fit_state_feature_mixture(
                values,
                random_state=random_state + 100 + state_index,
                max_components=max_components,
            )
        )

    return TournamentModel(
        model_name="conditional_state_mixture_residual",
        feature_columns=feature_columns,
        payload={
            "state_columns": state_columns,
            "state_fill": state_fill.to_numpy(float),
            "state_mean": state_mean,
            "state_std": state_std,
            "state_centers": centers,
            "state_models": state_models,
            "state_weights": np.asarray(state_weights, dtype=float),
            "state_count": int(state_count),
            "source_row_count": int(len(frame)),
            "max_components": int(max_components),
        },
    )


def _available_conditional_state_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in CONDITIONAL_STATE_CANDIDATE_COLUMNS
        if column in frame.columns or column in RECENT_STATE_FEATURES
    ]
    usable = []
    min_non_null = max(8, int(len(frame) * 0.20))
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if int(values.notna().sum()) < min_non_null:
            continue
        if float(values.std(skipna=True) or 0.0) <= 1e-8:
            continue
        usable.append(column)
    return usable


def _fit_state_feature_mixture(
    values: np.ndarray,
    *,
    random_state: int,
    max_components: int,
) -> dict[str, np.ndarray | int | str]:
    values = np.asarray(values, dtype=float)
    component_count = int(min(max_components, max(1, len(values) // 30)))
    if component_count < 2 or len(values) < values.shape[1] + 8:
        return _single_component_state_model(values)
    try:
        mixture = BayesianGaussianMixture(
            n_components=component_count,
            covariance_type="full",
            reg_covar=1e-4,
            weight_concentration_prior_type="dirichlet_process",
            max_iter=500,
            random_state=random_state,
        )
        mixture.fit(values)
    except ValueError:
        return _single_component_state_model(values)

    weights = np.asarray(mixture.weights_, dtype=float)
    active = weights > 1e-3
    if not active.any():
        return _single_component_state_model(values)
    weights = weights[active]
    weights = weights / weights.sum()
    return {
        "kind": "bayesian_gaussian_mixture",
        "weights": weights,
        "means": np.asarray(mixture.means_[active], dtype=float),
        "covariances": np.asarray(mixture.covariances_[active], dtype=float)
        + np.eye(values.shape[1]) * 1e-5,
        "component_count": int(active.sum()),
        "source_row_count": int(len(values)),
    }


def _single_component_state_model(values: np.ndarray) -> dict[str, np.ndarray | int | str]:
    return {
        "kind": "single_gaussian",
        "weights": np.ones(1, dtype=float),
        "means": values.mean(axis=0, keepdims=True),
        "covariances": _robust_cov(values)[None, :, :],
        "component_count": 1,
        "source_row_count": int(len(values)),
    }


def sample_tournament_model(
    model: TournamentModel,
    n: int,
    context_df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if model.model_name == "pca_latent_residual":
        values = _sample_pca_latent_model(model, n=n, rng=rng)
    elif model.model_name == "context_neighbor_residual":
        values = _sample_context_neighbor_model(model, n=n, context_df=context_df, rng=rng)
    elif model.model_name == "derived_joint_gaussian":
        values = _sample_derived_joint_gaussian_model(model, n=n, rng=rng)
    elif model.model_name == "conditional_state_mixture_residual":
        values = _sample_conditional_state_mixture_model(
            model,
            n=n,
            context_df=context_df,
            rng=rng,
        )
    else:
        raise KeyError(f"Unknown tournament model: {model.model_name}")
    samples = pd.DataFrame(values, columns=model.feature_columns)
    return _normalize_spin_axis_components(samples)


def _sample_pca_latent_model(
    model: TournamentModel,
    *,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    payload = model.payload
    latent = rng.multivariate_normal(
        payload["latent_mean"],
        payload["latent_cov"],
        size=n,
        check_valid="ignore",
    )
    residuals = rng.multivariate_normal(
        np.zeros(len(model.feature_columns)),
        payload["residual_cov"],
        size=n,
        check_valid="ignore",
    )
    standardized = latent @ payload["components"] + residuals
    return standardized * payload["std"] + payload["mean"]


def _sample_context_neighbor_model(
    model: TournamentModel,
    *,
    n: int,
    context_df: pd.DataFrame | None,
    rng: np.random.Generator,
) -> np.ndarray:
    payload = model.payload
    feature_pool = payload["feature_pool"]
    context_pool = payload["context_pool"]
    context_columns = list(payload["context_columns"])
    context_z = (context_pool - payload["context_mean"]) / payload["context_std"]
    if context_df is None or not set(context_columns).issubset(context_df.columns):
        selected = rng.choice(len(feature_pool), size=n, replace=True)
        return feature_pool[selected]

    target_context = context_df[context_columns].dropna().to_numpy(float)
    if len(target_context) == 0:
        selected = rng.choice(len(feature_pool), size=n, replace=True)
        return feature_pool[selected]
    if len(target_context) < n:
        target_context = target_context[rng.choice(len(target_context), size=n, replace=True)]
    else:
        target_context = target_context[:n]

    values = []
    k_neighbors = int(payload["k_neighbors"])
    for target in target_context:
        target_z = (target - payload["context_mean"]) / payload["context_std"]
        distances = np.linalg.norm(context_z - target_z, axis=1)
        neighbor_idx = np.argsort(distances)[:k_neighbors]
        local_features = feature_pool[neighbor_idx]
        weights = np.exp(-0.5 * np.square(distances[neighbor_idx]))
        if weights.sum() <= 0:
            weights = np.ones(len(neighbor_idx)) / len(neighbor_idx)
        else:
            weights = weights / weights.sum()
        mean = np.average(local_features, axis=0, weights=weights)
        centered = local_features - mean
        cov = (centered * weights[:, None]).T @ centered
        cov += np.eye(local_features.shape[1]) * 1e-5
        values.append(rng.multivariate_normal(mean, cov, check_valid="ignore"))
    return np.asarray(values)


def _sample_derived_joint_gaussian_model(
    model: TournamentModel,
    *,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    payload = model.payload
    joint_values = rng.multivariate_normal(
        payload["mean"],
        payload["cov"],
        size=n,
        check_valid="ignore",
    )
    return joint_values[:, : len(model.feature_columns)]


def _sample_conditional_state_mixture_model(
    model: TournamentModel,
    *,
    n: int,
    context_df: pd.DataFrame | None,
    rng: np.random.Generator,
) -> np.ndarray:
    payload = model.payload
    state_indices = _select_conditional_state_indices(model, n=n, context_df=context_df, rng=rng)
    values = []
    for state_index in state_indices:
        state_model = payload["state_models"][int(state_index)]
        weights = np.asarray(state_model["weights"], dtype=float)
        component_index = int(rng.choice(len(weights), p=weights / weights.sum()))
        mean = np.asarray(state_model["means"][component_index], dtype=float)
        cov = np.asarray(state_model["covariances"][component_index], dtype=float)
        values.append(rng.multivariate_normal(mean, cov, check_valid="ignore"))
    return np.asarray(values)


def _select_conditional_state_indices(
    model: TournamentModel,
    *,
    n: int,
    context_df: pd.DataFrame | None,
    rng: np.random.Generator,
) -> np.ndarray:
    payload = model.payload
    state_count = int(payload["state_count"])
    if context_df is None:
        weights = np.asarray(payload["state_weights"], dtype=float)
        return rng.choice(state_count, size=n, replace=True, p=weights / weights.sum())

    context = add_recent_pitcher_state_features(context_df)
    state_columns = list(payload["state_columns"])
    if not set(state_columns).issubset(context.columns):
        weights = np.asarray(payload["state_weights"], dtype=float)
        return rng.choice(state_count, size=n, replace=True, p=weights / weights.sum())
    state_frame = context[state_columns].apply(pd.to_numeric, errors="coerce")
    state_values = state_frame.fillna(
        pd.Series(payload["state_fill"], index=state_columns)
    ).to_numpy(float)
    if len(state_values) == 0:
        weights = np.asarray(payload["state_weights"], dtype=float)
        return rng.choice(state_count, size=n, replace=True, p=weights / weights.sum())
    if len(state_values) < n:
        state_values = state_values[rng.choice(len(state_values), size=n, replace=True)]
    else:
        state_values = state_values[:n]
    state_z = (state_values - payload["state_mean"]) / payload["state_std"]
    centers = np.asarray(payload["state_centers"], dtype=float)
    distances = np.linalg.norm(state_z[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


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


def pitch_family_for_pitch_type(pitch_type: str) -> str:
    """Map Statcast pitch type codes to broad physics families."""
    return PITCH_FAMILY_BY_PITCH_TYPE.get(str(pitch_type).upper(), "unknown")


def pitch_family_release_spin_settings(pitch_type: str) -> dict[str, float]:
    """Return release/spin treatment strengths for a pitch type."""
    family = pitch_family_for_pitch_type(pitch_type)
    return {
        key: float(value)
        for key, value in PITCH_FAMILY_RELEASE_SPIN_SETTINGS[family].items()
    } | {"pitch_family": family}


def fit_release_geometry_constraint(train: pd.DataFrame) -> dict[str, float | int]:
    """Learn the stable release-y plus extension relationship from real pitches."""
    required = ["release_pos_y", "release_extension"]
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"Missing release geometry columns: {', '.join(missing)}")

    frame = _sort_pitch_rows(train)
    keep_columns = required + [
        column for column in ["game_pk", "game_date"] if column in frame.columns
    ]
    frame = frame[keep_columns].dropna(subset=required).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 complete release geometry rows are required.")

    release_sum = (
        pd.to_numeric(frame["release_pos_y"], errors="coerce")
        + pd.to_numeric(frame["release_extension"], errors="coerce")
    ).to_numpy(float)
    extension = pd.to_numeric(frame["release_extension"], errors="coerce").to_numpy(float)
    sum_mean = float(np.mean(release_sum))
    sum_std = float(np.std(release_sum, ddof=0))
    predicted_sum_mean = sum_mean
    residual_std = sum_std
    recent_game_count = 0

    if "game_pk" in frame.columns:
        game_order = frame[
            ["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])
        ].drop_duplicates()
        if len(game_order) >= 2:
            game_values = frame["game_pk"].to_numpy()
            game_means = []
            game_counts = []
            aligned_game_means = np.zeros(len(frame), dtype=float)
            for game_pk in game_order["game_pk"].tolist():
                mask = game_values == game_pk
                game_mean = float(release_sum[mask].mean())
                game_means.append(game_mean)
                game_counts.append(int(mask.sum()))
                aligned_game_means[mask] = game_mean
            positions = np.arange(len(game_means), dtype=float)
            half_life_games = 1.25
            recency_weights = np.exp(-(positions.max() - positions) / half_life_games)
            recency_weights *= np.sqrt(np.asarray(game_counts, dtype=float))
            recency_weights = recency_weights / recency_weights.sum()
            predicted_sum_mean = float(np.average(game_means, weights=recency_weights))
            residual_std = float(np.std(release_sum - aligned_game_means, ddof=0))
            recent_game_count = int(len(game_means))

    target_sum_std = max(residual_std, sum_std * 0.15, 0.01)
    extension_q01, extension_q99 = np.quantile(extension, [0.01, 0.99])
    extension_padding = max(float(np.std(extension, ddof=0)) * 0.25, 0.02)
    return {
        "sum_mean": sum_mean,
        "sum_std": sum_std,
        "predicted_sum_mean": predicted_sum_mean,
        "target_sum_std": float(target_sum_std),
        "extension_lower": float(extension_q01 - extension_padding),
        "extension_upper": float(extension_q99 + extension_padding),
        "source_row_count": int(len(frame)),
        "recent_game_count": recent_game_count,
    }


def apply_release_geometry_constraint(
    samples: pd.DataFrame,
    constraint: Mapping[str, float | int],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Project generated release samples back onto plausible mound geometry."""
    required = ["release_pos_y", "release_extension"]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing release geometry columns: {', '.join(missing)}")

    result = samples.copy()
    rng = np.random.default_rng(random_state)
    extension = pd.to_numeric(result["release_extension"], errors="coerce").to_numpy(float)
    extension = np.clip(
        extension,
        float(constraint["extension_lower"]),
        float(constraint["extension_upper"]),
    )
    target_sum = rng.normal(
        float(constraint["predicted_sum_mean"]),
        float(constraint["target_sum_std"]),
        size=len(result),
    )
    result["release_extension"] = extension
    result["release_pos_y"] = target_sum - extension
    return result


def apply_release_geometry_blend(
    samples: pd.DataFrame,
    constraint: Mapping[str, float | int],
    *,
    alpha: float = 0.50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Partially project generated samples toward plausible y/extension geometry."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")
    constrained = apply_release_geometry_constraint(
        samples,
        constraint,
        random_state=random_state,
    )
    result = samples.copy()
    for column in ("release_pos_y", "release_extension"):
        result[column] = result[column] + alpha * (constrained[column] - result[column])

    release_sum = result["release_pos_y"] + result["release_extension"]
    extension = np.clip(
        pd.to_numeric(result["release_extension"], errors="coerce").to_numpy(float),
        float(constraint["extension_lower"]),
        float(constraint["extension_upper"]),
    )
    result["release_extension"] = extension
    result["release_pos_y"] = release_sum - extension
    return result


def fit_recent_state_anchor(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    half_life_games: float = 1.25,
) -> dict[str, object]:
    """Estimate the next-state feature center from the pitcher's recent games."""
    missing = [column for column in feature_columns if column not in train.columns]
    if missing:
        raise ValueError(f"Missing recent state feature columns: {', '.join(missing)}")

    frame = _sort_pitch_rows(train)
    keep_columns = feature_columns + [
        column for column in ["game_pk", "game_date"] if column in frame.columns
    ]
    frame = frame[keep_columns].dropna(subset=feature_columns).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 complete recent state rows are required.")

    values = frame[feature_columns].to_numpy(float)
    game_count = 0
    if "game_pk" not in frame.columns:
        recent_n = max(10, int(np.ceil(len(frame) * 0.35)))
        means = values[-recent_n:].mean(axis=0)
    else:
        game_order = frame[
            ["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])
        ].drop_duplicates()
        game_count = int(len(game_order))
        if game_count < 2:
            means = values.mean(axis=0)
        else:
            game_values = frame["game_pk"].to_numpy()
            game_means = []
            game_counts = []
            for game_pk in game_order["game_pk"].tolist():
                mask = game_values == game_pk
                game_means.append(values[mask].mean(axis=0))
                game_counts.append(int(mask.sum()))
            positions = np.arange(len(game_means), dtype=float)
            recency_weights = np.exp(-(positions.max() - positions) / half_life_games)
            recency_weights *= np.sqrt(np.asarray(game_counts, dtype=float))
            recency_weights = recency_weights / recency_weights.sum()
            means = np.average(np.asarray(game_means), axis=0, weights=recency_weights)

    return {
        "feature_columns": list(feature_columns),
        "means": np.asarray(means, dtype=float),
        "source_row_count": int(len(frame)),
        "game_count": game_count,
        "half_life_games": float(half_life_games),
    }


def fit_recent_trend_state_anchor(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    half_life_games: float = 5.0,
    horizon_games: float = 1.0,
    trend_shrinkage: float = 0.75,
    max_recent_std: float = 1.5,
) -> dict[str, object]:
    """Estimate a bounded next-state center from recent per-game feature trends."""
    missing = [column for column in feature_columns if column not in train.columns]
    if missing:
        raise ValueError(f"Missing recent trend feature columns: {', '.join(missing)}")
    if half_life_games <= 0:
        raise ValueError("half_life_games must be positive.")
    if trend_shrinkage < 0:
        raise ValueError("trend_shrinkage must be non-negative.")
    if max_recent_std <= 0:
        raise ValueError("max_recent_std must be positive.")

    frame = _sort_pitch_rows(train)
    keep_columns = feature_columns + [
        column for column in ["game_pk", "game_date"] if column in frame.columns
    ]
    frame = frame[keep_columns].dropna(subset=feature_columns).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 complete recent trend rows are required.")

    values = frame[feature_columns].to_numpy(float)
    if "game_pk" not in frame.columns:
        recent_n = max(10, int(np.ceil(len(frame) * 0.35)))
        means = values[-recent_n:].mean(axis=0)
        game_count = 0
    else:
        game_order = frame[
            ["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])
        ].drop_duplicates()
        game_count = int(len(game_order))
        if game_count < 3:
            means = values.mean(axis=0)
        else:
            game_values = frame["game_pk"].to_numpy()
            game_means = []
            game_counts = []
            for game_pk in game_order["game_pk"].tolist():
                mask = game_values == game_pk
                game_means.append(values[mask].mean(axis=0))
                game_counts.append(int(mask.sum()))

            game_mean_matrix = np.asarray(game_means, dtype=float)
            positions = np.arange(game_count, dtype=float)
            recency_weights = np.exp(-(positions.max() - positions) / half_life_games)
            recency_weights *= np.sqrt(np.asarray(game_counts, dtype=float))
            recency_weights = recency_weights / recency_weights.sum()
            x_mean = float(np.sum(recency_weights * positions))
            centered_x = positions - x_mean
            denominator = float(np.sum(recency_weights * centered_x * centered_x) + 1e-8)

            means = []
            recent_window = game_mean_matrix[-min(8, game_count) :]
            recent_mean = recent_window.mean(axis=0)
            recent_std = recent_window.std(axis=0, ddof=0)
            for column_index in range(game_mean_matrix.shape[1]):
                y = game_mean_matrix[:, column_index]
                y_mean = float(np.sum(recency_weights * y))
                slope = float(np.sum(recency_weights * centered_x * (y - y_mean)) / denominator)
                prediction = y_mean + trend_shrinkage * slope * (
                    (game_count - 1 + horizon_games) - x_mean
                )
                lower = recent_mean[column_index] - max_recent_std * recent_std[column_index]
                upper = recent_mean[column_index] + max_recent_std * recent_std[column_index]
                means.append(float(np.clip(prediction, lower, upper)))
            means = np.asarray(means, dtype=float)

    return {
        "feature_columns": list(feature_columns),
        "means": np.asarray(means, dtype=float),
        "source_row_count": int(len(frame)),
        "game_count": game_count,
        "half_life_games": float(half_life_games),
        "horizon_games": float(horizon_games),
        "trend_shrinkage": float(trend_shrinkage),
        "max_recent_std": float(max_recent_std),
    }


def apply_recent_state_anchor(
    samples: pd.DataFrame,
    anchor: Mapping[str, object],
    *,
    alpha: float = 0.70,
) -> pd.DataFrame:
    """Recenter a generated cloud toward the learned recent state mean."""
    columns = list(anchor["feature_columns"])
    missing = [column for column in columns if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing recent state sample columns: {', '.join(missing)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    result = samples.copy()
    target_means = np.asarray(anchor["means"], dtype=float)
    for column, target_mean in zip(columns, target_means, strict=True):
        current_mean = float(pd.to_numeric(result[column], errors="coerce").mean())
        if np.isfinite(current_mean) and np.isfinite(target_mean):
            result[column] = result[column] + alpha * (target_mean - current_mean)
    return result


def _circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        weights = np.ones(len(angles), dtype=float)
    weights = np.asarray(weights, dtype=float)
    vector = np.sum(weights * np.exp(1j * angles))
    if abs(vector) < 1e-12:
        return float(np.angle(np.exp(1j * angles[-1])))
    return float(np.angle(vector))


def fit_spin_axis_angle_anchor(
    train: pd.DataFrame,
    *,
    half_life_games: float = 1.25,
) -> dict[str, object]:
    """Estimate recent spin-axis direction as a circular game-state mean."""
    required = ["spin_axis_cos", "spin_axis_sin"]
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"Missing spin axis columns: {', '.join(missing)}")

    frame = _sort_pitch_rows(train)
    keep_columns = required + [
        column for column in ["game_pk", "game_date"] if column in frame.columns
    ]
    frame = frame[keep_columns].dropna(subset=required).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 complete spin axis rows are required.")

    angles = np.arctan2(
        pd.to_numeric(frame["spin_axis_sin"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame["spin_axis_cos"], errors="coerce").to_numpy(float),
    )
    game_count = 0
    if "game_pk" not in frame.columns:
        recent_n = max(10, int(np.ceil(len(frame) * 0.35)))
        angle_mean = _circular_mean(angles[-recent_n:])
    else:
        game_order = frame[
            ["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])
        ].drop_duplicates()
        game_count = int(len(game_order))
        if game_count < 2:
            angle_mean = _circular_mean(angles)
        else:
            game_values = frame["game_pk"].to_numpy()
            game_angles = []
            game_counts = []
            for game_pk in game_order["game_pk"].tolist():
                mask = game_values == game_pk
                game_angles.append(_circular_mean(angles[mask]))
                game_counts.append(int(mask.sum()))
            positions = np.arange(len(game_angles), dtype=float)
            recency_weights = np.exp(-(positions.max() - positions) / half_life_games)
            recency_weights *= np.sqrt(np.asarray(game_counts, dtype=float))
            recency_weights = recency_weights / recency_weights.sum()
            angle_mean = _circular_mean(np.asarray(game_angles), recency_weights)

    return {
        "angle_mean": float(angle_mean),
        "source_row_count": int(len(frame)),
        "game_count": game_count,
        "half_life_games": float(half_life_games),
    }


def apply_spin_axis_angle_anchor(
    samples: pd.DataFrame,
    anchor: Mapping[str, object],
    *,
    alpha: float = 0.70,
) -> pd.DataFrame:
    """Rotate generated spin-axis samples toward a circular recent-state anchor."""
    required = ["spin_axis_cos", "spin_axis_sin"]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing spin axis sample columns: {', '.join(missing)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    result = samples.copy()
    angles = np.arctan2(
        pd.to_numeric(result["spin_axis_sin"], errors="coerce").to_numpy(float),
        pd.to_numeric(result["spin_axis_cos"], errors="coerce").to_numpy(float),
    )
    angle_mean = float(anchor["angle_mean"])
    delta = np.angle(np.exp(1j * (angle_mean - angles)))
    anchored_angles = angles + alpha * delta
    result["spin_axis_cos"] = np.cos(anchored_angles)
    result["spin_axis_sin"] = np.sin(anchored_angles)
    return result


def _angular_difference(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def fit_spin_axis_residual_model(
    train: pd.DataFrame,
    *,
    half_life_games: float = 1.25,
    recent_fraction: float = 0.50,
) -> dict[str, object]:
    """Fit an empirical circular residual model for spin-axis samples."""
    required = ["spin_axis_cos", "spin_axis_sin"]
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise ValueError(f"Missing spin axis columns: {', '.join(missing)}")
    if not 0.0 < recent_fraction <= 1.0:
        raise ValueError("recent_fraction must be in (0.0, 1.0].")

    frame = _sort_pitch_rows(train)
    keep_columns = required + [
        column for column in ["game_pk", "game_date"] if column in frame.columns
    ]
    frame = frame[keep_columns].dropna(subset=required).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 complete spin axis rows are required.")

    angles = np.arctan2(
        pd.to_numeric(frame["spin_axis_sin"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame["spin_axis_cos"], errors="coerce").to_numpy(float),
    )
    anchor = fit_spin_axis_angle_anchor(frame, half_life_games=half_life_games)
    recent_n = max(10, int(np.ceil(len(angles) * recent_fraction)))
    recent_angles = angles[-recent_n:]
    residuals = _angular_difference(recent_angles, float(anchor["angle_mean"]))
    if len(residuals) < 10:
        residuals = _angular_difference(angles, float(anchor["angle_mean"]))
    return {
        "angle_mean": float(anchor["angle_mean"]),
        "residuals": np.asarray(residuals, dtype=float),
        "source_row_count": int(len(frame)),
        "residual_count": int(len(residuals)),
        "game_count": int(anchor["game_count"]),
        "half_life_games": float(half_life_games),
        "recent_fraction": float(recent_fraction),
    }


def apply_spin_axis_residual_model(
    samples: pd.DataFrame,
    model: Mapping[str, object],
    *,
    alpha: float = 0.65,
    random_state: int = 42,
) -> pd.DataFrame:
    """Move samples toward recent spin state while preserving empirical angular spread."""
    required = ["spin_axis_cos", "spin_axis_sin"]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing spin axis sample columns: {', '.join(missing)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    result = samples.copy()
    rng = np.random.default_rng(random_state)
    sample_angles = np.arctan2(
        pd.to_numeric(result["spin_axis_sin"], errors="coerce").to_numpy(float),
        pd.to_numeric(result["spin_axis_cos"], errors="coerce").to_numpy(float),
    )
    residuals = np.asarray(model["residuals"], dtype=float)
    sampled_residuals = residuals[rng.integers(0, len(residuals), size=len(result))]
    target_angles = float(model["angle_mean"]) + sampled_residuals
    adjusted_angles = sample_angles + alpha * _angular_difference(target_angles, sample_angles)
    result["spin_axis_cos"] = np.cos(adjusted_angles)
    result["spin_axis_sin"] = np.sin(adjusted_angles)
    return result


def _available_tournament_context_columns(train: pd.DataFrame) -> list[str]:
    return [column for column in CONTEXT_FEATURES if column in train.columns]


def _fit_tournament_candidate_models(train: pd.DataFrame) -> list[TournamentModel]:
    feature_columns = FEATURE_GROUPS["physics_core"]
    context_columns = _available_tournament_context_columns(train)
    models = []
    try:
        models.append(
            fit_conditional_state_mixture_model(
                train,
                feature_columns=feature_columns,
            )
        )
    except ValueError:
        pass
    models.extend(
        [
            fit_pca_latent_model(train, feature_columns=feature_columns),
            fit_derived_joint_gaussian_model(train, feature_columns=feature_columns),
        ]
    )
    if context_columns:
        models.insert(
            len(models) - 1,
            fit_context_neighbor_model(
                train,
                feature_columns=feature_columns,
                context_columns=context_columns,
            ),
        )
    return models


def _sort_pitch_rows(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if column in frame.columns
    ]
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _fit_short_memory_release_model(
    train: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    half_life_games: float = 1.0,
) -> GeneratorModel | None:
    if "game_pk" not in train.columns:
        return None
    keep_columns = ["game_pk", *PITCH_PHYSICS_FEATURES]
    if "game_date" in train.columns:
        keep_columns.insert(1, "game_date")
    frame = _sort_pitch_rows(train)[keep_columns].dropna(subset=PITCH_PHYSICS_FEATURES)
    if len(frame) < 40:
        return None

    game_order = (
        frame[["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(game_order) < 2:
        return None

    feature_values = frame[PITCH_PHYSICS_FEATURES].to_numpy(float)
    game_means = []
    game_counts = []
    aligned_game_means = np.zeros_like(feature_values)
    for game_pk in game_order["game_pk"].tolist():
        mask = frame["game_pk"].to_numpy() == game_pk
        game_mean = feature_values[mask].mean(axis=0)
        game_means.append(game_mean)
        game_counts.append(int(mask.sum()))
        aligned_game_means[mask] = game_mean

    game_means_array = np.asarray(game_means)
    positions = np.arange(len(game_means_array), dtype=float)
    recency_weights = np.exp(-(positions.max() - positions) / half_life_games)
    recency_weights *= np.sqrt(np.asarray(game_counts, dtype=float))
    recency_weights = recency_weights / recency_weights.sum()
    predicted_game_mean = np.average(game_means_array, axis=0, weights=recency_weights)

    residuals = feature_values - aligned_game_means
    recent_values = frame[PITCH_PHYSICS_FEATURES].tail(
        max(10, int(len(frame) * 0.35))
    ).to_numpy(float)
    cov = 0.70 * _robust_cov(residuals) + 0.30 * _robust_cov(recent_values)
    return GeneratorModel(
        "player_recent_weighted_game_drift_gaussian",
        pitcher_name,
        pitch_type,
        "release_only",
        PITCH_PHYSICS_FEATURES,
        {
            "predicted_game_mean": predicted_game_mean,
            "cov": cov,
            "context_columns": [],
            "source_row_count": int(len(feature_values)),
            "half_life_games": float(half_life_games),
        },
    )


def _fit_factorized_variants(
    factorized,
    train: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    random_state: int,
) -> dict[str, object]:
    variants: dict[str, object] = {"factorized_v2_1": factorized}
    release_suite = fit_generator_suite(
        train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        feature_group="release_only",
        random_state=random_state,
    )
    release_variant_specs = [
        ("factorized_release_game_drift_gaussian", "player_recent_weighted_game_drift_gaussian"),
        ("factorized_release_game_drift_copula", "player_recent_weighted_game_drift_copula"),
        ("factorized_release_recent_gaussian", "player_recent_multivariate_gaussian"),
    ]
    for variant_name, release_model_name in release_variant_specs:
        if release_model_name not in release_suite:
            continue
        variants[variant_name] = replace(
            factorized,
            model_name=variant_name,
            release_model=release_suite[release_model_name],
            release_model_name=release_model_name,
        )
    short_memory_release = _fit_short_memory_release_model(
        train,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        half_life_games=1.0,
    )
    if short_memory_release is not None:
        short_memory_payload = dict(short_memory_release.payload)
        short_memory_payload["cov"] = short_memory_payload["cov"] * 2.0
        short_memory_release = replace(short_memory_release, payload=short_memory_payload)
        variants["factorized_short_memory_wide_residual"] = replace(
            factorized,
            model_name="factorized_short_memory_wide_residual",
            release_model=short_memory_release,
            release_model_name="short_memory_release_half_life_1.0",
            downstream_residual_cov=factorized.downstream_residual_cov * 30.0,
            downstream_residual_copula=None,
            downstream_residual_offset=factorized.downstream_residual_offset * 0.25,
        )
        variants["factorized_short_memory_more_uncertain"] = replace(
            factorized,
            model_name="factorized_short_memory_more_uncertain",
            release_model=short_memory_release,
            release_model_name="short_memory_release_half_life_1.0",
            downstream_residual_cov=factorized.downstream_residual_cov * 40.0,
            downstream_residual_copula=None,
            downstream_residual_offset=factorized.downstream_residual_offset * 0.15,
        )
        variants["factorized_recent_state_anchored"] = replace(
            variants["factorized_short_memory_wide_residual"],
            model_name="factorized_recent_state_anchored",
        )
        variants["factorized_trend_state_anchored"] = replace(
            variants["factorized_short_memory_wide_residual"],
            model_name="factorized_trend_state_anchored",
        )
        variants["factorized_release_state_anchored"] = replace(
            variants["factorized_short_memory_wide_residual"],
            model_name="factorized_release_state_anchored",
        )
        variants["factorized_pitch_family_release_spin"] = replace(
            variants["factorized_short_memory_wide_residual"],
            model_name="factorized_pitch_family_release_spin",
        )
        variants["factorized_physics_constrained_state"] = replace(
            variants["factorized_short_memory_wide_residual"],
            model_name="factorized_physics_constrained_state",
        )
    return variants


def _evaluate_samples_by_layer(
    holdout: pd.DataFrame,
    model_samples: dict[str, pd.DataFrame],
    *,
    random_state: int,
) -> dict[str, dict[str, dict[str, object]]]:
    layer_results: dict[str, dict[str, dict[str, object]]] = {}
    for layer_index, feature_group in enumerate(VALIDATION_LAYERS):
        columns = FEATURE_GROUPS[feature_group]
        layer_results[feature_group] = {}
        for model_index, (model_name, samples) in enumerate(model_samples.items()):
            metrics = classifier_two_sample_test(
                holdout,
                samples,
                columns,
                random_state=random_state + layer_index * 100 + model_index,
            )
            layer_results[feature_group][model_name] = {
                "auc": float(metrics["auc"]),
                "raw_auc": float(metrics["raw_auc"]),
                "train_auc": float(metrics["train_auc"]),
                "n_real": int(metrics["n_real"]),
                "n_simulated": int(metrics["n_simulated"]),
                "top_leakage_features": metrics["top_leakage_features"],
            }
    return layer_results


def _aggregate_repeat_results(
    repeated_results: list[dict[str, dict[str, dict[str, object]]]],
    model_names: list[str],
    *,
    target_auc: float,
) -> dict[str, dict[str, dict[str, object]]]:
    if not repeated_results:
        raise ValueError("At least one repeated result is required.")
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for feature_group in VALIDATION_LAYERS:
        summary[feature_group] = {}
        for model_name in model_names:
            auc_values = np.asarray(
                [
                    float(repeat[feature_group][model_name]["auc"])
                    for repeat in repeated_results
                ],
                dtype=float,
            )
            first = repeated_results[0][feature_group][model_name]
            summary[feature_group][model_name] = {
                "features": FEATURE_GROUPS[feature_group],
                "mean_auc": float(auc_values.mean()),
                "std_auc": float(auc_values.std(ddof=0)),
                "min_auc": float(auc_values.min()),
                "max_auc": float(auc_values.max()),
                "pass_rate": float((auc_values <= target_auc).mean()),
                "repeat_count": int(len(auc_values)),
                "n_real": int(first["n_real"]),
                "n_simulated": int(first["n_simulated"]),
                "top_leakage_features": first["top_leakage_features"],
            }
    return summary


def evaluate_model_tournament(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    n_samples: int = 300,
    repeats: int = 12,
    random_state: int = 42,
    target_auc: float = 0.60,
    target_pass_rate: float = 0.80,
) -> dict[str, object]:
    factorized = fit_factorized_physics_model(
        train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        random_state=random_state,
    )
    factorized_variants = _fit_factorized_variants(
        factorized,
        train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        random_state=random_state,
    )
    candidate_models = _fit_tournament_candidate_models(train)
    model_names = list(factorized_variants) + [model.model_name for model in candidate_models]
    sample_count = max(n_samples, len(holdout))
    release_geometry_constraint = fit_release_geometry_constraint(train)
    recent_state_anchor = fit_recent_state_anchor(train, FEATURE_GROUPS["physics_core"])
    trend_state_anchor = fit_recent_trend_state_anchor(
        train,
        FEATURE_GROUPS["physics_core"],
        half_life_games=5.0,
        horizon_games=1.0,
        trend_shrinkage=0.75,
    )
    release_state_anchor = fit_recent_state_anchor(train, RELEASE_STATE_ANCHOR_COLUMNS)
    spin_axis_anchor = fit_spin_axis_angle_anchor(train)
    pitch_family_settings = pitch_family_release_spin_settings(pitch_type)
    spin_axis_residual_model = fit_spin_axis_residual_model(
        train,
        half_life_games=1.25,
        recent_fraction=float(pitch_family_settings["spin_recent_fraction"]),
    )
    repeated_results = []
    for repeat_index in range(repeats):
        repeat_seed = random_state + repeat_index * 1000
        model_samples: dict[str, pd.DataFrame] = {}
        for model_index, (model_name, model) in enumerate(factorized_variants.items()):
            samples = sample_factorized_physics(
                model,
                n=sample_count,
                context_df=holdout,
                random_state=repeat_seed + 10 + model_index,
            )
            if model_name == "factorized_physics_constrained_state":
                samples = apply_release_geometry_constraint(
                    samples,
                    release_geometry_constraint,
                    random_state=repeat_seed + 500 + model_index,
                )
            elif model_name == "factorized_recent_state_anchored":
                samples = apply_recent_state_anchor(
                    samples,
                    recent_state_anchor,
                    alpha=0.70,
                )
            elif model_name == "factorized_trend_state_anchored":
                samples = apply_recent_state_anchor(
                    samples,
                    trend_state_anchor,
                    alpha=0.85,
                )
            elif model_name == "factorized_release_state_anchored":
                samples = apply_recent_state_anchor(
                    samples,
                    recent_state_anchor,
                    alpha=0.70,
                )
                samples = apply_recent_state_anchor(
                    samples,
                    release_state_anchor,
                    alpha=0.70,
                )
                samples = apply_spin_axis_angle_anchor(
                    samples,
                    spin_axis_anchor,
                    alpha=0.70,
                )
            elif model_name == "factorized_pitch_family_release_spin":
                samples = apply_recent_state_anchor(
                    samples,
                    recent_state_anchor,
                    alpha=float(pitch_family_settings["physics_anchor_alpha"]),
                )
                samples = apply_recent_state_anchor(
                    samples,
                    release_state_anchor,
                    alpha=float(pitch_family_settings["release_anchor_alpha"]),
                )
                samples = apply_release_geometry_blend(
                    samples,
                    release_geometry_constraint,
                    alpha=float(pitch_family_settings["release_geometry_alpha"]),
                    random_state=repeat_seed + 700 + model_index,
                )
                samples = apply_spin_axis_residual_model(
                    samples,
                    spin_axis_residual_model,
                    alpha=float(pitch_family_settings["spin_residual_alpha"]),
                    random_state=repeat_seed + 900 + model_index,
                )
            model_samples[model_name] = samples
        factorized_count = len(model_samples)
        for model_index, model in enumerate(candidate_models):
            model_samples[model.model_name] = sample_tournament_model(
                model,
                n=sample_count,
                context_df=holdout,
                random_state=repeat_seed + 100 + factorized_count + model_index,
            )
        repeated_results.append(
            _evaluate_samples_by_layer(
                holdout,
                model_samples,
                random_state=repeat_seed + 300,
            )
        )

    layer_results = _aggregate_repeat_results(
        repeated_results,
        model_names,
        target_auc=target_auc,
    )
    best_by_layer = {
        layer: min(rows.items(), key=lambda item: float(item[1]["mean_auc"]))[0]
        for layer, rows in layer_results.items()
    }
    best_physics_core_model = best_by_layer["physics_core"]
    candidate_default = (
        float(layer_results["physics_core"][best_physics_core_model]["mean_auc"]) <= target_auc
        and all(
            float(layer_results[layer][best_physics_core_model]["mean_auc"]) <= target_auc
            and float(layer_results[layer][best_physics_core_model]["pass_rate"])
            >= target_pass_rate
            for layer in VALIDATION_LAYERS
        )
    )
    candidate_notes = {
        "factorized_short_memory_wide_residual": {
            "release_half_life_games": 1.0,
            "release_covariance_scale": 2.0,
            "downstream_residual_covariance_scale": 30.0,
            "downstream_residual_offset_scale": 0.25,
            "interpretation": "tests whether the factorized model was underestimating late-season downstream uncertainty",
        },
        "factorized_physics_constrained_state": {
            "description": "Short-memory factorized model with release_pos_y + extension geometry constraint.",
            "release_geometry_constraint": "release_pos_y_plus_extension",
            "release_geometry_source_rows": int(release_geometry_constraint["source_row_count"]),
            "release_geometry_recent_game_count": int(
                release_geometry_constraint["recent_game_count"]
            ),
        },
        "factorized_short_memory_more_uncertain": {
            "release_half_life_games": 1.0,
            "release_covariance_scale": 2.0,
            "downstream_residual_covariance_scale": 40.0,
            "downstream_residual_offset_scale": 0.15,
            "interpretation": "tests whether the previous best model was still underestimating downstream physics uncertainty",
        },
        "factorized_recent_state_anchored": {
            "base_model": "factorized_short_memory_wide_residual",
            "recent_state_anchor_alpha": 0.70,
            "recent_state_anchor_half_life_games": float(
                recent_state_anchor["half_life_games"]
            ),
            "recent_state_anchor_source_rows": int(recent_state_anchor["source_row_count"]),
            "recent_state_anchor_game_count": int(recent_state_anchor["game_count"]),
            "interpretation": "keeps the generated cloud centered on the pitcher's recent learned game state while preserving pitch-level variation",
        },
        "factorized_trend_state_anchored": {
            "base_model": "factorized_short_memory_wide_residual",
            "trend_state_anchor_alpha": 0.85,
            "trend_state_anchor_half_life_games": float(
                trend_state_anchor["half_life_games"]
            ),
            "trend_state_anchor_horizon_games": float(trend_state_anchor["horizon_games"]),
            "trend_state_anchor_shrinkage": float(trend_state_anchor["trend_shrinkage"]),
            "trend_state_anchor_max_recent_std": float(trend_state_anchor["max_recent_std"]),
            "trend_state_anchor_source_rows": int(trend_state_anchor["source_row_count"]),
            "trend_state_anchor_game_count": int(trend_state_anchor["game_count"]),
            "interpretation": "tests whether short-horizon game-state extrapolation improves future physical realism",
        },
        "factorized_release_state_anchored": {
            "base_model": "factorized_recent_state_anchored",
            "release_anchor_columns": RELEASE_STATE_ANCHOR_COLUMNS,
            "release_anchor_alpha": 0.70,
            "release_anchor_source_rows": int(release_state_anchor["source_row_count"]),
            "release_anchor_game_count": int(release_state_anchor["game_count"]),
            "spin_axis_anchor": "circular_recent_game_mean",
            "spin_axis_anchor_alpha": 0.70,
            "spin_axis_anchor_source_rows": int(spin_axis_anchor["source_row_count"]),
            "spin_axis_anchor_game_count": int(spin_axis_anchor["game_count"]),
            "interpretation": "tests whether release-specific centering and circular spin-axis anchoring improve release-layer robustness",
        },
        "factorized_pitch_family_release_spin": {
            "base_model": "factorized_short_memory_wide_residual",
            "pitch_family": str(pitch_family_settings["pitch_family"]),
            "physics_anchor_alpha": float(pitch_family_settings["physics_anchor_alpha"]),
            "release_anchor_alpha": float(pitch_family_settings["release_anchor_alpha"]),
            "release_geometry_blend_alpha": float(
                pitch_family_settings["release_geometry_alpha"]
            ),
            "spin_axis_model": "empirical_recent_circular_residual",
            "spin_residual_alpha": float(pitch_family_settings["spin_residual_alpha"]),
            "spin_recent_fraction": float(pitch_family_settings["spin_recent_fraction"]),
            "spin_axis_residual_source_rows": int(spin_axis_residual_model["source_row_count"]),
            "spin_axis_residual_count": int(spin_axis_residual_model["residual_count"]),
            "interpretation": (
                "tests pitch-family-specific release centering, release geometry, "
                "and spin-axis residual sampling without collapsing spin to one mean"
            ),
        },
    }
    conditional_state_model = next(
        (
            model
            for model in candidate_models
            if model.model_name == "conditional_state_mixture_residual"
        ),
        None,
    )
    if conditional_state_model is not None:
        candidate_notes["conditional_state_mixture_residual"] = {
            "description": "V2.4 state-conditioned physics-core mixture model.",
            "state_conditioned": True,
            "state_count": int(conditional_state_model.payload["state_count"]),
            "state_columns": list(conditional_state_model.payload["state_columns"]),
            "max_components": int(conditional_state_model.payload["max_components"]),
            "source_row_count": int(conditional_state_model.payload["source_row_count"]),
            "interpretation": "tests whether recent pitcher state and context-specific mixture covariances improve full-physics realism",
        }
    return {
        "model_name": "pitcher_twin_model_tournament",
        "pitcher_name": pitcher_name,
        "pitch_type": pitch_type,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "sample_count": int(sample_count),
        "repeat_count": int(repeats),
        "model_names": model_names,
        "target_auc": float(target_auc),
        "target_pass_rate": float(target_pass_rate),
        "acceptance_rule": {
            "lower_auc_is_better": True,
            "validated_layer_target_auc": float(target_auc),
            "validated_layer_target_pass_rate": float(target_pass_rate),
            "candidate_default_requires_physics_core_at_or_below_target": True,
            "candidate_default_requires_same_model_all_layers_at_or_below_target_and_pass_rate": True,
        },
        "candidate_default": bool(candidate_default),
        "best_by_layer": best_by_layer,
        "best_physics_core_model": best_physics_core_model,
        "candidate_source_rows": {
            model_name: int(model.source_row_count)
            for model_name, model in factorized_variants.items()
        }
        | {
            model.model_name: int(model.payload["source_row_count"])
            for model in candidate_models
        },
        "factorized_release_models": {
            model_name: model.release_model_name
            for model_name, model in factorized_variants.items()
        },
        "candidate_notes": candidate_notes,
        "factorized_variant_notes": candidate_notes,
        "layer_results": layer_results,
        "repeat_results": repeated_results,
    }
