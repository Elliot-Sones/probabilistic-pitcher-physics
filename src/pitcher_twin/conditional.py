"""Conditional pitch distribution helpers for side-by-side comparisons."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from pitcher_twin.features import COUNT_BUCKET_CODES
from pitcher_twin.models import GeneratorModel, fit_generator_suite, sample_generator
from pitcher_twin.validator import classifier_two_sample_test


CONDITIONAL_MODEL_PRIORITY = [
    "player_recent_weighted_game_drift_copula",
    "player_recent_weighted_game_drift_gaussian",
    "player_context_weighted_gaussian",
    "player_recent_multivariate_gaussian",
    "player_multivariate_gaussian",
    "player_gmm",
    "player_recent_game_window_empirical",
    "player_recent_empirical_bootstrap",
    "player_empirical_bootstrap",
]

DEFAULT_SUMMARY_COLUMNS = [
    "release_speed",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
]


def _count_bucket_code(balls: int, strikes: int) -> float:
    if balls == 0 and strikes == 0:
        bucket = "first_pitch"
    elif balls == 3 and strikes == 2:
        bucket = "full"
    elif balls > strikes:
        bucket = "behind"
    elif strikes > balls:
        bucket = "ahead"
    else:
        bucket = "even"
    return float(COUNT_BUCKET_CODES[bucket])


def make_context_dataframe(
    *,
    inning: int,
    pitcher_game_pitch_count: int | float,
    balls: int,
    strikes: int,
    batter_hand: str,
    pitcher_score_diff: int | float,
    repeat: int = 1,
) -> pd.DataFrame:
    """Build a model context dataframe from UI-style controls."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1.")
    hand_code = {"R": 0.0, "L": 1.0}.get(str(batter_hand).upper())
    if hand_code is None:
        raise ValueError("batter_hand must be 'R' or 'L'.")

    row = {
        "balls": float(balls),
        "strikes": float(strikes),
        "count_bucket_code": _count_bucket_code(int(balls), int(strikes)),
        "inning": float(inning),
        "pitcher_game_pitch_count": float(pitcher_game_pitch_count),
        "batter_stand_code": hand_code,
        "pitcher_score_diff": float(pitcher_score_diff),
    }
    return pd.DataFrame([row] * repeat)


def select_conditional_model(
    suite: dict[str, GeneratorModel],
    priority: Iterable[str] = CONDITIONAL_MODEL_PRIORITY,
) -> tuple[GeneratorModel, str, dict[str, object]]:
    """Select the best available conditional-capable generator from a fitted suite."""
    attempted = list(priority)
    for model_name in attempted:
        if model_name in suite:
            return suite[model_name], model_name, {
                "selected_model": model_name,
                "attempted_models": attempted,
                "used_fallback": model_name != attempted[0],
            }
    if not suite:
        raise ValueError("At least one fitted generator is required.")
    model_name = next(iter(suite))
    return suite[model_name], model_name, {
        "selected_model": model_name,
        "attempted_models": attempted,
        "used_fallback": True,
    }


def sample_conditional_distribution(
    suite: dict[str, GeneratorModel],
    context_df: pd.DataFrame,
    n: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Sample a fitted suite under a selected game context."""
    model, model_name, metadata = select_conditional_model(suite)
    samples = sample_generator(model, n=n, random_state=random_state, context_df=context_df)
    metadata = {
        **metadata,
        "feature_group": model.feature_group,
        "feature_columns": model.feature_columns,
        "requested_samples": int(n),
    }
    return samples, metadata


def summarize_distribution(
    samples: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Summarize generated samples with mean/std and central quantiles."""
    selected_columns = list(columns) if columns is not None else list(samples.columns)
    summary: dict[str, dict[str, float]] = {}
    for column in selected_columns:
        if column not in samples.columns:
            continue
        values = pd.to_numeric(samples[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "p10": float(values.quantile(0.10)),
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
        }
    return summary


def derive_miss_tendency(
    samples: pd.DataFrame,
    pitcher_hand: str | None = None,
    zone_half_width: float = 0.83,
    zone_bottom: float = 1.50,
    zone_top: float = 3.50,
) -> dict[str, object]:
    """Derive readable miss tendencies from generated plate-location samples."""
    required = {"plate_x", "plate_z"}
    if not required.issubset(samples.columns):
        return {
            "sample_count": 0,
            "zone_rate": float("nan"),
            "chase_rate": float("nan"),
            "spike_risk_rate": float("nan"),
            "primary_horizontal": "unavailable",
            "primary_vertical": "unavailable",
        }

    plate = samples[["plate_x", "plate_z"]].dropna().astype(float)
    if plate.empty:
        return {
            "sample_count": 0,
            "zone_rate": float("nan"),
            "chase_rate": float("nan"),
            "spike_risk_rate": float("nan"),
            "primary_horizontal": "unavailable",
            "primary_vertical": "unavailable",
        }

    in_zone = (
        plate["plate_x"].abs().le(zone_half_width)
        & plate["plate_z"].between(zone_bottom, zone_top)
    )
    mean_x = float(plate["plate_x"].mean())
    mean_z = float(plate["plate_z"].mean())
    hand = str(pitcher_hand or "").upper()
    if abs(mean_x) < 0.10:
        primary_horizontal = "balanced"
    elif hand == "L":
        primary_horizontal = "arm-side" if mean_x > 0 else "glove-side"
    else:
        primary_horizontal = "arm-side" if mean_x < 0 else "glove-side"

    if mean_z > 2.70:
        primary_vertical = "up"
    elif mean_z < 2.30:
        primary_vertical = "down"
    else:
        primary_vertical = "balanced"

    return {
        "sample_count": int(len(plate)),
        "zone_rate": float(in_zone.mean()),
        "chase_rate": float((~in_zone).mean()),
        "spike_risk_rate": float((plate["plate_z"] < 1.0).mean()),
        "primary_horizontal": primary_horizontal,
        "primary_vertical": primary_vertical,
        "zone_definition": {
            "plate_x_abs_max": float(zone_half_width),
            "plate_z_min": float(zone_bottom),
            "plate_z_max": float(zone_top),
        },
    }


def compare_context_distributions(
    context_a_samples: pd.DataFrame,
    context_b_samples: pd.DataFrame,
    pitcher_hand: str | None = None,
    columns: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build a dashboard-friendly side-by-side comparison payload."""
    selected_columns = list(columns) if columns is not None else DEFAULT_SUMMARY_COLUMNS
    summary_a = summarize_distribution(context_a_samples, selected_columns)
    summary_b = summarize_distribution(context_b_samples, selected_columns)
    common = sorted(set(summary_a) & set(summary_b))
    delta = {
        column: {
            "mean_delta": float(summary_b[column]["mean"] - summary_a[column]["mean"]),
            "std_delta": float(summary_b[column]["std"] - summary_a[column]["std"]),
        }
        for column in common
    }
    return {
        "context_a": {
            "summary": summary_a,
            "miss_tendency": derive_miss_tendency(context_a_samples, pitcher_hand=pitcher_hand),
        },
        "context_b": {
            "summary": summary_b,
            "miss_tendency": derive_miss_tendency(context_b_samples, pitcher_hand=pitcher_hand),
        },
        "delta": delta,
    }


def layer_status_from_report(report: dict[str, object]) -> dict[str, str]:
    """Map feature groups to validation status labels from a validation report."""
    statuses: dict[str, str] = {}
    for status in ("validated", "borderline", "diagnostic"):
        for layer in report.get(f"{status}_layers", []):
            feature_group = layer.get("feature_group")
            if feature_group:
                statuses[str(feature_group)] = status
    return statuses


def _model_or_fallback(
    suite: dict[str, GeneratorModel],
    requested_model: str,
) -> tuple[GeneratorModel, str, bool]:
    if requested_model in suite:
        return suite[requested_model], requested_model, False
    model, model_name, _ = select_conditional_model(suite)
    return model, model_name, True


def _evaluate_model_against_holdout(
    model: GeneratorModel,
    holdout: pd.DataFrame,
    n_samples: int,
    random_state: int,
    context_df: pd.DataFrame | None,
) -> dict[str, object]:
    samples = sample_generator(
        model,
        n=max(n_samples, len(holdout)),
        random_state=random_state,
        context_df=context_df,
    )
    metrics = classifier_two_sample_test(
        holdout,
        samples,
        model.feature_columns,
        random_state=random_state + 1000,
    )
    return metrics


def validate_conditional_layers(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    league_df: pd.DataFrame,
    pitcher_name: str,
    pitch_type: str,
    feature_groups: Iterable[str],
    n_samples: int = 200,
    random_state: int = 42,
) -> dict[str, object]:
    """Validate conditional generation against current game-drift baselines by layer."""
    comparisons = {
        "player_recent_weighted_game_drift_gaussian": {
            "requested_model": "player_recent_weighted_game_drift_gaussian",
            "conditional": False,
        },
        "player_recent_weighted_game_drift_copula": {
            "requested_model": "player_recent_weighted_game_drift_copula",
            "conditional": False,
        },
        "conditional_game_drift_copula": {
            "requested_model": "player_recent_weighted_game_drift_copula",
            "conditional": True,
        },
    }
    feature_group_results: dict[str, object] = {}
    for group_index, feature_group in enumerate(feature_groups):
        suite = fit_generator_suite(
            train,
            league_df,
            pitcher_name=pitcher_name,
            pitch_type=pitch_type,
            feature_group=feature_group,
            random_state=random_state,
        )
        model_results: dict[str, object] = {}
        for comparison_index, (comparison_name, comparison) in enumerate(comparisons.items()):
            model, fallback_model, used_fallback = _model_or_fallback(
                suite,
                str(comparison["requested_model"]),
            )
            metrics = _evaluate_model_against_holdout(
                model,
                holdout,
                n_samples=n_samples,
                random_state=random_state + group_index * 100 + comparison_index,
                context_df=holdout if comparison["conditional"] else None,
            )
            model_results[comparison_name] = {
                **metrics,
                "requested_model": comparison["requested_model"],
                "fallback_model": fallback_model,
                "used_fallback": used_fallback,
                "conditional": bool(comparison["conditional"]),
            }

        best_model = min(model_results, key=lambda name: float(model_results[name]["auc"]))
        feature_group_results[feature_group] = {
            "best_model": best_model,
            "best_auc": float(model_results[best_model]["auc"]),
            "model_results": model_results,
        }

    return {
        "pitcher_name": pitcher_name,
        "pitch_type": pitch_type,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "feature_group_results": feature_group_results,
    }
