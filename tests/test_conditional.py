from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_twin.candidates import CandidateThresholds, rank_pitcher_pitch_candidates
from pitcher_twin.conditional import (
    compare_context_distributions,
    derive_miss_tendency,
    layer_status_from_report,
    make_context_dataframe,
    sample_conditional_distribution,
    select_conditional_model,
    summarize_distribution,
    validate_conditional_layers,
)
from pitcher_twin.features import COUNT_BUCKET_CODES, clean_pitch_features
from pitcher_twin.models import fit_generator_suite
from pitcher_twin.validator import temporal_train_holdout


REAL_SAMPLE = Path(__file__).parent / "fixtures" / "real_statcast_sample.csv"


def _real_candidate_suite(feature_group: str = "command_representation"):
    df = clean_pitch_features(pd.read_csv(REAL_SAMPLE), pitch_types=None)
    ranking = rank_pitcher_pitch_candidates(
        df,
        thresholds=CandidateThresholds(min_pitches=20, min_holdout=5, min_games=1),
    )
    candidate = ranking.iloc[0].to_dict()
    subset = df[
        (df["pitcher"] == candidate["pitcher"])
        & (df["pitch_type"] == candidate["pitch_type"])
    ].copy()
    train, holdout = temporal_train_holdout(subset, train_fraction=0.7)
    suite = fit_generator_suite(
        train,
        df,
        pitcher_name=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        feature_group=feature_group,
    )
    return suite, holdout


def test_make_context_dataframe_encodes_ui_inputs() -> None:
    context = make_context_dataframe(
        inning=7,
        pitcher_game_pitch_count=88,
        balls=2,
        strikes=2,
        batter_hand="L",
        pitcher_score_diff=1,
        repeat=3,
    )

    assert context.shape[0] == 3
    assert context["count_bucket_code"].iloc[0] == COUNT_BUCKET_CODES["even"]
    assert context["batter_stand_code"].iloc[0] == 1.0
    assert context["pitcher_game_pitch_count"].iloc[0] == 88.0
    assert context["pitcher_score_diff"].iloc[0] == 1.0


def test_select_conditional_model_prefers_copula_then_game_drift() -> None:
    suite, _ = _real_candidate_suite()
    model, model_name, fallback = select_conditional_model(suite)

    assert model_name in {
        "player_recent_weighted_game_drift_copula",
        "player_recent_weighted_game_drift_gaussian",
        "player_context_weighted_gaussian",
        "player_recent_multivariate_gaussian",
        "player_multivariate_gaussian",
    }
    assert model.model_name == model_name
    assert fallback["selected_model"] == model_name


def test_sample_conditional_distribution_returns_finite_samples() -> None:
    suite, _ = _real_candidate_suite()
    context = make_context_dataframe(
        inning=5,
        pitcher_game_pitch_count=64,
        balls=1,
        strikes=2,
        batter_hand="R",
        pitcher_score_diff=-1,
        repeat=12,
    )

    samples, metadata = sample_conditional_distribution(suite, context, n=12, random_state=11)

    assert len(samples) == 12
    assert samples.columns.tolist() == suite[metadata["selected_model"]].feature_columns
    assert np.isfinite(samples.to_numpy(float)).all()
    assert metadata["requested_samples"] == 12


def test_summarize_distribution_reports_quantiles() -> None:
    samples = pd.DataFrame({"release_speed": [82.0, 84.0, 86.0], "plate_x": [-1.0, 0.0, 1.0]})
    summary = summarize_distribution(samples)

    assert summary["release_speed"]["mean"] == 84.0
    assert summary["release_speed"]["p50"] == 84.0
    assert summary["plate_x"]["p10"] < 0


def test_derive_miss_tendency_reports_zone_chase_and_spike() -> None:
    samples = pd.DataFrame({"plate_x": [-1.2, 0.0, 0.4, 1.4], "plate_z": [0.7, 2.4, 3.0, 4.1]})
    tendency = derive_miss_tendency(samples, pitcher_hand="R")

    assert tendency["sample_count"] == 4
    assert tendency["chase_rate"] > 0
    assert tendency["spike_risk_rate"] > 0
    assert tendency["primary_horizontal"] in {"arm-side", "glove-side", "balanced"}
    assert tendency["primary_vertical"] in {"up", "down", "balanced"}


def test_compare_context_distributions_returns_dashboard_payload() -> None:
    a = pd.DataFrame({"release_speed": [84.0, 85.0], "plate_x": [-0.2, 0.1], "plate_z": [2.0, 2.5]})
    b = pd.DataFrame({"release_speed": [82.0, 83.0], "plate_x": [0.4, 0.6], "plate_z": [1.2, 1.4]})
    payload = compare_context_distributions(a, b, pitcher_hand="L")

    assert payload["context_a"]["summary"]["release_speed"]["mean"] == 84.5
    assert payload["context_b"]["summary"]["release_speed"]["mean"] == 82.5
    assert payload["delta"]["release_speed"]["mean_delta"] == -2.0
    assert "miss_tendency" in payload["context_a"]


def test_layer_status_from_report_uses_validation_buckets() -> None:
    report = {
        "validated_layers": [{"feature_group": "command_representation"}],
        "borderline_layers": [{"feature_group": "trajectory_only"}],
        "diagnostic_layers": [{"feature_group": "physics_core"}],
    }

    status = layer_status_from_report(report)

    assert status["command_representation"] == "validated"
    assert status["trajectory_only"] == "borderline"
    assert status["physics_core"] == "diagnostic"


def test_conditional_layer_validation_reports_layers_and_models() -> None:
    df = clean_pitch_features(pd.read_csv(REAL_SAMPLE), pitch_types=None)
    ranking = rank_pitcher_pitch_candidates(
        df,
        thresholds=CandidateThresholds(min_pitches=20, min_holdout=5, min_games=1),
    )
    candidate = ranking.iloc[0].to_dict()
    subset = df[
        (df["pitcher"] == candidate["pitcher"])
        & (df["pitch_type"] == candidate["pitch_type"])
    ]
    train, holdout = temporal_train_holdout(subset, train_fraction=0.7)

    report = validate_conditional_layers(
        train,
        holdout,
        df,
        pitcher_name=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        feature_groups=["command_representation", "movement_only"],
        n_samples=20,
        random_state=13,
    )

    assert set(report["feature_group_results"]) == {"command_representation", "movement_only"}
    for row in report["feature_group_results"].values():
        assert "player_recent_weighted_game_drift_gaussian" in row["model_results"]
        assert "player_recent_weighted_game_drift_copula" in row["model_results"]
        assert "conditional_game_drift_copula" in row["model_results"]
        for metrics in row["model_results"].values():
            assert "auc" in metrics
            assert "fallback_model" in metrics
