from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_twin.features import (
    FEATURE_GROUPS,
    RECENT_STATE_FEATURES,
    add_count_bucket,
    add_pitcher_game_pitch_count,
    add_pitcher_score_diff,
    add_real_context_features,
    add_recent_pitcher_state_features,
    add_spin_axis_components,
    build_feature_matrix,
    clean_pitch_features,
    feature_availability_report,
)


REAL_SAMPLE = Path(__file__).parent / "fixtures" / "real_statcast_sample.csv"


def test_spin_axis_components_are_derived_from_real_spin_axis() -> None:
    df = pd.read_csv(REAL_SAMPLE, nrows=20)
    result = add_spin_axis_components(df)
    assert {"spin_axis_cos", "spin_axis_sin"}.issubset(result.columns)
    assert result["spin_axis_cos"].dropna().between(-1, 1).all()
    assert result["spin_axis_sin"].dropna().between(-1, 1).all()
    assert result["spin_axis_cos"].isna().sum() == df["spin_axis"].isna().sum()


def test_count_bucket_uses_real_balls_and_strikes() -> None:
    df = pd.DataFrame({"balls": [0, 3, 2, 1, 1], "strikes": [0, 2, 0, 2, 1]})
    result = add_count_bucket(df)
    assert list(result["count_bucket"]) == ["first_pitch", "full", "behind", "ahead", "even"]


def test_real_context_features_do_not_fabricate_missing_columns() -> None:
    df = pd.read_csv(REAL_SAMPLE, nrows=50)
    result = add_real_context_features(df)
    assert {
        "count_bucket",
        "batter_stand_code",
        "score_diff",
        "pitcher_game_pitch_count",
        "pitcher_score_diff",
    }.issubset(result.columns)
    if "pitcher_days_since_prev_game" in df.columns:
        assert "days_rest" in result.columns
    assert "fake_weather" not in result.columns


def test_pitcher_game_pitch_count_is_cumulative_within_pitcher_game() -> None:
    df = pd.DataFrame(
        {
            "game_date": ["2026-04-01"] * 6,
            "game_pk": [1] * 6,
            "pitcher": [10, 10, 10, 10, 10, 11],
            "at_bat_number": [1, 1, 2, 2, 2, 1],
            "pitch_number": [1, 2, 1, 2, 3, 1],
        }
    )
    result = add_pitcher_game_pitch_count(df)
    assert result["pitcher_game_pitch_count"].tolist() == [1, 2, 3, 4, 5, 1]


def test_pitcher_score_diff_is_positive_when_pitchers_team_leads() -> None:
    df = pd.DataFrame(
        {
            "pitcher_team": ["HOME", "AWAY", None],
            "home_team": ["HOME", "HOME", "HOME"],
            "away_team": ["AWAY", "AWAY", "AWAY"],
            "home_score": [5, 5, 4],
            "away_score": [3, 8, 2],
            "bat_score": [3, 5, 2],
            "fld_score": [5, 8, 4],
        }
    )
    result = add_pitcher_score_diff(df)
    assert result["pitcher_score_diff"].tolist() == [2.0, 3.0, 2.0]


def test_recent_pitcher_state_features_use_prior_pitches_only() -> None:
    df = pd.DataFrame(
        {
            "game_date": ["2026-04-01"] * 6,
            "game_pk": [1] * 6,
            "pitcher": [10] * 6,
            "at_bat_number": [1, 1, 2, 2, 3, 3],
            "pitch_number": [1, 2, 1, 2, 1, 2],
            "pitch_type": ["FF", "SL", "FF", "FF", "SL", "FF"],
            "release_speed": [94.0, 95.0, 96.0, 97.0, 98.0, 99.0],
            "release_spin_rate": [2200, 2210, 2220, 2230, 2240, 2250],
            "pfx_x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "pfx_z": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "plate_x": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
            "plate_z": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        }
    )

    result = add_recent_pitcher_state_features(df)

    assert set(RECENT_STATE_FEATURES).issubset(result.columns)
    assert pd.isna(result.loc[0, "previous_release_speed"])
    assert result.loc[1, "previous_release_speed"] == 94.0
    assert result.loc[2, "previous_plate_x"] == -0.2
    assert result.loc[3, "rolling_5_release_speed_mean"] == 95.0
    assert result.loc[5, "rolling_5_pfx_z_mean"] == np.mean([1.1, 1.2, 1.3, 1.4, 1.5])
    assert result.loc[2, "previous_pitch_type_code"] == result.loc[1, "pitch_type_code"]


def test_build_feature_matrix_returns_declared_group_columns() -> None:
    df = clean_pitch_features(pd.read_csv(REAL_SAMPLE), pitch_types=None)
    matrix = build_feature_matrix(df, feature_group="physics_core")
    assert list(matrix.columns) == FEATURE_GROUPS["physics_core"]
    assert np.isfinite(matrix.to_numpy(float)).all()


def test_feature_availability_report_counts_retained_rows() -> None:
    df = clean_pitch_features(pd.read_csv(REAL_SAMPLE), pitch_types=None)
    report = feature_availability_report(df)
    assert "physics_core" in report
    assert report["physics_core"]["rows_retained"] > 0
    assert 0 <= report["physics_core"]["feature_completeness"] <= 1
