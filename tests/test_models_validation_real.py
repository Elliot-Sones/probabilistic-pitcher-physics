from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from pitcher_twin.candidates import CandidateThresholds, rank_pitcher_pitch_candidates
from pitcher_twin.features import clean_pitch_features
from pitcher_twin.models import fit_generator_suite, sample_generator
from pitcher_twin.validator import classifier_two_sample_test, temporal_train_holdout


REAL_SAMPLE = Path(__file__).parent / "fixtures" / "real_statcast_sample.csv"


def _candidate_data():
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
    return df, subset, candidate


def test_temporal_train_holdout_preserves_time_order() -> None:
    _, subset, _ = _candidate_data()
    train, holdout = temporal_train_holdout(subset, train_fraction=0.7)
    assert len(train) > 0
    assert len(holdout) > 0
    assert train["game_date"].max() <= holdout["game_date"].min()


def test_fit_generator_suite_samples_real_feature_shapes() -> None:
    league_df, subset, candidate = _candidate_data()
    train, _ = temporal_train_holdout(subset, train_fraction=0.7)
    suite = fit_generator_suite(
        train,
        league_df,
        pitcher_name=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        feature_group="physics_core",
        random_state=4,
    )
    assert {"random_independent_noise", "league_same_pitch_empirical", "player_empirical_bootstrap", "player_multivariate_gaussian", "player_recent_empirical_bootstrap", "player_recent_multivariate_gaussian", "player_gmm"}.issubset(
        suite
    )
    for model in suite.values():
        samples = sample_generator(model, n=12, random_state=5)
        assert samples.shape == (12, len(model.feature_columns))
        assert np.isfinite(samples.to_numpy(float)).all()


def test_recent_generators_use_latest_training_window() -> None:
    league_df, subset, candidate = _candidate_data()
    train, _ = temporal_train_holdout(subset, train_fraction=0.7)
    suite = fit_generator_suite(
        train,
        league_df,
        pitcher_name=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        feature_group="physics_core",
        random_state=4,
    )

    full_rows = suite["player_empirical_bootstrap"].payload["source_row_count"]
    recent_rows = suite["player_recent_empirical_bootstrap"].payload["source_row_count"]
    assert 10 <= recent_rows < full_rows


def test_game_drift_generator_learns_recent_game_shift() -> None:
    feature_columns = [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
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
    rows = []
    for game_index, shift in enumerate([0.0, 2.0, 4.0], start=1):
        for pitch_number in range(1, 16):
            row = {
                "pitcher": 1,
                "pitcher_name": "Pitcher, Test",
                "pitch_type": "FF",
                "game_pk": game_index,
                "game_date": f"2026-04-0{game_index}",
                "balls": pitch_number % 4,
                "strikes": pitch_number % 3,
                "inning": 1 + pitch_number // 5,
                "pitch_number": pitch_number,
                "stand": "R" if pitch_number % 2 else "L",
                "bat_score": 1,
                "fld_score": 0,
            }
            for column_index, column in enumerate(feature_columns):
                row[column] = 10.0 + column_index + shift + pitch_number * 0.01
            rows.append(row)
    df = clean_pitch_features(pd.DataFrame(rows), pitch_types=None)
    suite = fit_generator_suite(
        df,
        df,
        pitcher_name="Test Pitcher",
        pitch_type="FF",
        feature_group="physics_core",
        random_state=4,
    )

    model = suite["player_recent_weighted_game_drift_gaussian"]
    samples = sample_generator(model, n=30, random_state=5)

    assert model.payload["game_count"] == 3
    assert model.payload["predicted_game_mean"][0] > model.payload["baseline_mean"][0]
    assert abs(samples["release_speed"].mean() - model.payload["predicted_game_mean"][0]) < 1.0


def test_game_drift_copula_generator_preserves_empirical_residual_margins() -> None:
    feature_columns = [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
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
    rows = []
    for game_index, shift in enumerate([0.0, 1.0, 2.0, 3.0], start=1):
        for pitch_number in range(1, 21):
            row = {
                "pitcher": 1,
                "pitcher_name": "Pitcher, Test",
                "pitch_type": "FF",
                "game_pk": game_index,
                "game_date": f"2026-04-0{game_index}",
                "balls": pitch_number % 4,
                "strikes": pitch_number % 3,
                "inning": 1 + pitch_number // 5,
                "pitch_number": pitch_number,
                "stand": "R" if pitch_number % 2 else "L",
                "bat_score": 1,
                "fld_score": 0,
            }
            skewed_residual = np.log1p(pitch_number) * 0.12
            for column_index, column in enumerate(feature_columns):
                row[column] = 20.0 + column_index + shift + skewed_residual
            rows.append(row)
    df = clean_pitch_features(pd.DataFrame(rows), pitch_types=None)
    suite = fit_generator_suite(
        df,
        df,
        pitcher_name="Test Pitcher",
        pitch_type="FF",
        feature_group="physics_core",
        random_state=4,
    )

    model = suite["player_recent_weighted_game_drift_copula"]
    samples = sample_generator(model, n=40, random_state=5)

    assert model.payload["copula_kind"] == "gaussian_empirical_margins"
    assert model.payload["residual_margins"].shape[1] == len(model.feature_columns)
    assert model.payload["copula_corr"].shape == (
        len(model.feature_columns),
        len(model.feature_columns),
    )
    assert samples.shape == (40, len(model.feature_columns))
    assert np.isfinite(samples.to_numpy(float)).all()


def test_game_drift_copula_handles_constant_residual_columns_without_runtime_warning() -> None:
    rows = []
    for game_index, shift in enumerate([0.0, 1.0, 2.0], start=1):
        for pitch_number in range(1, 16):
            rows.append(
                {
                    "pitcher": 1,
                    "pitcher_name": "Pitcher, Test",
                    "pitch_type": "FF",
                    "game_pk": game_index,
                    "game_date": f"2026-04-0{game_index}",
                    "release_speed": 90 + shift,
                    "release_spin_rate": 2200 + shift,
                    "spin_axis": 180,
                    "release_pos_x": 1.0 + shift,
                    "release_pos_y": 54.0,
                    "release_pos_z": 6.0 + shift,
                    "release_extension": 6.5,
                    "pfx_x": 0.1 + shift,
                    "pfx_z": 1.0 + shift,
                    "plate_x": 0.0 + shift,
                    "plate_z": 2.5 + shift,
                    "vx0": 1.0 + shift,
                    "vy0": -130.0 - shift,
                    "vz0": -4.0,
                    "ax": 5.0 + shift,
                    "ay": 25.0,
                    "az": -20.0,
                    "balls": 0,
                    "strikes": 0,
                    "inning": 1,
                    "pitch_number": pitch_number,
                    "stand": "R",
                    "bat_score": 0,
                    "fld_score": 0,
                }
            )
    df = clean_pitch_features(pd.DataFrame(rows), pitch_types=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suite = fit_generator_suite(
            df,
            df,
            pitcher_name="Test Pitcher",
            pitch_type="FF",
            feature_group="physics_core",
            random_state=4,
        )

    assert not [warning for warning in caught if "invalid value encountered" in str(warning.message)]
    samples = sample_generator(
        suite["player_recent_weighted_game_drift_copula"],
        n=10,
        random_state=5,
    )
    assert np.isfinite(samples.to_numpy(float)).all()


def test_recent_game_window_generator_uses_whole_appearances() -> None:
    rows = []
    for game_index in range(1, 5):
        for pitch_number in range(1, 8):
            rows.append(
                {
                    "pitcher": 1,
                    "pitcher_name": "Pitcher, Test",
                    "pitch_type": "FF",
                    "game_pk": game_index,
                    "game_date": f"2026-04-0{game_index}",
                    "release_speed": 90 + game_index,
                    "release_spin_rate": 2200 + game_index,
                    "spin_axis": 180,
                    "release_pos_x": 1.0,
                    "release_pos_y": 54.0,
                    "release_pos_z": 6.0,
                    "release_extension": 6.5,
                    "pfx_x": 0.1,
                    "pfx_z": 1.0,
                    "plate_x": 0.0,
                    "plate_z": 2.5,
                    "vx0": 1.0,
                    "vy0": -130.0,
                    "vz0": -4.0,
                    "ax": 5.0,
                    "ay": 25.0,
                    "az": -20.0,
                    "balls": 0,
                    "strikes": 0,
                    "inning": 1,
                    "pitch_number": pitch_number,
                    "stand": "R",
                    "bat_score": 0,
                    "fld_score": 0,
                }
            )
    df = clean_pitch_features(pd.DataFrame(rows), pitch_types=None)
    suite = fit_generator_suite(
        df,
        df,
        pitcher_name="Test Pitcher",
        pitch_type="FF",
        feature_group="physics_core",
        random_state=4,
    )

    model = suite["player_recent_game_window_empirical"]
    samples = sample_generator(model, n=20, random_state=5)

    assert model.payload["game_window"] == 4
    assert model.payload["game_count"] == 4
    assert samples["release_speed"].between(91, 94).all()


def test_classifier_two_sample_test_returns_auc_and_leakage_features() -> None:
    league_df, subset, candidate = _candidate_data()
    train, holdout = temporal_train_holdout(subset, train_fraction=0.7)
    suite = fit_generator_suite(
        train,
        league_df,
        pitcher_name=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        feature_group="physics_core",
        random_state=6,
    )
    generated = sample_generator(suite["player_multivariate_gaussian"], n=len(holdout), random_state=7)
    result = classifier_two_sample_test(holdout, generated, suite["player_multivariate_gaussian"].feature_columns)
    assert 0.5 <= result["auc"] <= 1.0
    assert 0.0 <= result["raw_auc"] <= 1.0
    assert result["n_real"] == len(holdout)
    assert result["top_leakage_features"]
    assert result["classifier_split"]["train_rows"] > 0
    assert result["classifier_split"]["test_rows"] > 0
    assert result["classifier_split"]["train_rows"] + result["classifier_split"]["test_rows"] == (
        result["n_real"] + result["n_simulated"]
    )


def test_classifier_two_sample_test_reports_held_out_classifier_auc() -> None:
    real = pd.DataFrame(
        {
            "release_speed": np.r_[np.zeros(20), np.ones(20) * 0.2],
            "plate_x": np.r_[np.zeros(20), np.ones(20) * 0.2],
        }
    )
    fake = pd.DataFrame(
        {
            "release_speed": np.r_[np.ones(20) * 6.0, np.ones(20) * 6.2],
            "plate_x": np.r_[np.ones(20) * -6.0, np.ones(20) * -6.2],
        }
    )
    result = classifier_two_sample_test(real, fake, ["release_speed", "plate_x"], random_state=11)
    assert result["classifier_split"]["strategy"] == "stratified_holdout"
    assert result["classifier_split"]["train_rows"] == 56
    assert result["classifier_split"]["test_rows"] == 24
    assert result["auc"] > 0.95


def test_public_residual_copula_helpers_sample_finite_residuals() -> None:
    from pitcher_twin.models import (
        fit_residual_gaussian_copula,
        sample_residual_gaussian_copula,
    )

    residuals = np.array(
        [
            [-1.0, -0.5, 0.2],
            [-0.2, -0.1, 0.0],
            [0.0, 0.1, 0.1],
            [0.5, 0.4, -0.2],
            [1.2, 0.8, -0.1],
        ]
        * 5,
        dtype=float,
    )

    payload = fit_residual_gaussian_copula(residuals)
    samples = sample_residual_gaussian_copula(payload, n=20, random_state=7)

    assert payload["copula_kind"] == "gaussian_empirical_margins"
    assert samples.shape == (20, 3)
    assert np.isfinite(samples).all()


def test_public_residual_copula_returns_none_for_tiny_training_sets() -> None:
    from pitcher_twin.models import fit_residual_gaussian_copula

    residuals = np.ones((3, 2), dtype=float)
    assert fit_residual_gaussian_copula(residuals) is None
