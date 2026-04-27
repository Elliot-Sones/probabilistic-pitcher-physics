from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_twin.validation_board import (
    CandidateCriteria,
    candidate_pitcher_pitches,
    render_scorecard_markdown,
    rolling_game_windows,
    summarize_tournament_report,
)


def _synthetic_board_frame() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    rows = []
    configs = [
        (101, "Ace, Alpha", "FF", 8, 8),
        (202, "Starter, Beta", "SL", 6, 7),
        (303, "Short, Gamma", "FF", 3, 5),
    ]
    for pitcher, pitcher_name, pitch_type, game_count, pitches_per_game in configs:
        for game_index in range(game_count):
            for pitch_index in range(pitches_per_game):
                base_speed = 96.0 if pitch_type == "FF" else 84.0
                rows.append(
                    {
                        "pitcher": pitcher,
                        "pitcher_name": pitcher_name,
                        "player_name": pitcher_name,
                        "pitch_type": pitch_type,
                        "game_pk": 1000 + pitcher + game_index,
                        "game_date": f"2026-04-{game_index + 1:02d}",
                        "pitcher_game_pitch_count": float(pitch_index + 1),
                        "release_speed": base_speed + rng.normal(0, 0.5),
                        "release_spin_rate": 2300.0 + rng.normal(0, 40.0),
                        "spin_axis_cos": -0.9 + rng.normal(0, 0.02),
                        "spin_axis_sin": 0.4 + rng.normal(0, 0.02),
                        "release_pos_x": 1.4 + rng.normal(0, 0.04),
                        "release_pos_y": 54.0 + rng.normal(0, 0.08),
                        "release_pos_z": 6.0 + rng.normal(0, 0.05),
                        "release_extension": 6.4 + rng.normal(0, 0.06),
                        "pfx_x": -1.1 + rng.normal(0, 0.05),
                        "pfx_z": 1.2 + rng.normal(0, 0.05),
                        "plate_x": rng.normal(0, 0.3),
                        "plate_z": 2.5 + rng.normal(0, 0.3),
                        "vx0": -5.0 + rng.normal(0, 0.1),
                        "vy0": -138.0 + rng.normal(0, 0.2),
                        "vz0": -4.0 + rng.normal(0, 0.1),
                        "ax": -12.0 + rng.normal(0, 0.2),
                        "ay": 25.0 + rng.normal(0, 0.2),
                        "az": -18.0 + rng.normal(0, 0.2),
                    }
                )
    return pd.DataFrame(rows)


def _fake_tournament_report(
    *,
    pitcher_name: str = "Ace, Alpha",
    pitch_type: str = "FF",
    physics_auc: float = 0.572,
    physics_pass_rate: float = 0.79,
) -> dict[str, object]:
    model_names = ["factorized_trend_state_anchored", "context_neighbor_residual"]
    layer_results = {}
    layer_values = {
        "command_representation": (0.530, 1.00),
        "movement_only": (0.541, 1.00),
        "release_only": (0.579, 0.83),
        "trajectory_only": (0.543, 1.00),
        "physics_core": (physics_auc, physics_pass_rate),
    }
    for layer, (auc, pass_rate) in layer_values.items():
        layer_results[layer] = {
            "factorized_trend_state_anchored": {
                "mean_auc": auc,
                "std_auc": 0.01,
                "min_auc": auc - 0.01,
                "max_auc": auc + 0.01,
                "pass_rate": pass_rate,
                "top_leakage_features": [{"feature": "release_pos_z", "importance": 0.8}],
            },
            "context_neighbor_residual": {
                "mean_auc": auc + 0.05,
                "std_auc": 0.02,
                "min_auc": auc + 0.03,
                "max_auc": auc + 0.07,
                "pass_rate": max(0.0, pass_rate - 0.3),
                "top_leakage_features": [{"feature": "release_speed", "importance": 0.5}],
            },
        }
    return {
        "pitcher_name": pitcher_name,
        "pitch_type": pitch_type,
        "n_train": 584,
        "n_holdout": 251,
        "repeat_count": 3,
        "sample_count": 260,
        "model_names": model_names,
        "target_auc": 0.60,
        "target_pass_rate": 0.80,
        "candidate_default": False,
        "best_by_layer": {
            layer: "factorized_trend_state_anchored" for layer in layer_values
        },
        "best_physics_core_model": "factorized_trend_state_anchored",
        "layer_results": layer_results,
    }


def test_candidate_pitcher_pitches_filters_temporal_quality_and_ranks_volume() -> None:
    frame = _synthetic_board_frame()

    candidates = candidate_pitcher_pitches(
        frame,
        CandidateCriteria(min_pitches=30, min_games=5, min_holdout=10, top=2),
    )

    assert candidates[0]["pitcher_name"] == "Ace, Alpha"
    assert candidates[0]["pitch_type"] == "FF"
    assert candidates[0]["pitch_count"] == 64
    assert candidates[0]["game_count"] == 8
    assert candidates[0]["holdout_count"] >= 10
    assert candidates[0]["physics_core_complete_count"] == 64
    assert [row["pitcher_name"] for row in candidates] == ["Ace, Alpha", "Starter, Beta"]


def test_rolling_game_windows_use_only_future_holdout_games() -> None:
    frame = _synthetic_board_frame()
    subset = frame[(frame["pitcher"] == 101) & (frame["pitch_type"] == "FF")]

    windows = rolling_game_windows(
        subset,
        min_train_games=4,
        holdout_games=2,
        max_windows=2,
    )

    assert len(windows) == 2
    for window in windows:
        assert int(window.train["game_pk"].max()) < int(window.holdout["game_pk"].min())
        assert window.train_game_count >= 4
        assert window.holdout_game_count == 2
        assert window.holdout_row_count == 16


def test_summarize_tournament_report_marks_candidate_when_pass_rate_is_short() -> None:
    report = _fake_tournament_report(physics_auc=0.572, physics_pass_rate=0.79)
    candidate = {
        "pitcher": 101,
        "pitcher_name": "Ace, Alpha",
        "pitch_type": "FF",
        "pitch_count": 64,
        "game_count": 8,
        "holdout_count": 20,
    }

    summary = summarize_tournament_report(report, candidate=candidate)

    assert summary["artifact_status"] == "physics_core_candidate"
    assert summary["best_physics_core_model"] == "factorized_trend_state_anchored"
    assert summary["physics_core_mean_auc"] == 0.572
    assert summary["physics_core_pass_rate"] == 0.79
    assert summary["layer_statuses"]["command_representation"]["status"] == "validated"
    assert summary["layer_statuses"]["physics_core"]["status"] == "candidate"
    assert summary["model_route"]["route_status"] == "candidate"
    assert summary["model_route"]["recommended_physics_model"] == (
        "factorized_trend_state_anchored"
    )
    assert summary["top_leakage_features"][0]["feature"] == "release_pos_z"


def test_summarize_tournament_report_marks_validated_only_when_auc_and_pass_rate_clear() -> None:
    report = _fake_tournament_report(physics_auc=0.558, physics_pass_rate=0.91)

    summary = summarize_tournament_report(report, candidate=None)

    assert summary["artifact_status"] == "validated_temporal_success"
    assert summary["layer_statuses"]["physics_core"]["status"] == "validated"


def test_render_scorecard_markdown_explains_score_and_rolling_windows() -> None:
    summary = summarize_tournament_report(
        _fake_tournament_report(physics_auc=0.572, physics_pass_rate=0.79),
        candidate={
            "pitcher": 101,
            "pitcher_name": "Ace, Alpha",
            "pitch_type": "FF",
            "pitch_count": 64,
            "game_count": 8,
            "holdout_count": 20,
        },
    )

    markdown = render_scorecard_markdown(
        summary,
        rolling_rows=[
            {
                "window_index": 1,
                "train_games": 4,
                "holdout_games": 2,
                "best_physics_core_model": "factorized_trend_state_anchored",
                "physics_core_mean_auc": 0.581,
                "physics_core_pass_rate": 1.0,
                "artifact_status": "validated_temporal_success",
            }
        ],
    )

    assert "# Pitcher Twin Scorecard: Ace, Alpha FF" in markdown
    assert "C2ST AUC" in markdown
    assert "## Model Route" in markdown
    assert "Recommended physics model" in markdown
    assert "physics_core_candidate" in markdown
    expected_row = (
        "| 1 | 4 | 2 | factorized_trend_state_anchored | "
        "0.581 | 1.00 | validated_temporal_success |"
    )
    assert expected_row in markdown


def test_validation_board_script_writes_leaderboard_and_scorecard_with_stub_tournament(
    tmp_path,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_validation_board.py"
    spec = importlib.util.spec_from_file_location("run_validation_board", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def stub_tournament(train, holdout, league_df, **kwargs):
        assert len(train) > 0
        assert len(holdout) > 0
        return _fake_tournament_report(
            pitcher_name=kwargs["pitcher_name"],
            pitch_type=kwargs["pitch_type"],
            physics_auc=0.572,
            physics_pass_rate=0.79,
        )

    outputs = module.run(
        data_path=None,
        output_dir=tmp_path,
        raw_df=_synthetic_board_frame(),
        top=1,
        min_pitches=30,
        min_games=5,
        min_holdout=10,
        repeats=1,
        samples=40,
        rolling=True,
        rolling_repeats=1,
        max_rolling_windows=1,
        tournament_fn=stub_tournament,
    )

    leaderboard = pd.read_csv(tmp_path / "leaderboard.csv")
    assert outputs["leaderboard_csv"] == str(tmp_path / "leaderboard.csv")
    assert leaderboard.loc[0, "pitcher_name"] == "Ace, Alpha"
    assert leaderboard.loc[0, "artifact_status"] == "physics_core_candidate"
    assert (tmp_path / "validation_board.md").exists()
    assert (tmp_path / "scorecards" / "ace_alpha_ff.md").exists()
    assert (tmp_path / "rolling_windows.csv").exists()
