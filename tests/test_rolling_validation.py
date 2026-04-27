from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from pitcher_twin.rolling_validation import (
    build_pitch_type_failure_explanations,
    evaluate_rolling_temporal_board,
    explain_detection_features,
    rolling_game_splits,
    score_rolling_validation_goals,
    write_rolling_board_outputs,
)


def _game_frame(game_count: int = 14, pitches_per_game: int = 5) -> pd.DataFrame:
    rows = []
    for game_index in range(game_count):
        for pitch_index in range(pitches_per_game):
            rows.append(
                {
                    "game_date": f"2026-04-{game_index + 1:02d}",
                    "game_pk": 1000 + game_index,
                    "pitcher": 10,
                    "pitcher_name": "Pitcher, Test",
                    "pitch_type": "FF",
                    "at_bat_number": pitch_index + 1,
                    "pitch_number": 1,
                    "release_speed": 94.0 + game_index * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _fake_tournament_report(*args, **kwargs) -> dict[str, object]:
    train, holdout = args[:2]
    fold_offset = int(holdout["game_pk"].min() - 1000)
    physics_auc = 0.56 + fold_offset * 0.01
    layer_results = {
        "command_representation": {
            "fake_model": {
                "mean_auc": 0.55,
                "pass_rate": 1.0,
                "std_auc": 0.01,
                "top_leakage_features": [
                    {"feature": "plate_x", "importance": 0.2},
                    {"feature": "plate_z", "importance": 0.1},
                ],
            }
        },
        "physics_core": {
            "fake_model": {
                "mean_auc": physics_auc,
                "pass_rate": 0.50,
                "std_auc": 0.03,
                "top_leakage_features": [
                    {"feature": "spin_axis_cos", "importance": 0.5},
                    {"feature": "release_extension", "importance": 0.4},
                    {"feature": "az", "importance": 0.3},
                ],
            }
        },
    }
    return {
        "pitcher_name": kwargs["pitcher_name"],
        "pitch_type": kwargs["pitch_type"],
        "n_train": len(train),
        "n_holdout": len(holdout),
        "target_auc": kwargs.get("target_auc", 0.60),
        "target_pass_rate": kwargs.get("target_pass_rate", 0.80),
        "repeat_count": kwargs["repeats"],
        "best_by_layer": {
            "command_representation": "fake_model",
            "physics_core": "fake_model",
        },
        "best_physics_core_model": "fake_model",
        "candidate_default": False,
        "layer_results": layer_results,
    }


def test_rolling_game_splits_train_cumulative_and_test_future_games() -> None:
    frame = _game_frame(game_count=14, pitches_per_game=2)

    splits = rolling_game_splits(
        frame,
        initial_train_games=4,
        test_games=2,
        step_games=2,
    )

    assert [(split.train_game_range, split.test_game_range) for split in splits] == [
        ("1-4", "5-6"),
        ("1-6", "7-8"),
        ("1-8", "9-10"),
        ("1-10", "11-12"),
        ("1-12", "13-14"),
    ]
    assert splits[0].train_game_pks == [1000, 1001, 1002, 1003]
    assert splits[0].test_game_pks == [1004, 1005]


def test_failure_explainer_labels_classifier_detection_features() -> None:
    release_explanation = explain_detection_features(
        ["spin_axis_cos", "release_extension", "release_spin_rate"]
    )
    movement_explanation = explain_detection_features(["az", "ax", "pfx_z"])

    assert release_explanation["primary_mode"] == "release/spin signature"
    assert "spin_axis_cos + release_extension + release_spin_rate" == release_explanation["signal"]
    assert movement_explanation["primary_mode"] == "acceleration/movement consistency"


def test_build_pitch_type_failure_explanations_reports_failed_layers() -> None:
    report = _fake_tournament_report(
        _game_frame().head(10),
        _game_frame().tail(5),
        None,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        repeats=2,
        target_auc=0.60,
        target_pass_rate=0.80,
    )

    failures = build_pitch_type_failure_explanations(
        report,
        pitch_type="FF",
        fold_index=1,
        target_auc=0.60,
        target_pass_rate=0.80,
    )

    assert len(failures) == 1
    assert failures[0]["pitch_type"] == "FF"
    assert failures[0]["layer"] == "physics_core"
    assert failures[0]["primary_mode"] == "release/spin signature"
    assert failures[0]["classifier_signal"] == "spin_axis_cos + release_extension + az"


def test_evaluate_rolling_temporal_board_uses_injected_tournament_evaluator() -> None:
    frame = _game_frame(game_count=8, pitches_per_game=4)

    board = evaluate_rolling_temporal_board(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        initial_train_games=4,
        test_games=2,
        step_games=2,
        repeats=2,
        random_state=7,
        tournament_evaluator=_fake_tournament_report,
    )

    assert board["pitcher_name"] == "Pitcher, Test"
    assert board["pitch_type"] == "FF"
    assert board["fold_count"] == 2
    assert board["folds"][0]["train_game_range"] == "1-4"
    assert board["folds"][0]["test_game_range"] == "5-6"
    assert board["folds"][0]["physics_core_mean_auc"] > 0.0
    assert board["consistency"]["physics_core_mean_auc_mean"] > 0.0
    assert board["primary_scoreboard"]["status"] == "rolling_candidate"
    assert board["primary_scoreboard"]["passed"] is False
    assert board["failure_explanations"]


def test_score_rolling_validation_goals_marks_current_baseline_diagnostic() -> None:
    scoreboard = score_rolling_validation_goals(
        {
            "physics_core_mean_auc_mean": 0.6959410364883786,
            "physics_core_mean_auc_min": 0.5933333333333333,
            "physics_core_mean_auc_max": 0.9285714285714286,
            "physics_core_target_hit_rate": 0.10,
        }
    )

    checks_by_metric = {check["metric"]: check for check in scoreboard["checks"]}
    assert scoreboard["status"] == "rolling_diagnostic"
    assert scoreboard["passed"] is False
    assert scoreboard["cleared_count"] == 0
    assert checks_by_metric["Mean rolling physics-core AUC"]["goal"] == "<= 0.620"
    assert checks_by_metric["Mean rolling physics-core AUC"]["passed"] is False
    assert checks_by_metric["Target hit rate"]["goal"] == ">= 0.40"
    assert checks_by_metric["Target hit rate"]["passed"] is False
    assert checks_by_metric["Worst fold physics-core AUC"]["goal"] == "< 0.800"
    assert checks_by_metric["Worst fold physics-core AUC"]["passed"] is False


def test_score_rolling_validation_goals_marks_cleared_gate_validated() -> None:
    scoreboard = score_rolling_validation_goals(
        {
            "physics_core_mean_auc_mean": 0.619,
            "physics_core_mean_auc_min": 0.552,
            "physics_core_mean_auc_max": 0.799,
            "physics_core_target_hit_rate": 0.40,
        }
    )

    assert scoreboard["status"] == "rolling_validated"
    assert scoreboard["passed"] is True
    assert scoreboard["cleared_count"] == 3


def test_write_rolling_board_outputs_writes_json_markdown_and_csv(tmp_path: Path) -> None:
    frame = _game_frame(game_count=8, pitches_per_game=4)
    board = evaluate_rolling_temporal_board(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        initial_train_games=4,
        test_games=2,
        step_games=2,
        repeats=2,
        tournament_evaluator=_fake_tournament_report,
    )

    outputs = write_rolling_board_outputs(board, tmp_path)

    assert Path(outputs["json"]).exists()
    markdown = Path(outputs["markdown"]).read_text()
    assert markdown.startswith("# Rolling Temporal Validation Board")
    assert "## Primary Rolling Scoreboard" in markdown
    assert "<= 0.620" in markdown
    assert ">= 0.40" in markdown
    assert "< 0.800" in markdown
    assert Path(outputs["failure_csv"]).exists()


def test_rolling_temporal_board_script_exposes_main() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_rolling_temporal_board.py"
    spec = importlib.util.spec_from_file_location("run_rolling_temporal_board", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "main")
    assert hasattr(module, "run")
