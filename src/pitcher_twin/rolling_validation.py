"""Rolling temporal validation and classifier failure explanations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_twin.tournament import evaluate_model_tournament


ROLLING_SCOREBOARD_GOALS = {
    "mean_physics_core_auc_max": 0.620,
    "target_hit_rate_min": 0.40,
    "worst_fold_physics_core_auc_max_exclusive": 0.800,
}


@dataclass(frozen=True)
class RollingGameSplit:
    fold_index: int
    train_game_pks: list[int]
    test_game_pks: list[int]
    train_game_range: str
    test_game_range: str


def rolling_game_splits(
    frame: pd.DataFrame,
    *,
    initial_train_games: int = 10,
    test_games: int = 2,
    step_games: int = 2,
) -> list[RollingGameSplit]:
    if initial_train_games < 1 or test_games < 1 or step_games < 1:
        raise ValueError("initial_train_games, test_games, and step_games must be positive.")
    game_order = _ordered_game_pks(frame)
    splits = []
    fold_index = 1
    train_end = initial_train_games
    while train_end + test_games <= len(game_order):
        test_end = train_end + test_games
        splits.append(
            RollingGameSplit(
                fold_index=fold_index,
                train_game_pks=game_order[:train_end],
                test_game_pks=game_order[train_end:test_end],
                train_game_range=f"1-{train_end}",
                test_game_range=f"{train_end + 1}-{test_end}",
            )
        )
        fold_index += 1
        train_end += step_games
    return splits


def _ordered_game_pks(frame: pd.DataFrame) -> list[int]:
    if "game_pk" not in frame.columns:
        raise ValueError("Rolling validation requires game_pk.")
    columns = ["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])
    games = frame[columns].drop_duplicates("game_pk").reset_index(drop=True)
    sort_columns = [column for column in ["game_date", "game_pk"] if column in games.columns]
    games = games.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return [int(game_pk) for game_pk in games["game_pk"].tolist()]


def score_rolling_validation_goals(
    consistency: dict[str, object],
    *,
    goals: dict[str, float] | None = None,
) -> dict[str, object]:
    """Score the rolling board against the presentation-grade acceptance gate."""
    goals = dict(ROLLING_SCOREBOARD_GOALS if goals is None else goals)
    mean_goal = float(goals["mean_physics_core_auc_max"])
    hit_rate_goal = float(goals["target_hit_rate_min"])
    worst_goal = float(goals["worst_fold_physics_core_auc_max_exclusive"])

    mean_auc = float(consistency["physics_core_mean_auc_mean"])
    best_auc = float(consistency["physics_core_mean_auc_min"])
    worst_auc = float(consistency["physics_core_mean_auc_max"])
    hit_rate = float(consistency["physics_core_target_hit_rate"])

    checks = [
        {
            "metric": "Mean rolling physics-core AUC",
            "current": mean_auc,
            "current_display": f"{mean_auc:.3f}",
            "goal": f"<= {mean_goal:.3f}",
            "passed": bool(mean_auc <= mean_goal),
            "gap_to_goal": max(0.0, mean_auc - mean_goal),
            "direction": "lower_is_better",
        },
        {
            "metric": "Target hit rate",
            "current": hit_rate,
            "current_display": f"{hit_rate:.2f}",
            "goal": f">= {hit_rate_goal:.2f}",
            "passed": bool(hit_rate >= hit_rate_goal),
            "gap_to_goal": max(0.0, hit_rate_goal - hit_rate),
            "direction": "higher_is_better",
        },
        {
            "metric": "Worst fold physics-core AUC",
            "current": worst_auc,
            "current_display": f"{worst_auc:.3f}",
            "goal": f"< {worst_goal:.3f}",
            "passed": bool(worst_auc < worst_goal),
            "gap_to_goal": max(0.0, worst_auc - worst_goal),
            "direction": "lower_is_better",
        },
    ]
    cleared_count = sum(1 for check in checks if check["passed"])
    check_count = len(checks)
    if cleared_count == check_count:
        status = "rolling_validated"
    elif cleared_count >= 2:
        status = "rolling_candidate"
    else:
        status = "rolling_diagnostic"

    return {
        "scoreboard_name": "primary_rolling_scoreboard",
        "metric_family": "future_game_physics_core_c2st_auc",
        "interpretation": (
            "Rolling temporal validation is the main truth test because each fold "
            "asks whether the model can match a later, unseen game window."
        ),
        "goals": goals,
        "current": {
            "mean_rolling_physics_core_auc": mean_auc,
            "best_fold_physics_core_auc": best_auc,
            "worst_fold_physics_core_auc": worst_auc,
            "target_hit_rate": hit_rate,
        },
        "checks": checks,
        "cleared_count": int(cleared_count),
        "check_count": int(check_count),
        "passed": bool(cleared_count == check_count),
        "status": status,
    }


def explain_detection_features(features: list[str]) -> dict[str, object]:
    categories = []
    feature_set = set(features)
    if any(feature.startswith("spin_axis") for feature in features) or any(
        feature in {"release_speed", "release_spin_rate", "release_extension"}
        or feature.startswith("release_pos")
        for feature in features
    ):
        categories.append("release/spin signature")
    if any(feature in {"ax", "ay", "az", "vx0", "vy0", "vz0"} for feature in features):
        categories.append("trajectory/acceleration")
    if any(feature.startswith("pfx_") for feature in features):
        categories.append("movement")
    if any(feature.startswith("plate_") for feature in features):
        categories.append("command/location")

    if "trajectory/acceleration" in categories and (
        "movement" in categories or feature_set.intersection({"pfx_x", "pfx_z"})
    ):
        primary_mode = "acceleration/movement consistency"
    elif categories:
        primary_mode = categories[0]
    else:
        primary_mode = "general distribution shape"

    return {
        "primary_mode": primary_mode,
        "categories": categories or ["general distribution shape"],
        "signal": " + ".join(features),
    }


def build_pitch_type_failure_explanations(
    tournament_report: dict[str, object],
    *,
    pitch_type: str,
    fold_index: int | None = None,
    target_auc: float | None = None,
    target_pass_rate: float | None = None,
) -> list[dict[str, object]]:
    target_auc = float(target_auc if target_auc is not None else tournament_report["target_auc"])
    target_pass_rate = float(
        target_pass_rate
        if target_pass_rate is not None
        else tournament_report.get("target_pass_rate", 0.80)
    )
    failures = []
    best_by_layer = dict(tournament_report["best_by_layer"])
    for layer, model_name in best_by_layer.items():
        row = tournament_report["layer_results"][layer][model_name]
        mean_auc = float(row["mean_auc"])
        pass_rate = float(row["pass_rate"])
        if mean_auc <= target_auc and pass_rate >= target_pass_rate:
            continue
        features = [
            str(item["feature"])
            for item in row.get("top_leakage_features", [])
            if "feature" in item
        ][:5]
        explanation = explain_detection_features(features)
        failures.append(
            {
                "fold_index": fold_index,
                "pitch_type": pitch_type,
                "layer": layer,
                "model": model_name,
                "mean_auc": mean_auc,
                "pass_rate": pass_rate,
                "target_auc": target_auc,
                "target_pass_rate": target_pass_rate,
                "classifier_signal": explanation["signal"],
                "primary_mode": explanation["primary_mode"],
                "categories": explanation["categories"],
            }
        )
    return failures


def evaluate_rolling_temporal_board(
    player_frame: pd.DataFrame,
    league_frame: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    initial_train_games: int = 10,
    test_games: int = 2,
    step_games: int = 2,
    n_samples: int = 300,
    repeats: int = 6,
    random_state: int = 42,
    target_auc: float = 0.60,
    target_pass_rate: float = 0.80,
    tournament_evaluator: Callable[..., dict[str, object]] = evaluate_model_tournament,
) -> dict[str, object]:
    splits = rolling_game_splits(
        player_frame,
        initial_train_games=initial_train_games,
        test_games=test_games,
        step_games=step_games,
    )
    if not splits:
        raise ValueError("Not enough games for rolling temporal validation.")

    fold_summaries = []
    failure_rows = []
    for split in splits:
        train = player_frame[player_frame["game_pk"].isin(split.train_game_pks)].copy()
        holdout = player_frame[player_frame["game_pk"].isin(split.test_game_pks)].copy()
        report = tournament_evaluator(
            train,
            holdout,
            league_frame,
            pitcher_name=pitcher_name,
            pitch_type=pitch_type,
            n_samples=max(n_samples, len(holdout)),
            repeats=repeats,
            random_state=random_state + split.fold_index * 1000,
            target_auc=target_auc,
            target_pass_rate=target_pass_rate,
        )
        layer_scoreboard = _layer_scoreboard(report)
        physics_row = report["layer_results"]["physics_core"][report["best_by_layer"]["physics_core"]]
        fold_failures = build_pitch_type_failure_explanations(
            report,
            pitch_type=pitch_type,
            fold_index=split.fold_index,
            target_auc=target_auc,
            target_pass_rate=target_pass_rate,
        )
        failure_rows.extend(fold_failures)
        fold_summaries.append(
            {
                "fold_index": split.fold_index,
                "train_game_range": split.train_game_range,
                "test_game_range": split.test_game_range,
                "train_game_pks": split.train_game_pks,
                "test_game_pks": split.test_game_pks,
                "train_rows": int(len(train)),
                "holdout_rows": int(len(holdout)),
                "best_physics_core_model": str(report["best_by_layer"]["physics_core"]),
                "physics_core_mean_auc": float(physics_row["mean_auc"]),
                "physics_core_pass_rate": float(physics_row["pass_rate"]),
                "candidate_default": bool(report.get("candidate_default", False)),
                "layer_scoreboard": layer_scoreboard,
                "failure_count": int(len(fold_failures)),
            }
        )

    physics_values = np.asarray(
        [float(fold["physics_core_mean_auc"]) for fold in fold_summaries],
        dtype=float,
    )
    pass_values = np.asarray(
        [float(fold["physics_core_pass_rate"]) for fold in fold_summaries],
        dtype=float,
    )
    consistency = {
        "physics_core_mean_auc_mean": float(physics_values.mean()),
        "physics_core_mean_auc_std": float(physics_values.std(ddof=0)),
        "physics_core_mean_auc_min": float(physics_values.min()),
        "physics_core_mean_auc_max": float(physics_values.max()),
        "physics_core_target_hit_rate": float((physics_values <= target_auc).mean()),
        "physics_core_pass_rate_mean": float(pass_values.mean()),
    }
    return {
        "model_name": "rolling_temporal_validation_board",
        "pitcher_name": pitcher_name,
        "pitch_type": pitch_type,
        "initial_train_games": int(initial_train_games),
        "test_games": int(test_games),
        "step_games": int(step_games),
        "repeat_count": int(repeats),
        "target_auc": float(target_auc),
        "target_pass_rate": float(target_pass_rate),
        "fold_count": int(len(fold_summaries)),
        "folds": fold_summaries,
        "failure_explanations": failure_rows,
        "consistency": consistency,
        "primary_scoreboard": score_rolling_validation_goals(consistency),
    }


def _layer_scoreboard(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for layer, model_name in report["best_by_layer"].items():
        row = report["layer_results"][layer][model_name]
        rows.append(
            {
                "layer": layer,
                "best_model": model_name,
                "mean_auc": float(row["mean_auc"]),
                "pass_rate": float(row["pass_rate"]),
                "std_auc": float(row.get("std_auc", 0.0)),
            }
        )
    return rows


def write_rolling_board_outputs(board: dict[str, object], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rolling_validation_board.json"
    markdown_path = output_dir / "rolling_validation_board.md"
    failure_csv_path = output_dir / "pitch_type_failure_explainer.csv"

    json_path.write_text(json.dumps(board, indent=2) + "\n")
    markdown_path.write_text(_rolling_board_markdown(board))
    pd.DataFrame(board["failure_explanations"]).to_csv(failure_csv_path, index=False)
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "failure_csv": str(failure_csv_path),
    }


def _rolling_board_markdown(board: dict[str, object]) -> str:
    consistency = board["consistency"]
    scoreboard = board.get("primary_scoreboard") or score_rolling_validation_goals(consistency)
    lines = [
        "# Rolling Temporal Validation Board",
        "",
        f"- Pitcher: `{board['pitcher_name']}`",
        f"- Pitch type: `{board['pitch_type']}`",
        f"- Folds: `{board['fold_count']}`",
        f"- Repeats per fold: `{board['repeat_count']}`",
        f"- Target AUC: `<={board['target_auc']:.3f}`",
        f"- Target pass rate: `>={board['target_pass_rate']:.2f}`",
        "",
        "## Primary Rolling Scoreboard",
        "",
        f"- Status: `{scoreboard['status']}`",
        f"- Goals cleared: `{scoreboard['cleared_count']}/{scoreboard['check_count']}`",
        "- Best fold physics-core AUC: "
        f"`{scoreboard['current']['best_fold_physics_core_auc']:.3f}`",
        "",
        "| Metric | Current | Goal | Result | Gap |",
        "|---|---:|---:|---|---:|",
    ]
    for check in scoreboard["checks"]:
        result = "clear" if check["passed"] else "miss"
        lines.append(
            "| {metric} | {current_display} | {goal} | {result} | {gap_to_goal:.3f} |".format(
                result=result,
                **check,
            )
        )
    lines.extend(
        [
            "",
            "Rolling validation is the main scoreboard because it tests repeated "
            "future-game windows, not one favorable temporal split.",
            "",
            "## Fold Results",
            "",
        "| Fold | Train games | Test games | Train rows | Holdout rows | "
        "Best physics model | Physics AUC | Pass rate | Failures |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for fold in board["folds"]:
        lines.append(
            "| {fold_index} | {train_game_range} | {test_game_range} | {train_rows} | "
            "{holdout_rows} | {best_physics_core_model} | {physics_core_mean_auc:.3f} | "
            "{physics_core_pass_rate:.2f} | {failure_count} |".format(**fold)
        )
    lines.extend(
        [
            "",
            "## Consistency",
            "",
            f"- Mean physics-core AUC: `{consistency['physics_core_mean_auc_mean']:.3f}`",
            "- Physics-core AUC range: "
            f"`{consistency['physics_core_mean_auc_min']:.3f}` to "
            f"`{consistency['physics_core_mean_auc_max']:.3f}`",
            f"- Target hit rate: `{consistency['physics_core_target_hit_rate']:.2f}`",
            f"- Mean pass rate: `{consistency['physics_core_pass_rate_mean']:.2f}`",
            "",
            "## Pitch-Type Failure Explainer",
            "",
            "| Fold | Pitch | Layer | Model | AUC | Pass rate | Classifier signal | Failure mode |",
            "|---:|---|---|---|---:|---:|---|---|",
        ]
    )
    for failure in board["failure_explanations"]:
        lines.append(
            "| {fold_index} | {pitch_type} | {layer} | {model} | {mean_auc:.3f} | "
            "{pass_rate:.2f} | {classifier_signal} | {primary_mode} |".format(**failure)
        )
    lines.append("")
    return "\n".join(lines)
