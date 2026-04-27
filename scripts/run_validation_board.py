#!/usr/bin/env python3
"""Run a cross-candidate Pitcher Twin validation board on real Statcast data."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.features import clean_pitch_features  # noqa: E402
from pitcher_twin.tournament import evaluate_model_tournament  # noqa: E402
from pitcher_twin.validation_board import (  # noqa: E402
    CandidateCriteria,
    candidate_pitcher_pitches,
    leaderboard_frame,
    render_scorecard_markdown,
    render_validation_board_markdown,
    rolling_game_windows,
    slugify_label,
    summarize_tournament_report,
)
from pitcher_twin.validator import temporal_train_holdout  # noqa: E402


TournamentFn = Callable[..., dict[str, Any]]


def run(
    data_path: Path | None,
    output_dir: Path,
    *,
    raw_df: pd.DataFrame | None = None,
    top: int = 5,
    min_pitches: int = 300,
    min_games: int = 8,
    min_holdout: int = 60,
    repeats: int = 3,
    samples: int = 260,
    random_state: int = 42,
    rolling: bool = False,
    rolling_repeats: int = 1,
    max_rolling_windows: int = 3,
    tournament_fn: TournamentFn = evaluate_model_tournament,
) -> dict[str, str]:
    """Run the validation board and write durable artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    scorecard_dir = output_dir / "scorecards"
    report_dir.mkdir(parents=True, exist_ok=True)
    scorecard_dir.mkdir(parents=True, exist_ok=True)

    raw = (
        raw_df.copy()
        if raw_df is not None
        else load_statcast_cache(_require_data_path(data_path))
    )
    clean = _clean_or_use_feature_frame(raw)
    criteria = CandidateCriteria(
        min_pitches=min_pitches,
        min_games=min_games,
        min_holdout=min_holdout,
        top=top,
    )
    candidates = candidate_pitcher_pitches(clean, criteria)
    if not candidates:
        raise RuntimeError(
            "No pitcher/pitch candidates met the validation-board thresholds. "
            f"min_pitches={min_pitches}, min_games={min_games}, min_holdout={min_holdout}."
        )

    summaries: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        subset = _candidate_subset(clean, candidate)
        train, holdout = temporal_train_holdout(subset, train_fraction=criteria.train_fraction)
        report = tournament_fn(
            train,
            holdout,
            clean,
            pitcher_name=str(candidate["pitcher_name"]),
            pitch_type=str(candidate["pitch_type"]),
            n_samples=max(samples, len(holdout)),
            repeats=repeats,
            random_state=random_state + candidate_index * 100,
        )
        report.update(
            {
                "data_path": str(data_path) if data_path is not None else "<in-memory>",
                "pitcher_id": candidate["pitcher"],
                "rows_clean": int(len(clean)),
                "rows_subset": int(len(subset)),
                "validation_board_candidate": candidate,
            }
        )
        slug = _candidate_slug(candidate)
        candidate_report_dir = report_dir / slug
        candidate_report_dir.mkdir(parents=True, exist_ok=True)
        (candidate_report_dir / "model_tournament_report.json").write_text(
            json.dumps(report, indent=2, default=_json_default) + "\n"
        )

        candidate_rolling_rows: list[dict[str, Any]] = []
        if rolling:
            windows = rolling_game_windows(
                subset,
                min_train_games=max(3, min_games // 2),
                holdout_games=2,
                max_windows=max_rolling_windows,
            )
            for window in windows:
                window_report = tournament_fn(
                    window.train,
                    window.holdout,
                    clean,
                    pitcher_name=str(candidate["pitcher_name"]),
                    pitch_type=str(candidate["pitch_type"]),
                    n_samples=max(samples, len(window.holdout)),
                    repeats=rolling_repeats,
                    random_state=random_state + 5000 + window.window_index,
                )
                window_summary = summarize_tournament_report(window_report, candidate=candidate)
                row = {
                    "pitcher": candidate["pitcher"],
                    "pitcher_name": candidate["pitcher_name"],
                    "pitch_type": candidate["pitch_type"],
                    "window_index": window.window_index,
                    "train_games": window.train_game_count,
                    "holdout_games": window.holdout_game_count,
                    "train_rows": window.train_row_count,
                    "holdout_rows": window.holdout_row_count,
                    "train_end_game": window.train_end_game,
                    "holdout_start_game": window.holdout_start_game,
                    "holdout_end_game": window.holdout_end_game,
                    "best_physics_core_model": window_summary["best_physics_core_model"],
                    "physics_core_mean_auc": window_summary["physics_core_mean_auc"],
                    "physics_core_pass_rate": window_summary["physics_core_pass_rate"],
                    "artifact_status": window_summary["artifact_status"],
                }
                candidate_rolling_rows.append(row)
                rolling_rows.append(row)

        summary = summarize_tournament_report(report, candidate=candidate)
        summaries.append(summary)
        (candidate_report_dir / "scorecard.md").write_text(
            render_scorecard_markdown(summary, rolling_rows=candidate_rolling_rows)
        )
        scorecard_path = scorecard_dir / f"{slug}.md"
        scorecard_path.write_text(
            render_scorecard_markdown(summary, rolling_rows=candidate_rolling_rows)
        )

    leaderboard = leaderboard_frame(summaries)
    leaderboard_csv = output_dir / "leaderboard.csv"
    leaderboard_json = output_dir / "leaderboard.json"
    board_markdown = output_dir / "validation_board.md"
    leaderboard.to_csv(leaderboard_csv, index=False)
    leaderboard_json.write_text(
        json.dumps(leaderboard.to_dict("records"), indent=2, default=_json_default) + "\n"
    )
    board_markdown.write_text(render_validation_board_markdown(leaderboard, summaries))

    outputs = {
        "leaderboard_csv": str(leaderboard_csv),
        "leaderboard_json": str(leaderboard_json),
        "validation_board": str(board_markdown),
    }
    if rolling_rows:
        rolling_path = output_dir / "rolling_windows.csv"
        pd.DataFrame(rolling_rows).to_csv(rolling_path, index=False)
        outputs["rolling_windows_csv"] = str(rolling_path)

    run_config = {
        "data_path": str(data_path) if data_path is not None else "<in-memory>",
        "output_dir": str(output_dir),
        "top": top,
        "min_pitches": min_pitches,
        "min_games": min_games,
        "min_holdout": min_holdout,
        "repeats": repeats,
        "samples": samples,
        "random_state": random_state,
        "rolling": rolling,
        "rolling_repeats": rolling_repeats,
        "max_rolling_windows": max_rolling_windows,
        "candidate_count": len(candidates),
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    outputs["run_config"] = str(config_path)
    return outputs


def _require_data_path(data_path: Path | None) -> Path:
    if data_path is None:
        raise ValueError("data_path is required when raw_df is not provided.")
    return data_path


def _clean_or_use_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if "spin_axis" in raw.columns:
        return clean_pitch_features(raw, pitch_types=None)
    return raw.reset_index(drop=True).copy()


def _candidate_subset(clean: pd.DataFrame, candidate: dict[str, Any]) -> pd.DataFrame:
    return clean[
        (clean["pitcher"] == candidate["pitcher"])
        & (clean["pitch_type"] == candidate["pitch_type"])
    ].copy()


def _candidate_slug(candidate: dict[str, Any]) -> str:
    return slugify_label(f"{candidate['pitcher_name']}_{candidate['pitch_type']}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation_board"))
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-pitches", type=int, default=300)
    parser.add_argument("--min-games", type=int, default=8)
    parser.add_argument("--min-holdout", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=260)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--rolling-repeats", type=int, default=1)
    parser.add_argument("--max-rolling-windows", type=int, default=3)
    args = parser.parse_args()
    outputs = run(
        args.data,
        args.output_dir,
        top=args.top,
        min_pitches=args.min_pitches,
        min_games=args.min_games,
        min_holdout=args.min_holdout,
        repeats=args.repeats,
        samples=args.samples,
        random_state=args.random_state,
        rolling=args.rolling,
        rolling_repeats=args.rolling_repeats,
        max_rolling_windows=args.max_rolling_windows,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
