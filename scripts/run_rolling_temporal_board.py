#!/usr/bin/env python3
"""Run rolling temporal validation board reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.features import clean_pitch_features  # noqa: E402
from pitcher_twin.rolling_validation import (  # noqa: E402
    evaluate_rolling_temporal_board,
    write_rolling_board_outputs,
)


def _pitcher_name_from_subset(subset) -> str:
    for column in ("pitcher_name", "player_name"):
        if column in subset.columns:
            names = subset[column].dropna()
            if not names.empty:
                return str(names.iloc[0])
    return str(subset["pitcher"].dropna().iloc[0])


def run(
    data_path: Path,
    output_dir: Path,
    pitcher_id: int,
    pitch_type: str,
    *,
    initial_train_games: int = 10,
    test_games: int = 2,
    step_games: int = 2,
    n_samples: int = 300,
    repeats: int = 6,
    random_state: int = 42,
) -> dict[str, str]:
    raw = load_statcast_cache(data_path)
    clean = clean_pitch_features(raw, pitch_types=None)
    subset = clean[(clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == pitch_type)].copy()
    if subset.empty:
        raise RuntimeError(f"No rows found for pitcher={pitcher_id}, pitch_type={pitch_type}.")
    pitcher_name = _pitcher_name_from_subset(subset)
    board = evaluate_rolling_temporal_board(
        subset,
        clean,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        initial_train_games=initial_train_games,
        test_games=test_games,
        step_games=step_games,
        n_samples=n_samples,
        repeats=repeats,
        random_state=random_state,
    )
    board.update(
        {
            "data_path": str(data_path),
            "pitcher_id": int(pitcher_id),
            "rows_raw": int(len(raw)),
            "rows_clean": int(len(clean)),
            "rows_subset": int(len(subset)),
        }
    )
    return write_rolling_board_outputs(board, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rolling_validation"))
    parser.add_argument("--pitcher-id", type=int, default=669373)
    parser.add_argument("--pitch-type", type=str, default="FF")
    parser.add_argument("--initial-train-games", type=int, default=10)
    parser.add_argument("--test-games", type=int, default=2)
    parser.add_argument("--step-games", type=int, default=2)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    outputs = run(
        args.data,
        args.output_dir,
        args.pitcher_id,
        args.pitch_type,
        initial_train_games=args.initial_train_games,
        test_games=args.test_games,
        step_games=args.step_games,
        n_samples=args.samples,
        repeats=args.repeats,
        random_state=args.random_state,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
