#!/usr/bin/env python3
"""Rank real pitcher/pitch candidates from a Statcast cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.candidates import CandidateThresholds, rank_pitcher_pitch_candidates, write_selected_candidates  # noqa: E402
from pitcher_twin.data import load_statcast_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/candidate_rankings.csv"))
    parser.add_argument("--selected-output", type=Path, default=Path("outputs/selected_candidates.json"))
    parser.add_argument("--min-pitches", type=int, default=600)
    parser.add_argument("--min-holdout", type=int, default=150)
    parser.add_argument("--min-games", type=int, default=4)
    args = parser.parse_args()

    df = load_statcast_cache(args.input)
    ranking = rank_pitcher_pitch_candidates(
        df,
        CandidateThresholds(
            min_pitches=args.min_pitches,
            min_holdout=args.min_holdout,
            min_games=args.min_games,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output, index=False)
    if len(ranking) >= 2:
        write_selected_candidates(ranking, args.selected_output, args.input)
    print(f"Wrote {args.output}")
    if len(ranking):
        print(ranking.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
