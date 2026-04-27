#!/usr/bin/env python3
"""Fetch real Open-Meteo game-time weather for Statcast game IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.weather import fetch_weather_for_game_pks, write_weather_cache  # noqa: E402


def run(data_path: Path, output_path: Path, limit_games: int | None = None) -> Path:
    pitches = load_statcast_cache(data_path)
    if "game_pk" not in pitches.columns:
        raise RuntimeError("Input Statcast file must include game_pk.")
    game_pks = sorted(int(game_pk) for game_pk in pitches["game_pk"].dropna().unique())
    if limit_games is not None:
        game_pks = game_pks[:limit_games]
    if not game_pks:
        raise RuntimeError("No game_pk values found in Statcast file.")
    weather = fetch_weather_for_game_pks(game_pks)
    return write_weather_cache(weather, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-games", type=int)
    args = parser.parse_args()
    print(run(args.data, args.output, limit_games=args.limit_games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
