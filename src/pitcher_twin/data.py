"""Real Statcast loading and cache validation for Pitcher Twin."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_STATCAST_COLUMNS = [
    "pitch_type",
    "game_date",
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "player_name",
    "batter",
    "pitcher",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "balls",
    "strikes",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "inning",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "release_spin_rate",
    "release_extension",
    "game_pk",
    "pitch_number",
    "release_pos_y",
    "spin_axis",
]


def ensure_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_STATCAST_COLUMNS,
) -> None:
    """Raise with the exact missing real Statcast columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required Statcast columns: {', '.join(missing)}")


def _missing_cache_message(path: Path) -> str:
    return (
        f"Real Statcast cache is missing: {path}. "
        "Run `python3 scripts/fetch_real_statcast.py --start YYYY-MM-DD "
        "--end YYYY-MM-DD --output data/raw/statcast.parquet` or pass an existing "
        "real public Statcast cache."
    )


def load_statcast_cache(path: str | Path) -> pd.DataFrame:
    """Load a real Statcast cache from CSV or parquet, never fabricating rows."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(_missing_cache_message(cache_path))

    if cache_path.suffix.lower() == ".csv":
        df = pd.read_csv(cache_path, low_memory=False)
    elif cache_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(cache_path)
    else:
        raise ValueError(f"Unsupported Statcast cache format: {cache_path.suffix}")

    ensure_required_columns(df)
    return df.reset_index(drop=True)


def write_statcast_cache(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a validated real Statcast cache to CSV or parquet."""
    ensure_required_columns(df)
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix.lower() == ".csv":
        df.to_csv(cache_path, index=False)
    elif cache_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(cache_path, index=False)
    else:
        raise ValueError(f"Unsupported Statcast cache format: {cache_path.suffix}")
    return cache_path


def load_existing_statcast_sources(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate existing real caches."""
    frames = [load_statcast_cache(path) for path in paths]
    if not frames:
        raise ValueError("At least one real Statcast cache path is required.")
    return pd.concat(frames, ignore_index=True)


def fetch_statcast_range(start_date: str, end_date: str, output_path: str | Path) -> Path:
    """Fetch public Statcast data with pybaseball and cache it locally."""
    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise RuntimeError(
            "pybaseball is required to fetch real Statcast data. "
            "Install with `pip install -e '.[data]'`."
        ) from exc

    df = statcast(start_dt=start_date, end_dt=end_date)
    ensure_required_columns(df)
    return write_statcast_cache(df, output_path)
