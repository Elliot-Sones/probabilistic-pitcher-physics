"""Real pitcher/pitch candidate ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CORE_COMPLETENESS_COLUMNS = [
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


@dataclass(frozen=True)
class CandidateThresholds:
    min_pitches: int = 600
    min_holdout: int = 150
    min_games: int = 4
    min_completeness: float = 0.95


def _scale(series: pd.Series, target: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return (values / target).clip(upper=1.0)


def _pitcher_name_column(df: pd.DataFrame) -> str:
    if "pitcher_name" in df.columns:
        return "pitcher_name"
    return "player_name"


def rank_pitcher_pitch_candidates(
    df: pd.DataFrame,
    thresholds: CandidateThresholds = CandidateThresholds(),
) -> pd.DataFrame:
    """Rank real `(pitcher, pitch_type)` candidates for modeling viability."""
    required = {"pitcher", "pitch_type", "game_pk", "game_date", _pitcher_name_column(df)}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing candidate ranking columns: {', '.join(missing)}")

    name_col = _pitcher_name_column(df)
    rows: list[dict[str, object]] = []
    for (pitcher, pitcher_name, pitch_type), group in df.groupby(
        ["pitcher", name_col, "pitch_type"], dropna=True
    ):
        group = group.sort_values("game_date")
        n = len(group)
        train_n = int(n * 0.7)
        holdout_n = n - train_n
        games = int(group["game_pk"].nunique())
        completeness = float(group[CORE_COMPLETENESS_COLUMNS].notna().mean().mean())
        variability_values = [
            pd.to_numeric(group["release_speed"], errors="coerce").std(),
            pd.to_numeric(group["pfx_x"], errors="coerce").std(),
            pd.to_numeric(group["pfx_z"], errors="coerce").std(),
            pd.to_numeric(group["plate_x"], errors="coerce").std(),
            pd.to_numeric(group["plate_z"], errors="coerce").std(),
        ]
        finite_variability = [value for value in variability_values if np.isfinite(value)]
        if finite_variability:
            variability_signal = float(np.mean(finite_variability))
        else:
            variability_signal = 0.0
        context_columns = [
            column
            for column in ["balls", "strikes", "inning", "pitch_number", "stand", "p_throws"]
            if column in group.columns
        ]
        context_coverage = (
            float(group[context_columns].notna().mean().mean()) if context_columns else 0.0
        )

        sample_size_score = min(n / thresholds.min_pitches, 1.0)
        games_score = min(games / thresholds.min_games, 1.0)
        holdout_viability = min(holdout_n / thresholds.min_holdout, 1.0)
        variability_score = min(max(variability_signal, 0.0) / 2.0, 1.0)
        candidate_score = (
            0.30 * sample_size_score
            + 0.20 * games_score
            + 0.15 * completeness
            + 0.15 * holdout_viability
            + 0.10 * context_coverage
            + 0.10 * variability_score
        )
        passes_thresholds = (
            n >= thresholds.min_pitches
            and holdout_n >= thresholds.min_holdout
            and games >= thresholds.min_games
            and completeness >= thresholds.min_completeness
        )

        rows.append(
            {
                "pitcher": int(pitcher),
                "pitcher_name": str(pitcher_name),
                "pitch_type": str(pitch_type),
                "n": int(n),
                "games": games,
                "train_n": int(train_n),
                "holdout_n": int(holdout_n),
                "feature_completeness": completeness,
                "context_coverage": context_coverage,
                "variability_signal": variability_signal,
                "candidate_score": float(candidate_score),
                "passes_thresholds": bool(passes_thresholds),
                "core_columns_checked": len(CORE_COMPLETENESS_COLUMNS),
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values(
        ["passes_thresholds", "candidate_score", "n", "games"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def write_selected_candidates(
    ranking: pd.DataFrame,
    output_path: str | Path,
    data_path: str | Path,
) -> dict[str, object]:
    """Write primary/backup selected real candidates to JSON."""
    if len(ranking) < 2:
        raise ValueError("At least two ranked candidates are required to choose a backup.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "primary": ranking.iloc[0].to_dict(),
        "backup": ranking.iloc[1].to_dict(),
        "top_candidates": ranking.head(10).to_dict(orient="records"),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
