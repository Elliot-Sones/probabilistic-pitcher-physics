from __future__ import annotations

from pathlib import Path

import pandas as pd

from pitcher_twin.candidates import (
    CORE_COMPLETENESS_COLUMNS,
    CandidateThresholds,
    rank_pitcher_pitch_candidates,
    write_selected_candidates,
)


REAL_SAMPLE = Path(__file__).parent / "fixtures" / "real_statcast_sample.csv"


def test_candidate_ranking_returns_scored_real_pitcher_pitch_pairs() -> None:
    df = pd.read_csv(REAL_SAMPLE)
    ranking = rank_pitcher_pitch_candidates(
        df,
        thresholds=CandidateThresholds(min_pitches=20, min_holdout=5, min_games=1),
    )
    assert not ranking.empty
    assert {"pitcher_name", "pitcher", "pitch_type", "candidate_score", "feature_completeness"}.issubset(
        ranking.columns
    )
    assert ranking["candidate_score"].is_monotonic_decreasing
    assert ranking["feature_completeness"].between(0, 1).all()


def test_candidate_ranking_uses_core_completeness_columns() -> None:
    df = pd.read_csv(REAL_SAMPLE)
    ranking = rank_pitcher_pitch_candidates(
        df,
        thresholds=CandidateThresholds(min_pitches=20, min_holdout=5, min_games=1),
    )
    assert set(CORE_COMPLETENESS_COLUMNS).issubset(df.columns)
    assert (ranking["core_columns_checked"] == len(CORE_COMPLETENESS_COLUMNS)).all()


def test_write_selected_candidates_creates_primary_and_backup_json(tmp_path) -> None:
    df = pd.read_csv(REAL_SAMPLE)
    ranking = rank_pitcher_pitch_candidates(
        df,
        thresholds=CandidateThresholds(min_pitches=20, min_holdout=5, min_games=1),
    )
    output = tmp_path / "selected_candidates.json"
    payload = write_selected_candidates(ranking, output, data_path=REAL_SAMPLE)
    assert output.exists()
    assert payload["primary"]["candidate_score"] >= payload["backup"]["candidate_score"]
    assert payload["data_path"] == str(REAL_SAMPLE)
