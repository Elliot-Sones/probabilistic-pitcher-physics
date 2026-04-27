from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pitcher_twin.candidates import CandidateThresholds, rank_pitcher_pitch_candidates
from pitcher_twin.features import clean_pitch_features
from pitcher_twin.models import fit_generator_suite
from pitcher_twin.sampler import sample_pitch_session
from pitcher_twin.machine_session_format import to_machine_session_json, write_machine_session_json
from pitcher_twin.validator import temporal_train_holdout


REAL_SAMPLE = Path(__file__).parent / "fixtures" / "real_statcast_sample.csv"


def _session():
    df = clean_pitch_features(pd.read_csv(REAL_SAMPLE), pitch_types=None)
    ranking = rank_pitcher_pitch_candidates(
        df,
        thresholds=CandidateThresholds(min_pitches=20, min_holdout=5, min_games=1),
    )
    candidate = ranking.iloc[0].to_dict()
    subset = df[
        (df["pitcher"] == candidate["pitcher"])
        & (df["pitch_type"] == candidate["pitch_type"])
    ].copy()
    train, _ = temporal_train_holdout(subset, train_fraction=0.7)
    suite = fit_generator_suite(
        train,
        df,
        pitcher_name=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        random_state=11,
    )
    return candidate, sample_pitch_session(suite["player_gmm"], n=5, random_state=12)


def test_sample_pitch_session_labels_simulated_source() -> None:
    candidate, session = _session()
    assert len(session) == 5
    assert set(session["source"]) == {"simulated_from_real_model"}
    assert set(session["pitcher_name"]) == {candidate["pitcher_name"]}
    assert set(session["pitch_type"]) == {candidate["pitch_type"]}


def test_pitch_json_contains_pitch_targets_and_metadata(tmp_path) -> None:
    candidate, session = _session()
    payload = to_machine_session_json(
        session,
        pitcher=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        metadata={"data_path": str(REAL_SAMPLE), "model_name": "player_gmm"},
    )
    assert payload["schema_version"] == "pitcher-twin.real.v1"
    assert payload["pitcher"] == candidate["pitcher_name"]
    assert payload["pitches"][0]["source"] == "simulated_from_real_model"
    path = tmp_path / "session.json"
    write_machine_session_json(payload, path)
    assert json.loads(path.read_text())["pitches"][0]["source"] == "simulated_from_real_model"
