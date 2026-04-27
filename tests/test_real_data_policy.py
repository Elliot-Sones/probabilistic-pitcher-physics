from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pitcher_twin.data import (
    REQUIRED_STATCAST_COLUMNS,
    ensure_required_columns,
    load_existing_statcast_sources,
    load_statcast_cache,
    write_statcast_cache,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
REAL_SAMPLE = FIXTURE_DIR / "real_statcast_sample.csv"
PROVENANCE = FIXTURE_DIR / "real_statcast_sample.provenance.json"


def test_real_fixture_files_exist_with_provenance() -> None:
    assert REAL_SAMPLE.exists()
    assert PROVENANCE.exists()
    provenance = json.loads(PROVENANCE.read_text())
    assert provenance["source_kind"] == "real_public_statcast_cache"
    assert provenance["row_count"] > 0
    assert Path(provenance["source_path"]).exists()


def test_real_fixture_contains_required_statcast_columns() -> None:
    df = pd.read_csv(REAL_SAMPLE, nrows=50)
    assert set(REQUIRED_STATCAST_COLUMNS).issubset(df.columns)


def test_missing_cache_raises_with_fetch_instruction(tmp_path) -> None:
    missing = tmp_path / "missing_statcast.csv"
    with pytest.raises(FileNotFoundError, match="fetch_real_statcast"):
        load_statcast_cache(missing)


def test_required_column_error_lists_missing_columns() -> None:
    df = pd.read_csv(REAL_SAMPLE, nrows=10).drop(columns=["release_speed", "pitch_type"])
    with pytest.raises(ValueError, match="pitch_type.*release_speed|release_speed.*pitch_type"):
        ensure_required_columns(df)


def test_cache_round_trip_csv_and_multi_source_load(tmp_path) -> None:
    original = pd.read_csv(REAL_SAMPLE, nrows=25)
    output = tmp_path / "round_trip.csv"
    write_statcast_cache(original, output)
    loaded = load_statcast_cache(output)
    pd.testing.assert_frame_equal(loaded, original)

    combined = load_existing_statcast_sources([output, output])
    assert len(combined) == 50
