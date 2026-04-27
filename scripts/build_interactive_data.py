"""Pre-sample model output for the interactive Try It panel on the static site.

For each (pitcher, pitch_type) candidate, fit the generator suite once and
sample N pitches per (count_bucket, batter_hand) game-context combination.
Also exports the real held-out pitches so the JS layer can overlay them.

Output: site/data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitcher_twin.conditional import (  # noqa: E402
    make_context_dataframe,
    sample_conditional_distribution,
)
from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.features import clean_pitch_features  # noqa: E402
from pitcher_twin.models import fit_generator_suite  # noqa: E402
from pitcher_twin.validator import temporal_train_holdout  # noqa: E402

OUTPUT_PATH = ROOT / "site" / "data.json"
SKUBAL_CSV = ROOT / "data" / "processed" / "skubal_2025.csv"
LATEST_CSV = Path(
    "/Users/elliot18/assistant/projects/trajekt-scout/data/processed/latest_statcast.csv"
)

CANDIDATES = [
    {"key": "skubal_ff", "label": "Tarik Skubal · FF (4-seam)", "data": SKUBAL_CSV, "pitcher": 669373, "pitch_type": "FF", "pitcher_name": "Skubal, Tarik", "status": "validated"},
    {"key": "skubal_si", "label": "Tarik Skubal · SI (sinker)", "data": SKUBAL_CSV, "pitcher": 669373, "pitch_type": "SI", "pitcher_name": "Skubal, Tarik", "status": "diagnostic"},
    {"key": "skubal_ch", "label": "Tarik Skubal · CH (changeup)", "data": SKUBAL_CSV, "pitcher": 669373, "pitch_type": "CH", "pitcher_name": "Skubal, Tarik", "status": "diagnostic"},
    {"key": "mattson_ff", "label": "Isaac Mattson · FF (4-seam)", "data": LATEST_CSV, "pitcher": 642547, "pitch_type": "FF", "pitcher_name": "Mattson, Isaac", "status": "candidate"},
    {"key": "peralta_ff", "label": "Freddy Peralta · FF (4-seam)", "data": LATEST_CSV, "pitcher": 642547, "pitch_type": "FF", "pitcher_name": "Peralta, Freddy", "status": "diagnostic"},
    {"key": "bradley_ff", "label": "Taj Bradley · FF (4-seam)", "data": LATEST_CSV, "pitcher": 681867, "pitch_type": "FF", "pitcher_name": "Bradley, Taj", "status": "diagnostic"},
]

# Resolve pitcher IDs from names if possible.
PITCHER_NAME_FIXUPS = {
    "Mattson, Isaac": None,
    "Peralta, Freddy": None,
    "Bradley, Taj": None,
}

COUNT_BUCKETS = [
    {"key": "first_pitch", "label": "First pitch (0-0)", "balls": 0, "strikes": 0},
    {"key": "ahead", "label": "Ahead (0-2)", "balls": 0, "strikes": 2},
    {"key": "even", "label": "Even (1-1)", "balls": 1, "strikes": 1},
    {"key": "behind", "label": "Behind (2-0)", "balls": 2, "strikes": 0},
    {"key": "full", "label": "Full count (3-2)", "balls": 3, "strikes": 2},
]
BATTER_HANDS = [
    {"key": "R", "label": "vs Right-handed batter"},
    {"key": "L", "label": "vs Left-handed batter"},
]
SAMPLES_PER_CONTEXT = 80
KEEP_COLUMNS = ["plate_x", "plate_z", "release_speed", "release_spin_rate", "pfx_x", "pfx_z"]


def serialize_frame(frame: pd.DataFrame) -> list[dict]:
    cols = [column for column in KEEP_COLUMNS if column in frame.columns]
    sub = frame[cols].dropna(subset=["plate_x", "plate_z"])
    rows = []
    for _, row in sub.iterrows():
        record = {}
        for col in cols:
            value = row[col]
            if pd.isna(value):
                continue
            record[col] = float(value)
        rows.append(record)
    return rows


def lookup_pitcher_id(clean: pd.DataFrame, target_name: str) -> int | None:
    """Try to match a pitcher name in the cleaned cache to a pitcher id."""
    if "player_name" not in clean.columns or "pitcher" not in clean.columns:
        return None
    matches = clean[clean["player_name"].fillna("").str.strip() == target_name.strip()]
    if matches.empty:
        return None
    counts = matches["pitcher"].value_counts()
    if counts.empty:
        return None
    return int(counts.index[0])


def build_for_candidate(spec: dict, clean_cache: dict[str, pd.DataFrame]) -> dict | None:
    data_path = Path(spec["data"])
    if not data_path.exists():
        print(f"  skip (missing data): {spec['key']}")
        return None
    cache_key = str(data_path)
    if cache_key not in clean_cache:
        print(f"  loading {data_path.name} ...", flush=True)
        clean_cache[cache_key] = clean_pitch_features(
            load_statcast_cache(data_path), pitch_types=None
        )
    clean = clean_cache[cache_key]

    pitcher_id = spec["pitcher"]
    if spec["pitcher_name"] in PITCHER_NAME_FIXUPS:
        looked_up = lookup_pitcher_id(clean, spec["pitcher_name"])
        if looked_up is not None:
            pitcher_id = looked_up

    subset = clean[
        (clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == spec["pitch_type"])
    ].copy()
    if len(subset) < 60:
        print(f"  skip {spec['key']} (only {len(subset)} pitches)")
        return None

    train, holdout = temporal_train_holdout(subset, train_fraction=0.7)
    if len(train) < 30:
        print(f"  skip {spec['key']} (train too small)")
        return None

    print(
        f"  fit suite for {spec['key']} (train={len(train)}, holdout={len(holdout)})...",
        flush=True,
    )
    suite = fit_generator_suite(
        train,
        clean,
        pitcher_name=spec["pitcher_name"],
        pitch_type=spec["pitch_type"],
        feature_group="physics_core",
    )

    contexts: dict[str, list[dict]] = {}
    for count in COUNT_BUCKETS:
        for hand in BATTER_HANDS:
            ctx_df = make_context_dataframe(
                inning=5,
                pitcher_game_pitch_count=45,
                balls=count["balls"],
                strikes=count["strikes"],
                batter_hand=hand["key"],
                pitcher_score_diff=0,
                repeat=1,
            )
            samples, _ = sample_conditional_distribution(
                suite,
                ctx_df,
                n=SAMPLES_PER_CONTEXT,
                random_state=int(np.uint32(hash((count["key"], hand["key"], spec["key"])) & 0xFFFFFFFF)),
            )
            samples_renamed = samples.rename(columns={"spin_rate": "release_spin_rate"})
            key = f"{count['key']}_{hand['key']}"
            contexts[key] = serialize_frame(samples_renamed)

    return {
        "key": spec["key"],
        "label": spec["label"],
        "pitcher_name": spec["pitcher_name"],
        "pitch_type": spec["pitch_type"],
        "status": spec["status"],
        "train_count": int(len(train)),
        "holdout_count": int(len(holdout)),
        "real_holdout": serialize_frame(holdout),
        "samples": contexts,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_cache: dict[str, pd.DataFrame] = {}
    candidates_out = []
    for spec in CANDIDATES:
        print(f"building {spec['key']} ...", flush=True)
        result = build_for_candidate(spec, clean_cache)
        if result is None:
            continue
        candidates_out.append(result)

    payload = {
        "schema_version": 1,
        "candidates": candidates_out,
        "count_buckets": COUNT_BUCKETS,
        "batter_hands": BATTER_HANDS,
        "samples_per_context": SAMPLES_PER_CONTEXT,
        "default_candidate": "skubal_ff",
        "default_count_bucket": "even",
        "default_batter_hand": "R",
    }
    OUTPUT_PATH.write_text(json.dumps(payload))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"wrote {OUTPUT_PATH} ({size_kb:.1f} KB, {len(candidates_out)} candidates)")


if __name__ == "__main__":
    main()
