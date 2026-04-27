#!/usr/bin/env python3
"""Build small test fixtures from an existing real public Statcast cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402


def build_real_fixtures(source: Path, output_dir: Path, rows: int = 500) -> dict[str, Path]:
    df = load_statcast_cache(source)
    sample = df.sort_values(["game_date", "game_pk", "pitch_number"]).head(rows).copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "real_statcast_sample.csv"
    provenance_path = output_dir / "real_statcast_sample.provenance.json"
    sample.to_csv(sample_path, index=False)
    provenance = {
        "source_kind": "real_public_statcast_cache",
        "source_path": str(source),
        "row_count": int(len(sample)),
        "source_row_count": int(len(df)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "date_min": str(sample["game_date"].min()),
        "date_max": str(sample["game_date"].max()),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return {"sample": sample_path, "provenance": provenance_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--rows", type=int, default=500)
    args = parser.parse_args()

    outputs = build_real_fixtures(args.source, args.output_dir, args.rows)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
