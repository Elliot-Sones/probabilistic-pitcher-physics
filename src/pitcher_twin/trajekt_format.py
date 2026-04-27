"""Trajekt-shaped JSON export utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _float(row: pd.Series, column: str) -> float | None:
    if column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def to_trajekt_json(
    samples: pd.DataFrame,
    pitcher: str,
    pitch_type: str,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    pitches = []
    for row in samples.to_dict(orient="records"):
        series = pd.Series(row)
        pitches.append(
            {
                "index": int(row["sample_index"]),
                "source": row.get("source", "simulated_from_real_model"),
                "release": {
                    "pos_x": _float(series, "release_pos_x"),
                    "pos_y": _float(series, "release_pos_y"),
                    "pos_z": _float(series, "release_pos_z"),
                    "extension": _float(series, "release_extension"),
                },
                "velocity": {
                    "release_speed": _float(series, "release_speed"),
                    "vx0": _float(series, "vx0"),
                    "vy0": _float(series, "vy0"),
                    "vz0": _float(series, "vz0"),
                },
                "spin": {
                    "rate": _float(series, "release_spin_rate"),
                    "axis_cos": _float(series, "spin_axis_cos"),
                    "axis_sin": _float(series, "spin_axis_sin"),
                },
                "movement": {
                    "pfx_x": _float(series, "pfx_x"),
                    "pfx_z": _float(series, "pfx_z"),
                    "ax": _float(series, "ax"),
                    "ay": _float(series, "ay"),
                    "az": _float(series, "az"),
                },
                "plate_target": {
                    "x": _float(series, "plate_x"),
                    "z": _float(series, "plate_z"),
                },
                "model_metadata": {
                    "model_name": row.get("model_name"),
                    "feature_group": row.get("feature_group"),
                },
            }
        )
    return {
        "schema_version": "pitcher-twin.real.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pitcher": pitcher,
        "pitch_type": pitch_type,
        "metadata": metadata or {},
        "pitches": pitches,
    }


def write_trajekt_json(payload: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output
