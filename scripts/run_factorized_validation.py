#!/usr/bin/env python3
"""Run real-data factorized physics validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.factorized import validate_factorized_physics  # noqa: E402
from pitcher_twin.features import clean_pitch_features  # noqa: E402
from pitcher_twin.validator import temporal_train_holdout  # noqa: E402


def _pitcher_name_from_subset(subset) -> str:
    for column in ("pitcher_name", "player_name"):
        if column in subset.columns:
            names = subset[column].dropna()
            if not names.empty:
                return str(names.iloc[0])
    return str(subset["pitcher"].dropna().iloc[0])


def run(data_path: Path, output_dir: Path, pitcher_id: int, pitch_type: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_statcast_cache(data_path)
    clean = clean_pitch_features(raw, pitch_types=None)
    subset = clean[(clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == pitch_type)].copy()
    if subset.empty:
        raise RuntimeError(f"No rows found for pitcher={pitcher_id}, pitch_type={pitch_type}.")

    pitcher_name = _pitcher_name_from_subset(subset)
    train, holdout = temporal_train_holdout(subset, train_fraction=0.70)
    report = validate_factorized_physics(
        train,
        holdout,
        clean,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        n_samples=max(300, len(holdout)),
        random_state=42,
    )
    report.update(
        {
            "data_path": str(data_path),
            "pitcher_id": int(pitcher_id),
            "rows_raw": int(len(raw)),
            "rows_clean": int(len(clean)),
            "rows_subset": int(len(subset)),
        }
    )

    report_path = output_dir / "factorized_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary_lines = [
        "# Factorized Physics Validation",
        "",
        f"- Pitcher: `{pitcher_name}`",
        f"- Pitch type: `{pitch_type}`",
        f"- Train rows: `{report['n_train']}`",
        f"- Holdout rows: `{report['n_holdout']}`",
        "",
        "| Layer | Factorized | Game-Drift Gaussian | Game-Drift Copula |",
        "|---|---:|---:|---:|",
    ]
    for layer, row in report["layer_results"].items():
        summary_lines.append(
            f"| {layer} | {row['factorized_auc']:.3f} | "
            f"{row['game_drift_gaussian_auc']:.3f} | {row['game_drift_copula_auc']:.3f} |"
        )
    summary_path = output_dir / "factorized_validation_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return {"report": str(report_path), "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/factorized_skubal_2025_ff"))
    parser.add_argument("--pitcher-id", type=int, default=669373)
    parser.add_argument("--pitch-type", type=str, default="FF")
    args = parser.parse_args()
    outputs = run(args.data, args.output_dir, args.pitcher_id, args.pitch_type)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
