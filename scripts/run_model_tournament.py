#!/usr/bin/env python3
"""Run a repeated model tournament against real temporal holdout data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.features import clean_pitch_features  # noqa: E402
from pitcher_twin.tournament import evaluate_model_tournament  # noqa: E402
from pitcher_twin.validator import temporal_train_holdout  # noqa: E402

SUMMARY_TITLE = "# Model Tournament"


def _pitcher_name_from_subset(subset) -> str:
    for column in ("pitcher_name", "player_name"):
        if column in subset.columns:
            names = subset[column].dropna()
            if not names.empty:
                return str(names.iloc[0])
    return str(subset["pitcher"].dropna().iloc[0])


def _write_summary(report: dict[str, object], output_dir: Path) -> Path:
    model_names = list(report["model_names"])
    header = "| Layer | Best model | " + " | ".join(model_names) + " |"
    separator = "|---|---|" + "---:|" * len(model_names)
    summary_lines = [
        SUMMARY_TITLE,
        "",
        f"- Pitcher: `{report['pitcher_name']}`",
        f"- Pitch type: `{report['pitch_type']}`",
        f"- Train rows: `{report['n_train']}`",
        f"- Holdout rows: `{report['n_holdout']}`",
        f"- Repeats: `{report['repeat_count']}`",
        f"- Target AUC: `<={report['target_auc']:.3f}`",
        f"- Target pass rate: `>={report['target_pass_rate']:.2f}`",
        f"- Candidate default: `{report['candidate_default']}`",
        f"- Best physics-core model: `{report['best_physics_core_model']}`",
        "",
        header,
        separator,
    ]
    for layer, rows in report["layer_results"].items():
        values = " | ".join(f"{rows[model]['mean_auc']:.3f}" for model in model_names)
        summary_lines.append(f"| {layer} | {report['best_by_layer'][layer]} | {values} |")
    summary_lines.extend(
        [
            "",
            "Lower C2ST AUC is better; `0.50` is ideal real-vs-generated indistinguishability.",
            "A model becomes a candidate default only if the same model clears the mean AUC and pass-rate targets for every layer.",
        ]
    )
    summary_path = output_dir / "model_tournament_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return summary_path


def run(
    data_path: Path,
    output_dir: Path,
    pitcher_id: int,
    pitch_type: str,
    n_samples: int = 300,
    repeats: int = 12,
    random_state: int = 42,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_statcast_cache(data_path)
    clean = clean_pitch_features(raw, pitch_types=None)
    subset = clean[(clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == pitch_type)].copy()
    if subset.empty:
        raise RuntimeError(f"No rows found for pitcher={pitcher_id}, pitch_type={pitch_type}.")

    pitcher_name = _pitcher_name_from_subset(subset)
    train, holdout = temporal_train_holdout(subset, train_fraction=0.70)
    report = evaluate_model_tournament(
        train,
        holdout,
        clean,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        n_samples=max(n_samples, len(holdout)),
        repeats=repeats,
        random_state=random_state,
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

    report_path = output_dir / "model_tournament_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary_path = _write_summary(report, output_dir)
    return {"report": str(report_path), "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_tournament_skubal_2025_ff"))
    parser.add_argument("--pitcher-id", type=int, default=669373)
    parser.add_argument("--pitch-type", type=str, default="FF")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    outputs = run(
        args.data,
        args.output_dir,
        args.pitcher_id,
        args.pitch_type,
        n_samples=args.samples,
        repeats=args.repeats,
        random_state=args.random_state,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
