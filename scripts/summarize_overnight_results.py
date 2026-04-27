#!/usr/bin/env python3
"""Summarize Pitcher Twin overnight validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_LABELS = {
    "random_independent_noise": "random independent noise",
    "league_same_pitch_empirical": "league same-pitch empirical",
    "player_empirical_bootstrap": "player empirical bootstrap",
    "player_multivariate_gaussian": "player multivariate Gaussian",
    "player_recent_empirical_bootstrap": "player recent empirical bootstrap",
    "player_recent_multivariate_gaussian": "player recent multivariate Gaussian",
    "player_context_weighted_gaussian": "player context-weighted Gaussian",
}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def aggregate_auc(report: dict) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for evaluation in report.get("evaluations", []):
        for model_name, metrics in evaluation.get("models", {}).items():
            auc = metrics.get("auc_mean")
            if isinstance(auc, (int, float)):
                values.setdefault(model_name, []).append(float(auc))
    return {model_name: mean(model_values) for model_name, model_values in values.items()}


def auc_table(title: str, report: dict | None) -> str:
    if not report:
        return f"## {title}\n\nMissing report.\n"

    rows = aggregate_auc(report)
    lines = [
        f"## {title}",
        "",
        "| Model | Mean detectability C2ST AUC |",
        "|---|---:|",
    ]
    for model_name, auc in sorted(rows.items()):
        label = MODEL_LABELS.get(model_name, model_name)
        lines.append(f"| {label} | {auc:.3f} |")
    lines.append("")
    return "\n".join(lines)


def candidate_table(report: dict | None) -> str:
    if not report:
        return "## Candidate Snapshot\n\nMissing candidate report.\n"

    lines = [
        "## Candidate Snapshot",
        "",
        "| Pitcher | Pitch | Real pitches | Games | Train | Holdout |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in report.get("candidates", [])[:10]:
        lines.append(
            "| {pitcher_name} | {pitch_type} | {n} | {games} | {train_n} | {holdout_n} |".format(
                **candidate
            )
        )
    lines.append("")
    return "\n".join(lines)


def conclusion(random_report: dict | None, temporal_report: dict | None) -> str:
    if not random_report or not temporal_report:
        return (
            "## Conclusion\n\n"
            "The overnight run did not produce both random and temporal split reports. "
            "Check `run_status.json` and `BLOCKED.md`.\n"
        )

    random_auc = aggregate_auc(random_report)
    temporal_auc = aggregate_auc(temporal_report)
    random_player = random_auc.get("player_multivariate_gaussian")
    temporal_player = temporal_auc.get("player_multivariate_gaussian")
    league_random = random_auc.get("league_same_pitch_empirical")

    lines = ["## Conclusion", ""]
    if random_player is not None and random_player <= 0.60:
        lines.append(
            "The random split supports the project: player-specific generated pitches "
            "are hard to distinguish from held-out real pitches "
            f"(`detectability AUC={random_player:.3f}`)."
        )
    elif random_player is not None:
        lines.append(
            "The random split is not yet strong enough: the player model remains "
            f"detectable (`detectability AUC={random_player:.3f}`)."
        )

    if league_random is not None and league_random >= 0.80:
        lines.append(
            "The league-average baseline is clearly detectable, which supports the claim "
            f"that pitcher-specific structure matters (`detectability AUC={league_random:.3f}`)."
        )

    if temporal_player is not None and temporal_player > 0.70:
        lines.append(
            "The temporal split is the current weakness: a static model trained on earlier "
            f"games does not match later games (`detectability AUC={temporal_player:.3f}`). The next "
            "work should model count, fatigue, game-level drift, and real weather/context."
        )
    elif temporal_player is not None:
        lines.append(
            "The temporal split is promising enough to continue toward the full contextual model "
            f"(`detectability AUC={temporal_player:.3f}`)."
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/overnight"))
    parser.add_argument("--output", type=Path, default=Path("outputs/overnight/morning_report.md"))
    args = parser.parse_args()

    input_dir = args.input_dir
    random_report = load_json(input_dir / "random_split_report.json")
    temporal_report = load_json(input_dir / "temporal_split_report.json")
    status = load_json(input_dir / "run_status.json")

    data_report = random_report or temporal_report or {}
    thresholds = data_report.get("validation_thresholds", {})
    candidate_thresholds = thresholds.get("candidate_thresholds", {})
    classifier_split = thresholds.get("classifier_split", {})
    lines = [
        "# Pitcher Twin Overnight Report",
        "",
        "This report summarizes real-data validation outputs. It does not include mock data.",
        "",
        "## Data",
        "",
        f"- Data path: `{data_report.get('data_path', 'unknown')}`",
        f"- Rows after cleaning: `{data_report.get('rows_clean', 'unknown')}`",
        f"- Date range: `{data_report.get('date_min', 'unknown')}` to `{data_report.get('date_max', 'unknown')}`",
        f"- Feature count: `{data_report.get('feature_count', 'unknown')}`",
        f"- Min train rows: `{candidate_thresholds.get('min_train', 'unknown')}`",
        f"- Min holdout rows: `{candidate_thresholds.get('min_holdout', 'unknown')}`",
        f"- Temporal success target: `<= {thresholds.get('temporal_success_auc', 'unknown')}`",
        f"- Classifier split: `{classifier_split.get('strategy', 'unknown')}`",
        "",
        candidate_table(data_report),
        auc_table("Random Split", random_report),
        auc_table("Temporal Split", temporal_report),
        conclusion(random_report, temporal_report),
    ]

    if status:
        lines.extend(
            [
                "## Run Status",
                "",
                f"- OK: `{status.get('ok')}`",
                f"- Started: `{status.get('started_at')}`",
                f"- Finished: `{status.get('finished_at')}`",
                "",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
