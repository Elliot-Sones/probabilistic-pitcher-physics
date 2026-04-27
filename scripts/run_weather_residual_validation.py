#!/usr/bin/env python3
"""Run V2.2 weather residual validation against the V2.1 factorized baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.factorized import (  # noqa: E402
    VALIDATION_LAYERS,
    fit_factorized_physics_model,
    sample_factorized_physics,
)
from pitcher_twin.features import FEATURE_GROUPS, clean_pitch_features  # noqa: E402
from pitcher_twin.validator import classifier_two_sample_test, temporal_train_holdout  # noqa: E402
from pitcher_twin.weather import (  # noqa: E402
    WEATHER_FEATURE_COLUMNS,
    join_weather_by_game_pitch_rows,
    read_weather_cache,
)


def _pitcher_name_from_subset(subset: pd.DataFrame) -> str:
    for column in ("pitcher_name", "player_name"):
        if column in subset.columns:
            names = subset[column].dropna()
            if not names.empty:
                return str(names.iloc[0])
    return str(subset["pitcher"].dropna().iloc[0])


def _assert_weather_coverage(subset: pd.DataFrame) -> None:
    missing_columns = [column for column in WEATHER_FEATURE_COLUMNS if column not in subset.columns]
    if missing_columns:
        raise RuntimeError(f"Weather cache is missing columns: {', '.join(missing_columns)}")
    coverage = subset[WEATHER_FEATURE_COLUMNS].notna().all(axis=1).mean()
    if coverage < 1.0:
        missing_rows = int((~subset[WEATHER_FEATURE_COLUMNS].notna().all(axis=1)).sum())
        raise RuntimeError(
            f"Weather coverage is incomplete for selected rows: {missing_rows} rows missing."
        )


def _evaluate_layers(
    holdout: pd.DataFrame,
    baseline_samples: pd.DataFrame,
    weather_samples: pd.DataFrame,
    random_state: int,
) -> dict[str, dict[str, object]]:
    layer_results: dict[str, dict[str, object]] = {}
    for index, feature_group in enumerate(VALIDATION_LAYERS):
        columns = FEATURE_GROUPS[feature_group]
        baseline_metrics = classifier_two_sample_test(
            holdout,
            baseline_samples,
            columns,
            random_state=random_state + index * 10,
        )
        weather_metrics = classifier_two_sample_test(
            holdout,
            weather_samples,
            columns,
            random_state=random_state + index * 10 + 1,
        )
        baseline_auc = float(baseline_metrics["auc"])
        weather_auc = float(weather_metrics["auc"])
        layer_results[feature_group] = {
            "features": columns,
            "baseline_auc": baseline_auc,
            "weather_auc": weather_auc,
            "delta_auc": float(weather_auc - baseline_auc),
            "weather_improved": bool(weather_auc < baseline_auc),
            "baseline_top_leakage": baseline_metrics["top_leakage_features"],
            "weather_top_leakage": weather_metrics["top_leakage_features"],
        }
    return layer_results


def _aggregate_repeated_layer_results(
    repeated_results: list[dict[str, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    if not repeated_results:
        raise ValueError("At least one repeated result is required.")
    layers = repeated_results[0].keys()
    summary: dict[str, dict[str, object]] = {}
    for layer in layers:
        baseline_values = np.asarray(
            [float(result[layer]["baseline_auc"]) for result in repeated_results],
            dtype=float,
        )
        weather_values = np.asarray(
            [float(result[layer]["weather_auc"]) for result in repeated_results],
            dtype=float,
        )
        baseline_mean = float(baseline_values.mean())
        weather_mean = float(weather_values.mean())
        summary[layer] = {
            "baseline_auc": baseline_mean,
            "weather_auc": weather_mean,
            "delta_auc": float(weather_mean - baseline_mean),
            "baseline_std_auc": float(baseline_values.std(ddof=0)),
            "weather_std_auc": float(weather_values.std(ddof=0)),
            "weather_improved": bool(weather_mean < baseline_mean),
            "repeat_count": int(len(repeated_results)),
        }
    return summary


def run(
    data_path: Path,
    weather_cache: Path,
    output_dir: Path,
    pitcher_id: int,
    pitch_type: str,
    n_samples: int = 300,
    random_state: int = 42,
    repeats: int = 1,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_statcast_cache(data_path)
    clean = clean_pitch_features(raw, pitch_types=None)
    weather = read_weather_cache(weather_cache)
    clean_weather = join_weather_by_game_pitch_rows(clean, weather)
    subset = clean_weather[
        (clean_weather["pitcher"] == pitcher_id) & (clean_weather["pitch_type"] == pitch_type)
    ].copy()
    if subset.empty:
        raise RuntimeError(f"No rows found for pitcher={pitcher_id}, pitch_type={pitch_type}.")
    _assert_weather_coverage(subset)

    pitcher_name = _pitcher_name_from_subset(subset)
    train, holdout = temporal_train_holdout(subset, train_fraction=0.70)
    model = fit_factorized_physics_model(
        train,
        clean_weather,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        random_state=random_state,
        weather_feature_columns=WEATHER_FEATURE_COLUMNS,
    )
    if model.weather_residual_adjustment is None:
        raise RuntimeError("Weather residual adjustment was unavailable after fitting.")

    sample_count = max(n_samples, len(holdout))
    repeated_layer_results = []
    for repeat_index in range(repeats):
        repeat_seed = random_state + repeat_index * 1000
        baseline_samples = sample_factorized_physics(
            model,
            n=sample_count,
            context_df=holdout,
            random_state=repeat_seed + 10,
            use_weather=False,
        )
        weather_samples = sample_factorized_physics(
            model,
            n=sample_count,
            context_df=holdout,
            random_state=repeat_seed + 10,
            use_weather=True,
        )
        repeated_layer_results.append(
            _evaluate_layers(
                holdout,
                baseline_samples,
                weather_samples,
                random_state=repeat_seed + 100,
            )
        )
    layer_results = _aggregate_repeated_layer_results(repeated_layer_results)
    report = {
        "model_name": "player_factorized_weather_residual_ablation",
        "pitcher_name": pitcher_name,
        "pitch_type": pitch_type,
        "pitcher_id": int(pitcher_id),
        "data_path": str(data_path),
        "weather_cache": str(weather_cache),
        "rows_raw": int(len(raw)),
        "rows_subset": int(len(subset)),
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "weather_feature_columns": WEATHER_FEATURE_COLUMNS,
        "weather_adjustment_source_row_count": int(
            model.weather_residual_adjustment["source_row_count"]
        ),
        "repeat_count": int(repeats),
        "acceptance_rule": {
            "target_physics_core_auc": 0.600,
            "weather_must_improve_baseline": True,
        },
        "layer_results": layer_results,
        "repeat_results": repeated_layer_results,
    }
    report_path = output_dir / "weather_residual_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    summary_lines = [
        "# Weather Residual Validation",
        "",
        f"- Pitcher: `{pitcher_name}`",
        f"- Pitch type: `{pitch_type}`",
        f"- Train rows: `{report['n_train']}`",
        f"- Holdout rows: `{report['n_holdout']}`",
        f"- Weather rows used for fit: `{report['weather_adjustment_source_row_count']}`",
        "",
        "| Layer | V2.1 Baseline Mean | V2.2 Weather Mean | Delta |",
        "|---|---:|---:|---:|",
    ]
    for layer, row in layer_results.items():
        summary_lines.append(
            f"| {layer} | {row['baseline_auc']:.3f} | "
            f"{row['weather_auc']:.3f} | {row['delta_auc']:+.3f} |"
        )
    summary_path = output_dir / "weather_residual_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return {"report": str(report_path), "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weather-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/weather_residual_skubal_2025_ff"))
    parser.add_argument("--pitcher-id", type=int, default=669373)
    parser.add_argument("--pitch-type", type=str, default="FF")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    outputs = run(
        args.data,
        args.weather_cache,
        args.output_dir,
        args.pitcher_id,
        args.pitch_type,
        n_samples=args.samples,
        repeats=args.repeats,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
