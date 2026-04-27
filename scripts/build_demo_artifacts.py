#!/usr/bin/env python3
"""Build real Pitcher Twin demo artifacts from a Statcast cache."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.candidates import CandidateThresholds, rank_pitcher_pitch_candidates  # noqa: E402
from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.features import FEATURE_GROUPS, clean_pitch_features  # noqa: E402
from pitcher_twin.models import fit_generator_suite  # noqa: E402
from pitcher_twin.sampler import sample_pitch_session  # noqa: E402
from pitcher_twin.trajekt_format import to_trajekt_json, write_trajekt_json  # noqa: E402
from pitcher_twin.validator import classifier_two_sample_test, temporal_train_holdout  # noqa: E402


MODEL_SELECTION_PRIORITY = [
    "player_recent_weighted_game_drift_copula",
    "player_recent_weighted_game_drift_gaussian",
    "player_context_weighted_gaussian",
    "player_recent_multivariate_gaussian",
    "player_multivariate_gaussian",
    "player_gmm",
    "player_recent_game_window_empirical",
    "player_recent_empirical_bootstrap",
    "player_empirical_bootstrap",
    "random_independent_noise",
    "league_same_pitch_empirical",
]

ROBUST_VALIDATION_PASS_RATE = 0.80


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _morning_report(report: dict) -> str:
    candidate = report["selected_candidate"]
    lines = [
        "# Pitcher Twin Real Demo Report",
        "",
        "This report uses real public Statcast rows only.",
        "",
        "## Selected Candidate",
        "",
        f"- Pitcher: `{candidate['pitcher_name']}`",
        f"- Pitch type: `{candidate['pitch_type']}`",
        f"- Real pitches in cache: `{candidate['n']}`",
        f"- Games: `{candidate['games']}`",
        f"- Holdout rows: `{candidate['holdout_n']}`",
        "",
        "## Model Validation",
        "",
        f"- Artifact status: `{report['artifact_status']}`",
        f"- Selected feature group: `{report['selected_feature_group']}`",
        f"- Selected model: `{report['selected_model']}`",
        f"- Selected temporal C2ST AUC: `{report['selected_auc']:.3f}`",
        f"- Physics-core temporal C2ST AUC: `{report['physics_temporal_auc']:.3f}`",
        f"- Temporal success target: `<= {report['validation_thresholds']['temporal_success_auc']:.2f}`",
        "- C2ST classifier split: held-out stratified classifier rows",
        f"- Export strategy: `{report['export']['export_strategy']}`",
        f"- Export physics model: `{report['export'].get('export_physics_model', 'n/a')}`",
        "- Scope: real-data proof-of-concept validation",
        f"- Model selection policy: {report['validation_thresholds']['model_selection']['policy']}",
        "",
        "| Feature group | Selected model | Detectability C2ST AUC | Minimum C2ST AUC |",
        "|---|---|---:|---:|",
    ]
    for feature_group, row in sorted(report["feature_group_results"].items()):
        lines.append(
            f"| {feature_group} | {row['best_model']} | {row['best_auc']:.3f} | {row['minimum_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Detectability AUC is folded around `0.50`; lower is better and `0.50` means "
            "a held-out classifier cannot tell generated pitches from held-out real pitches.",
            "",
            "## Layer Status",
            "",
            "| Status | Feature group | Model | AUC |",
            "|---|---|---|---:|",
        ]
    )
    for layer in report["validated_layers"]:
        lines.append(
            f"| validated | {layer['feature_group']} | {layer['model']} | {layer['auc']:.3f} |"
        )
    for layer in report["borderline_layers"]:
        lines.append(
            f"| borderline | {layer['feature_group']} | {layer['model']} | {layer['auc']:.3f} |"
        )
    for layer in report["diagnostic_layers"]:
        lines.append(
            f"| diagnostic | {layer['feature_group']} | {layer['model']} | {layer['auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Robustness Checks",
            "",
            f"Repeated over `{report['robustness_checks']['seed_count']}` sample/classifier seeds.",
            "",
            "| Feature group | Model | Mean AUC | Std | Pass rate <= target |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for feature_group, row in report["robustness_checks"]["results"].items():
        lines.append(
            f"| {feature_group} | {row['model']} | {row['mean_auc']:.3f} | "
            f"{row['std_auc']:.3f} | {row['pass_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Generator",
            "",
            f"- `{report['selected_model']}` on `{report['selected_feature_group']}`",
            "",
            "## Caveat",
            "",
            "Validated means repeated-seed mean AUC is at or below target and pass rate is high. "
            "Borderline layers pass a single split or mean threshold but are not stable enough yet. "
            "Full physics remains diagnostic until `physics_core` reaches the configured temporal target.",
            "",
        ]
    )
    return "\n".join(lines)


def _model_priority(model_name: str) -> int:
    if model_name in MODEL_SELECTION_PRIORITY:
        return MODEL_SELECTION_PRIORITY.index(model_name)
    return len(MODEL_SELECTION_PRIORITY)


def _select_model_within_tolerance(
    model_results: dict[str, dict[str, object]],
    tolerance: float = 0.05,
) -> tuple[str, float, str, float]:
    minimum_model = min(model_results, key=lambda name: model_results[name]["auc"])
    minimum_auc = float(model_results[minimum_model]["auc"])
    eligible = [
        (model_name, float(metrics["auc"]))
        for model_name, metrics in model_results.items()
        if float(metrics["auc"]) <= minimum_auc + tolerance
    ]
    selected_model, selected_auc = sorted(
        eligible,
        key=lambda item: (_model_priority(item[0]), item[1]),
    )[0]
    return selected_model, selected_auc, minimum_model, minimum_auc


def _select_global_model(
    feature_group_results: dict[str, dict[str, object]],
    tolerance: float = 0.05,
) -> tuple[str, str, float]:
    minimum_auc = min(row["minimum_auc"] for row in feature_group_results.values())
    eligible = []
    for feature_group, row in feature_group_results.items():
        for model_name, metrics in row["model_results"].items():
            auc = float(metrics["auc"])
            if auc <= minimum_auc + tolerance:
                eligible.append(
                    (
                        feature_group,
                        model_name,
                        auc,
                        _model_priority(model_name),
                        int(row["feature_count"]),
                    )
                )
    feature_group, model_name, auc, _, _ = sorted(
        eligible,
        key=lambda item: (item[3], item[2], item[4], item[0]),
    )[0]
    return feature_group, model_name, auc


def _artifact_status(
    selected_feature_group: str,
    selected_auc: float,
    physics_auc: float | None,
    temporal_success_auc: float,
    validated_layer_count: int = 0,
) -> str:
    selected_valid = selected_auc <= temporal_success_auc
    physics_valid = physics_auc is not None and physics_auc <= temporal_success_auc
    if selected_valid and physics_valid:
        return "validated_full_temporal_success"
    if selected_valid and validated_layer_count >= 2:
        return "validated_component_layers_physics_diagnostic"
    if selected_valid and selected_feature_group == "command_representation":
        return "validated_command_layer_physics_diagnostic"
    if selected_valid:
        return "partial_temporal_success"
    return "diagnostic_not_final"


def _layer_row(feature_group: str, row: dict[str, object]) -> dict[str, object]:
    return {
        "feature_group": feature_group,
        "features": FEATURE_GROUPS[feature_group],
        "model": row["best_model"],
        "auc": row["best_auc"],
        "minimum_auc": row["minimum_auc"],
    }


def _layer_report(
    feature_group_results: dict[str, dict[str, object]],
    temporal_success_auc: float,
    selected_feature_group: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validated = []
    diagnostic = []
    for feature_group, row in feature_group_results.items():
        layer = {
            "feature_group": feature_group,
            "features": FEATURE_GROUPS[feature_group],
            "model": row["best_model"],
            "auc": row["best_auc"],
            "minimum_auc": row["minimum_auc"],
        }
        if float(row["best_auc"]) <= temporal_success_auc:
            validated.append(layer)
        else:
            diagnostic.append(layer)

    validated.sort(key=lambda layer: layer["feature_group"] != selected_feature_group)
    diagnostic.sort(key=lambda layer: layer["feature_group"] != "physics_core")
    return validated, diagnostic


def _classify_layers_by_robustness(
    feature_group_results: dict[str, dict[str, object]],
    robustness_checks: dict[str, object],
    temporal_success_auc: float,
    robust_pass_rate: float = ROBUST_VALIDATION_PASS_RATE,
) -> dict[str, list[dict[str, object]]]:
    robust_results = robustness_checks.get("results", {})
    groups = sorted(
        feature_group_results,
        key=lambda group: (group != "physics_core", group),
    )
    classified = {"validated": [], "borderline": [], "diagnostic": []}
    for feature_group in groups:
        row = feature_group_results[feature_group]
        layer = _layer_row(feature_group, row)
        robust = robust_results.get(feature_group)
        single_pass = float(row["best_auc"]) <= temporal_success_auc
        if robust:
            robust_mean = float(robust["mean_auc"])
            robust_pass = float(robust["pass_rate"])
            layer["robust_mean_auc"] = robust_mean
            layer["robust_pass_rate"] = robust_pass
            if robust_mean <= temporal_success_auc and robust_pass >= robust_pass_rate:
                classified["validated"].append(layer)
            elif single_pass or robust_mean <= temporal_success_auc:
                classified["borderline"].append(layer)
            else:
                classified["diagnostic"].append(layer)
        elif single_pass:
            classified["borderline"].append(layer)
        else:
            classified["diagnostic"].append(layer)

    for status in classified:
        classified[status].sort(
            key=lambda layer: (
                layer["feature_group"] != "command_representation",
                layer["feature_group"] != "movement_only",
                layer["feature_group"] != "trajectory_only",
                layer["feature_group"] != "release_only",
                layer["feature_group"] != "physics_core",
                layer["feature_group"],
            )
        )
    return classified


def _summarize_aucs(aucs: list[float], temporal_success_auc: float) -> dict[str, object]:
    return {
        "mean_auc": mean(aucs),
        "std_auc": stdev(aucs) if len(aucs) > 1 else 0.0,
        "min_auc": min(aucs),
        "max_auc": max(aucs),
        "pass_rate": sum(auc <= temporal_success_auc for auc in aucs) / len(aucs),
        "aucs": aucs,
    }


def _robustness_checks(
    fitted_suites: dict[str, dict[str, object]],
    feature_group_results: dict[str, dict[str, object]],
    selected_feature_group: str,
    selected_model: str,
    holdout,
    n_samples: int,
    temporal_success_auc: float,
    seed_count: int,
) -> dict[str, object]:
    groups = [
        selected_feature_group,
        "physics_core",
        "release_only",
        "movement_only",
        "trajectory_only",
    ]
    unique_groups = []
    for group in groups:
        if group in fitted_suites and group not in unique_groups:
            unique_groups.append(group)

    results = {}
    for group in unique_groups:
        model_name = (
            selected_model
            if group == selected_feature_group
            else feature_group_results[group]["best_model"]
        )
        model = fitted_suites[group][model_name]
        aucs = []
        for offset in range(seed_count):
            seed = 1000 + offset
            samples = sample_pitch_session(
                model,
                n=max(n_samples, len(holdout)),
                random_state=seed,
                context_df=holdout,
            )
            metrics = classifier_two_sample_test(
                holdout,
                samples,
                model.feature_columns,
                random_state=5000 + offset,
            )
            aucs.append(float(metrics["auc"]))
        results[group] = {
            "model": model_name,
            **_summarize_aucs(aucs, temporal_success_auc),
        }
    return {
        "seed_count": seed_count,
        "temporal_success_auc": temporal_success_auc,
        "results": results,
    }


def _build_export_session(
    fitted_suites: dict[str, dict[str, object]],
    feature_group_results: dict[str, dict[str, object]],
    selected_feature_group: str,
    selected_model: str,
    holdout,
    n_samples: int,
):
    selected_session = sample_pitch_session(
        fitted_suites[selected_feature_group][selected_model],
        n=n_samples,
        random_state=99,
        context_df=holdout,
    )
    if "physics_core" not in fitted_suites:
        return selected_session, {
            "export_strategy": "selected_layer_only",
            "export_selected_feature_group": selected_feature_group,
            "export_selected_model": selected_model,
        }

    physics_model = feature_group_results["physics_core"]["best_model"]
    physics_session = sample_pitch_session(
        fitted_suites["physics_core"][physics_model],
        n=n_samples,
        random_state=101,
        context_df=holdout,
    )
    for column in FEATURE_GROUPS[selected_feature_group]:
        if column in selected_session.columns:
            physics_session[column] = selected_session[column].to_numpy()
    physics_session["model_name"] = f"{selected_model}_over_{physics_model}"
    physics_session["feature_group"] = f"{selected_feature_group}_over_physics_core"
    return physics_session, {
        "export_strategy": "validated_layer_over_physics_core",
        "export_selected_feature_group": selected_feature_group,
        "export_selected_model": selected_model,
        "export_physics_feature_group": "physics_core",
        "export_physics_model": physics_model,
        "export_physics_auc": feature_group_results["physics_core"]["best_auc"],
        "physics_temporal_auc": feature_group_results["physics_core"]["best_auc"],
    }


def build_real_demo_artifacts(
    data_path: Path,
    output_dir: Path = Path("outputs/real_demo"),
    min_pitches: int = 600,
    min_holdout: int = 150,
    min_games: int = 4,
    n_samples: int = 15,
    temporal_success_auc: float = 0.60,
    robustness_seeds: int = 30,
    target_pitcher_id: int | None = None,
    target_pitch_type: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_statcast_cache(data_path)
    clean = clean_pitch_features(raw, pitch_types=None)
    candidate_pool = clean
    if target_pitcher_id is not None:
        candidate_pool = candidate_pool[candidate_pool["pitcher"] == target_pitcher_id]
    if target_pitch_type is not None:
        candidate_pool = candidate_pool[candidate_pool["pitch_type"] == target_pitch_type]
    candidate_thresholds = CandidateThresholds(
        min_pitches=min_pitches,
        min_holdout=min_holdout,
        min_games=min_games,
    )
    ranking = rank_pitcher_pitch_candidates(
        candidate_pool,
        thresholds=candidate_thresholds,
    )
    if ranking.empty:
        raise RuntimeError("No viable real pitcher/pitch candidates found in the provided cache.")

    candidate_rankings = output_dir / "candidate_rankings.csv"
    ranking.to_csv(candidate_rankings, index=False)
    candidate = ranking.iloc[0].to_dict()
    subset = clean[
        (clean["pitcher"] == candidate["pitcher"])
        & (clean["pitch_type"] == candidate["pitch_type"])
    ].copy()
    train, holdout = temporal_train_holdout(subset, train_fraction=0.7)
    feature_group_results: dict[str, dict[str, object]] = {}
    fitted_suites = {}
    for feature_group in FEATURE_GROUPS:
        try:
            suite = fit_generator_suite(
                train,
                clean,
                pitcher_name=candidate["pitcher_name"],
                pitch_type=candidate["pitch_type"],
                feature_group=feature_group,
            )
        except (KeyError, ValueError):
            continue
        model_results = {}
        for model_name, model in suite.items():
            samples = sample_pitch_session(
                model,
                n=max(n_samples, len(holdout)),
                random_state=42,
                context_df=holdout,
            )
            model_results[model_name] = classifier_two_sample_test(
                holdout,
                samples,
                model.feature_columns,
            )
        best_model, best_auc, minimum_model, minimum_auc = _select_model_within_tolerance(
            model_results
        )
        feature_group_results[feature_group] = {
            "best_model": best_model,
            "best_auc": best_auc,
            "minimum_auc_model": minimum_model,
            "minimum_auc": minimum_auc,
            "model_results": model_results,
            "feature_count": len(FEATURE_GROUPS[feature_group]),
        }
        fitted_suites[feature_group] = suite

    selected_feature_group, selected_model, selected_auc = _select_global_model(
        feature_group_results
    )
    physics_auc = (
        float(feature_group_results["physics_core"]["best_auc"])
        if "physics_core" in feature_group_results
        else None
    )
    robustness_checks = _robustness_checks(
        fitted_suites,
        feature_group_results,
        selected_feature_group,
        selected_model,
        holdout,
        n_samples,
        temporal_success_auc,
        robustness_seeds,
    )
    official_layers = _classify_layers_by_robustness(
        feature_group_results,
        robustness_checks,
        temporal_success_auc,
        ROBUST_VALIDATION_PASS_RATE,
    )
    validated_layers = official_layers["validated"]
    borderline_layers = official_layers["borderline"]
    diagnostic_layers = official_layers["diagnostic"]
    artifact_status = _artifact_status(
        selected_feature_group,
        selected_auc,
        physics_auc,
        temporal_success_auc,
        len(validated_layers),
    )
    final_session, export_metadata = _build_export_session(
        fitted_suites,
        feature_group_results,
        selected_feature_group,
        selected_model,
        holdout,
        n_samples,
    )
    payload = to_trajekt_json(
        final_session,
        pitcher=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        metadata={
            "data_path": str(data_path),
            "selected_model": selected_model,
            "selected_feature_group": selected_feature_group,
            "artifact_status": artifact_status,
            "validated_feature_groups": [layer["feature_group"] for layer in validated_layers],
            "borderline_feature_groups": [
                layer["feature_group"] for layer in borderline_layers
            ],
            "diagnostic_feature_groups": [
                layer["feature_group"] for layer in diagnostic_layers
            ],
            "physics_temporal_auc": physics_auc,
            "source_rows": int(len(raw)),
            **export_metadata,
        },
    )

    validation_report = output_dir / "validation_report.json"
    session_json = output_dir / "final_session.json"
    morning_report = output_dir / "morning_report.md"
    report = {
        "data_path": str(data_path),
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "selected_candidate": candidate,
        "selected_feature_group": selected_feature_group,
        "selected_model": selected_model,
        "selected_auc": selected_auc,
        "physics_temporal_auc": physics_auc,
        "artifact_status": artifact_status,
        "validated_layers": validated_layers,
        "borderline_layers": borderline_layers,
        "diagnostic_layers": diagnostic_layers,
        "robustness_checks": robustness_checks,
        "export": export_metadata,
        "validation_thresholds": {
            "candidate_thresholds": asdict(candidate_thresholds),
            "temporal_success_auc": float(temporal_success_auc),
            "classifier_split": {
                "strategy": "stratified_holdout",
                "test_fraction": 0.30,
            },
            "official_layer_validation": {
                "mean_auc_max": float(temporal_success_auc),
                "pass_rate_min": ROBUST_VALIDATION_PASS_RATE,
            },
            "model_selection": {
                "policy": "prefer contextual/parametric models within 0.05 C2ST AUC of the minimum",
                "tolerance": 0.05,
                "priority": MODEL_SELECTION_PRIORITY,
            },
        },
        "feature_group_results": feature_group_results,
        "model_results": feature_group_results[selected_feature_group]["model_results"],
    }
    _write_json(validation_report, report)
    write_trajekt_json(payload, session_json)
    morning_report.write_text(_morning_report(report))
    return {
        "candidate_rankings": candidate_rankings,
        "validation_report": validation_report,
        "session_json": session_json,
        "morning_report": morning_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_demo"))
    parser.add_argument("--min-pitches", type=int, default=600)
    parser.add_argument("--min-holdout", type=int, default=150)
    parser.add_argument("--min-games", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--temporal-success-auc", type=float, default=0.60)
    parser.add_argument("--robustness-seeds", type=int, default=30)
    parser.add_argument("--target-pitcher-id", type=int, default=None)
    parser.add_argument("--target-pitch-type", type=str, default=None)
    args = parser.parse_args()
    outputs = build_real_demo_artifacts(
        args.data,
        args.output_dir,
        args.min_pitches,
        args.min_holdout,
        args.min_games,
        args.samples,
        args.temporal_success_auc,
        args.robustness_seeds,
        args.target_pitcher_id,
        args.target_pitch_type,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
