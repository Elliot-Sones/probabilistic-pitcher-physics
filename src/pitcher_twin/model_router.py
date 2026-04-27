"""Model-routing summaries for Pitcher Twin validation reports."""

from __future__ import annotations

from typing import Any

from pitcher_twin.tournament import pitch_family_for_pitch_type


def build_model_route(report: dict[str, Any]) -> dict[str, Any]:
    """Choose trusted/candidate/diagnostic model routes from a tournament report."""
    target_auc = float(report.get("target_auc", 0.60))
    target_pass_rate = float(report.get("target_pass_rate", 0.80))
    best_by_layer = dict(report.get("best_by_layer", {}))
    layer_results = dict(report.get("layer_results", {}))
    layer_routes: dict[str, dict[str, Any]] = {}
    validated: list[str] = []
    candidate: list[str] = []
    diagnostic: list[str] = []

    for layer, model_name in best_by_layer.items():
        metrics = dict(layer_results.get(layer, {}).get(model_name, {}))
        mean_auc = float(metrics.get("mean_auc", float("nan")))
        pass_rate = float(metrics.get("pass_rate", 0.0))
        status = _route_status(mean_auc, pass_rate, target_auc, target_pass_rate)
        if status == "validated":
            validated.append(layer)
        elif status == "candidate":
            candidate.append(layer)
        else:
            diagnostic.append(layer)
        layer_routes[layer] = {
            "feature_group": layer,
            "status": status,
            "model": model_name,
            "mean_auc": mean_auc,
            "pass_rate": pass_rate,
            "top_leakage_features": list(metrics.get("top_leakage_features", [])),
        }

    physics_route = layer_routes.get("physics_core", {})
    route_status = str(physics_route.get("status", "diagnostic"))
    return {
        "pitcher_name": report.get("pitcher_name", "unknown"),
        "pitch_type": report.get("pitch_type", "unknown"),
        "pitch_family": pitch_family_for_pitch_type(str(report.get("pitch_type", ""))),
        "target_auc": target_auc,
        "target_pass_rate": target_pass_rate,
        "route_status": route_status,
        "recommended_physics_model": physics_route.get("model", "unknown"),
        "validated_feature_groups": validated,
        "candidate_feature_groups": candidate,
        "diagnostic_feature_groups": diagnostic,
        "layer_routes": layer_routes,
    }


def _route_status(
    mean_auc: float,
    pass_rate: float,
    target_auc: float,
    target_pass_rate: float,
) -> str:
    if mean_auc <= target_auc and pass_rate >= target_pass_rate:
        return "validated"
    if mean_auc <= target_auc:
        return "candidate"
    return "diagnostic"
