from __future__ import annotations

from pitcher_twin.model_router import build_model_route


def _route_report() -> dict[str, object]:
    return {
        "pitcher_name": "Skubal, Tarik",
        "pitch_type": "CH",
        "target_auc": 0.60,
        "target_pass_rate": 0.80,
        "best_by_layer": {
            "command_representation": "factorized_pitch_family_release_spin",
            "movement_only": "factorized_trend_state_anchored",
            "release_only": "factorized_pitch_family_release_spin",
            "physics_core": "factorized_pitch_family_release_spin",
        },
        "layer_results": {
            "command_representation": {
                "factorized_pitch_family_release_spin": {
                    "mean_auc": 0.531,
                    "pass_rate": 1.0,
                    "top_leakage_features": [{"feature": "plate_x", "importance": 0.2}],
                }
            },
            "movement_only": {
                "factorized_trend_state_anchored": {
                    "mean_auc": 0.548,
                    "pass_rate": 1.0,
                    "top_leakage_features": [{"feature": "pfx_x", "importance": 0.4}],
                }
            },
            "release_only": {
                "factorized_pitch_family_release_spin": {
                    "mean_auc": 0.586,
                    "pass_rate": 0.67,
                    "top_leakage_features": [
                        {"feature": "spin_axis_sin", "importance": 0.8}
                    ],
                }
            },
            "physics_core": {
                "factorized_pitch_family_release_spin": {
                    "mean_auc": 0.641,
                    "pass_rate": 0.0,
                    "top_leakage_features": [
                        {"feature": "release_pos_y", "importance": 0.6}
                    ],
                }
            },
        },
    }


def test_build_model_route_groups_layers_by_trust_status() -> None:
    route = build_model_route(_route_report())

    assert route["pitcher_name"] == "Skubal, Tarik"
    assert route["pitch_type"] == "CH"
    assert route["pitch_family"] == "changeup"
    assert route["route_status"] == "diagnostic"
    assert route["recommended_physics_model"] == "factorized_pitch_family_release_spin"
    assert route["validated_feature_groups"] == [
        "command_representation",
        "movement_only",
    ]
    assert route["candidate_feature_groups"] == ["release_only"]
    assert route["diagnostic_feature_groups"] == ["physics_core"]
    assert route["layer_routes"]["release_only"]["status"] == "candidate"
    assert route["layer_routes"]["physics_core"]["top_leakage_features"][0]["feature"] == (
        "release_pos_y"
    )


def test_build_model_route_marks_validated_physics_when_auc_and_pass_rate_clear() -> None:
    report = _route_report()
    report["layer_results"]["physics_core"]["factorized_pitch_family_release_spin"][
        "mean_auc"
    ] = 0.552
    report["layer_results"]["physics_core"]["factorized_pitch_family_release_spin"][
        "pass_rate"
    ] = 1.0

    route = build_model_route(report)

    assert route["route_status"] == "validated"
    assert "physics_core" in route["validated_feature_groups"]
