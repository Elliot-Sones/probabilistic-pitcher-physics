from __future__ import annotations

import json
import importlib.util
from pathlib import Path


REAL_SAMPLE = Path(__file__).parent / "fixtures" / "real_statcast_sample.csv"
SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_demo_artifacts.py"


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("build_demo_artifacts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_builder():
    return _load_builder_module().build_real_demo_artifacts


def test_artifact_status_does_not_claim_full_success_when_physics_fails() -> None:
    module = _load_builder_module()
    assert (
        module._artifact_status(
            selected_feature_group="command_representation",
            selected_auc=0.546,
            physics_auc=0.735,
            temporal_success_auc=0.60,
        )
        == "validated_command_layer_physics_diagnostic"
    )
    assert (
        module._artifact_status(
            selected_feature_group="command_representation",
            selected_auc=0.541,
            physics_auc=0.683,
            temporal_success_auc=0.60,
            validated_layer_count=2,
        )
        == "validated_component_layers_physics_diagnostic"
    )


def test_robustness_policy_splits_validated_borderline_and_diagnostic_layers() -> None:
    module = _load_builder_module()
    feature_group_results = {
        "command_representation": {"best_model": "m", "best_auc": 0.541, "minimum_auc": 0.541},
        "movement_only": {"best_model": "m", "best_auc": 0.546, "minimum_auc": 0.546},
        "trajectory_only": {"best_model": "m", "best_auc": 0.589, "minimum_auc": 0.589},
        "release_only": {"best_model": "m", "best_auc": 0.597, "minimum_auc": 0.597},
        "physics_core": {"best_model": "m", "best_auc": 0.683, "minimum_auc": 0.683},
    }
    robustness = {
        "results": {
            "command_representation": {"mean_auc": 0.539, "pass_rate": 0.97},
            "movement_only": {"mean_auc": 0.545, "pass_rate": 0.93},
            "trajectory_only": {"mean_auc": 0.592, "pass_rate": 0.57},
            "release_only": {"mean_auc": 0.631, "pass_rate": 0.17},
            "physics_core": {"mean_auc": 0.670, "pass_rate": 0.07},
        }
    }

    layers = module._classify_layers_by_robustness(
        feature_group_results,
        robustness,
        temporal_success_auc=0.60,
        robust_pass_rate=0.80,
    )

    assert [layer["feature_group"] for layer in layers["validated"]] == [
        "command_representation",
        "movement_only",
    ]
    assert [layer["feature_group"] for layer in layers["borderline"]] == [
        "trajectory_only",
        "release_only",
    ]
    assert [layer["feature_group"] for layer in layers["diagnostic"]] == ["physics_core"]


def test_copula_model_is_prioritized_for_demo_selection() -> None:
    module = _load_builder_module()

    assert "player_recent_weighted_game_drift_copula" in module.MODEL_SELECTION_PRIORITY
    assert module._model_priority("player_recent_weighted_game_drift_copula") < module._model_priority(
        "player_recent_weighted_game_drift_gaussian"
    )


def test_build_real_demo_artifacts_writes_reports_and_json(tmp_path) -> None:
    build_real_demo_artifacts = _load_builder()
    result = build_real_demo_artifacts(
        data_path=REAL_SAMPLE,
        output_dir=tmp_path,
        min_pitches=20,
        min_holdout=5,
        min_games=1,
        n_samples=8,
    )
    assert result["candidate_rankings"].exists()
    assert result["validation_report"].exists()
    assert result["session_json"].exists()
    assert result["morning_report"].exists()

    report = json.loads(result["validation_report"].read_text())
    assert report["selected_candidate"]["pitcher_name"]
    assert report["validation_thresholds"]["candidate_thresholds"] == {
        "min_pitches": 20,
        "min_holdout": 5,
        "min_games": 1,
        "min_completeness": 0.95,
    }
    assert report["validation_thresholds"]["temporal_success_auc"] == 0.60
    assert report["artifact_status"] in {
        "validated_command_layer_physics_diagnostic",
        "validated_component_layers_physics_diagnostic",
        "validated_full_temporal_success",
        "partial_temporal_success",
        "diagnostic_not_final",
    }
    physics_auc = report["feature_group_results"]["physics_core"]["best_auc"]
    if physics_auc > report["validation_thresholds"]["temporal_success_auc"]:
        assert report["artifact_status"] != "validated_full_temporal_success"
        assert report["artifact_status"] != "validated_temporal_success"
        assert report["artifact_status"] in {
            "validated_command_layer_physics_diagnostic",
            "validated_component_layers_physics_diagnostic",
        }
    if report["selected_auc"] > report["validation_thresholds"]["temporal_success_auc"]:
        assert report["artifact_status"] == "diagnostic_not_final"
    assert "validated_layers" in report
    assert "borderline_layers" in report
    assert report["diagnostic_layers"]
    if report["validated_layers"]:
        assert report["validated_layers"][0]["feature_group"] == report["selected_feature_group"]
    if physics_auc > report["validation_thresholds"]["temporal_success_auc"]:
        assert any(layer["feature_group"] == "physics_core" for layer in report["diagnostic_layers"])
    assert report["robustness_checks"]["seed_count"] == 30
    assert report["selected_feature_group"]
    assert report["selected_auc"] <= 0.70
    assert report["feature_group_results"]
    assert "player_context_weighted_gaussian" in report["feature_group_results"]["physics_core"]["model_results"]
    assert "player_gmm" in report["model_results"]
    payload = json.loads(result["session_json"].read_text())
    assert payload["schema_version"] == "pitcher-twin.real.v1"
    assert payload["metadata"]["export_strategy"] == "validated_layer_over_physics_core"
    assert payload["metadata"]["artifact_status"] == report["artifact_status"]
    assert "validated_feature_groups" in payload["metadata"]
    assert "validated_layers" not in payload["metadata"]
    assert "borderline_feature_groups" in payload["metadata"]
    assert payload["metadata"]["diagnostic_feature_groups"]
    assert "diagnostic_layers" not in payload["metadata"]
    assert payload["metadata"]["physics_temporal_auc"] == physics_auc
    assert payload["pitches"][0]["source"] == "simulated_from_real_model"
    assert payload["pitches"][0]["velocity"]["release_speed"] is not None
    assert payload["pitches"][0]["plate_target"]["x"] is not None
