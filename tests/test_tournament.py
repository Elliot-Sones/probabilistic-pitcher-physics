from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_twin.features import FEATURE_GROUPS
from pitcher_twin.tournament import (
    DERIVED_FEATURE_COLUMNS,
    apply_release_geometry_blend,
    apply_release_geometry_constraint,
    apply_spin_axis_residual_model,
    apply_recent_state_anchor,
    apply_spin_axis_angle_anchor,
    build_derived_physics_features,
    evaluate_model_tournament,
    fit_conditional_state_mixture_model,
    fit_release_geometry_constraint,
    fit_spin_axis_residual_model,
    fit_recent_state_anchor,
    fit_recent_trend_state_anchor,
    fit_spin_axis_angle_anchor,
    pitch_family_for_pitch_type,
    pitch_family_release_spin_settings,
    fit_context_neighbor_model,
    fit_derived_joint_gaussian_model,
    fit_pca_latent_model,
    sample_tournament_model,
)


def _synthetic_pitch_frame(n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    game_ids = np.repeat(np.arange(1, 9), n // 8)
    frame = pd.DataFrame(
        {
            "pitcher": 1,
            "pitcher_name": "Pitcher, Test",
            "pitch_type": "FF",
            "game_pk": game_ids,
            "game_date": np.repeat(pd.date_range("2026-04-01", periods=8).astype(str), n // 8),
            "at_bat_number": np.arange(n) // 4,
            "pitch_number": np.tile([1, 2, 3, 4], n // 4),
            "balls": rng.integers(0, 4, n),
            "strikes": rng.integers(0, 3, n),
            "count_bucket_code": rng.integers(0, 5, n).astype(float),
            "inning": rng.integers(1, 8, n).astype(float),
            "pitcher_game_pitch_count": np.arange(1, n + 1).astype(float),
            "batter_stand_code": rng.integers(0, 2, n).astype(float),
            "pitcher_score_diff": rng.normal(0.0, 2.0, n),
            "release_speed": rng.normal(95.0, 1.0, n),
            "release_spin_rate": rng.normal(2350.0, 80.0, n),
            "spin_axis_cos": rng.normal(-0.93, 0.03, n),
            "spin_axis_sin": rng.normal(0.35, 0.04, n),
            "release_pos_x": rng.normal(1.4, 0.08, n),
            "release_pos_y": rng.normal(54.0, 0.2, n),
            "release_pos_z": rng.normal(6.1, 0.1, n),
            "release_extension": rng.normal(6.5, 0.15, n),
        }
    )
    frame["pfx_x"] = -0.02 * frame["release_speed"] + rng.normal(0, 0.05, n)
    frame["pfx_z"] = 0.002 * frame["release_spin_rate"] + rng.normal(0, 0.05, n)
    frame["plate_x"] = 0.45 * frame["pfx_x"] + rng.normal(0, 0.3, n)
    frame["plate_z"] = 2.4 + 0.20 * frame["pfx_z"] + rng.normal(0, 0.3, n)
    frame["vx0"] = -0.05 * frame["release_speed"] + rng.normal(0, 0.1, n)
    frame["vy0"] = -1.45 * frame["release_speed"] + rng.normal(0, 0.2, n)
    frame["vz0"] = -4.0 + 0.2 * frame["pfx_z"] + rng.normal(0, 0.1, n)
    frame["ax"] = 15 * frame["pfx_x"] + rng.normal(0, 0.2, n)
    frame["ay"] = 25 + rng.normal(0, 0.4, n)
    frame["az"] = -20 + 4 * frame["pfx_z"] + rng.normal(0, 0.2, n)
    return frame


def _circular_mean_for_test(angles: np.ndarray) -> float:
    return float(np.angle(np.exp(1j * angles).mean()))


def test_build_derived_physics_features_returns_geometry_columns() -> None:
    frame = _synthetic_pitch_frame()
    derived = build_derived_physics_features(frame)

    assert DERIVED_FEATURE_COLUMNS == [
        "derived_spin_axis_angle",
        "derived_spin_axis_norm_error",
        "derived_release_xz_radius",
        "derived_movement_magnitude",
        "derived_movement_angle",
        "derived_velocity_adjusted_pfx_x",
        "derived_velocity_adjusted_pfx_z",
        "derived_plate_radius",
    ]
    assert set(DERIVED_FEATURE_COLUMNS).issubset(derived.columns)
    assert np.isfinite(derived[DERIVED_FEATURE_COLUMNS].to_numpy(float)).all()
    first_angle = np.arctan2(frame.loc[0, "spin_axis_sin"], frame.loc[0, "spin_axis_cos"])
    assert np.isclose(derived.loc[0, "derived_spin_axis_angle"], first_angle)


def test_release_geometry_constraint_restores_learned_extension_sum() -> None:
    frame = _synthetic_pitch_frame(n=80)
    drift = np.linspace(-0.04, 0.04, len(frame))
    frame["release_pos_y"] = 54.0 + drift
    frame["release_extension"] = 6.5 - drift

    constraint = fit_release_geometry_constraint(frame)

    samples = frame[["release_pos_y", "release_extension", "release_pos_x"]].head(24).copy()
    samples["release_pos_y"] = 53.25
    samples["release_extension"] = 6.10

    constrained = apply_release_geometry_constraint(samples, constraint, random_state=11)

    release_sum = constrained["release_pos_y"] + constrained["release_extension"]
    assert abs(float(release_sum.mean()) - constraint["sum_mean"]) < 0.08
    assert release_sum.std(ddof=0) > 0.0
    assert release_sum.std(ddof=0) < 0.15
    assert np.isfinite(constrained["release_pos_y"]).all()
    assert np.isfinite(constrained["release_extension"]).all()


def test_recent_state_anchor_moves_generated_cloud_toward_recent_games() -> None:
    frame = _synthetic_pitch_frame(n=80)
    frame["release_speed"] = np.r_[np.full(40, 94.0), np.full(40, 97.0)]
    frame["release_spin_rate"] = np.r_[np.full(40, 2280.0), np.full(40, 2380.0)]

    anchor = fit_recent_state_anchor(
        frame,
        ["release_speed", "release_spin_rate"],
        half_life_games=0.75,
    )

    samples = frame[["release_speed", "release_spin_rate"]].head(20).copy()
    samples["release_speed"] = 92.0
    samples["release_spin_rate"] = 2200.0

    anchored = apply_recent_state_anchor(samples, anchor, alpha=0.5)

    assert anchor["game_count"] == 8
    assert anchor["source_row_count"] == 80
    assert anchored["release_speed"].mean() > samples["release_speed"].mean()
    assert anchored["release_spin_rate"].mean() > samples["release_spin_rate"].mean()
    assert np.isclose(
        anchored["release_speed"].mean(),
        samples["release_speed"].mean()
        + 0.5 * (anchor["means"][0] - samples["release_speed"].mean()),
    )


def test_spin_axis_angle_anchor_rotates_samples_and_preserves_unit_norm() -> None:
    frame = _synthetic_pitch_frame(n=80)
    train_angles = np.r_[np.full(40, 0.25), np.full(40, 1.05)]
    frame["spin_axis_cos"] = np.cos(train_angles)
    frame["spin_axis_sin"] = np.sin(train_angles)

    anchor = fit_spin_axis_angle_anchor(frame, half_life_games=0.75)

    sample_angles = np.full(24, -0.35)
    samples = pd.DataFrame(
        {
            "spin_axis_cos": 2.0 * np.cos(sample_angles),
            "spin_axis_sin": 2.0 * np.sin(sample_angles),
        }
    )

    anchored = apply_spin_axis_angle_anchor(samples, anchor, alpha=0.50)

    before_angle = np.arctan2(samples["spin_axis_sin"].mean(), samples["spin_axis_cos"].mean())
    after_angle = np.arctan2(
        anchored["spin_axis_sin"].mean(),
        anchored["spin_axis_cos"].mean(),
    )
    before_gap = abs(np.angle(np.exp(1j * (before_angle - anchor["angle_mean"]))))
    after_gap = abs(np.angle(np.exp(1j * (after_angle - anchor["angle_mean"]))))
    norm = np.sqrt(anchored["spin_axis_cos"] ** 2 + anchored["spin_axis_sin"] ** 2)

    assert anchor["game_count"] == 8
    assert after_gap < before_gap
    assert np.allclose(norm, 1.0)


def test_pitch_family_release_spin_settings_are_specific_to_pitch_shape() -> None:
    ff_settings = pitch_family_release_spin_settings("FF")
    changeup_settings = pitch_family_release_spin_settings("CH")
    sinker_settings = pitch_family_release_spin_settings("SI")

    assert pitch_family_for_pitch_type("FF") == "rising_fastball"
    assert pitch_family_for_pitch_type("SI") == "sinker"
    assert pitch_family_for_pitch_type("CH") == "changeup"
    assert pitch_family_for_pitch_type("SL") == "breaking"
    assert ff_settings["physics_anchor_alpha"] > changeup_settings["physics_anchor_alpha"]
    assert changeup_settings["spin_residual_alpha"] > ff_settings["spin_residual_alpha"]
    assert sinker_settings["release_geometry_alpha"] >= ff_settings["release_geometry_alpha"]


def test_spin_axis_residual_model_recenters_without_collapsing_angular_spread() -> None:
    frame = _synthetic_pitch_frame(n=80)
    train_angles = np.r_[np.linspace(0.15, 0.45, 40), np.linspace(0.95, 1.45, 40)]
    frame["spin_axis_cos"] = np.cos(train_angles)
    frame["spin_axis_sin"] = np.sin(train_angles)

    model = fit_spin_axis_residual_model(frame, half_life_games=0.75, recent_fraction=0.55)

    sample_angles = np.linspace(-0.35, 0.05, 32)
    samples = pd.DataFrame(
        {
            "spin_axis_cos": np.cos(sample_angles),
            "spin_axis_sin": np.sin(sample_angles),
        }
    )
    adjusted = apply_spin_axis_residual_model(
        samples,
        model,
        alpha=0.75,
        random_state=23,
    )
    adjusted_angles = np.unwrap(
        np.arctan2(adjusted["spin_axis_sin"], adjusted["spin_axis_cos"])
    )
    adjusted_norm = np.sqrt(adjusted["spin_axis_cos"] ** 2 + adjusted["spin_axis_sin"] ** 2)

    before_gap = abs(
        np.angle(np.exp(1j * (_circular_mean_for_test(sample_angles) - model["angle_mean"])))
    )
    after_gap = abs(
        np.angle(np.exp(1j * (_circular_mean_for_test(adjusted_angles) - model["angle_mean"])))
    )
    assert model["source_row_count"] == 80
    assert model["residual_count"] >= 10
    assert after_gap < before_gap
    assert adjusted_angles.std(ddof=0) > 0.05
    assert np.allclose(adjusted_norm, 1.0)


def test_release_geometry_blend_partially_projects_y_extension_sum() -> None:
    frame = _synthetic_pitch_frame(n=80)
    constraint = fit_release_geometry_constraint(frame)
    samples = frame[["release_pos_y", "release_extension"]].head(20).copy()
    samples["release_pos_y"] = 52.5
    samples["release_extension"] = 5.5
    before_gap = abs(
        float((samples["release_pos_y"] + samples["release_extension"]).mean())
        - float(constraint["predicted_sum_mean"])
    )

    blended = apply_release_geometry_blend(
        samples,
        constraint,
        alpha=0.50,
        random_state=31,
    )
    after_gap = abs(
        float((blended["release_pos_y"] + blended["release_extension"]).mean())
        - float(constraint["predicted_sum_mean"])
    )

    assert after_gap < before_gap
    assert after_gap > 0.0
    assert blended["release_extension"].between(
        constraint["extension_lower"],
        constraint["extension_upper"],
    ).all()


def test_recent_trend_state_anchor_extrapolates_bounded_game_drift() -> None:
    frame = _synthetic_pitch_frame(n=80)
    game_index = frame["game_pk"].astype(float) - 1.0
    frame["release_speed"] = 94.0 + 0.25 * game_index
    frame["release_spin_rate"] = 2280.0 + 12.0 * game_index

    anchor = fit_recent_trend_state_anchor(
        frame,
        ["release_speed", "release_spin_rate"],
        half_life_games=4.0,
        horizon_games=1.0,
        trend_shrinkage=0.75,
    )
    recent_anchor = fit_recent_state_anchor(
        frame,
        ["release_speed", "release_spin_rate"],
        half_life_games=4.0,
    )

    assert anchor["game_count"] == 8
    assert anchor["source_row_count"] == 80
    assert anchor["means"][0] > recent_anchor["means"][0]
    assert anchor["means"][1] > recent_anchor["means"][1]


def test_pca_latent_model_samples_finite_physics_core_rows() -> None:
    frame = _synthetic_pitch_frame()
    model = fit_pca_latent_model(frame, feature_columns=FEATURE_GROUPS["physics_core"])
    samples = sample_tournament_model(model, n=20, context_df=frame.head(20), random_state=4)

    assert model.model_name == "pca_latent_residual"
    assert samples.columns.tolist() == FEATURE_GROUPS["physics_core"]
    assert samples.shape == (20, len(FEATURE_GROUPS["physics_core"]))
    assert np.isfinite(samples.to_numpy(float)).all()
    assert model.payload["component_count"] >= 1


def test_context_neighbor_model_samples_from_similar_contexts() -> None:
    frame = _synthetic_pitch_frame()
    model = fit_context_neighbor_model(
        frame,
        feature_columns=FEATURE_GROUPS["physics_core"],
        context_columns=["inning", "pitcher_game_pitch_count", "balls", "strikes"],
    )
    context = frame.tail(12)
    samples = sample_tournament_model(model, n=12, context_df=context, random_state=5)

    assert model.model_name == "context_neighbor_residual"
    assert samples.columns.tolist() == FEATURE_GROUPS["physics_core"]
    assert samples.shape == (12, len(FEATURE_GROUPS["physics_core"]))
    assert np.isfinite(samples.to_numpy(float)).all()


def test_derived_joint_gaussian_samples_raw_physics_from_enriched_fit() -> None:
    frame = _synthetic_pitch_frame()
    model = fit_derived_joint_gaussian_model(frame, feature_columns=FEATURE_GROUPS["physics_core"])
    samples = sample_tournament_model(model, n=18, context_df=frame.head(18), random_state=9)

    assert model.model_name == "derived_joint_gaussian"
    assert samples.columns.tolist() == FEATURE_GROUPS["physics_core"]
    assert samples.shape == (18, len(FEATURE_GROUPS["physics_core"]))
    assert np.isfinite(samples.to_numpy(float)).all()
    assert set(DERIVED_FEATURE_COLUMNS).issubset(model.payload["joint_columns"])


def test_conditional_state_mixture_model_samples_state_conditioned_physics() -> None:
    frame = _synthetic_pitch_frame()
    late_mask = frame["pitcher_game_pitch_count"] > frame["pitcher_game_pitch_count"].median()
    frame.loc[late_mask, "release_speed"] += 2.5
    frame.loc[late_mask, "release_spin_rate"] += 120.0
    frame.loc[late_mask, "pfx_z"] += 0.35

    model = fit_conditional_state_mixture_model(
        frame,
        feature_columns=FEATURE_GROUPS["physics_core"],
        random_state=17,
    )
    context = frame.tail(16).copy()
    samples = sample_tournament_model(model, n=16, context_df=context, random_state=19)

    assert model.model_name == "conditional_state_mixture_residual"
    assert samples.columns.tolist() == FEATURE_GROUPS["physics_core"]
    assert samples.shape == (16, len(FEATURE_GROUPS["physics_core"]))
    assert np.isfinite(samples.to_numpy(float)).all()
    assert model.payload["state_count"] >= 2
    assert model.payload["source_row_count"] == len(frame)
    norms = np.sqrt(samples["spin_axis_cos"] ** 2 + samples["spin_axis_sin"] ** 2)
    assert np.allclose(norms, 1.0)


def test_model_tournament_reports_repeated_layer_results() -> None:
    frame = _synthetic_pitch_frame()
    train = frame.head(68).copy()
    holdout = frame.tail(28).copy()

    report = evaluate_model_tournament(
        train,
        holdout,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        n_samples=48,
        repeats=2,
        random_state=11,
    )

    assert report["model_name"] == "pitcher_twin_model_tournament"
    assert report["repeat_count"] == 2
    assert report["model_names"] == [
        "factorized_v2_1",
        "factorized_release_game_drift_gaussian",
        "factorized_release_game_drift_copula",
        "factorized_release_recent_gaussian",
        "factorized_short_memory_wide_residual",
        "factorized_short_memory_more_uncertain",
        "factorized_recent_state_anchored",
        "factorized_trend_state_anchored",
        "factorized_release_state_anchored",
        "factorized_pitch_family_release_spin",
        "factorized_physics_constrained_state",
        "conditional_state_mixture_residual",
        "pca_latent_residual",
        "context_neighbor_residual",
        "derived_joint_gaussian",
    ]
    assert set(report["layer_results"]) == {
        "command_representation",
        "movement_only",
        "release_only",
        "trajectory_only",
        "physics_core",
    }
    for layer, rows in report["layer_results"].items():
        assert set(rows) == set(report["model_names"])
        assert report["best_by_layer"][layer] in report["model_names"]
        for row in rows.values():
            assert 0.5 <= row["mean_auc"] <= 1.0
            assert row["repeat_count"] == 2
    assert report["best_physics_core_model"] in report["model_names"]
    assert (
        report["candidate_notes"]["factorized_physics_constrained_state"][
            "release_geometry_constraint"
        ]
        == "release_pos_y_plus_extension"
    )
    assert (
        report["candidate_notes"]["factorized_release_state_anchored"][
            "release_anchor_alpha"
        ]
        == 0.70
    )
    assert (
        report["candidate_notes"]["factorized_release_state_anchored"]["spin_axis_anchor"]
        == "circular_recent_game_mean"
    )
    assert (
        report["candidate_notes"]["factorized_pitch_family_release_spin"]["pitch_family"]
        == "rising_fastball"
    )
    assert (
        report["candidate_notes"]["factorized_pitch_family_release_spin"][
            "spin_axis_model"
        ]
        == "empirical_recent_circular_residual"
    )
    assert (
        report["candidate_notes"]["factorized_pitch_family_release_spin"][
            "release_geometry_blend_alpha"
        ]
        > 0
    )
    assert (
        report["candidate_notes"]["factorized_trend_state_anchored"][
            "trend_state_anchor_horizon_games"
        ]
        == 1.0
    )
    assert report["candidate_notes"]["conditional_state_mixture_residual"]["state_conditioned"]


def test_model_tournament_script_exposes_main() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_model_tournament.py"
    spec = importlib.util.spec_from_file_location("run_model_tournament", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "main")
    assert hasattr(module, "run")
    assert module.SUMMARY_TITLE == "# Model Tournament"
