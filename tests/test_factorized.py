from __future__ import annotations

import numpy as np
import pandas as pd

from pitcher_twin.features import PITCH_PHYSICS_FEATURES, TRAJECTORY_FEATURES
from pitcher_twin.factorized import fit_residual_layer, sample_residual_layer
from pitcher_twin.factorized import fit_factorized_physics_model, sample_factorized_physics
from pitcher_twin.factorized import validate_factorized_physics


def test_residual_layer_learns_linear_relationship_and_returns_finite_samples() -> None:
    frame = pd.DataFrame(
        {
            "release_speed": np.linspace(90.0, 99.0, 40),
            "release_spin_rate": np.linspace(2200.0, 2400.0, 40),
            "inning": np.tile([1.0, 5.0], 20),
        }
    )
    frame["pfx_z"] = 0.04 * frame["release_speed"] + 0.001 * frame["release_spin_rate"]
    frame["pfx_x"] = -0.02 * frame["release_speed"] + 0.03 * frame["inning"]

    layer = fit_residual_layer(
        frame,
        name="movement",
        conditioning_columns=["release_speed", "release_spin_rate", "inning"],
        target_columns=["pfx_x", "pfx_z"],
        ridge=1.0,
    )
    context = frame[["release_speed", "release_spin_rate", "inning"]].head(8)
    samples = sample_residual_layer(layer, context, random_state=9)

    assert samples.shape == (8, 2)
    assert samples.columns.tolist() == ["pfx_x", "pfx_z"]
    assert np.isfinite(samples.to_numpy(float)).all()
    assert abs(samples["pfx_z"].mean() - frame["pfx_z"].head(8).mean()) < 0.4


def _synthetic_factorized_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    release_speed = rng.normal(95.0, 1.0, n)
    spin = rng.normal(2350.0, 80.0, n)
    frame = pd.DataFrame(
        {
            "pitcher": 1,
            "pitcher_name": "Pitcher, Test",
            "pitch_type": "FF",
            "game_pk": np.repeat(np.arange(1, 9), n // 8),
            "game_date": np.repeat(pd.date_range("2026-04-01", periods=8).astype(str), n // 8),
            "at_bat_number": np.arange(n) // 4,
            "pitch_number": np.tile([1, 2, 3, 4], n // 4),
            "balls": rng.integers(0, 4, n),
            "strikes": rng.integers(0, 3, n),
            "count_bucket_code": 2.0,
            "inning": rng.integers(1, 8, n).astype(float),
            "pitcher_game_pitch_count": np.arange(1, n + 1).astype(float),
            "batter_stand_code": rng.integers(0, 2, n).astype(float),
            "pitcher_score_diff": rng.normal(0.0, 2.0, n),
            "release_speed": release_speed,
            "release_spin_rate": spin,
            "spin_axis_cos": rng.normal(-0.95, 0.02, n),
            "spin_axis_sin": rng.normal(0.25, 0.03, n),
            "release_pos_x": rng.normal(1.4, 0.08, n),
            "release_pos_y": rng.normal(54.0, 0.2, n),
            "release_pos_z": rng.normal(6.1, 0.1, n),
            "release_extension": rng.normal(6.5, 0.15, n),
        }
    )
    frame["pfx_x"] = -0.02 * frame["release_speed"] + rng.normal(0, 0.05, n)
    frame["pfx_z"] = 0.002 * frame["release_spin_rate"] + rng.normal(0, 0.05, n)
    frame["vx0"] = -0.05 * frame["release_speed"] + rng.normal(0, 0.1, n)
    frame["vy0"] = -1.45 * frame["release_speed"] + rng.normal(0, 0.2, n)
    frame["vz0"] = -4.0 + 0.2 * frame["pfx_z"] + rng.normal(0, 0.1, n)
    frame["ax"] = 15 * frame["pfx_x"] + rng.normal(0, 0.2, n)
    frame["ay"] = 25 + rng.normal(0, 0.4, n)
    frame["az"] = -20 + 4 * frame["pfx_z"] + rng.normal(0, 0.2, n)
    frame["plate_x"] = (
        0.5 * frame["pfx_x"] + 0.03 * frame["pitcher_score_diff"] + rng.normal(0, 0.1, n)
    )
    frame["plate_z"] = (
        2.2
        + 0.25 * frame["pfx_z"]
        - 0.004 * frame["pitcher_game_pitch_count"]
        + rng.normal(0, 0.1, n)
    )
    return frame


def test_factorized_physics_model_samples_full_physics_core_columns() -> None:
    frame = _synthetic_factorized_frame()
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
    )
    context = frame.head(12)
    samples = sample_factorized_physics(model, n=12, context_df=context, random_state=6)

    expected = PITCH_PHYSICS_FEATURES + ["pfx_x", "pfx_z"] + TRAJECTORY_FEATURES + [
        "plate_x",
        "plate_z",
    ]
    assert samples.columns.tolist() == expected
    assert samples.shape == (12, len(expected))
    assert np.isfinite(samples.to_numpy(float)).all()
    assert model.movement_layer.source_row_count >= 20
    assert model.trajectory_layer.source_row_count >= 20
    assert model.command_layer.source_row_count >= 20


def test_factorized_physics_samples_normalize_spin_axis_components() -> None:
    frame = _synthetic_factorized_frame()
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
    )
    samples = sample_factorized_physics(model, n=24, context_df=frame.head(24), random_state=6)

    norms = np.sqrt(samples["spin_axis_cos"] ** 2 + samples["spin_axis_sin"] ** 2)
    assert np.allclose(norms, 1.0)


def test_factorized_physics_model_preserves_joint_downstream_residual_shape() -> None:
    frame = _synthetic_factorized_frame()
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
    )

    assert model.downstream_residual_columns == [
        "pfx_x",
        "pfx_z",
        *TRAJECTORY_FEATURES,
        "plate_x",
        "plate_z",
    ]
    assert model.downstream_residual_cov.shape == (10, 10)
    assert model.downstream_residual_copula is not None


def test_factorized_physics_model_tracks_recent_downstream_residual_offset() -> None:
    frame = _synthetic_factorized_frame()
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
    )

    assert model.downstream_residual_offset.shape == (10,)
    assert np.isfinite(model.downstream_residual_offset).all()


def test_factorized_physics_model_tracks_release_variance_floor() -> None:
    frame = _synthetic_factorized_frame()
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
    )

    assert model.release_variance_floor_columns == PITCH_PHYSICS_FEATURES
    assert model.release_variance_floor_std.shape == (len(PITCH_PHYSICS_FEATURES),)
    assert np.isfinite(model.release_variance_floor_std).all()


def test_factorized_validation_reports_baselines_and_layer_metrics() -> None:
    frame = _synthetic_factorized_frame()
    train = frame.iloc[:56].copy()
    holdout = frame.iloc[56:].copy()

    report = validate_factorized_physics(
        train,
        holdout,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        n_samples=40,
        random_state=8,
    )

    assert report["model_name"] == "player_factorized_physics_residual"
    assert set(report["layer_results"]) == {
        "command_representation",
        "movement_only",
        "release_only",
        "trajectory_only",
        "physics_core",
    }
    for row in report["layer_results"].values():
        assert "factorized_auc" in row
        assert "game_drift_gaussian_auc" in row
        assert "game_drift_copula_auc" in row
        assert 0.5 <= row["factorized_auc"] <= 1.0


def test_factorized_validation_script_exposes_main() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "run_factorized_validation.py"
    spec = importlib.util.spec_from_file_location("run_factorized_validation", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)


def test_factorized_validation_script_uses_player_name_fallback() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "run_factorized_validation.py"
    spec = importlib.util.spec_from_file_location("run_factorized_validation", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    subset = pd.DataFrame({"player_name": ["Skubal, Tarik"]})
    assert module._pitcher_name_from_subset(subset) == "Skubal, Tarik"
