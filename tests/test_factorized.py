from __future__ import annotations

import numpy as np
import pandas as pd

from pitcher_twin.features import PITCH_PHYSICS_FEATURES, TRAJECTORY_FEATURES
from pitcher_twin.factorized import fit_residual_layer, sample_residual_layer
from pitcher_twin.factorized import fit_factorized_physics_model, sample_factorized_physics
from pitcher_twin.factorized import validate_factorized_physics
from pitcher_twin.weather import WEATHER_FEATURE_COLUMNS


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


def _add_synthetic_weather(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    game_weather = result.groupby("game_pk", sort=True)["game_pk"].transform("first").astype(float)
    result["weather_temperature_2m_f"] = 55.0 + game_weather
    result["weather_relative_humidity_2m"] = 45.0 + game_weather
    result["weather_pressure_msl_hpa"] = 1010.0 - game_weather
    result["weather_precipitation_mm"] = (game_weather % 2) * 0.1
    result["weather_wind_speed_10m_mph"] = 3.0 + game_weather * 0.2
    radians = np.deg2rad(game_weather * 20.0)
    result["weather_wind_dir_sin"] = np.sin(radians)
    result["weather_wind_dir_cos"] = np.cos(radians)
    result["weather_precip_flag"] = (result["weather_precipitation_mm"] > 0).astype(float)
    result["weather_roof_open"] = 1.0
    result["pfx_z"] = result["pfx_z"] + 0.01 * result["weather_temperature_2m_f"]
    result["az"] = result["az"] - 0.03 * result["weather_relative_humidity_2m"]
    return result


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


def test_factorized_physics_model_fits_optional_weather_residual_adjustment() -> None:
    frame = _add_synthetic_weather(_synthetic_factorized_frame())
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
        weather_feature_columns=WEATHER_FEATURE_COLUMNS,
    )

    assert model.weather_feature_columns == WEATHER_FEATURE_COLUMNS
    assert model.weather_residual_adjustment is not None
    assert model.weather_residual_adjustment["source_row_count"] >= 20


def test_factorized_weather_sampling_requires_weather_context_when_enabled() -> None:
    frame = _add_synthetic_weather(_synthetic_factorized_frame())
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
        weather_feature_columns=WEATHER_FEATURE_COLUMNS,
    )

    try:
        sample_factorized_physics(
            model,
            n=10,
            context_df=frame.drop(columns=WEATHER_FEATURE_COLUMNS).head(10),
            random_state=6,
            use_weather=True,
        )
    except ValueError as exc:
        assert "weather columns" in str(exc)
    else:
        raise AssertionError("Weather sampling should require weather columns.")


def test_factorized_weather_sampling_changes_samples_when_enabled() -> None:
    frame = _add_synthetic_weather(_synthetic_factorized_frame())
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
        weather_feature_columns=WEATHER_FEATURE_COLUMNS,
    )

    weather_off = sample_factorized_physics(
        model,
        n=12,
        context_df=frame.head(12),
        random_state=6,
        use_weather=False,
    )
    weather_on = sample_factorized_physics(
        model,
        n=12,
        context_df=frame.head(12),
        random_state=6,
        use_weather=True,
    )

    assert not weather_off[["pfx_z", "az"]].equals(weather_on[["pfx_z", "az"]])


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


def test_weather_residual_validation_script_exposes_main() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "run_weather_residual_validation.py"
    spec = importlib.util.spec_from_file_location("run_weather_residual_validation", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)


def test_weather_residual_validation_aggregates_repeated_layer_results() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "run_weather_residual_validation.py"
    spec = importlib.util.spec_from_file_location("run_weather_residual_validation", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    runs = [
        {
            "physics_core": {"baseline_auc": 0.60, "weather_auc": 0.58},
            "movement_only": {"baseline_auc": 0.55, "weather_auc": 0.53},
        },
        {
            "physics_core": {"baseline_auc": 0.62, "weather_auc": 0.64},
            "movement_only": {"baseline_auc": 0.57, "weather_auc": 0.51},
        },
    ]

    summary = module._aggregate_repeated_layer_results(runs)

    assert summary["physics_core"]["baseline_auc"] == 0.61
    assert summary["physics_core"]["weather_auc"] == 0.61
    assert summary["physics_core"]["delta_auc"] == 0.0
    assert summary["movement_only"]["weather_improved"] is True
