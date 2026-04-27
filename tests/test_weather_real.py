from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_twin.weather import (
    WEATHER_FEATURE_COLUMNS,
    build_open_meteo_archive_url,
    join_weather_by_game_pitch_rows,
    nearest_hourly_weather,
    normalize_open_meteo_hourly,
    open_meteo_query_date,
)


def test_build_open_meteo_archive_url_uses_real_archive_endpoint_and_hourly_fields() -> None:
    url = build_open_meteo_archive_url(
        latitude=41.495861,
        longitude=-81.685255,
        start_date="2025-09-23",
        end_date="2025-09-23",
    )

    assert url.startswith("https://archive-api.open-meteo.com/v1/archive?")
    assert "latitude=41.495861" in url
    assert "longitude=-81.685255" in url
    assert "hourly=temperature_2m%2Crelative_humidity_2m%2Cpressure_msl" in url
    assert "temperature_unit=fahrenheit" in url
    assert "wind_speed_unit=mph" in url
    assert "timezone=UTC" in url


def test_normalize_open_meteo_hourly_returns_prefixed_numeric_features() -> None:
    payload = {
        "hourly": {
            "time": ["2025-09-23T22:00", "2025-09-23T23:00"],
            "temperature_2m": [72.0, 70.5],
            "relative_humidity_2m": [51.0, 55.0],
            "pressure_msl": [1012.1, 1011.9],
            "precipitation": [0.0, 0.1],
            "wind_speed_10m": [2.0, 3.5],
            "wind_direction_10m": [180.0, 270.0],
        }
    }

    hourly = normalize_open_meteo_hourly(payload)

    assert hourly["weather_time_utc"].dt.tz is not None
    hourly_feature_columns = set(WEATHER_FEATURE_COLUMNS) - {"weather_roof_open"}
    assert hourly_feature_columns.issubset(hourly.columns)
    assert hourly["weather_temperature_2m_f"].tolist() == [72.0, 70.5]
    assert hourly["weather_precip_flag"].tolist() == [0.0, 1.0]
    assert np.isclose(hourly.loc[0, "weather_wind_dir_sin"], 0.0, atol=1e-8)
    assert np.isclose(hourly.loc[0, "weather_wind_dir_cos"], -1.0, atol=1e-8)


def test_nearest_hourly_weather_selects_closest_hour_and_records_delta_minutes() -> None:
    hourly = normalize_open_meteo_hourly(
        {
            "hourly": {
                "time": ["2025-09-23T21:00", "2025-09-23T22:00", "2025-09-23T23:00"],
                "temperature_2m": [73.0, 72.0, 71.0],
                "relative_humidity_2m": [50.0, 51.0, 52.0],
                "pressure_msl": [1012.0, 1012.1, 1012.2],
                "precipitation": [0.0, 0.0, 0.0],
                "wind_speed_10m": [2.0, 2.0, 2.0],
                "wind_direction_10m": [90.0, 180.0, 270.0],
            }
        }
    )

    row = nearest_hourly_weather(hourly, "2025-09-23T22:40:00Z")

    assert row["weather_temperature_2m_f"] == 71.0
    assert row["weather_time_delta_minutes"] == 20.0


def test_open_meteo_query_date_uses_utc_date_not_local_official_date() -> None:
    assert open_meteo_query_date("2025-08-26T02:05:00Z") == "2025-08-26"


def test_join_weather_by_game_pitch_rows_left_joins_real_game_weather() -> None:
    pitches = pd.DataFrame(
        {
            "game_pk": [1, 1, 2],
            "release_speed": [95.0, 96.0, 97.0],
        }
    )
    game_weather = pd.DataFrame(
        {
            "game_pk": [1],
            "weather_temperature_2m_f": [72.0],
            "weather_relative_humidity_2m": [51.0],
            "weather_pressure_msl_hpa": [1012.1],
            "weather_precipitation_mm": [0.0],
            "weather_wind_speed_10m_mph": [2.0],
            "weather_wind_dir_sin": [0.0],
            "weather_wind_dir_cos": [-1.0],
            "weather_precip_flag": [0.0],
            "weather_roof_open": [1.0],
        }
    )

    joined = join_weather_by_game_pitch_rows(pitches, game_weather)

    assert joined.loc[0, "weather_temperature_2m_f"] == 72.0
    assert joined.loc[1, "weather_temperature_2m_f"] == 72.0
    assert pd.isna(joined.loc[2, "weather_temperature_2m_f"])


def test_fetch_open_meteo_weather_script_exposes_main() -> None:
    script = Path(__file__).parents[1] / "scripts" / "fetch_open_meteo_weather.py"
    spec = importlib.util.spec_from_file_location("fetch_open_meteo_weather", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)
