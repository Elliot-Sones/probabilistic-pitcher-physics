"""Real weather fetch and join helpers."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OPEN_METEO_ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
MLB_LIVE_FEED_ENDPOINT = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

WEATHER_FEATURE_COLUMNS = [
    "weather_temperature_2m_f",
    "weather_relative_humidity_2m",
    "weather_pressure_msl_hpa",
    "weather_precipitation_mm",
    "weather_wind_speed_10m_mph",
    "weather_wind_dir_sin",
    "weather_wind_dir_cos",
    "weather_precip_flag",
    "weather_roof_open",
]


def build_open_meteo_archive_url(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> str:
    params = {
        "latitude": f"{float(latitude):.6f}",
        "longitude": f"{float(longitude):.6f}",
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "mm",
        "timezone": "UTC",
    }
    return f"{OPEN_METEO_ARCHIVE_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _read_json_url(url: str, timeout_seconds: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "pitcher-twin-weather/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_open_meteo_hourly(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Open-Meteo payload is missing hourly time data.")

    frame = pd.DataFrame(hourly).rename(
        columns={
            "time": "weather_time_utc",
            "temperature_2m": "weather_temperature_2m_f",
            "relative_humidity_2m": "weather_relative_humidity_2m",
            "pressure_msl": "weather_pressure_msl_hpa",
            "precipitation": "weather_precipitation_mm",
            "wind_speed_10m": "weather_wind_speed_10m_mph",
            "wind_direction_10m": "weather_wind_direction_10m_deg",
        }
    )
    frame["weather_time_utc"] = pd.to_datetime(frame["weather_time_utc"], utc=True)
    numeric_columns = [column for column in frame.columns if column != "weather_time_utc"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    radians = np.deg2rad(frame["weather_wind_direction_10m_deg"])
    frame["weather_wind_dir_sin"] = np.sin(radians)
    frame["weather_wind_dir_cos"] = np.cos(radians)
    frame["weather_precip_flag"] = (frame["weather_precipitation_mm"].fillna(0.0) > 0).astype(float)
    return frame


def nearest_hourly_weather(hourly: pd.DataFrame, target_time_utc: str | pd.Timestamp) -> dict[str, Any]:
    if hourly.empty:
        raise ValueError("Hourly weather frame is empty.")
    target = pd.Timestamp(target_time_utc)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    deltas = (hourly["weather_time_utc"] - target).abs()
    index = int(deltas.idxmin())
    row = hourly.loc[index].to_dict()
    row["weather_time_delta_minutes"] = float(deltas.loc[index].total_seconds() / 60.0)
    return row


def open_meteo_query_date(game_time_utc: str | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(game_time_utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.date().isoformat()


def join_weather_by_game_pitch_rows(
    pitches: pd.DataFrame,
    game_weather: pd.DataFrame,
) -> pd.DataFrame:
    if "game_pk" not in pitches.columns:
        raise ValueError("Pitch rows must include game_pk to join weather.")
    if "game_pk" not in game_weather.columns:
        raise ValueError("Game weather rows must include game_pk.")
    weather_columns = [
        column for column in game_weather.columns if column == "game_pk" or column.startswith("weather_")
    ]
    return pitches.merge(game_weather[weather_columns], on="game_pk", how="left")


def fetch_mlb_game_weather_metadata(game_pk: int) -> dict[str, Any]:
    payload = _read_json_url(MLB_LIVE_FEED_ENDPOINT.format(game_pk=int(game_pk)))
    game_data = payload["gameData"]
    venue = game_data["venue"]
    coordinates = venue["location"]["defaultCoordinates"]
    roof_type = str(venue.get("fieldInfo", {}).get("roofType", "Unknown"))
    return {
        "game_pk": int(game_pk),
        "game_time_utc": game_data["datetime"]["dateTime"],
        "game_date": game_data["datetime"]["officialDate"],
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "latitude": float(coordinates["latitude"]),
        "longitude": float(coordinates["longitude"]),
        "roof_type": roof_type,
        "weather_roof_open": 0.0 if roof_type.lower() in {"dome", "closed"} else 1.0,
        "mlb_weather_condition": game_data.get("weather", {}).get("condition"),
        "mlb_weather_temp": game_data.get("weather", {}).get("temp"),
        "mlb_weather_wind": game_data.get("weather", {}).get("wind"),
    }


def fetch_open_meteo_game_weather(game_metadata: dict[str, Any]) -> dict[str, Any]:
    date = open_meteo_query_date(game_metadata["game_time_utc"])
    url = build_open_meteo_archive_url(
        latitude=float(game_metadata["latitude"]),
        longitude=float(game_metadata["longitude"]),
        start_date=date,
        end_date=date,
    )
    hourly = normalize_open_meteo_hourly(_read_json_url(url))
    weather = nearest_hourly_weather(hourly, game_metadata["game_time_utc"])
    return {
        **game_metadata,
        **weather,
        "weather_source": "open-meteo-historical",
    }


def fetch_weather_for_game_pks(game_pks: list[int]) -> pd.DataFrame:
    rows = []
    for game_pk in game_pks:
        metadata = fetch_mlb_game_weather_metadata(int(game_pk))
        rows.append(fetch_open_meteo_game_weather(metadata))
    return pd.DataFrame(rows)


def write_weather_cache(weather: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        weather.to_parquet(path, index=False)
    else:
        weather.to_csv(path, index=False)
    return path


def read_weather_cache(path: str | Path) -> pd.DataFrame:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Weather cache not found: {cache_path}")
    if cache_path.suffix.lower() == ".parquet":
        return pd.read_parquet(cache_path)
    return pd.read_csv(cache_path)
