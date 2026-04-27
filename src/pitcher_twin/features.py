"""Real Statcast feature engineering for pitch variability modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_PHYSICS_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "spin_axis_cos",
    "spin_axis_sin",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
]

MOVEMENT_FEATURES = ["pfx_x", "pfx_z", "plate_x", "plate_z"]

TRAJECTORY_FEATURES = ["vx0", "vy0", "vz0", "ax", "ay", "az"]

CONTEXT_FEATURES = [
    "balls",
    "strikes",
    "count_bucket_code",
    "inning",
    "pitcher_game_pitch_count",
    "batter_stand_code",
    "pitcher_score_diff",
]

OPTIONAL_CONTEXT_FEATURES = [
    "days_rest",
    "times_through_order",
    "recent_workload",
]

FEATURE_GROUPS = {
    "release_only": PITCH_PHYSICS_FEATURES,
    "movement_only": MOVEMENT_FEATURES,
    "trajectory_only": TRAJECTORY_FEATURES,
    "shape_representation": PITCH_PHYSICS_FEATURES + MOVEMENT_FEATURES,
    "command_representation": ["plate_x", "plate_z"],
    "physics_core": PITCH_PHYSICS_FEATURES + MOVEMENT_FEATURES + TRAJECTORY_FEATURES,
    "physics_count": PITCH_PHYSICS_FEATURES
    + MOVEMENT_FEATURES
    + TRAJECTORY_FEATURES
    + ["balls", "strikes", "count_bucket_code"],
    "physics_fatigue": PITCH_PHYSICS_FEATURES
    + MOVEMENT_FEATURES
    + TRAJECTORY_FEATURES
    + ["pitcher_game_pitch_count", "inning", "days_rest", "times_through_order"],
    "physics_batter_context": PITCH_PHYSICS_FEATURES
    + MOVEMENT_FEATURES
    + TRAJECTORY_FEATURES
    + ["batter_stand_code", "pitcher_score_diff"],
}

COUNT_BUCKET_CODES = {
    "first_pitch": 0,
    "behind": 1,
    "even": 2,
    "ahead": 3,
    "full": 4,
}


def add_spin_axis_components(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    radians = np.deg2rad(pd.to_numeric(result["spin_axis"], errors="coerce"))
    result["spin_axis_cos"] = np.cos(radians)
    result["spin_axis_sin"] = np.sin(radians)
    return result


def add_count_bucket(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    balls = pd.to_numeric(result["balls"], errors="coerce")
    strikes = pd.to_numeric(result["strikes"], errors="coerce")
    buckets = np.select(
        [
            (balls == 0) & (strikes == 0),
            (balls == 3) & (strikes == 2),
            balls > strikes,
            strikes > balls,
        ],
        ["first_pitch", "full", "behind", "ahead"],
        default="even",
    )
    result["count_bucket"] = buckets
    result["count_bucket_code"] = result["count_bucket"].map(COUNT_BUCKET_CODES).astype(float)
    return result


def add_pitcher_game_pitch_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative pitcher pitch count within each real game."""
    result = df.copy()
    required = {"game_date", "game_pk", "pitcher", "at_bat_number", "pitch_number"}
    if not required.issubset(result.columns):
        result["pitcher_game_pitch_count"] = np.nan
        return result

    ordered = result.sort_values(
        ["game_date", "game_pk", "pitcher", "at_bat_number", "pitch_number"],
        kind="mergesort",
    )
    counts = ordered.groupby(["game_pk", "pitcher"], dropna=False).cumcount() + 1
    result["pitcher_game_pitch_count"] = counts.reindex(result.index).astype(float)
    return result


def add_pitcher_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Add score differential from the pitcher's team perspective."""
    result = df.copy()
    result["pitcher_score_diff"] = np.nan

    score_columns = {"pitcher_team", "home_team", "away_team", "home_score", "away_score"}
    if score_columns.issubset(result.columns):
        home_score = pd.to_numeric(result["home_score"], errors="coerce")
        away_score = pd.to_numeric(result["away_score"], errors="coerce")
        pitcher_team = result["pitcher_team"].astype("string")
        home_team = result["home_team"].astype("string")
        away_team = result["away_team"].astype("string")
        home_pitcher = pitcher_team == home_team
        away_pitcher = pitcher_team == away_team
        result.loc[home_pitcher, "pitcher_score_diff"] = (
            home_score[home_pitcher] - away_score[home_pitcher]
        )
        result.loc[away_pitcher, "pitcher_score_diff"] = (
            away_score[away_pitcher] - home_score[away_pitcher]
        )

    missing = result["pitcher_score_diff"].isna()
    if {"bat_score", "fld_score"}.issubset(result.columns) and missing.any():
        bat_score = pd.to_numeric(result["bat_score"], errors="coerce")
        fld_score = pd.to_numeric(result["fld_score"], errors="coerce")
        result.loc[missing, "pitcher_score_diff"] = fld_score[missing] - bat_score[missing]
    return result


def add_real_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add context features only when derivable from real Statcast columns."""
    result = add_count_bucket(df)
    result = add_pitcher_game_pitch_count(result)
    result = add_pitcher_score_diff(result)
    result["batter_stand_code"] = result.get("stand", pd.Series(index=result.index)).map(
        {"R": 0.0, "L": 1.0}
    )
    if {"bat_score", "fld_score"}.issubset(result.columns):
        result["score_diff"] = pd.to_numeric(result["bat_score"], errors="coerce") - pd.to_numeric(
            result["fld_score"], errors="coerce"
        )
    elif "bat_score_diff" in result.columns:
        result["score_diff"] = pd.to_numeric(result["bat_score_diff"], errors="coerce")
    else:
        result["score_diff"] = np.nan

    if "pitcher_days_since_prev_game" in result.columns:
        result["days_rest"] = pd.to_numeric(result["pitcher_days_since_prev_game"], errors="coerce")
    if "n_thruorder_pitcher" in result.columns:
        result["times_through_order"] = pd.to_numeric(result["n_thruorder_pitcher"], errors="coerce")
    if "pitcher_days_until_next_game" in result.columns:
        result["recent_workload"] = pd.to_numeric(
            result["pitcher_days_until_next_game"], errors="coerce"
        )
    return result


def filter_pitch_types(df: pd.DataFrame, pitch_types: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    if pitch_types is None:
        return df.reset_index(drop=True)
    return df[df["pitch_type"].isin(pitch_types)].reset_index(drop=True)


def clean_pitch_features(
    df: pd.DataFrame,
    pitch_types: list[str] | tuple[str, ...] | None = ("FF", "SI"),
) -> pd.DataFrame:
    result = add_spin_axis_components(df)
    result = add_real_context_features(result)
    result = filter_pitch_types(result, pitch_types)
    numeric_columns = sorted({column for columns in FEATURE_GROUPS.values() for column in columns})
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame, feature_group: str = "physics_core") -> pd.DataFrame:
    if feature_group not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group: {feature_group}")
    columns = FEATURE_GROUPS[feature_group]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {feature_group}: {', '.join(missing)}")
    return df[columns].dropna().reset_index(drop=True)


def feature_availability_report(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for group_name, columns in FEATURE_GROUPS.items():
        available = [column for column in columns if column in df.columns]
        if len(available) != len(columns):
            report[group_name] = {
                "feature_count": len(columns),
                "available_feature_count": len(available),
                "rows_retained": 0,
                "feature_completeness": 0.0,
            }
            continue
        matrix = df[columns]
        report[group_name] = {
            "feature_count": len(columns),
            "available_feature_count": len(available),
            "rows_retained": int(matrix.dropna().shape[0]),
            "feature_completeness": float(matrix.notna().mean().mean()),
        }
    return report
