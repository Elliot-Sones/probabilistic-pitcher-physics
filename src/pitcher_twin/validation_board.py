"""Generalized validation board utilities for Pitcher Twin model tournaments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pitcher_twin.features import FEATURE_GROUPS
from pitcher_twin.model_router import build_model_route
from pitcher_twin.validator import temporal_train_holdout


@dataclass(frozen=True)
class CandidateCriteria:
    """Minimum real-data evidence required before running a pitcher/pitch board row."""

    min_pitches: int = 300
    min_games: int = 8
    min_holdout: int = 60
    top: int = 5
    train_fraction: float = 0.70
    feature_group: str = "physics_core"


@dataclass(frozen=True)
class RollingWindow:
    """A chronological train/holdout game window for temporal validation."""

    window_index: int
    train: pd.DataFrame
    holdout: pd.DataFrame
    train_game_count: int
    holdout_game_count: int
    train_row_count: int
    holdout_row_count: int
    train_end_game: str
    holdout_start_game: str
    holdout_end_game: str


def slugify_label(value: str) -> str:
    """Return a stable lowercase file-system label."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def candidate_pitcher_pitches(
    frame: pd.DataFrame,
    criteria: CandidateCriteria | None = None,
) -> list[dict[str, Any]]:
    """Rank pitcher/pitch pairs that have enough real temporal evidence."""
    criteria = criteria or CandidateCriteria()
    if criteria.feature_group not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group: {criteria.feature_group}")
    required = {"pitcher", "pitch_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing candidate columns: {', '.join(missing)}")

    feature_columns = FEATURE_GROUPS[criteria.feature_group]
    candidates: list[dict[str, Any]] = []
    for (pitcher, pitch_type), group in frame.groupby(["pitcher", "pitch_type"], dropna=False):
        group = group.copy()
        pitch_count = int(len(group))
        game_count = _game_count(group)
        if pitch_count < criteria.min_pitches or game_count < criteria.min_games:
            continue
        _, holdout = temporal_train_holdout(group, train_fraction=criteria.train_fraction)
        holdout_count = int(len(holdout))
        if holdout_count < criteria.min_holdout:
            continue
        physics_core_complete_count = _complete_count(group, feature_columns)
        if physics_core_complete_count < criteria.min_pitches:
            continue
        pitcher_name = _pitcher_name(group)
        candidate_score = float(
            physics_core_complete_count
            + game_count * 10.0
            + holdout_count * 0.25
        )
        candidates.append(
            {
                "pitcher": _python_scalar(pitcher),
                "pitcher_name": pitcher_name,
                "pitch_type": str(pitch_type),
                "pitch_count": pitch_count,
                "game_count": game_count,
                "holdout_count": holdout_count,
                "physics_core_complete_count": physics_core_complete_count,
                "candidate_score": candidate_score,
                "train_fraction": float(criteria.train_fraction),
            }
        )

    return sorted(
        candidates,
        key=lambda row: (
            row["candidate_score"],
            row["physics_core_complete_count"],
            row["game_count"],
        ),
        reverse=True,
    )[: criteria.top]


def rolling_game_windows(
    frame: pd.DataFrame,
    *,
    min_train_games: int = 8,
    holdout_games: int = 2,
    max_windows: int = 4,
) -> list[RollingWindow]:
    """Create chronological rolling windows with future games held out."""
    if min_train_games <= 0:
        raise ValueError("min_train_games must be positive.")
    if holdout_games <= 0:
        raise ValueError("holdout_games must be positive.")
    if max_windows <= 0:
        return []
    games = _ordered_games(frame)
    if len(games) < min_train_games + holdout_games:
        return []

    windows: list[RollingWindow] = []
    for start in range(0, len(games) - min_train_games - holdout_games + 1):
        train_games = games[: min_train_games + start]
        holdout_slice = games[min_train_games + start : min_train_games + start + holdout_games]
        train = _rows_for_games(frame, train_games)
        holdout = _rows_for_games(frame, holdout_slice)
        windows.append(
            RollingWindow(
                window_index=len(windows) + 1,
                train=train,
                holdout=holdout,
                train_game_count=len(train_games),
                holdout_game_count=len(holdout_slice),
                train_row_count=int(len(train)),
                holdout_row_count=int(len(holdout)),
                train_end_game=str(train_games[-1]),
                holdout_start_game=str(holdout_slice[0]),
                holdout_end_game=str(holdout_slice[-1]),
            )
        )
        if len(windows) >= max_windows:
            break
    return windows


def summarize_tournament_report(
    report: dict[str, Any],
    *,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Condense a tournament report into one leaderboard row plus layer detail."""
    target_auc = float(report.get("target_auc", 0.60))
    target_pass_rate = float(report.get("target_pass_rate", 0.80))
    best_by_layer = dict(report.get("best_by_layer", {}))
    layer_results = dict(report.get("layer_results", {}))
    layer_statuses: dict[str, dict[str, Any]] = {}
    for layer, model_name in best_by_layer.items():
        metrics = dict(layer_results.get(layer, {}).get(model_name, {}))
        mean_auc = float(metrics.get("mean_auc", float("nan")))
        pass_rate = float(metrics.get("pass_rate", 0.0))
        layer_statuses[layer] = {
            "status": _validation_status(mean_auc, pass_rate, target_auc, target_pass_rate),
            "best_model": model_name,
            "mean_auc": mean_auc,
            "std_auc": float(metrics.get("std_auc", 0.0)),
            "pass_rate": pass_rate,
            "top_leakage_features": list(metrics.get("top_leakage_features", [])),
        }

    physics = layer_statuses.get("physics_core", {})
    physics_status = physics.get("status", "diagnostic")
    artifact_status = {
        "validated": "validated_temporal_success",
        "candidate": "physics_core_candidate",
        "diagnostic": "physics_core_diagnostic",
    }.get(str(physics_status), "physics_core_diagnostic")
    candidate = candidate or {}
    model_route = build_model_route(report)
    summary = {
        "pitcher": candidate.get("pitcher", report.get("pitcher_id")),
        "pitcher_name": candidate.get("pitcher_name", report.get("pitcher_name", "unknown")),
        "pitch_type": candidate.get("pitch_type", report.get("pitch_type", "unknown")),
        "pitch_count": int(candidate.get("pitch_count", report.get("rows_subset", 0) or 0)),
        "game_count": int(candidate.get("game_count", 0) or 0),
        "holdout_count": int(candidate.get("holdout_count", report.get("n_holdout", 0) or 0)),
        "n_train": int(report.get("n_train", 0) or 0),
        "n_holdout": int(report.get("n_holdout", 0) or 0),
        "repeat_count": int(report.get("repeat_count", 0) or 0),
        "sample_count": int(report.get("sample_count", 0) or 0),
        "target_auc": target_auc,
        "target_pass_rate": target_pass_rate,
        "candidate_default": bool(report.get("candidate_default", False)),
        "artifact_status": artifact_status,
        "best_by_layer": best_by_layer,
        "best_physics_core_model": report.get(
            "best_physics_core_model",
            physics.get("best_model", "unknown"),
        ),
        "physics_core_mean_auc": float(physics.get("mean_auc", float("nan"))),
        "physics_core_pass_rate": float(physics.get("pass_rate", 0.0)),
        "layer_statuses": layer_statuses,
        "model_route": model_route,
        "top_leakage_features": list(physics.get("top_leakage_features", [])),
    }
    return summary


def leaderboard_frame(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a compact sortable board from candidate summaries."""
    rows = []
    for summary in summaries:
        rows.append(
            {
                "pitcher": summary["pitcher"],
                "pitcher_name": summary["pitcher_name"],
                "pitch_type": summary["pitch_type"],
                "pitch_count": summary["pitch_count"],
                "game_count": summary["game_count"],
                "holdout_count": summary["holdout_count"],
                "best_physics_core_model": summary["best_physics_core_model"],
                "physics_core_mean_auc": summary["physics_core_mean_auc"],
                "physics_core_pass_rate": summary["physics_core_pass_rate"],
                "artifact_status": summary["artifact_status"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["physics_core_mean_auc", "physics_core_pass_rate"],
        ascending=[True, False],
    )


def render_scorecard_markdown(
    summary: dict[str, Any],
    *,
    rolling_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Render a presentation-ready per-candidate scorecard."""
    title = f"{summary['pitcher_name']} {summary['pitch_type']}"
    lines = [
        f"# Pitcher Twin Scorecard: {title}",
        "",
        "## Verdict",
        "",
        f"- Artifact status: `{summary['artifact_status']}`",
        f"- Best physics-core model: `{summary['best_physics_core_model']}`",
        f"- Physics-core C2ST AUC: `{summary['physics_core_mean_auc']:.3f}`",
        f"- Physics-core pass rate: `{summary['physics_core_pass_rate']:.2f}`",
        (
            f"- Target: C2ST AUC <= `{summary['target_auc']:.3f}` "
            f"and pass rate >= `{summary['target_pass_rate']:.2f}`"
        ),
        "",
        (
            "C2ST AUC is the held-out classifier two-sample score. Lower is better; "
            "`0.50` means the classifier cannot reliably distinguish real held-out "
            "pitches from generated pitches."
        ),
        "",
        "## Model Route",
        "",
        f"- Route status: `{summary['model_route']['route_status']}`",
        f"- Pitch family: `{summary['model_route']['pitch_family']}`",
        (
            "- Recommended physics model: "
            f"`{summary['model_route']['recommended_physics_model']}`"
        ),
        "- Validated layers: "
        + _format_route_groups(summary["model_route"]["validated_feature_groups"]),
        "- Candidate layers: "
        + _format_route_groups(summary["model_route"]["candidate_feature_groups"]),
        "- Diagnostic layers: "
        + _format_route_groups(summary["model_route"]["diagnostic_feature_groups"]),
        "",
        "## Data",
        "",
        f"- Pitcher id: `{summary['pitcher']}`",
        f"- Pitch count: `{summary['pitch_count']}`",
        f"- Games: `{summary['game_count']}`",
        f"- Temporal train rows: `{summary['n_train']}`",
        f"- Temporal holdout rows: `{summary['n_holdout']}`",
        f"- Repeats: `{summary['repeat_count']}`",
        "",
        "## Layer Results",
        "",
        "| Layer | Status | Best model | Mean AUC | Pass rate |",
        "|---|---|---|---:|---:|",
    ]
    for layer, row in summary["layer_statuses"].items():
        lines.append(
            (
                f"| {layer} | {row['status']} | {row['best_model']} | "
                f"{row['mean_auc']:.3f} | {row['pass_rate']:.2f} |"
            )
        )

    leakage = summary.get("top_leakage_features", [])
    if leakage:
        lines.extend(["", "## Main Classifier Clues", ""])
        for row in leakage[:5]:
            lines.append(f"- `{row['feature']}` importance `{float(row['importance']):.3f}`")

    if rolling_rows:
        lines.extend(
            [
                "",
                "## Rolling Temporal Windows",
                "",
                (
                    "| Window | Train games | Holdout games | Best model | "
                    "Physics AUC | Pass rate | Status |"
                ),
                "|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in rolling_rows:
            lines.append(
                "| "
                f"{row['window_index']} | "
                f"{row['train_games']} | "
                f"{row['holdout_games']} | "
                f"{row['best_physics_core_model']} | "
                f"{float(row['physics_core_mean_auc']):.3f} | "
                f"{float(row['physics_core_pass_rate']):.2f} | "
                f"{row['artifact_status']} |"
            )
    return "\n".join(lines) + "\n"


def render_validation_board_markdown(
    leaderboard: pd.DataFrame,
    summaries: list[dict[str, Any]],
) -> str:
    """Render the cross-candidate board as one clean markdown artifact."""
    lines = [
        "# Pitcher Twin Validation Board",
        "",
        (
            "This board runs the same temporal model-tournament protocol across selected "
            "real pitcher/pitch candidates."
        ),
        "",
        (
            "| Rank | Pitcher | Pitch | Games | Holdout | Best physics model | "
            "Physics AUC | Pass rate | Status |"
        ),
        "|---:|---|---|---:|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(leaderboard.to_dict("records"), start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{row['pitcher_name']} | "
            f"{row['pitch_type']} | "
            f"{int(row['game_count'])} | "
            f"{int(row['holdout_count'])} | "
            f"{row['best_physics_core_model']} | "
            f"{float(row['physics_core_mean_auc']):.3f} | "
            f"{float(row['physics_core_pass_rate']):.2f} | "
            f"{row['artifact_status']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    status_counts = pd.Series([summary["artifact_status"] for summary in summaries]).value_counts()
    for status, count in status_counts.items():
        lines.append(f"- `{status}`: `{int(count)}` candidate(s)")
    lines.append("")
    lines.append(
        "Use this board to separate one-off success from repeatable pitcher-twin generalization."
    )
    return "\n".join(lines) + "\n"


def _validation_status(
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


def _format_route_groups(groups: list[str]) -> str:
    if not groups:
        return "`none`"
    return ", ".join(f"`{group}`" for group in groups)


def _game_count(frame: pd.DataFrame) -> int:
    if "game_pk" in frame.columns:
        return int(frame["game_pk"].nunique(dropna=True))
    if "game_date" in frame.columns:
        return int(frame["game_date"].nunique(dropna=True))
    return 1


def _complete_count(frame: pd.DataFrame, columns: list[str]) -> int:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return 0
    return int(frame[columns].dropna().shape[0])


def _pitcher_name(frame: pd.DataFrame) -> str:
    for column in ("pitcher_name", "player_name"):
        if column in frame.columns:
            names = frame[column].dropna()
            if not names.empty:
                return str(names.iloc[0])
    return str(frame["pitcher"].dropna().iloc[0])


def _ordered_games(frame: pd.DataFrame) -> list[Any]:
    if "game_pk" not in frame.columns:
        dates = (
            frame["game_date"].drop_duplicates().tolist()
            if "game_date" in frame.columns
            else [0]
        )
        return dates
    columns = ["game_pk"] + (["game_date"] if "game_date" in frame.columns else [])
    games = frame[columns].drop_duplicates()
    sort_columns = [column for column in ["game_date", "game_pk"] if column in games.columns]
    games = games.sort_values(sort_columns, kind="mergesort")
    return games["game_pk"].tolist()


def _rows_for_games(frame: pd.DataFrame, games: list[Any]) -> pd.DataFrame:
    if "game_pk" in frame.columns:
        return frame[frame["game_pk"].isin(games)].copy()
    return frame[frame["game_date"].isin(games)].copy()


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
