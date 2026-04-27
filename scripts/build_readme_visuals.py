#!/usr/bin/env python3
"""Build README visuals from real Pitcher Twin data and artifacts.

The README is meant to explain the product, not decorate it. These figures
therefore use real Statcast rows, generated app samples, validation boards,
and rolling validation artifacts.
"""

from __future__ import annotations

import io
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "readme"
SKUBAL_DATA_PATH = ROOT / "data" / "processed" / "skubal_2025.csv"
SITE_DATA_PATH = ROOT / "site" / "data.json"
SKUBAL_BOARD_PATH = ROOT / "outputs" / "validation_board_skubal_2025_top3_v4" / "leaderboard.csv"
LATEST_BOARD_PATH = ROOT / "outputs" / "validation_board_latest_statcast_top3_v4" / "leaderboard.csv"
TOURNAMENT_REPORT_PATH = ROOT / "outputs" / "model_tournament_skubal_2025_ff" / "model_tournament_report.json"
ROLLING_BOARD_PATH = ROOT / "outputs" / "rolling_validation_skubal_2025_ff" / "rolling_validation_board.json"

FAMILY_FEATURES = [
    "plate_x",
    "plate_z",
    "pfx_x",
    "pfx_z",
    "release_speed",
    "release_spin_rate",
]

FAMILY_COLORS = ["#197a4d", "#2f6f82", "#d7a531", "#a43d32", "#5b4d91"]
REAL_COLOR = "#197a4d"
GEN_COLOR = "#d7a531"
INK = "#191815"
MUTED = "#625b50"
GRID = "#d8cdb8"
TARGET = "#d7a531"
FAIL = "#a43d32"


@dataclass(frozen=True)
class FamilyModel:
    scaler: StandardScaler
    kmeans: KMeans
    label_map: dict[int, int]
    centers: pd.DataFrame
    labels: list[str]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#3a3731")
        spine.set_linewidth(1.1)
    ax.tick_params(colors=INK, labelsize=9)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fig_to_image(fig: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=145, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _save_gif(path: Path, frames: list[Image.Image], *, duration: int = 1100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)


def _load_skubal() -> pd.DataFrame:
    data = pd.read_csv(SKUBAL_DATA_PATH)
    data["game_date"] = pd.to_datetime(data["game_date"])
    return data


def _load_skubal_ff() -> pd.DataFrame:
    data = _load_skubal()
    ff = data[(data["pitcher"] == 669373) & (data["pitch_type"] == "FF")].copy()
    ff = ff.dropna(subset=FAMILY_FEATURES)
    ff = ff.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    return ff


def _fit_family_model(ff: pd.DataFrame, *, n_families: int = 5) -> tuple[pd.DataFrame, FamilyModel]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(ff[FAMILY_FEATURES])
    kmeans = KMeans(n_clusters=n_families, n_init=40, random_state=19)
    original_labels = kmeans.fit_predict(scaled)

    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FAMILY_FEATURES)
    centers["original_label"] = range(n_families)
    # Order families visually: high locations first, then left-to-right.
    centers = centers.sort_values(["plate_z", "plate_x"], ascending=[False, True]).reset_index(drop=True)
    label_map = {int(row.original_label): int(index) for index, row in centers.iterrows()}

    labeled = ff.copy()
    labeled["style_family"] = [label_map[int(label)] for label in original_labels]
    centers["style_family"] = range(n_families)
    centers = centers.set_index("style_family")
    labels = [_family_label(index, centers.loc[index]) for index in range(n_families)]
    return labeled, FamilyModel(scaler=scaler, kmeans=kmeans, label_map=label_map, centers=centers, labels=labels)


def _predict_families(frame: pd.DataFrame, model: FamilyModel) -> np.ndarray:
    clean = frame[FAMILY_FEATURES].astype(float)
    original = model.kmeans.predict(model.scaler.transform(clean))
    return np.array([model.label_map[int(label)] for label in original])


def _family_label(index: int, center: pd.Series) -> str:
    if center["plate_z"] >= 3.25:
        height = "high zone"
    elif center["plate_z"] <= 2.45:
        height = "low zone"
    else:
        height = "middle zone"

    if center["plate_x"] <= -0.45:
        lane = "left edge"
    elif center["plate_x"] >= 0.45:
        lane = "right edge"
    else:
        lane = "center lane"

    return f"F{index + 1}: {height}, {lane}"


def _draw_strike_zone(ax: plt.Axes) -> None:
    zone = plt.Rectangle((-0.83, 1.5), 1.66, 2.0, fill=False, color=INK, linewidth=1.8)
    ax.add_patch(zone)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.7, 4.9)
    ax.set_xlabel("plate_x (ft)")
    ax.set_ylabel("plate_z (ft)")
    ax.set_aspect("equal", adjustable="box")


def _site_candidate(key: str = "skubal_ff") -> dict[str, Any]:
    site_data = _load_json(SITE_DATA_PATH)
    for candidate in site_data["candidates"]:
        if candidate["key"] == key:
            return candidate
    raise KeyError(f"Missing candidate {key!r} in {SITE_DATA_PATH}")


def build_pitch_family_map(ff: pd.DataFrame, family_model: FamilyModel) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.07, right=0.98, wspace=0.26)
    fig.suptitle("Pitcher Twin learns style families inside one pitch type", fontsize=20, fontweight="bold", color=INK, y=0.98)
    fig.text(
        0.5,
        0.91,
        "Tarik Skubal 2025 four-seam fastballs only: clusters are learned from real pitch location, movement, velocity, and spin.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    for family_id in range(len(family_model.labels)):
        subset = ff[ff["style_family"] == family_id]
        color = FAMILY_COLORS[family_id]
        label = f"{family_model.labels[family_id]} ({len(subset)} pitches)"
        axes[0].scatter(subset["plate_x"], subset["plate_z"], s=18, alpha=0.63, color=color, label=label, edgecolor="none")
        axes[1].scatter(subset["pfx_x"] * 12, subset["pfx_z"] * 12, s=18, alpha=0.63, color=color, edgecolor="none")

        center = family_model.centers.loc[family_id]
        axes[0].scatter(center["plate_x"], center["plate_z"], s=165, color=color, edgecolor=INK, linewidth=1.4)
        axes[0].text(center["plate_x"] + 0.05, center["plate_z"] + 0.05, f"F{family_id + 1}", fontsize=10, weight="bold")
        axes[1].scatter(center["pfx_x"] * 12, center["pfx_z"] * 12, s=165, color=color, edgecolor=INK, linewidth=1.4)
        axes[1].text(center["pfx_x"] * 12 + 0.25, center["pfx_z"] * 12 + 0.25, f"F{family_id + 1}", fontsize=10, weight="bold")

    _draw_strike_zone(axes[0])
    axes[0].set_title("Where the FF families finish at the plate", fontsize=13, fontweight="bold", color=INK)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.01, -0.33), fontsize=8, frameon=False, ncol=1)

    axes[1].axhline(0, color=GRID, linewidth=1.1)
    axes[1].axvline(0, color=GRID, linewidth=1.1)
    axes[1].set_title("How those same families differ in movement", fontsize=13, fontweight="bold", color=INK)
    axes[1].set_xlabel("horizontal movement, pfx_x (inches)")
    axes[1].set_ylabel("vertical movement, pfx_z (inches)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlim(-4.0, 11.0)
    axes[1].set_ylim(10.5, 22.0)

    for ax in axes:
        _style_axes(ax)

    _save_fig(fig, ASSET_DIR / "pitch-family-inside-ff.png")


def build_real_vs_generated_diagnostics(family_model: FamilyModel) -> None:
    candidate = _site_candidate()
    real = pd.DataFrame(candidate["real_holdout"]).dropna(subset=FAMILY_FEATURES)
    generated = pd.DataFrame(candidate["samples"]["even_R"]).dropna(subset=FAMILY_FEATURES)
    generated["style_family"] = _predict_families(generated, family_model)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.08, right=0.98, hspace=0.35, wspace=0.18)
    fig.suptitle("Generated Skubal FF samples compared with later real holdout pitches", fontsize=19, fontweight="bold", color=INK, y=0.975)
    fig.text(
        0.5,
        0.915,
        "Real holdout rows come from site/data.json; generated rows are model samples for one app context.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    axes[0, 0].scatter(real["plate_x"], real["plate_z"], s=18, alpha=0.52, color=REAL_COLOR, label=f"real holdout ({len(real)})")
    axes[0, 0].scatter(generated["plate_x"], generated["plate_z"], s=36, alpha=0.76, color=GEN_COLOR, marker="x", label=f"generated ({len(generated)})")
    _draw_strike_zone(axes[0, 0])
    axes[0, 0].set_title("Plate-location envelope", fontsize=12, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=9)

    axes[0, 1].scatter(real["pfx_x"] * 12, real["pfx_z"] * 12, s=18, alpha=0.52, color=REAL_COLOR)
    axes[0, 1].scatter(generated["pfx_x"] * 12, generated["pfx_z"] * 12, s=36, alpha=0.76, color=GEN_COLOR, marker="x")
    axes[0, 1].set_title("Movement envelope", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("pfx_x (inches)")
    axes[0, 1].set_ylabel("pfx_z (inches)")

    bins = np.linspace(93.0, 102.5, 24)
    axes[1, 0].hist(real["release_speed"], bins=bins, color=REAL_COLOR, alpha=0.48, density=True, label="real")
    axes[1, 0].hist(generated["release_speed"], bins=bins, color=GEN_COLOR, alpha=0.55, density=True, label="generated")
    axes[1, 0].set_title("Velocity distribution", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("release_speed (mph)")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend(frameon=False, fontsize=9)

    counts = generated["style_family"].value_counts(normalize=True).reindex(range(len(family_model.labels)), fill_value=0)
    axes[1, 1].bar(range(len(counts)), counts.values, color=FAMILY_COLORS, edgecolor=INK, linewidth=0.8)
    axes[1, 1].set_title("Generated sample mix across learned FF families", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(range(len(counts)))
    axes[1, 1].set_xticklabels([f"F{i + 1}" for i in range(len(counts))])
    axes[1, 1].set_ylabel("sample share")
    axes[1, 1].set_ylim(0, max(0.45, counts.max() + 0.1))
    for index, value in enumerate(counts.values):
        axes[1, 1].text(index, value + 0.015, f"{value:.0%}", ha="center", fontsize=9, fontweight="bold")

    for ax in axes.ravel():
        _style_axes(ax)

    _save_fig(fig, ASSET_DIR / "real-vs-generated-diagnostics.png")


def build_family_probability_shift_gif(family_model: FamilyModel) -> None:
    candidate = _site_candidate()
    contexts = [
        ("0-0 vs RHB", "first_pitch_R"),
        ("0-2 vs RHB", "ahead_R"),
        ("1-1 vs RHB", "even_R"),
        ("2-0 vs RHB", "behind_R"),
        ("3-2 vs LHB", "full_L"),
    ]
    frames: list[Image.Image] = []
    for label, sample_key in contexts:
        generated = pd.DataFrame(candidate["samples"][sample_key]).dropna(subset=FAMILY_FEATURES)
        generated["style_family"] = _predict_families(generated, family_model)
        probs = generated["style_family"].value_counts(normalize=True).reindex(range(len(family_model.labels)), fill_value=0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
        fig.subplots_adjust(top=0.80, bottom=0.19, left=0.06, right=0.98, wspace=0.30)
        fig.suptitle(f"Conditional generator shifts the mix of FF style families: {label}", fontsize=17, fontweight="bold", color=INK, y=0.975)
        fig.text(
            0.5,
            0.885,
            "Each frame uses actual generated samples from site/data.json and assigns them to the learned Skubal FF families.",
            ha="center",
            fontsize=10.5,
            color=MUTED,
        )

        for family_id in range(len(family_model.labels)):
            subset = generated[generated["style_family"] == family_id]
            color = FAMILY_COLORS[family_id]
            axes[0].scatter(subset["plate_x"], subset["plate_z"], s=34, alpha=0.78, color=color, label=f"F{family_id + 1}")
            axes[1].scatter(subset["pfx_x"] * 12, subset["pfx_z"] * 12, s=34, alpha=0.78, color=color)

        _draw_strike_zone(axes[0])
        axes[0].set_title("Generated plate cloud", fontsize=12, fontweight="bold")
        axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))

        axes[1].set_title("Generated movement cloud", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("pfx_x (inches)")
        axes[1].set_ylabel("pfx_z (inches)")
        axes[1].set_xlim(-4.0, 11.0)
        axes[1].set_ylim(10.5, 22.0)
        axes[1].set_aspect("equal", adjustable="box")

        axes[2].barh(range(len(probs)), probs.values, color=FAMILY_COLORS, edgecolor=INK, linewidth=0.8)
        axes[2].set_yticks(range(len(probs)))
        axes[2].set_yticklabels([f"F{i + 1}" for i in range(len(probs))])
        axes[2].invert_yaxis()
        axes[2].set_xlim(0, 0.55)
        axes[2].set_xlabel("share of generated samples")
        axes[2].set_title("Family probability mix", fontsize=12, fontweight="bold")
        for index, value in enumerate(probs.values):
            axes[2].text(value + 0.015, index, f"{value:.0%}", va="center", fontsize=9, fontweight="bold")

        for ax in axes:
            _style_axes(ax)
        frames.append(_fig_to_image(fig))

    _save_gif(ASSET_DIR / "family-probability-shift.gif", frames, duration=1250)


def _load_layer_results() -> pd.DataFrame:
    report = _load_json(TOURNAMENT_REPORT_PATH)
    rows = []
    labels = {
        "command_representation": "command",
        "movement_only": "movement",
        "trajectory_only": "trajectory",
        "release_only": "release",
        "physics_core": "physics core",
    }
    for layer, label in labels.items():
        model_name = report["best_by_layer"][layer]
        result = report["layer_results"][layer][model_name]
        rows.append(
            {
                "layer": label,
                "mean_auc": float(result["mean_auc"]),
                "pass_rate": float(result["pass_rate"]),
                "model": model_name,
            }
        )
    return pd.DataFrame(rows)


def build_overall_results_dashboard() -> None:
    skubal_board = pd.read_csv(SKUBAL_BOARD_PATH)
    latest_board = pd.read_csv(LATEST_BOARD_PATH)
    layer_results = _load_layer_results()
    rolling = _load_json(ROLLING_BOARD_PATH)
    folds = pd.DataFrame(rolling["folds"])

    fig = plt.figure(figsize=(15.5, 12))
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.07, right=0.98, hspace=0.55, wspace=0.28)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05])
    fig.suptitle("Full overall results: what validated, what is candidate, what is still diagnostic", fontsize=19, fontweight="bold", color=INK, y=0.985)
    fig.text(
        0.5,
        0.94,
        "C2ST AUC: 0.50 is ideal, <= 0.60 is the realism target, lower is better.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    ax_pitch = fig.add_subplot(grid[0, 0])
    labels = [f"Skubal {row.pitch_type}" for _, row in skubal_board.iterrows()]
    colors = [REAL_COLOR if row.physics_core_mean_auc <= 0.60 else TARGET if row.physics_core_mean_auc < 0.67 else FAIL for _, row in skubal_board.iterrows()]
    ax_pitch.barh(labels, skubal_board["physics_core_mean_auc"], color=colors, edgecolor=INK, linewidth=0.7)
    ax_pitch.axvline(0.60, color=TARGET, linewidth=2, linestyle="--", label="target 0.60")
    ax_pitch.axvline(0.50, color=REAL_COLOR, linewidth=2, linestyle=":", label="ideal 0.50")
    ax_pitch.set_xlim(0.48, 0.72)
    ax_pitch.set_xlabel("physics-core mean AUC")
    ax_pitch.set_title("Skubal 2025 pitch-type board", fontsize=12, fontweight="bold")
    for index, row in skubal_board.iterrows():
        ax_pitch.text(row.physics_core_mean_auc + 0.006, index, f"{row.physics_core_mean_auc:.3f} / {row.physics_core_pass_rate:.0%}", va="center", fontsize=9)
    ax_pitch.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axes(ax_pitch)

    ax_latest = fig.add_subplot(grid[0, 1])
    latest_labels = [f"{row.pitcher_name.split()[-1]} {row.pitch_type}" for _, row in latest_board.iterrows()]
    colors = [REAL_COLOR if row.physics_core_mean_auc <= 0.60 else TARGET if row.physics_core_mean_auc < 0.67 else FAIL for _, row in latest_board.iterrows()]
    ax_latest.barh(latest_labels, latest_board["physics_core_mean_auc"], color=colors, edgecolor=INK, linewidth=0.7)
    ax_latest.axvline(0.60, color=TARGET, linewidth=2, linestyle="--")
    ax_latest.axvline(0.50, color=REAL_COLOR, linewidth=2, linestyle=":")
    ax_latest.set_xlim(0.48, 0.72)
    ax_latest.set_xlabel("physics-core mean AUC")
    ax_latest.set_title("Latest-Statcast candidate board", fontsize=12, fontweight="bold")
    for index, row in latest_board.iterrows():
        ax_latest.text(row.physics_core_mean_auc + 0.006, index, f"{row.physics_core_mean_auc:.3f} / {row.physics_core_pass_rate:.0%}", va="center", fontsize=9)
    _style_axes(ax_latest)

    ax_layers = fig.add_subplot(grid[1, :])
    layer_colors = [REAL_COLOR if value <= 0.60 else TARGET for value in layer_results["mean_auc"]]
    ax_layers.bar(layer_results["layer"], layer_results["mean_auc"], color=layer_colors, edgecolor=INK, linewidth=0.8)
    ax_layers.axhline(0.60, color=TARGET, linewidth=2, linestyle="--", label="target 0.60")
    ax_layers.axhline(0.50, color=REAL_COLOR, linewidth=2, linestyle=":", label="ideal 0.50")
    ax_layers.set_ylim(0.48, 0.64)
    ax_layers.set_ylabel("mean AUC")
    ax_layers.set_title("Skubal FF layer tournament: each part of the pitch is scored separately", fontsize=12, fontweight="bold")
    for index, row in layer_results.iterrows():
        ax_layers.text(index, row.mean_auc + 0.006, f"{row.mean_auc:.3f}\npass {row.pass_rate:.0%}", ha="center", va="bottom", fontsize=9)
    ax_layers.legend(frameon=False, fontsize=8, loc="upper left")
    _style_axes(ax_layers)

    ax_roll = fig.add_subplot(grid[2, 0])
    ax_roll.plot(folds["fold_index"], folds["physics_core_mean_auc"], color=INK, linewidth=2.3, marker="o")
    ax_roll.axhline(0.60, color=TARGET, linewidth=2, linestyle="--", label="fold target 0.60")
    ax_roll.axhline(0.80, color=FAIL, linewidth=2, linestyle="--", label="worst-fold ceiling 0.80")
    ax_roll.set_ylim(0.55, 0.96)
    ax_roll.set_xlabel("rolling fold")
    ax_roll.set_ylabel("physics-core AUC")
    ax_roll.set_title("Rolling future-window stress test", fontsize=12, fontweight="bold")
    ax_roll.legend(frameon=False, fontsize=8)
    _style_axes(ax_roll)

    ax_summary = fig.add_subplot(grid[2, 1])
    ax_summary.axis("off")
    current = rolling["primary_scoreboard"]["current"]
    lines = [
        ("Built", "real Statcast pipeline, conditional sampler, validation boards, static app, Trajekt-style export"),
        ("Validated", "Skubal FF single temporal split and all Skubal FF component layers are under the 0.60 target"),
        ("Candidate", "Isaac Mattson FF candidate on latest Statcast board; Skubal SI improved but remains diagnostic"),
        ("Still hard", "rolling future-window robustness and non-FF physics-core validation"),
        ("Rolling mean", f"{current['mean_rolling_physics_core_auc']:.3f}"),
        ("Rolling target hit rate", f"{current['target_hit_rate']:.0%}"),
        ("Worst rolling fold", f"{current['worst_fold_physics_core_auc']:.3f}"),
    ]
    y = 0.97
    ax_summary.text(0.0, y, "Overall project status", fontsize=14, fontweight="bold", color=INK, transform=ax_summary.transAxes)
    y -= 0.12
    for label, value in lines:
        ax_summary.text(0.0, y, label, fontsize=10.5, fontweight="bold", color=INK, transform=ax_summary.transAxes)
        wrapped = "\n".join(textwrap.wrap(value, width=72))
        ax_summary.text(0.36, y, wrapped, fontsize=10.0, color=MUTED, transform=ax_summary.transAxes)
        y -= 0.095 + 0.055 * max(0, wrapped.count("\n"))

    _save_fig(fig, ASSET_DIR / "overall-results-dashboard.png")


def build_asset_readme() -> None:
    note = """# README Evidence Visual Assets

All displayed README visuals are generated from tracked real-data artifacts.

## Data Sources

- `data/processed/skubal_2025.csv`: real Tarik Skubal 2025 Statcast rows.
- `site/data.json`: real holdout rows and generated app samples.
- `outputs/validation_board_skubal_2025_top3_v4/leaderboard.csv`: Skubal pitch-type board.
- `outputs/validation_board_latest_statcast_top3_v4/leaderboard.csv`: latest-Statcast candidate board.
- `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json`: layer tournament.
- `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.json`: rolling future-window validation.

## Displayed Assets

- `pitch-family-inside-ff.png`: learned style families inside Skubal's FF.
- `overall-results-dashboard.png`: full project result summary.
- `real-vs-generated-diagnostics.png`: real holdout vs generated Skubal FF diagnostics.
- `family-probability-shift.gif`: context-driven family probability changes.
"""
    (ASSET_DIR / "README.md").write_text(note, encoding="utf-8")


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    ff = _load_skubal_ff()
    labeled_ff, family_model = _fit_family_model(ff)
    build_pitch_family_map(labeled_ff, family_model)
    build_real_vs_generated_diagnostics(family_model)
    build_family_probability_shift_gif(family_model)
    build_overall_results_dashboard()
    build_asset_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
