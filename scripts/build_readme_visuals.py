#!/usr/bin/env python3
"""Build README visuals from real validation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import imageio.v3 as iio  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pitcher_twin.features import clean_pitch_features  # noqa: E402


EXPECTED_ASSETS = [
    "skubal_ff_variation.png",
    "skubal_ff_pitch_sequence.gif",
    "v21_results_auc.png",
    "pitcher_twin_pipeline.svg",
    "pitcher_twin_pipeline.excalidraw",
    "v21_physics_chain.svg",
    "v21_physics_chain.excalidraw",
]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "readme"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "factorized_skubal_2025_ff" / "factorized_validation_report.json"
)
SKUBAL_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "skubal_2025.csv",
    Path("/Users/elliot18/Desktop/Home/Projects/pitch-pitcher-twin/data/processed/skubal_2025.csv"),
]


def _resolve_skubal_data(data_path: Path | None) -> Path:
    candidates = [data_path] if data_path is not None else SKUBAL_DATA_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(f"Could not find Skubal CSV. Searched: {searched}")


def _load_skubal_ff(data_path: Path | None = None) -> pd.DataFrame:
    source = _resolve_skubal_data(data_path)
    frame = clean_pitch_features(pd.read_csv(source), pitch_types=None)
    subset = frame[(frame["pitcher"] == 669373) & (frame["pitch_type"] == "FF")].copy()
    if subset.empty:
        raise RuntimeError("No Tarik Skubal FF rows found in the requested data file.")
    return subset.reset_index(drop=True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_zone(ax: plt.Axes) -> None:
    zone = Rectangle(
        (-0.83, 1.50),
        1.66,
        2.00,
        linewidth=1.5,
        edgecolor="#252a34",
        facecolor="none",
        alpha=0.9,
    )
    ax.add_patch(zone)


def build_variation_plot(skubal_ff: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "skubal_ff_variation.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    color = pd.to_numeric(skubal_ff["pitcher_game_pitch_count"], errors="coerce")

    axes[0].scatter(
        skubal_ff["plate_x"],
        skubal_ff["plate_z"],
        c=color,
        cmap="viridis",
        s=22,
        alpha=0.72,
        linewidths=0,
    )
    _draw_zone(axes[0])
    axes[0].axvline(0, color="#8a93a3", linewidth=0.8, alpha=0.6)
    axes[0].set_title("Command variation at the plate", fontsize=13, weight="bold")
    axes[0].set_xlabel("plate_x (ft)")
    axes[0].set_ylabel("plate_z (ft)")
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(0.0, 5.2)
    axes[0].grid(True, color="#e7e9ef", linewidth=0.8)

    scatter = axes[1].scatter(
        skubal_ff["pfx_x"],
        skubal_ff["pfx_z"],
        c=color,
        cmap="viridis",
        s=22,
        alpha=0.72,
        linewidths=0,
    )
    axes[1].set_title("Movement variation for the same FF", fontsize=13, weight="bold")
    axes[1].set_xlabel("pfx_x (ft)")
    axes[1].set_ylabel("pfx_z (ft)")
    axes[1].grid(True, color="#e7e9ef", linewidth=0.8)

    fig.suptitle("Tarik Skubal FF: same pitch type, real variation", fontsize=17, weight="bold")
    fig.text(
        0.5,
        0.01,
        f"{len(skubal_ff):,} FF pitches across {skubal_ff['game_pk'].nunique()} games, "
        "colored by pitcher game pitch count",
        ha="center",
        fontsize=10,
        color="#5b6472",
    )
    cbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("pitcher game pitch count")
    _save_figure(fig, path)
    return path


def _sequence_frame(skubal_ff: pd.DataFrame, label: str, mask: pd.Series) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    previous = skubal_ff.index < mask[mask].index.min() if mask.any() else np.zeros(len(skubal_ff), bool)
    ax.scatter(
        skubal_ff.loc[previous, "plate_x"],
        skubal_ff.loc[previous, "plate_z"],
        s=16,
        color="#bcc3cf",
        alpha=0.32,
        linewidths=0,
    )
    ax.scatter(
        skubal_ff.loc[mask, "plate_x"],
        skubal_ff.loc[mask, "plate_z"],
        s=28,
        color="#2d8cff",
        alpha=0.78,
        linewidths=0,
    )
    _draw_zone(ax)
    ax.axvline(0, color="#8a93a3", linewidth=0.8, alpha=0.6)
    ax.set_title(f"Skubal FF command variation: pitch count {label}", fontsize=13, weight="bold")
    ax.set_xlabel("plate_x (ft)")
    ax.set_ylabel("plate_z (ft)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.0, 5.2)
    ax.grid(True, color="#e7e9ef", linewidth=0.8)
    ax.text(
        -2.35,
        4.85,
        f"highlighted pitches: {int(mask.sum())}",
        fontsize=9,
        color="#485163",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d7dce5"},
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return iio.imread(buffer, extension=".png")


def build_pitch_sequence_gif(skubal_ff: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "skubal_ff_pitch_sequence.gif"
    pitch_count = pd.to_numeric(skubal_ff["pitcher_game_pitch_count"], errors="coerce")
    sections = [
        ("1-25", pitch_count.between(1, 25)),
        ("26-50", pitch_count.between(26, 50)),
        ("51-75", pitch_count.between(51, 75)),
        ("76+", pitch_count >= 76),
    ]
    frames = [_sequence_frame(skubal_ff, label, mask) for label, mask in sections]
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, duration=950, loop=0)
    return path


def build_results_plot(report_path: Path, output_dir: Path) -> Path:
    path = output_dir / "v21_results_auc.png"
    report = json.loads(report_path.read_text())
    layer_names = [
        "command_representation",
        "movement_only",
        "release_only",
        "trajectory_only",
        "physics_core",
    ]
    display = ["command", "movement", "release", "trajectory", "physics core"]
    rows = [report["layer_results"][layer] for layer in layer_names]
    values = {
        "V2.1 factorized": [row["factorized_auc"] for row in rows],
        "game-drift Gaussian": [row["game_drift_gaussian_auc"] for row in rows],
        "game-drift copula": [row["game_drift_copula_auc"] for row in rows],
    }

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    x = np.arange(len(display))
    width = 0.25
    colors = ["#2563eb", "#f59e0b", "#10b981"]
    for index, (name, aucs) in enumerate(values.items()):
        ax.bar(x + (index - 1) * width, aucs, width=width, label=name, color=colors[index])
    ax.axhline(0.50, color="#111827", linewidth=1.2, linestyle="--", label="ideal 0.50")
    ax.axhline(0.60, color="#ef4444", linewidth=1.1, linestyle=":", label="target 0.60")
    ax.set_ylim(0.48, 0.74)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel("C2ST detectability AUC (lower is better)")
    ax.set_title("V2.1 factorized model improved full-physics detectability", weight="bold")
    ax.grid(True, axis="y", color="#e7e9ef")
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    for index, aucs in enumerate(values.values()):
        for xpos, auc in zip(x + (index - 1) * width, aucs, strict=True):
            ax.text(xpos, auc + 0.004, f"{auc:.3f}", ha="center", va="bottom", fontsize=8)
    _save_figure(fig, path)
    return path


def _svg_box(x: int, y: int, w: int, h: int, title: str, subtitle: str, fill: str) -> str:
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="#1f2937" stroke-width="2"/>
    <text x="{x + w / 2}" y="{y + 34}" text-anchor="middle" font-size="17" font-weight="700" fill="#111827">{title}</text>
    <text x="{x + w / 2}" y="{y + 59}" text-anchor="middle" font-size="12" fill="#4b5563">{subtitle}</text>
    """


def _svg_arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f"""
    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" stroke-width="2.2" marker-end="url(#arrow)"/>
    """


def build_pipeline_svg(output_dir: Path) -> Path:
    path = output_dir / "pitcher_twin_pipeline.svg"
    boxes = [
        (40, 96, 160, 92, "Statcast", "real pitch rows", "#eff6ff"),
        (240, 96, 170, 92, "Feature layers", "release/move/cmd", "#ecfdf5"),
        (450, 96, 170, 92, "Generators", "drift + copula", "#fffbeb"),
        (660, 96, 170, 92, "Holdout", "later games", "#f5f3ff"),
        (870, 96, 170, 92, "C2ST", "can it tell?", "#fef2f2"),
        (1080, 96, 170, 92, "Machine Session JSON", "validated export", "#f8fafc"),
    ]
    arrows = [
        _svg_arrow(200, 142, 240, 142),
        _svg_arrow(410, 142, 450, 142),
        _svg_arrow(620, 142, 660, 142),
        _svg_arrow(830, 142, 870, 142),
        _svg_arrow(1040, 142, 1080, 142),
    ]
    content = "\n".join(_svg_box(*box) for box in boxes)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1290" height="300" viewBox="0 0 1290 300">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/>
    </marker>
  </defs>
  <rect width="1290" height="300" fill="white"/>
  <text x="40" y="44" font-size="24" font-weight="800" fill="#111827">Pitcher Twin technical pipeline</text>
  <text x="40" y="72" font-size="14" fill="#4b5563">Train on early real pitches, generate samples, then ask whether a classifier can distinguish them from later real pitches.</text>
  {content}
  {"".join(arrows)}
  <text x="645" y="246" text-anchor="middle" font-size="14" fill="#374151">Lower C2ST AUC means generated pitches look more like held-out real pitches.</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)
    return path


def build_v21_chain_svg(output_dir: Path) -> Path:
    path = output_dir / "v21_physics_chain.svg"
    boxes = [
        (70, 108, 210, 100, "Release", "velocity / spin / slot", "#eff6ff"),
        (350, 108, 210, 100, "Movement residual", "pfx_x / pfx_z", "#ecfdf5"),
        (630, 108, 210, 100, "Trajectory residual", "vx/vy/vz + accel", "#fffbeb"),
        (910, 108, 210, 100, "Command residual", "plate_x / plate_z", "#f5f3ff"),
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1190" height="360" viewBox="0 0 1190 360">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/>
    </marker>
  </defs>
  <rect width="1190" height="360" fill="white"/>
  <text x="70" y="48" font-size="24" font-weight="800" fill="#111827">V2.1 factorized physics-residual model</text>
  <text x="70" y="76" font-size="14" fill="#4b5563">Instead of sampling one flat physics vector, V2.1 samples a pitch as a coupled physics chain.</text>
  {"".join(_svg_box(*box) for box in boxes)}
  {_svg_arrow(280, 158, 350, 158)}
  {_svg_arrow(560, 158, 630, 158)}
  {_svg_arrow(840, 158, 910, 158)}
  <rect x="180" y="260" width="300" height="54" rx="14" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>
  <text x="330" y="283" text-anchor="middle" font-size="13" font-weight="700" fill="#111827">release variance floor</text>
  <text x="330" y="302" text-anchor="middle" font-size="11" fill="#475569">prevents collapsed release spread</text>
  <rect x="690" y="260" width="330" height="54" rx="14" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>
  <text x="855" y="283" text-anchor="middle" font-size="13" font-weight="700" fill="#111827">recent-game residual drift</text>
  <text x="855" y="302" text-anchor="middle" font-size="11" fill="#475569">later games can shift downstream physics</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)
    return path


def _excalidraw_element(kind: str, x: int, y: int, w: int, h: int, text: str = "") -> dict[str, object]:
    element_id = uuid4().hex[:16]
    base = {
        "id": element_id,
        "type": kind,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": "#1f2937",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if kind == "rectangle" else None,
        "seed": int(uuid4().int % 2_000_000_000),
        "version": 1,
        "versionNonce": int(uuid4().int % 2_000_000_000),
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }
    if kind == "text":
        base.update(
            {
                "text": text,
                "fontSize": 20,
                "fontFamily": 1,
                "textAlign": "center",
                "verticalAlign": "middle",
                "containerId": None,
                "originalText": text,
                "lineHeight": 1.25,
                "baseline": h - 8,
            }
        )
    if kind == "arrow":
        base.update(
            {
                "points": [[0, 0], [w, h]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow",
            }
        )
    return base


def _write_excalidraw(path: Path, elements: list[dict[str, object]]) -> Path:
    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "pitcher-twin-readme",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def build_excalidraw_sources(output_dir: Path) -> list[Path]:
    pipeline_labels = ["Statcast", "Feature layers", "Generators", "Holdout", "C2ST", "Machine Session JSON"]
    pipeline_elements: list[dict[str, object]] = []
    for index, label in enumerate(pipeline_labels):
        x = 40 + index * 190
        pipeline_elements.append(_excalidraw_element("rectangle", x, 90, 150, 78))
        pipeline_elements.append(_excalidraw_element("text", x + 10, 110, 130, 36, label))
        if index < len(pipeline_labels) - 1:
            pipeline_elements.append(_excalidraw_element("arrow", x + 154, 128, 34, 0))

    chain_labels = ["Release", "Movement residual", "Trajectory residual", "Command residual"]
    chain_elements: list[dict[str, object]] = []
    for index, label in enumerate(chain_labels):
        x = 60 + index * 240
        chain_elements.append(_excalidraw_element("rectangle", x, 100, 190, 82))
        chain_elements.append(_excalidraw_element("text", x + 12, 120, 166, 36, label))
        if index < len(chain_labels) - 1:
            chain_elements.append(_excalidraw_element("arrow", x + 194, 140, 42, 0))

    return [
        _write_excalidraw(output_dir / "pitcher_twin_pipeline.excalidraw", pipeline_elements),
        _write_excalidraw(output_dir / "v21_physics_chain.excalidraw", chain_elements),
    ]


def build_assets(data_path: Path | None, report_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    skubal_ff = _load_skubal_ff(data_path)
    built = [
        build_variation_plot(skubal_ff, output_dir),
        build_pitch_sequence_gif(skubal_ff, output_dir),
        build_results_plot(report_path, output_dir),
        build_pipeline_svg(output_dir),
        build_v21_chain_svg(output_dir),
    ]
    built.extend(build_excalidraw_sources(output_dir))
    return built


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    built = build_assets(args.data, args.report, args.output_dir)
    for path in built:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
