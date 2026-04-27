#!/usr/bin/env python3
"""Build README visuals from real Pitcher Twin artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "readme"
SITE_DATA_PATH = ROOT / "site" / "data.json"
TOURNAMENT_REPORT_PATH = ROOT / "outputs" / "model_tournament_skubal_2025_ff" / "model_tournament_report.json"
LEADERBOARD_PATH = ROOT / "outputs" / "validation_board_skubal_2025_top3" / "leaderboard.csv"
ROLLING_BOARD_PATH = ROOT / "outputs" / "rolling_validation_skubal_2025_ff" / "rolling_validation_board.json"

INK = (25, 24, 21)
MUTED = (98, 91, 80)
PAPER = (247, 244, 236)
CARD = (255, 255, 255)
CREAM = (255, 250, 240)
LINE = (216, 205, 184)
GREEN = (31, 122, 77)
BLUE = (47, 111, 130)
AMBER = (215, 165, 49)
RED = (164, 61, 50)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(round(a[i] * amount + b[i] * (1 - amount)) for i in range(3))


def _save_gif(path: Path, frames: list[Image.Image], *, duration: int = 120) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )


def _base_canvas(width: int = 1400, height: int = 760) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((42, 42, width - 42, height - 42), radius=30, fill=CARD)
    return image, draw


def _candidate(site_data: dict[str, Any], key: str = "skubal_ff") -> dict[str, Any]:
    for candidate in site_data["candidates"]:
        if candidate["key"] == key:
            return candidate
    raise KeyError(f"Missing candidate {key!r} in {SITE_DATA_PATH}")


def _frame(records: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records).dropna(subset=["plate_x", "plate_z"])


def _draw_plate(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 - 26, y0 - 68, x1 + 26, y1 + 42), radius=22, fill=PAPER, outline=LINE, width=2)
    draw.text((x0 - 6, y0 - 48), title, font=_font(25, bold=True), fill=INK)
    if subtitle:
        draw.text((x0 - 6, y0 - 18), subtitle, font=_font(18), fill=MUTED)
    draw.rectangle(box, outline=INK, width=4)
    for index in (1, 2):
        x = x0 + index * ((x1 - x0) // 3)
        y = y0 + index * ((y1 - y0) // 3)
        draw.line((x, y0, x, y1), fill=LINE, width=2)
        draw.line((x0, y, x1, y), fill=LINE, width=2)


def _plate_xy(row: pd.Series | dict[str, float], box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    plate_x = float(row["plate_x"])
    plate_z = float(row["plate_z"])
    x = x0 + (plate_x + 2.0) / 4.0 * (x1 - x0)
    y = y1 - (plate_z - 0.5) / 4.7 * (y1 - y0)
    return int(round(x)), int(round(y))


def _draw_points(
    draw: ImageDraw.ImageDraw,
    frame: pd.DataFrame,
    box: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    radius: int = 5,
    limit: int | None = None,
) -> None:
    if limit is not None and len(frame) > limit:
        frame = frame.sample(limit, random_state=7)
    for _, row in frame.iterrows():
        x, y = _plate_xy(row, box)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline)


def _zone_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    in_zone = (
        frame["plate_x"].between(-0.83, 0.83)
        & frame["plate_z"].between(1.5, 3.5)
    )
    return float(in_zone.mean())


def build_best_result_summary() -> None:
    leaderboard = pd.read_csv(LEADERBOARD_PATH)
    row = leaderboard[(leaderboard["pitcher_name"] == "Skubal, Tarik") & (leaderboard["pitch_type"] == "FF")].iloc[0]
    auc = float(row["physics_core_mean_auc"])
    pass_rate = float(row["physics_core_pass_rate"])

    image, draw = _base_canvas()
    draw.text((82, 82), "Best Real-Data Result", font=_font(58, bold=True), fill=INK)
    draw.text((82, 150), "Tarik Skubal 2025 FF, trained on earlier games and tested on later games.", font=_font(25), fill=MUTED)

    metrics = [
        ("Real FF pitches", f"{int(row['pitch_count']):,}", GREEN),
        ("Games", str(int(row["game_count"])), BLUE),
        ("Holdout pitches", str(int(row["holdout_count"])), AMBER),
        ("Pass rate", f"{pass_rate:.0%}", GREEN),
    ]
    for index, (label, value, color) in enumerate(metrics):
        x = 82 + index * 310
        draw.rounded_rectangle((x, 238, x + 270, 370), radius=22, fill=CREAM, outline=color, width=3)
        draw.text((x + 24, 262), label, font=_font(20, bold=True), fill=MUTED)
        draw.text((x + 24, 300), value, font=_font(48, bold=True), fill=color)

    draw.rounded_rectangle((120, 470, 1280, 600), radius=28, fill=_blend(GREEN, CARD, 0.13))
    draw.text((170, 492), "C2ST AUC", font=_font(28, bold=True), fill=MUTED)
    draw.text((170, 525), f"{auc:.3f}", font=_font(56, bold=True), fill=GREEN)
    draw.text((420, 512), "0.50 is ideal", font=_font(28, bold=True), fill=INK)
    draw.text((420, 548), "0.60 is the pass line; lower is better.", font=_font(24), fill=MUTED)

    x0, x1, y = 170, 1230, 665
    draw.line((x0, y, x1, y), fill=LINE, width=12)
    for value, color, label in [(0.50, GREEN, "ideal"), (0.60, AMBER, "target"), (1.00, RED, "easy to detect")]:
        x = x0 + int(((value - 0.50) / 0.50) * (x1 - x0))
        draw.line((x, y - 28, x, y + 28), fill=color, width=5)
        draw.text((x - 44, y + 42), label, font=_font(18, bold=True), fill=color)
    x = x0 + int(((auc - 0.50) / 0.50) * (x1 - x0))
    draw.ellipse((x - 22, y - 22, x + 22, y + 22), fill=GREEN, outline=INK, width=3)
    draw.text((x - 42, y - 72), f"{auc:.3f}", font=_font(27, bold=True), fill=INK)
    image.save(ASSET_DIR / "best-result-summary.png", optimize=True)


def build_real_vs_generated_cloud(site_data: dict[str, Any]) -> None:
    candidate = _candidate(site_data)
    real = _frame(candidate["real_holdout"])
    generated = _frame(candidate["samples"]["even_R"])
    leaderboard = pd.read_csv(LEADERBOARD_PATH)
    row = leaderboard[(leaderboard["pitcher_name"] == "Skubal, Tarik") & (leaderboard["pitch_type"] == "FF")].iloc[0]
    auc = float(row["physics_core_mean_auc"])
    pass_rate = float(row["physics_core_pass_rate"])
    frames: list[Image.Image] = []
    states = [
        ("Later real holdout", real, None),
        ("Generated samples", None, generated),
        ("Overlay: real vs generated", real, generated),
    ]

    for title, real_frame, generated_frame in states:
        image, draw = _base_canvas()
        draw.text((82, 82), "Best Validated Result: Skubal 2025 FF", font=_font(55, bold=True), fill=INK)
        draw.text((82, 148), "Holdout Statcast vs model samples.", font=_font(26), fill=MUTED)
        draw.rounded_rectangle((82, 205, 468, 646), radius=26, fill=CREAM)
        draw.text((118, 244), "Result", font=_font(22, bold=True), fill=MUTED)
        draw.text((118, 284), f"{auc:.3f}", font=_font(76, bold=True), fill=GREEN)
        draw.text((118, 360), "C2ST AUC", font=_font(27, bold=True), fill=INK)
        draw.text((118, 410), f"{pass_rate:.0%} pass rate", font=_font(32, bold=True), fill=GREEN)
        draw.text((118, 470), f"{int(row['pitch_count'])} real FF pitches", font=_font(24), fill=INK)
        draw.text((118, 510), f"{int(row['game_count'])} games", font=_font(24), fill=INK)
        draw.text((118, 550), f"{int(row['holdout_count'])} later holdout rows", font=_font(24), fill=INK)

        box = (670, 235, 1160, 635)
        subtitle = "green = real holdout, amber = generated"
        _draw_plate(draw, box, title=title, subtitle=subtitle)
        if real_frame is not None:
            _draw_points(draw, real_frame, box, color=_blend(GREEN, CARD, 0.45), outline=GREEN, radius=4, limit=180)
        if generated_frame is not None:
            _draw_points(draw, generated_frame, box, color=_blend(AMBER, CARD, 0.55), outline=AMBER, radius=6)
        draw.rounded_rectangle((670, 650, 1160, 690), radius=15, fill=_blend(GREEN, CARD, 0.14))
        draw.text((696, 660), "Generated samples occupy the same plate-location cloud.", font=_font(18, bold=True), fill=GREEN)
        frames.extend([image] * 10)
    _save_gif(ASSET_DIR / "real-vs-generated-cloud.gif", frames, duration=150)


def build_context_cloud_shift(site_data: dict[str, Any]) -> None:
    candidate = _candidate(site_data)
    contexts = [
        ("0-0 vs RHB", "first_pitch_R"),
        ("0-2 vs RHB", "ahead_R"),
        ("1-1 vs RHB", "even_R"),
        ("2-0 vs RHB", "behind_R"),
        ("3-2 vs LHB", "full_L"),
    ]
    frames: list[Image.Image] = []
    box = (820, 235, 1210, 635)

    for label, key in contexts:
        frame = _frame(candidate["samples"][key])
        image, draw = _base_canvas()
        draw.text((82, 82), "What The App Does", font=_font(56, bold=True), fill=INK)
        draw.text((82, 148), "Move the context controls; the pitch envelope updates.", font=_font(26), fill=MUTED)
        controls = [
            ("Pitcher", "Tarik Skubal"),
            ("Pitch", "FF"),
            ("Context", label),
            ("Samples", str(len(frame))),
        ]
        for index, (name, value) in enumerate(controls):
            x = 82 + (index % 2) * 310
            y = 240 + (index // 2) * 112
            draw.rounded_rectangle((x, y, x + 270, y + 78), radius=18, fill=CREAM)
            draw.text((x + 22, y + 14), name, font=_font(18), fill=MUTED)
            draw.text((x + 22, y + 40), value, font=_font(26, bold=True), fill=INK)
        draw.rounded_rectangle((82, 500, 670, 620), radius=22, fill=_blend(GREEN, CARD, 0.14))
        draw.text((112, 528), "Real output from site/data.json", font=_font(26, bold=True), fill=GREEN)
        draw.text(
            (112, 568),
            f"mean velo {frame['release_speed'].mean():.1f} mph  |  zone rate {_zone_rate(frame):.0%}",
            font=_font(21),
            fill=INK,
        )
        _draw_plate(draw, box, title=f"Generated cloud: {label}", subtitle="actual pre-sampled app payload")
        _draw_points(draw, frame, box, color=_blend(GREEN, CARD, 0.55), outline=GREEN, radius=6)
        frames.extend([image] * 12)
    _save_gif(ASSET_DIR / "context-cloud-shift.gif", frames, duration=150)


def build_model_architecture() -> None:
    image, draw = _base_canvas()
    draw.text((82, 82), "How The Model Learns A Pitcher's Style", font=_font(50, bold=True), fill=INK)
    draw.text(
        (82, 142),
        "The generator samples pitch physics in layers instead of averaging every feature together.",
        font=_font(25),
        fill=MUTED,
    )
    steps = [
        ("Real Statcast", "plate, release, spin, movement", GREEN),
        ("Release state", "velocity + spin + geometry", BLUE),
        ("Movement residual", "break around release", AMBER),
        ("Trajectory residual", "vx/vy/vz + acceleration", RED),
        ("Command cloud", "plate_x / plate_z", GREEN),
        ("Trajekt JSON", "sampled session export", BLUE),
    ]
    positions = [(78, 250), (500, 250), (922, 250), (922, 500), (500, 500), (78, 500)]
    for index, ((title, subtitle, color), (x, y)) in enumerate(zip(steps, positions, strict=True)):
        draw.rounded_rectangle((x, y, x + 310, y + 118), radius=22, fill=CREAM, outline=color, width=4)
        draw.ellipse((x + 22, y + 22, x + 66, y + 66), fill=color)
        draw.text((x + 36, y + 31), str(index + 1), font=_font(22, bold=True), fill=CREAM)
        draw.text((x + 88, y + 30), title, font=_font(27, bold=True), fill=INK)
        draw.text((x + 88, y + 70), subtitle, font=_font(18), fill=MUTED)
    arrows = [
        ((390, 309), (490, 309)),
        ((812, 309), (912, 309)),
        ((1077, 372), (1077, 488)),
        ((922, 559), (820, 559)),
        ((500, 559), (398, 559)),
    ]
    for start, end in arrows:
        _draw_arrow(draw, start, end, INK)
    image.save(ASSET_DIR / "model-architecture.png", optimize=True)
    build_excalidraw_architecture()


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    draw.line((start, end), fill=color, width=5)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 18
    points = [
        end,
        (
            round(end[0] - size * math.cos(angle - math.pi / 6)),
            round(end[1] - size * math.sin(angle - math.pi / 6)),
        ),
        (
            round(end[0] - size * math.cos(angle + math.pi / 6)),
            round(end[1] - size * math.sin(angle + math.pi / 6)),
        ),
    ]
    draw.polygon(points, fill=color)


def build_c2st_workflow() -> None:
    image, draw = _base_canvas()
    draw.text((82, 82), "How We Prove Realism", font=_font(56, bold=True), fill=INK)
    draw.text(
        (82, 148),
        "A classifier two-sample test asks whether generated pitches are detectable.",
        font=_font(26),
        fill=MUTED,
    )
    steps = [
        ("1", "Train", "early real pitches", GREEN),
        ("2", "Generate", "synthetic samples", BLUE),
        ("3", "Hold out", "later real pitches", AMBER),
        ("4", "Classifier", "tries to spot fakes", RED),
        ("5", "AUC", "0.533 best result", GREEN),
    ]
    for index, (number, title, subtitle, color) in enumerate(steps):
        x = 82 + index * 255
        y = 305
        draw.rounded_rectangle((x, y, x + 210, y + 140), radius=22, fill=CREAM, outline=color, width=4)
        draw.ellipse((x + 22, y + 24, x + 66, y + 68), fill=color)
        draw.text((x + 37, y + 33), number, font=_font(22, bold=True), fill=CREAM)
        draw.text((x + 24, y + 82), title, font=_font(28, bold=True), fill=INK)
        draw.text((x + 24, y + 116), subtitle, font=_font(18), fill=MUTED)
        if index < len(steps) - 1:
            _draw_arrow(draw, (x + 220, y + 70), (x + 248, y + 70), INK)
    draw.line((320, 575, 1080, 575), fill=LINE, width=12)
    for value, color, label in [(0.50, GREEN, "ideal 0.50"), (0.60, AMBER, "target 0.60"), (1.00, RED, "obvious fake")]:
        x = 320 + int(((value - 0.50) / 0.50) * 760)
        draw.line((x, 550, x, 600), fill=color, width=5)
        draw.text((x - 50, 614), label, font=_font(17, bold=True), fill=color)
    auc = 0.533
    x = 320 + int(((auc - 0.50) / 0.50) * 760)
    draw.ellipse((x - 19, 556, x + 19, 594), fill=GREEN, outline=INK, width=3)
    draw.text((x - 42, 512), f"{auc:.3f}", font=_font(28, bold=True), fill=INK)
    image.save(ASSET_DIR / "c2st-validation-workflow.png", optimize=True)


def build_layer_results() -> None:
    report = _load_json(TOURNAMENT_REPORT_PATH)
    order = [
        ("command_representation", "command"),
        ("movement_only", "movement"),
        ("trajectory_only", "trajectory"),
        ("release_only", "release"),
        ("physics_core", "physics core"),
    ]
    rows = []
    for layer, label in order:
        model_name = report["best_by_layer"][layer]
        result = report["layer_results"][layer][model_name]
        rows.append((label, float(result["mean_auc"]), float(result["pass_rate"]), model_name))

    image, draw = _base_canvas()
    draw.text((82, 82), "Layer Validation Results", font=_font(56, bold=True), fill=INK)
    draw.text(
        (82, 148),
        "Repeated-seed tournament on Skubal 2025 FF; lower C2ST AUC is better.",
        font=_font(26),
        fill=MUTED,
    )
    chart = (330, 245, 1160, 620)
    x0, y0, x1, y1 = chart
    draw.line((x0, y1, x1, y1), fill=INK, width=3)
    draw.line((x0, y0, x0, y1), fill=INK, width=3)
    for value, color, label in [(0.50, GREEN, "ideal"), (0.60, AMBER, "target")]:
        x = x0 + int(((value - 0.48) / 0.22) * (x1 - x0))
        draw.line((x, y0, x, y1), fill=color, width=3)
        draw.text((x - 28, y0 - 32), label, font=_font(16, bold=True), fill=color)
    bar_h = 48
    for index, (label, auc, pass_rate, model_name) in enumerate(rows):
        y = y0 + 35 + index * 68
        draw.text((82, y + 9), label, font=_font(25, bold=True), fill=INK)
        bar_x = x0
        bar_w = int(((auc - 0.48) / 0.22) * (x1 - x0))
        color = GREEN if auc <= 0.60 else AMBER
        draw.rounded_rectangle((bar_x, y, bar_x + bar_w, y + bar_h), radius=12, fill=_blend(color, CARD, 0.22))
        draw.text((bar_x + bar_w + 16, y + 9), f"{auc:.3f}", font=_font(24, bold=True), fill=INK)
        draw.text((1168, y + 11), f"{pass_rate:.0%}", font=_font(21, bold=True), fill=color)
    draw.text((1160, y0 - 32), "pass", font=_font(16, bold=True), fill=MUTED)
    draw.rounded_rectangle((82, 650, 1318, 700), radius=18, fill=_blend(GREEN, CARD, 0.12))
    draw.text(
        (112, 662),
        "This is the model evidence: command, movement, trajectory, release, and full physics are scored separately.",
        font=_font(22, bold=True),
        fill=GREEN,
    )
    image.save(ASSET_DIR / "layer-results.png", optimize=True)


def build_rolling_folds() -> None:
    board = _load_json(ROLLING_BOARD_PATH)
    folds = board["folds"]
    aucs = [float(fold["physics_core_mean_auc"]) for fold in folds]
    frames: list[Image.Image] = []
    width, height = 1400, 760
    plot_left, plot_top, plot_right, plot_bottom = 110, 260, 1260, 585
    min_auc, max_auc = 0.55, 0.95

    def xy(index: int, auc: float) -> tuple[int, int]:
        x = plot_left + int(index * ((plot_right - plot_left) / (len(aucs) - 1)))
        y = plot_bottom - int(((auc - min_auc) / (max_auc - min_auc)) * (plot_bottom - plot_top))
        return x, y

    for active in range(len(folds)):
        image, draw = _base_canvas(width, height)
        draw.text((82, 82), "Future-Window Stress Test", font=_font(56, bold=True), fill=INK)
        draw.text(
            (82, 148),
            "Train on past games, then test the next unseen game window.",
            font=_font(26),
            fill=MUTED,
        )
        fold = folds[active]
        draw.rounded_rectangle((935, 82, 1285, 170), radius=20, fill=CREAM)
        draw.text((965, 104), f"Fold {fold['fold_index']}/10", font=_font(28, bold=True), fill=INK)
        draw.text((965, 138), f"AUC {float(fold['physics_core_mean_auc']):.3f}", font=_font(24, bold=True), fill=AMBER)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=3)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=3)
        for value, label, color in [(0.60, "fold target 0.60", GREEN), (0.80, "worst-fold ceiling 0.80", RED)]:
            _, y = xy(0, value)
            draw.line((plot_left, y, plot_right, y), fill=color, width=3)
            draw.text((plot_right - 220, y - 28), label, font=_font(18, bold=True), fill=color)
        points = [xy(i, auc) for i, auc in enumerate(aucs)]
        for index in range(active):
            draw.line((points[index], points[index + 1]), fill=INK, width=5)
        for index, (x, y) in enumerate(points[: active + 1]):
            auc = aucs[index]
            color = GREEN if auc <= 0.60 else AMBER if auc < 0.80 else RED
            radius = 10 if index != active else 16
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=INK, width=2)
            draw.text((x - 10, plot_bottom + 18), str(index + 1), font=_font(16, bold=True), fill=MUTED)
        draw.rounded_rectangle((110, 625, 1260, 690), radius=18, fill=CREAM)
        draw.text(
            (140, 645),
            "Use this as the reliability frontier, not the headline result.",
            font=_font(26, bold=True),
            fill=INK,
        )
        frames.extend([image] * 8)
    _save_gif(ASSET_DIR / "rolling-folds.gif", frames, duration=120)


def build_excalidraw_architecture() -> None:
    def rect(element_id: str, x: int, y: int, w: int, h: int, text: str, color: str) -> list[dict[str, Any]]:
        box = {
            "id": element_id,
            "type": "rectangle",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": "#191815",
            "backgroundColor": color,
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": {"type": 3},
            "seed": _seed(element_id),
            "version": 1,
            "versionNonce": _seed(element_id + "v"),
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
        }
        label = {
            "id": f"{element_id}_label",
            "type": "text",
            "x": x + 18,
            "y": y + 20,
            "width": w - 36,
            "height": 52,
            "angle": 0,
            "strokeColor": "#191815",
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": None,
            "seed": _seed(element_id + "label"),
            "version": 1,
            "versionNonce": _seed(element_id + "labelv"),
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
            "text": text,
            "fontSize": 22,
            "fontFamily": 1,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": None,
            "originalText": text,
            "lineHeight": 1.25,
        }
        return [box, label]

    elements: list[dict[str, Any]] = []
    cards = [
        ("statcast", 80, 120, "Real Statcast\npitch rows", "#dcefe5"),
        ("release", 330, 120, "Release state\nvelocity + spin", "#e5edf0"),
        ("movement", 580, 120, "Movement\nresidual", "#f4e6bd"),
        ("trajectory", 830, 120, "Trajectory\nresidual", "#f0d3cd"),
        ("command", 1080, 120, "Command\ncloud", "#dcefe5"),
        ("export", 580, 310, "Trajekt-shaped\nsession JSON", "#e5edf0"),
    ]
    for card in cards:
        elements.extend(rect(card[0], card[1], card[2], 180, 96, card[3], card[4]))
    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#f7f4ec", "currentItemFontFamily": 1},
        "files": {},
    }
    (ASSET_DIR / "model-architecture.excalidraw").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_asset_readme() -> None:
    note = """# README Evidence Visual Assets

All displayed README visuals are generated from tracked project artifacts.

## Data Sources

- `site/data.json`: real holdout rows and generated app samples.
- `outputs/validation_board_skubal_2025_top3/leaderboard.csv`: best validated result.
- `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json`: layer AUCs.
- `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.json`: rolling fold AUCs.

## Displayed Assets

- `best-result-summary.png`: top-line Skubal FF validation summary.
- `real-vs-generated-cloud.gif`: real held-out Skubal FF vs generated samples.
- `context-cloud-shift.gif`: actual pre-sampled app contexts from `site/data.json`.
- `model-architecture.png`: factorized model structure.
- `c2st-validation-workflow.png`: classifier two-sample validation flow.
- `layer-results.png`: real repeated-seed tournament layer results.
- `rolling-folds.gif`: real rolling future-window stress test.
- `model-architecture.excalidraw`: editable architecture source.
"""
    (ASSET_DIR / "README.md").write_text(note, encoding="utf-8")


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    site_data = _load_json(SITE_DATA_PATH)
    build_best_result_summary()
    build_real_vs_generated_cloud(site_data)
    build_context_cloud_shift(site_data)
    build_model_architecture()
    build_c2st_workflow()
    build_layer_results()
    build_rolling_folds()
    build_asset_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
