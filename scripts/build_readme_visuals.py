#!/usr/bin/env python3
"""Build deterministic README visuals for the Pitcher Twin project."""

from __future__ import annotations

import json
import hashlib
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "readme"

ROLLING_AUCS = [0.593, 0.602, 0.929, 0.639, 0.608, 0.663, 0.725, 0.793, 0.765, 0.701]
ROLLING_GOALS = {
    "mean_auc": 0.620,
    "target_hit_rate": 0.40,
    "worst_fold": 0.800,
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Avenir Next.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size, index=1 if bold else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def build_scoreboard_svg() -> None:
    rows = [
        ("Single 70/30 split", "validated", "0.533 AUC", "#1f7a4d", "Skubal FF ceiling result"),
        (
            "Rolling truth test",
            "diagnostic",
            "0.702 mean AUC",
            "#b47f1a",
            "goal <= 0.620, hit rate >= 0.40",
        ),
        (
            "Worst rolling fold",
            "miss",
            "0.929 AUC",
            "#a43d32",
            "release/spin state drift remains visible",
        ),
    ]
    row_blocks = []
    y = 178
    for index, (name, status, value, color, note) in enumerate(rows):
        row_y = y + index * 118
        escaped_note = html.escape(note)
        row_blocks.append(
            f"""
  <g transform="translate(48 {row_y})">
    <rect x="0" y="0" width="1004" height="88" rx="14" fill="#ffffff" opacity="0.92"/>
    <rect x="0" y="0" width="7" height="88" rx="3" fill="{color}"/>
    <text x="34" y="35" class="row-title">{name}</text>
    <text x="34" y="64" class="row-note">{escaped_note}</text>
    <rect x="590" y="18" width="360" height="52" rx="18" fill="{color}" opacity="0.13"/>
    <text x="616" y="40" class="metric" fill="{color}">{value}</text>
    <text x="616" y="63" class="pill" fill="{color}">{status}</text>
  </g>"""
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="610" viewBox="0 0 1100 610" role="img" aria-label="Pitcher Twin honest scoreboard">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#161512"/>
      <stop offset="58%" stop-color="#24221d"/>
      <stop offset="100%" stop-color="#f7f4ec"/>
    </linearGradient>
    <style>
      .eyebrow {{ font: 700 18px ui-sans-serif, system-ui, sans-serif; fill: #d7a531; letter-spacing: 2px; }}
      .title {{ font: 800 50px ui-sans-serif, system-ui, sans-serif; fill: #f7f4ec; }}
      .subtitle {{ font: 500 22px ui-sans-serif, system-ui, sans-serif; fill: #d8d1c1; }}
      .row-title {{ font: 800 26px ui-sans-serif, system-ui, sans-serif; fill: #191815; }}
      .row-note {{ font: 500 17px ui-sans-serif, system-ui, sans-serif; fill: #625b50; }}
      .pill {{ font: 800 17px ui-sans-serif, system-ui, sans-serif; text-transform: uppercase; }}
      .metric {{ font: 850 34px ui-sans-serif, system-ui, sans-serif; fill: #191815; }}
    </style>
  </defs>
  <rect width="1100" height="610" fill="url(#bg)"/>
  <path d="M-120 500 C 120 315, 270 525, 470 365 S 760 155, 1180 345" fill="none" stroke="#1f7a4d" stroke-width="3" opacity="0.48"/>
  <path d="M-80 540 C 170 380, 330 530, 540 380 S 800 260, 1160 430" fill="none" stroke="#d7a531" stroke-width="3" opacity="0.42"/>
  <path d="M-40 575 C 210 425, 410 555, 630 410 S 840 340, 1120 525" fill="none" stroke="#a43d32" stroke-width="3" opacity="0.38"/>
  <text x="48" y="72" class="eyebrow">PRIMARY VALIDATION STORY</text>
  <text x="48" y="128" class="title">Honest Scoreboard</text>
  <text x="48" y="162" class="subtitle">Single split shows the ceiling. Rolling windows decide reliability.</text>
  {''.join(row_blocks)}
</svg>
"""
    _write(ASSET_DIR / "scoreboard.svg", svg)


def build_pipeline_svg() -> None:
    steps = [
        ("Real Statcast", "public pitch rows"),
        ("Feature layers", "release, spin, movement"),
        ("Generator suite", "factorized + drift models"),
        ("C2ST validator", "classifier realism test"),
        ("Rolling board", "future-game reliability"),
        ("Trajekt export", "sampled session JSON"),
    ]
    blocks = []
    for index, (title, subtitle) in enumerate(steps):
        x = 72 + (index % 3) * 330
        y = 190 + (index // 3) * 170
        color = ["#1f7a4d", "#2f6f82", "#d7a531", "#a43d32", "#b47f1a", "#1f7a4d"][index]
        blocks.append(
            f"""
  <g transform="translate({x} {y})">
    <rect width="250" height="120" rx="18" fill="#fffaf0" stroke="#211f1a" stroke-width="2"/>
    <circle cx="34" cy="34" r="16" fill="{color}"/>
    <text x="34" y="43" text-anchor="middle" class="number">{index + 1}</text>
    <text x="128" y="58" text-anchor="middle" class="step-title">{title}</text>
    <text x="128" y="86" text-anchor="middle" class="step-sub">{subtitle}</text>
  </g>"""
        )
    arrows = [
        (322, 250, 390, 250),
        (652, 250, 720, 250),
        (860, 310, 860, 360),
        (720, 420, 652, 420),
        (390, 420, 322, 420),
    ]
    for index, (x1, y1, x2, y2) in enumerate(arrows):
        blocks.append(
            f"""
  <path d="M{x1} {y1} L{x2} {y2}" stroke="#211f1a" stroke-width="3" stroke-linecap="round" marker-end="url(#arrowhead)"/>"""
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="560" viewBox="0 0 1100 560" role="img" aria-label="Pitcher Twin model pipeline">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L8,3 z" fill="#211f1a"/>
    </marker>
    <style>
      .title {{ font: 850 44px ui-sans-serif, system-ui, sans-serif; fill: #191815; }}
      .subtitle {{ font: 500 21px ui-sans-serif, system-ui, sans-serif; fill: #625b50; }}
      .number {{ font: 900 18px ui-sans-serif, system-ui, sans-serif; fill: #fffaf0; }}
      .step-title {{ font: 800 19px ui-sans-serif, system-ui, sans-serif; fill: #191815; }}
      .step-sub {{ font: 500 13px ui-sans-serif, system-ui, sans-serif; fill: #625b50; }}
    </style>
  </defs>
  <rect width="1100" height="560" fill="#f7f4ec"/>
  <rect x="28" y="28" width="1044" height="504" rx="26" fill="#ffffff" opacity="0.72" stroke="#d7cdb8"/>
  <text x="54" y="88" class="title">From Real Pitches To A Validated Practice Envelope</text>
  <text x="54" y="122" class="subtitle">Every generated pitch is tied back to a real-data split and a validation layer.</text>
  {''.join(blocks)}
</svg>
"""
    _write(ASSET_DIR / "pipeline.svg", svg)


def build_rolling_gif() -> None:
    width, height = 1120, 520
    background = (247, 244, 236)
    ink = (25, 24, 21)
    green = (31, 122, 77)
    amber = (180, 127, 26)
    red = (164, 61, 50)
    cream = (255, 250, 240)
    title_font = _font(34, bold=True)
    label_font = _font(18)
    small_font = _font(15)
    metric_font = _font(24, bold=True)
    frames: list[Image.Image] = []
    plot_left, plot_top, plot_right, plot_bottom = 80, 205, 1040, 405
    min_auc, max_auc = 0.55, 0.95

    def xy(index: int, auc: float) -> tuple[float, float]:
        x = plot_left + index * ((plot_right - plot_left) / (len(ROLLING_AUCS) - 1))
        y = plot_bottom - ((auc - min_auc) / (max_auc - min_auc)) * (plot_bottom - plot_top)
        return x, y

    for active in range(len(ROLLING_AUCS)):
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((28, 24, width - 28, height - 28), radius=24, fill=(255, 255, 255))
        draw.text((58, 54), "Rolling Temporal Validation", font=title_font, fill=ink)
        draw.text(
            (58, 98),
            "Train on past games, test on the next future window. Lower AUC is better.",
            font=label_font,
            fill=(98, 91, 80),
        )
        draw.rounded_rectangle((820, 50, 1030, 116), radius=14, fill=(250, 242, 224))
        draw.text((844, 67), f"Fold {active + 1}/10", font=metric_font, fill=ink)
        draw.text((844, 96), f"AUC {ROLLING_AUCS[active]:.3f}", font=label_font, fill=red if ROLLING_AUCS[active] > 0.8 else amber)

        for i in range(31):
            x = 66 + i * 31
            color = (216, 205, 184)
            if i < 10 + active * 2:
                color = green
            if 10 + active * 2 <= i < 12 + active * 2:
                color = amber
            draw.rounded_rectangle((x, 138, x + 22, 158), radius=5, fill=color)
        draw.text(
            (66, 164),
            "games 1-30: green=train, amber=test window",
            font=small_font,
            fill=(98, 91, 80),
        )

        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(45, 43, 38), width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(45, 43, 38), width=2)
        for value, label, color in [(0.60, "fold target 0.60", green), (0.80, "worst-fold ceiling 0.80", red)]:
            y = xy(0, value)[1]
            draw.line((plot_left, y, plot_right, y), fill=color, width=2)
            draw.text((plot_right - 170, y - 24), label, font=small_font, fill=color)

        points = [xy(i, auc) for i, auc in enumerate(ROLLING_AUCS)]
        for i in range(active):
            draw.line((points[i], points[i + 1]), fill=ink, width=4)
        for i, (x, y) in enumerate(points[: active + 1]):
            auc = ROLLING_AUCS[i]
            color = green if auc <= 0.60 else amber if auc < 0.80 else red
            radius = 8 if i != active else 13
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=ink, width=2)
            draw.text((x - 12, plot_bottom + 14), str(i + 1), font=small_font, fill=(98, 91, 80))
        draw.text((plot_left - 4, plot_top - 28), "C2ST AUC", font=label_font, fill=ink)
        draw.text((plot_right - 70, plot_bottom + 44), "fold", font=label_font, fill=ink)

        status = "passes fold target" if ROLLING_AUCS[active] <= 0.60 else "diagnostic fold"
        draw.rounded_rectangle((58, 430, 1060, 474), radius=14, fill=cream, outline=(216, 205, 184))
        draw.text(
            (80, 442),
            f"Status: rolling_diagnostic. Fold {active + 1} {status}; reliability means many future windows, not one lucky fold.",
            font=label_font,
            fill=ink,
        )
        frames.extend([image] * 7)
    frames[0].save(
        ASSET_DIR / "rolling-window-validation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=95,
        loop=0,
        optimize=True,
    )


def build_excalidraw() -> None:
    def rect(element_id: str, x: int, y: int, w: int, h: int, text: str, color: str) -> list[dict]:
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

    elements: list[dict] = []
    cards = [
        ("statcast", 80, 120, "Real Statcast\npitch rows", "#dcefe5"),
        ("features", 330, 120, "Layered\nfeatures", "#e5edf0"),
        ("generator", 580, 120, "Factorized\nphysics generator", "#f4e6bd"),
        ("validator", 830, 120, "C2ST\nvalidator", "#f0d3cd"),
        ("rolling", 1080, 120, "Rolling\nscoreboard", "#f3dfbd"),
    ]
    for card in cards:
        elements.extend(rect(card[0], card[1], card[2], 180, 96, card[3], card[4]))
    elements.extend(rect("export", 580, 310, 180, 96, "Trajekt-shaped\nsession JSON", "#dcefe5"))

    for index in range(len(cards) - 1):
        x1 = cards[index][1] + 184
        x2 = cards[index + 1][1] - 12
        y = 168
        elements.append(
            {
                "id": f"arrow_{index}",
                "type": "arrow",
                "x": x1,
                "y": y,
                "width": x2 - x1,
                "height": 0,
                "angle": 0,
                "strokeColor": "#191815",
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "strokeWidth": 2,
                "strokeStyle": "solid",
                "roughness": 1,
                "opacity": 100,
                "groupIds": [],
                "frameId": None,
                "roundness": {"type": 2},
                "seed": 1010 + index,
                "version": 1,
                "versionNonce": 2020 + index,
                "isDeleted": False,
                "boundElements": None,
                "updated": 1,
                "link": None,
                "locked": False,
                "points": [[0, 0], [x2 - x1, 0]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow",
            }
        )
    elements.append(
        {
            "id": "arrow_export",
            "type": "arrow",
            "x": 670,
            "y": 220,
            "width": 0,
            "height": 88,
            "angle": 0,
            "strokeColor": "#191815",
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": {"type": 2},
            "seed": 4040,
            "version": 1,
            "versionNonce": 5050,
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
            "points": [[0, 0], [0, 88]],
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
        }
    )
    elements.extend(rect("note", 80, 310, 370, 96, "The point is reliability:\nrolling validation is the truth test.", "#fffaf0"))

    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#f7f4ec",
            "currentItemFontFamily": 1,
        },
        "files": {},
    }
    (ASSET_DIR / "pitcher-twin-architecture.excalidraw").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_prompt_note() -> None:
    note = """# README Visual Assets

Generated and deterministic assets used by the README.

## `hero-pitcher-twin.png`

Built with the built-in image generation tool.

Prompt summary:

> Cinematic README hero image for a baseball ML project that models pitcher variability as probability clouds, with a baseball in motion, translucent trajectory envelopes, subtle strike-zone/data overlays, charcoal/emerald/amber/red palette, no text, no logos, no real player likeness.

## Deterministic Assets

- `scoreboard.svg`: exact primary-scoreboard numbers.
- `pipeline.svg`: model/data flow overview.
- `rolling-window-validation.gif`: animated rolling-window validation explanation.
- `pitcher-twin-architecture.excalidraw`: editable architecture source.
"""
    _write(ASSET_DIR / "README.md", note)


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    build_scoreboard_svg()
    build_pipeline_svg()
    build_rolling_gif()
    build_excalidraw()
    build_prompt_note()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
