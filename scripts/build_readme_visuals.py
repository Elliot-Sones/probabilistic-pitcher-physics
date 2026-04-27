#!/usr/bin/env python3
"""Build deterministic README visuals for the Pitcher Twin project."""

from __future__ import annotations

import json
import hashlib
import html
import math
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
BEST_RESULT = {
    "pitcher": "Tarik Skubal",
    "pitch_type": "FF",
    "season": "2025",
    "pitches": 835,
    "games": 31,
    "holdout": 251,
    "auc": 0.533,
    "pass_rate": 1.00,
}


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


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def build_scoreboard_svg() -> None:
    rows = [
        ("Single 70/30 split", "validated", "0.533", "#1f7a4d", "Skubal FF ceiling result"),
        (
            "Rolling truth test",
            "diagnostic",
            "0.702",
            "#b47f1a",
            "goal <= 0.620, hit rate >= 0.40",
        ),
        (
            "Worst rolling fold",
            "miss",
            "0.929",
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
    <rect x="555" y="18" width="260" height="52" rx="18" fill="{color}" opacity="0.13"/>
    <text x="584" y="40" class="metric" fill="{color}">{value}</text>
    <text x="584" y="63" class="pill" fill="{color}">{status}</text>
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


def build_scoreboard_png() -> None:
    width, height = 1400, 760
    image = Image.new("RGB", (width, height), (25, 24, 21))
    draw = ImageDraw.Draw(image)
    title_font = _font(64, bold=True)
    eyebrow_font = _font(28, bold=True)
    subtitle_font = _font(30)
    row_title_font = _font(38, bold=True)
    row_note_font = _font(25)
    metric_font = _font(50, bold=True)
    status_font = _font(25, bold=True)
    colors = [(31, 122, 77), (180, 127, 26), (164, 61, 50)]
    rows = [
        ("Single 70/30 split", "Skubal FF ceiling result", "0.533", "VALIDATED"),
        ("Rolling truth test", "goal <= 0.620, hit rate >= 0.40", "0.702", "DIAGNOSTIC"),
        ("Worst rolling fold", "release/spin state drift remains visible", "0.929", "MISS"),
    ]
    draw.text((70, 66), "PRIMARY VALIDATION STORY", font=eyebrow_font, fill=(215, 165, 49))
    draw.text((70, 125), "Honest Scoreboard", font=title_font, fill=(247, 244, 236))
    draw.text(
        (70, 205),
        "Single split shows the ceiling. Rolling windows decide reliability.",
        font=subtitle_font,
        fill=(216, 209, 193),
    )
    for line_index, color in enumerate(colors):
        y = 548 + line_index * 55
        draw.arc((-80, y - 260, 1320, y + 400), 200, 340, fill=color, width=4)
    for index, (title, note, value, status) in enumerate(rows):
        y = 280 + index * 130
        color = colors[index]
        draw.rounded_rectangle((70, y, 1330, y + 94), radius=14, fill=(242, 242, 241))
        draw.rounded_rectangle((70, y, 82, y + 94), radius=6, fill=color)
        draw.text((120, y + 18), title, font=row_title_font, fill=(25, 24, 21))
        draw.text((120, y + 59), note, font=row_note_font, fill=(98, 91, 80))
        draw.rounded_rectangle((1040, y + 17, 1292, y + 77), radius=26, fill=_blend(color, (242, 242, 241), 0.16))
        draw.text((1084, y + 15), value, font=metric_font, fill=(25, 24, 21))
        draw.text((1086, y + 62), status, font=status_font, fill=color)
    image.save(ASSET_DIR / "scoreboard.png", optimize=True)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(round(a[i] * amount + b[i] * (1 - amount)) for i in range(3))


def _save_gif(path: Path, frames: list[Image.Image], *, duration: int = 95) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )


def _draw_label_value(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    value: str,
    *,
    accent: tuple[int, int, int],
) -> None:
    x, y = xy
    draw.text((x, y), label.upper(), font=_font(17, bold=True), fill=(98, 91, 80))
    draw.text((x, y + 23), value, font=_font(36, bold=True), fill=accent)


def build_best_results_gif() -> None:
    width, height = 1400, 760
    ink = (25, 24, 21)
    cream = (247, 244, 236)
    white = (255, 255, 255)
    green = (31, 122, 77)
    amber = (215, 165, 49)
    red = (164, 61, 50)
    frames: list[Image.Image] = []
    title_font = _font(58, bold=True)
    sub_font = _font(28)
    metric_font = _font(62, bold=True)
    small_font = _font(21)
    hold_frames = 6

    for step in range(30):
        t = min(1.0, step / 20)
        image = Image.new("RGB", (width, height), cream)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((42, 42, width - 42, height - 42), radius=30, fill=white)
        draw.text((82, 82), "Best Validated Result", font=title_font, fill=ink)
        draw.text(
            (82, 152),
            "Tarik Skubal 2025 FF: generated pitches vs later real Statcast holdout",
            font=sub_font,
            fill=(98, 91, 80),
        )
        draw.rounded_rectangle((82, 220, 1318, 610), radius=28, fill=(250, 242, 224))
        draw.text((122, 260), "C2ST AUC", font=small_font, fill=(98, 91, 80))
        draw.text((122, 294), f"{BEST_RESULT['auc']:.3f}", font=metric_font, fill=green)
        draw.text((122, 372), "lower is better; 0.50 is ideal", font=small_font, fill=(98, 91, 80))
        draw.text((122, 450), "Pass rate", font=small_font, fill=(98, 91, 80))
        draw.text((122, 484), f"{BEST_RESULT['pass_rate']:.0%}", font=metric_font, fill=green)

        scale_x0, scale_y = 500, 355
        scale_w = 700
        draw.line((scale_x0, scale_y, scale_x0 + scale_w, scale_y), fill=(216, 205, 184), width=12)
        for value, color, label in [(0.50, green, "ideal"), (0.60, amber, "target"), (0.90, red, "easy to detect")]:
            x = scale_x0 + int(((value - 0.50) / 0.45) * scale_w)
            draw.line((x, scale_y - 28, x, scale_y + 28), fill=color, width=4)
            draw.text((x - 34, scale_y + 42), label, font=_font(17, bold=True), fill=color)
        current_auc = 0.90 + (BEST_RESULT["auc"] - 0.90) * t
        x = scale_x0 + int(((current_auc - 0.50) / 0.45) * scale_w)
        draw.ellipse((x - 20, scale_y - 20, x + 20, scale_y + 20), fill=green, outline=ink, width=3)
        draw.text((x - 44, scale_y - 72), f"{current_auc:.3f}", font=_font(26, bold=True), fill=ink)

        _draw_label_value(draw, (520, 460), "Real FF pitches", f"{BEST_RESULT['pitches']:,}", accent=ink)
        _draw_label_value(draw, (755, 460), "Games", str(BEST_RESULT["games"]), accent=ink)
        _draw_label_value(draw, (930, 460), "Holdout rows", str(BEST_RESULT["holdout"]), accent=ink)
        _draw_label_value(draw, (1160, 460), "Pitch", str(BEST_RESULT["pitch_type"]), accent=ink)

        draw.rounded_rectangle((82, 636, 1318, 694), radius=18, fill=(220, 239, 229))
        draw.text(
            (112, 654),
            "Result first: the classifier barely separates generated Skubal fastballs from later real ones.",
            font=_font(25, bold=True),
            fill=green,
        )
        frames.extend([image] * (hold_frames if step in {0, 21, 29} else 1))
    _save_gif(ASSET_DIR / "best-validated-result.gif", frames, duration=80)


def build_pitch_cloud_generator_gif() -> None:
    width, height = 1400, 760
    ink = (25, 24, 21)
    cream = (247, 244, 236)
    green = (31, 122, 77)
    amber = (215, 165, 49)
    red = (164, 61, 50)
    frames: list[Image.Image] = []
    title_font = _font(54, bold=True)
    sub_font = _font(26)
    label_font = _font(22, bold=True)
    small_font = _font(18)
    controls = [
        ("Pitcher", "Tarik Skubal"),
        ("Pitch", "FF"),
        ("Inning", "7"),
        ("Pitch count", "88"),
        ("Count", "2-2"),
        ("Batter", "LHB"),
    ]
    points = []
    for i in range(52):
        angle = i * 0.63
        radius = 1.0 + (i % 7) * 0.18
        px = math.cos(angle) * radius * 52 + math.sin(i) * 16
        pz = math.sin(angle * 0.8) * radius * 42 + math.cos(i * 0.4) * 14
        velo = 94.7 + math.sin(i * 0.5) * 1.1
        points.append((px, pz, velo))

    for frame_index in range(36):
        shown = min(len(points), 5 + frame_index * 2)
        image = Image.new("RGB", (width, height), cream)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((42, 42, width - 42, height - 42), radius=30, fill=(255, 255, 255))
        draw.text((82, 82), "Generate A Pitch Probability Cloud", font=title_font, fill=ink)
        draw.text(
            (82, 145),
            "The output is a distribution of likely pitch characteristics, not one guessed pitch.",
            font=sub_font,
            fill=(98, 91, 80),
        )
        for index, (label, value) in enumerate(controls):
            x = 82 + (index % 3) * 265
            y = 220 + (index // 3) * 86
            draw.rounded_rectangle((x, y, x + 228, y + 62), radius=16, fill=(250, 242, 224))
            draw.text((x + 18, y + 12), label, font=small_font, fill=(98, 91, 80))
            draw.text((x + 18, y + 34), value, font=label_font, fill=ink)

        zone = (900, 252, 1240, 610)
        draw.rounded_rectangle((845, 190, 1292, 654), radius=22, fill=(247, 244, 236), outline=(216, 205, 184), width=2)
        draw.text((870, 212), "Generated plate-location samples", font=label_font, fill=ink)
        draw.rectangle(zone, outline=ink, width=4)
        for i in range(1, 3):
            x = zone[0] + i * ((zone[2] - zone[0]) // 3)
            y = zone[1] + i * ((zone[3] - zone[1]) // 3)
            draw.line((x, zone[1], x, zone[3]), fill=(216, 205, 184), width=2)
            draw.line((zone[0], y, zone[2], y), fill=(216, 205, 184), width=2)
        for px, pz, velo in points[:shown]:
            x = int((zone[0] + zone[2]) / 2 + px)
            y = int((zone[1] + zone[3]) / 2 - pz)
            color = green if velo >= 95.0 else amber
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=ink)
        draw.rounded_rectangle((82, 520, 780, 640), radius=20, fill=(220, 239, 229))
        draw.text((112, 544), "Expected FF envelope", font=label_font, fill=green)
        draw.text((112, 582), "velocity 94-96 mph  |  command cloud  |  sampled session JSON", font=small_font, fill=ink)
        if frame_index > 25:
            draw.rounded_rectangle((112, 612, 430, 668), radius=18, fill=green)
            draw.text((142, 628), "Export samples", font=_font(24, bold=True), fill=(255, 250, 240))
        frames.append(image)
    _save_gif(ASSET_DIR / "pitch-cloud-generator.gif", frames, duration=85)


def build_factorized_chain_gif() -> None:
    width, height = 1400, 760
    ink = (25, 24, 21)
    cream = (247, 244, 236)
    colors = [(31, 122, 77), (47, 111, 130), (215, 165, 49), (164, 61, 50)]
    layers = [
        ("Release state", "velocity, spin, release point"),
        ("Movement", "break around that release"),
        ("Trajectory", "speed + acceleration consistency"),
        ("Command", "plate-location cloud"),
    ]
    frames: list[Image.Image] = []
    for frame_index in range(40):
        active = min(3, frame_index // 9)
        image = Image.new("RGB", (width, height), cream)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((42, 42, width - 42, height - 42), radius=30, fill=(255, 255, 255))
        draw.text((82, 82), "How The Model Learns A Pitcher's Style", font=_font(54, bold=True), fill=ink)
        draw.text(
            (82, 145),
            "It does not average every feature together; it samples physics in a chain.",
            font=_font(26),
            fill=(98, 91, 80),
        )
        start_x, y = 92, 330
        for index, (title, subtitle) in enumerate(layers):
            x = start_x + index * 318
            color = colors[index]
            fill = _blend(color, (255, 255, 255), 0.16 if index <= active else 0.05)
            outline = color if index <= active else (216, 205, 184)
            draw.rounded_rectangle((x, y, x + 260, y + 170), radius=26, fill=fill, outline=outline, width=4)
            draw.ellipse((x + 24, y + 24, x + 72, y + 72), fill=color if index <= active else (216, 205, 184))
            draw.text((x + 40, y + 34), str(index + 1), font=_font(24, bold=True), fill=(255, 250, 240))
            draw.text((x + 24, y + 88), title, font=_font(24, bold=True), fill=ink)
            draw.text((x + 24, y + 122), subtitle, font=_font(17), fill=(98, 91, 80))
            if index < len(layers) - 1:
                arrow_color = ink if index < active else (216, 205, 184)
                _draw_arrow(draw, (x + 268, y + 86), (x + 308, y + 86), arrow_color)
        draw.rounded_rectangle((82, 570, 1318, 652), radius=22, fill=(250, 242, 224))
        message = [
            "Start with the pitcher's release state.",
            "Then sample movement that is plausible for that release.",
            "Then preserve trajectory/acceleration consistency.",
            "Finally produce a command cloud and exportable samples.",
        ][active]
        draw.text((116, 596), message, font=_font(28, bold=True), fill=colors[active])
        frames.extend([image] * (5 if frame_index % 9 == 0 else 1))
    _save_gif(ASSET_DIR / "factorized-physics-chain.gif", frames, duration=90)


def build_c2st_validator_gif() -> None:
    width, height = 1400, 760
    ink = (25, 24, 21)
    cream = (247, 244, 236)
    green = (31, 122, 77)
    amber = (215, 165, 49)
    red = (164, 61, 50)
    stages = [
        ("Train", "early real pitches"),
        ("Generate", "synthetic pitch samples"),
        ("Hold out", "later real pitches"),
        ("Classifier", "tries to spot fakes"),
        ("Score", "AUC 0.533"),
    ]
    frames: list[Image.Image] = []
    for frame_index in range(42):
        active = min(4, frame_index // 8)
        image = Image.new("RGB", (width, height), cream)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((42, 42, width - 42, height - 42), radius=30, fill=(255, 255, 255))
        draw.text((82, 82), "How We Prove Realism", font=_font(56, bold=True), fill=ink)
        draw.text(
            (82, 148),
            "A classifier two-sample test asks: can a model tell generated pitches from future real ones?",
            font=_font(26),
            fill=(98, 91, 80),
        )
        for index, (title, subtitle) in enumerate(stages):
            x = 92 + index * 252
            y = 300
            color = green if index <= active else (216, 205, 184)
            draw.rounded_rectangle((x, y, x + 205, y + 138), radius=22, fill=_blend(color, (255, 250, 240), 0.14), outline=color, width=4)
            draw.text((x + 24, y + 34), title, font=_font(27, bold=True), fill=ink)
            draw.text((x + 24, y + 76), subtitle, font=_font(18), fill=(98, 91, 80))
            if index < len(stages) - 1:
                _draw_arrow(draw, (x + 214, y + 69), (x + 244, y + 69), ink if index < active else (216, 205, 184))

        draw.rounded_rectangle((220, 515, 1180, 630), radius=28, fill=(220, 239, 229))
        if active < 4:
            text = "The validator stays blind to the labels until the classifier test."
            color = ink
        else:
            text = "Best result: 0.533 AUC, close to coin-flip separation."
            color = green
        draw.text((260, 552), text, font=_font(34, bold=True), fill=color)
        draw.line((320, 654, 1080, 654), fill=(216, 205, 184), width=12)
        current = 0.90 + (BEST_RESULT["auc"] - 0.90) * min(1, max(0, (frame_index - 31) / 8))
        x = 320 + int(((current - 0.50) / 0.45) * 760)
        draw.ellipse((x - 16, 638, x + 16, 670), fill=green if active >= 4 else amber, outline=ink, width=2)
        draw.text((300, 676), "0.50 ideal", font=_font(16, bold=True), fill=green)
        draw.text((1010, 676), "1.00 obvious fake", font=_font(16, bold=True), fill=red)
        frames.extend([image] * (4 if frame_index % 8 == 0 else 1))
    _save_gif(ASSET_DIR / "c2st-validator.gif", frames, duration=90)


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


def build_pipeline_png() -> None:
    width, height = 1400, 720
    image = Image.new("RGB", (width, height), (247, 244, 236))
    draw = ImageDraw.Draw(image)
    title_font = _font(48, bold=True)
    subtitle_font = _font(26)
    step_title_font = _font(28, bold=True)
    step_sub_font = _font(20)
    number_font = _font(25, bold=True)
    ink = (25, 24, 21)
    draw.rounded_rectangle((42, 42, width - 42, height - 42), radius=30, fill=(255, 255, 255), outline=(215, 205, 184), width=2)
    draw.text((82, 82), "From Real Pitches To A Validated Practice Envelope", font=title_font, fill=ink)
    draw.text(
        (82, 142),
        "Every generated pitch is tied back to a real-data split and a validation layer.",
        font=subtitle_font,
        fill=(98, 91, 80),
    )
    steps = [
        ("Real Statcast", "public pitch rows", (31, 122, 77)),
        ("Feature layers", "release, spin, movement", (47, 111, 130)),
        ("Generator suite", "factorized + drift models", (215, 165, 49)),
        ("C2ST validator", "classifier realism test", (164, 61, 50)),
        ("Rolling board", "future-game reliability", (180, 127, 26)),
        ("Trajekt export", "sampled session JSON", (31, 122, 77)),
    ]
    positions = [(88, 230), (520, 230), (952, 230), (88, 470), (520, 470), (952, 470)]
    for index, ((title, subtitle, color), (x, y)) in enumerate(zip(steps, positions, strict=True)):
        draw.rounded_rectangle((x, y, x + 360, y + 142), radius=22, fill=(255, 250, 240), outline=ink, width=3)
        draw.ellipse((x + 30, y + 30, x + 78, y + 78), fill=color)
        draw.text((x + 45, y + 39), str(index + 1), font=number_font, fill=(255, 250, 240))
        draw.text((x + 108, y + 42), title, font=step_title_font, fill=ink)
        draw.text((x + 108, y + 86), subtitle, font=step_sub_font, fill=(98, 91, 80))
    arrows = [
        ((452, 301), (512, 301)),
        ((884, 301), (944, 301)),
        ((1132, 376), (1132, 462)),
        ((952, 541), (888, 541)),
        ((520, 541), (456, 541)),
    ]
    for start, end in arrows:
        _draw_arrow(draw, start, end, ink)
    image.save(ASSET_DIR / "pipeline.png", optimize=True)


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

- `best-validated-result.gif`: results-first animation for the top of the README.
- `pitch-cloud-generator.gif`: animated conditional generator explanation.
- `factorized-physics-chain.gif`: animated model-structure explanation.
- `c2st-validator.gif`: animated validation explanation.
- `pipeline.png`: model/data flow overview used by README.
- `pipeline.svg`: editable SVG source for the pipeline.
- `rolling-window-validation.gif`: animated rolling-window validation explanation.
- `pitcher-twin-architecture.excalidraw`: editable architecture source.
"""
    _write(ASSET_DIR / "README.md", note)


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    build_best_results_gif()
    build_pitch_cloud_generator_gif()
    build_factorized_chain_gif()
    build_c2st_validator_gif()
    build_pipeline_svg()
    build_pipeline_png()
    build_rolling_gif()
    build_excalidraw()
    build_prompt_note()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
