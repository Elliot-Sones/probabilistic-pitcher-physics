"""Build the static landing page used for the public hosted demo.

Generates `site/index.html` with embedded Plotly charts. The output is a
self-contained static directory deployable to Vercel/Netlify/GitHub Pages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.features import clean_pitch_features  # noqa: E402
from pitcher_twin.validator import temporal_train_holdout  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "skubal_2025.csv"
SESSION_PATH = ROOT / "docs" / "assets" / "final_session.json"
ROLLING_BOARD_PATH = (
    ROOT / "outputs" / "rolling_validation_skubal_2025_ff" / "rolling_validation_board.json"
)
VALIDATION_BOARD_PATH = ROOT / "outputs" / "validation_board_skubal_2025_top3" / "leaderboard.csv"
LATEST_BOARD_PATH = ROOT / "outputs" / "validation_board_latest_statcast_top3" / "leaderboard.csv"
OUTPUT_DIR = ROOT / "site"

REAL_COLOR = "#171512"
SIM_COLOR = "#1f7a4d"


def figure_to_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def style_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.86)",
        height=height,
        margin={"l": 12, "r": 12, "t": 36, "b": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        legend_title_text="",
        font={"color": "#171512", "family": "Inter, system-ui, sans-serif"},
    )
    return fig


def build_overlay_dataframe() -> pd.DataFrame:
    clean = clean_pitch_features(load_statcast_cache(DATA_PATH), pitch_types=None)
    subset = clean[(clean["pitcher"] == 669373) & (clean["pitch_type"] == "FF")].copy()
    _, holdout = temporal_train_holdout(subset, train_fraction=0.7)

    real_columns = [
        column
        for column in [
            "plate_x",
            "plate_z",
            "release_speed",
            "release_spin_rate",
            "pfx_x",
            "pfx_z",
        ]
        if column in holdout.columns
    ]
    real = holdout[real_columns].copy()
    real["source"] = "Real holdout"

    session = json.loads(SESSION_PATH.read_text())
    sim_rows = []
    for pitch in session.get("pitches", []):
        velocity = pitch.get("velocity", {})
        spin = pitch.get("spin", {})
        movement = pitch.get("movement", {})
        plate = pitch.get("plate_target", {})
        sim_rows.append(
            {
                "plate_x": plate.get("x"),
                "plate_z": plate.get("z"),
                "release_speed": velocity.get("release_speed"),
                "release_spin_rate": spin.get("rate"),
                "pfx_x": movement.get("pfx_x"),
                "pfx_z": movement.get("pfx_z"),
            }
        )
    sim = pd.DataFrame(sim_rows)
    sim["source"] = "Simulated"
    sim_columns = [column for column in real_columns if column in sim.columns]
    sim = sim[sim_columns + ["source"]]

    return pd.concat([real, sim], ignore_index=True)


def make_plate_overlay(combined: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        combined,
        x="plate_x",
        y="plate_z",
        color="source",
        color_discrete_map={"Real holdout": REAL_COLOR, "Simulated": SIM_COLOR},
        opacity=0.62,
        hover_data={
            "release_speed": ":.1f",
            "release_spin_rate": ":.0f",
            "source": True,
            "plate_x": False,
            "plate_z": False,
        },
        labels={"plate_x": "Plate X (ft)", "plate_z": "Plate Z (ft)"},
    )
    fig.update_traces(marker={"size": 11, "line": {"width": 0.6, "color": REAL_COLOR}})
    fig.add_shape(
        type="rect",
        x0=-0.83,
        x1=0.83,
        y0=1.5,
        y1=3.5,
        line={"color": REAL_COLOR, "width": 2},
    )
    fig.update_xaxes(range=[-2.2, 2.2], zeroline=False)
    fig.update_yaxes(range=[0.4, 4.6], zeroline=False, scaleanchor="x", scaleratio=1)
    return style_layout(fig, height=460)


def make_velocity_overlay(combined: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        combined,
        x="release_speed",
        color="source",
        color_discrete_map={"Real holdout": REAL_COLOR, "Simulated": SIM_COLOR},
        barmode="overlay",
        opacity=0.65,
        nbins=20,
        labels={"release_speed": "Release velocity (mph)"},
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color=REAL_COLOR)
    return style_layout(fig, height=320)


def make_spin_overlay(combined: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        combined,
        x="release_spin_rate",
        color="source",
        color_discrete_map={"Real holdout": REAL_COLOR, "Simulated": SIM_COLOR},
        barmode="overlay",
        opacity=0.65,
        nbins=20,
        labels={"release_spin_rate": "Spin rate (rpm)"},
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color=REAL_COLOR)
    return style_layout(fig, height=320)


def make_movement_overlay(combined: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        combined,
        x="pfx_x",
        y="pfx_z",
        color="source",
        color_discrete_map={"Real holdout": REAL_COLOR, "Simulated": SIM_COLOR},
        opacity=0.62,
        labels={"pfx_x": "Horizontal break (ft)", "pfx_z": "Vertical break (ft)"},
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 0.5, "color": REAL_COLOR}})
    return style_layout(fig, height=380)


def make_rolling_chart() -> go.Figure | None:
    if not ROLLING_BOARD_PATH.exists():
        return None
    board = json.loads(ROLLING_BOARD_PATH.read_text())
    folds = pd.DataFrame(board["folds"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=folds["fold_index"],
            y=folds["physics_core_mean_auc"],
            mode="lines+markers",
            marker={"size": 11, "color": "#a43d32", "line": {"width": 1, "color": REAL_COLOR}},
            line={"color": REAL_COLOR, "width": 2},
            customdata=folds[
                [
                    "train_game_range",
                    "test_game_range",
                    "best_physics_core_model",
                    "physics_core_pass_rate",
                ]
            ],
            hovertemplate=(
                "<b>Fold %{x}</b><br>"
                "Physics AUC=%{y:.3f}<br>"
                "Train games=%{customdata[0]}<br>"
                "Test games=%{customdata[1]}<br>"
                "Model=%{customdata[2]}<br>"
                "Pass=%{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=float(board["target_auc"]),
        line_dash="dash",
        line_color=SIM_COLOR,
        annotation_text="fold success target (≤ 0.60)",
    )
    fig.add_hline(
        y=0.800,
        line_dash="dot",
        line_color="#a43d32",
        annotation_text="worst-fold ceiling (< 0.80)",
    )
    fig.update_layout(
        xaxis_title="Rolling fold (future game window)",
        yaxis_title="Physics-core C2ST AUC",
    )
    return style_layout(fig, height=420)


def make_cross_pitcher_chart() -> go.Figure | None:
    frames = []
    for path, label in [
        (VALIDATION_BOARD_PATH, "Skubal pitch types"),
        (LATEST_BOARD_PATH, "Other pitchers (FF)"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["board"] = label
        frames.append(df)
    if not frames:
        return None
    leaderboard = pd.concat(frames, ignore_index=True)
    fig = px.scatter(
        leaderboard,
        x="holdout_count",
        y="physics_core_mean_auc",
        color="artifact_status",
        size="pitch_count",
        symbol="board",
        hover_data=["pitcher_name", "pitch_type", "best_physics_core_model"],
        color_discrete_map={
            "validated_temporal_success": "#1f7a4d",
            "physics_core_candidate": "#d7a531",
            "physics_core_diagnostic": "#a43d32",
        },
        labels={
            "holdout_count": "Temporal holdout rows",
            "physics_core_mean_auc": "Physics-core C2ST AUC",
            "artifact_status": "Status",
            "board": "Board",
        },
    )
    fig.add_hline(y=0.60, line_dash="dash", line_color=REAL_COLOR, annotation_text="target ≤ 0.60")
    return style_layout(fig, height=420)


HTML_TEMPLATE = dedent(
    """\
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Pitcher Twin · live demo</title>
    <meta name="description" content="A generative model that learns each MLB pitcher's full pitch distribution and produces synthetic pitches a classifier struggles to tell apart from real held-out ones.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <style>
    :root {{
      --bg: #f7f4ec;
      --ink: #171512;
      --muted: #5b5247;
      --green: #1f7a4d;
      --red: #a43d32;
      --gold: #d7a531;
      --card-bg: #ffffff;
      --card-border: rgba(28, 26, 23, .10);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background:
        linear-gradient(90deg, rgba(31, 122, 77, .07) 1px, transparent 1px),
        linear-gradient(180deg, rgba(31, 122, 77, .07) 1px, transparent 1px),
        var(--bg);
      background-size: 28px 28px;
      color: var(--ink);
      font-family: "Inter", system-ui, sans-serif;
      line-height: 1.55;
    }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 3rem 1.4rem 5rem; }}
    .kicker {{
      font-size: .78rem;
      letter-spacing: .22em;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: .35rem;
    }}
    h1 {{
      font-size: clamp(2.2rem, 4.6vw, 3.4rem);
      line-height: 1.04;
      letter-spacing: -.02em;
      margin: .1rem 0 1rem;
      color: var(--ink);
      font-weight: 800;
    }}
    h2 {{ font-size: 1.6rem; margin-top: 3rem; letter-spacing: -.01em; }}
    h3 {{ font-size: 1.15rem; margin-top: 1.6rem; letter-spacing: -.005em; }}
    .lede {{
      font-size: 1.18rem;
      max-width: 880px;
      color: #2a2722;
      margin: 0 0 2rem;
    }}
    .lede strong {{ color: var(--ink); }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1rem;
      margin: 1.4rem 0 2.4rem;
    }}
    .score-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.25rem 1.4rem;
      box-shadow: 0 18px 38px rgba(31, 27, 20, .07);
      position: relative;
      overflow: hidden;
    }}
    .score-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 5px;
      background: var(--green);
    }}
    .score-card.diag::before {{ background: var(--red); }}
    .score-card.tested::before {{ background: var(--gold); }}
    .score-label {{
      font-size: .75rem;
      letter-spacing: .14em;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: .3rem;
    }}
    .score-value {{
      font-size: 2.5rem;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -.02em;
    }}
    .score-unit {{
      font-size: 1rem;
      color: var(--muted);
      font-weight: 600;
      margin-left: .25rem;
    }}
    .score-meta {{
      margin-top: .65rem;
      font-size: .92rem;
      color: #3b362f;
      line-height: 1.45;
    }}
    .panel {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.25rem;
      box-shadow: 0 12px 28px rgba(31, 27, 20, .06);
      margin: 1.5rem 0;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 1.25rem;
      align-items: start;
    }}
    @media (max-width: 880px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .grid-2 > .panel {{ margin: 0; }}
    .legend-key {{
      display: inline-flex;
      align-items: center;
      gap: .35rem;
      margin-right: 1rem;
      font-size: .92rem;
    }}
    .legend-dot {{
      width: .65rem; height: .65rem; border-radius: 999px;
      display: inline-block;
    }}
    .legend-dot.real {{ background: var(--ink); }}
    .legend-dot.sim {{ background: var(--green); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: .5rem;
      font-size: .96rem;
    }}
    th, td {{
      padding: .6rem .8rem;
      border-bottom: 1px solid rgba(28, 26, 23, .08);
      text-align: left;
    }}
    th {{
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
      font-size: .78rem;
      color: var(--muted);
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    .pill {{
      display: inline-block;
      padding: .12rem .55rem;
      border-radius: 999px;
      font-size: .72rem;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: white;
    }}
    .pill.green {{ background: var(--green); }}
    .pill.red {{ background: var(--red); }}
    .pill.gold {{ background: var(--gold); color: #2a2117; }}
    code {{
      font-family: "JetBrains Mono", monospace;
      font-size: .9em;
      background: rgba(28, 26, 23, .05);
      padding: .12rem .35rem;
      border-radius: 4px;
    }}
    pre code {{
      display: block;
      padding: 1rem;
      background: rgba(28, 26, 23, .05);
      border-radius: 8px;
      overflow-x: auto;
      line-height: 1.5;
    }}
    a {{ color: var(--green); }}
    a:hover {{ color: #136a40; }}
    .footer {{
      margin-top: 4rem;
      padding-top: 1.5rem;
      border-top: 1px solid rgba(28, 26, 23, .12);
      font-size: .9rem;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      justify-content: space-between;
    }}
    .gh-button {{
      display: inline-block;
      padding: .55rem 1rem;
      border: 1px solid var(--ink);
      border-radius: 8px;
      color: var(--ink);
      text-decoration: none;
      font-weight: 600;
      background: white;
    }}
    .gh-button:hover {{ background: var(--ink); color: white; }}
    .caption {{
      font-size: .92rem;
      color: var(--muted);
      margin-top: .5rem;
    }}
    </style>
    </head>
    <body>
    <main class="container">

    <div class="kicker">Pitcher Twin · live demo</div>
    <h1>We model the cloud, not just the pitch.</h1>
    <p class="lede">
      A real pitcher doesn't throw one fastball &mdash; they throw a cloud of them,
      shaped by count, inning, batter, fatigue, score, and the simple fact that
      no human releases the ball the same way twice. <strong>Pitcher Twin</strong>
      learns that cloud from public Statcast and generates pitches a classifier
      struggles to tell apart from real held-out ones.
    </p>

    <div class="score-grid">
      <div class="score-card">
        <div class="score-label">Validated · single 70/30 split</div>
        <div class="score-value">0.533 <span class="score-unit">AUC</span></div>
        <div class="score-meta">100% pass rate &middot; Skubal 2025 FF &middot; <strong>0.50 = coin flip</strong></div>
      </div>
      <div class="score-card diag">
        <div class="score-label">Diagnostic · rolling stress test</div>
        <div class="score-value">{rolling_value} <span class="score-unit">mean AUC</span></div>
        <div class="score-meta">10 future-game folds &middot; goal &le; 0.620 &middot; the honest gap</div>
      </div>
      <div class="score-card tested">
        <div class="score-label">Tested across</div>
        <div class="score-value">4 <span class="score-unit">pitchers</span></div>
        <div class="score-meta">3 pitch types &middot; Skubal FF/SI/CH + Mattson, Peralta, Bradley FFs</div>
      </div>
    </div>

    <h2>Real held-out pitches vs generated samples</h2>
    <p class="lede">
      <span class="legend-key"><span class="legend-dot real"></span>Dark dots — real Skubal fastballs from the held-out 30% the model never saw.</span>
      <span class="legend-key"><span class="legend-dot sim"></span>Green dots — pitches the model generated from training data only.</span>
      A classifier trained specifically to spot the green ones lands at 0.533 ROC-AUC: barely above a coin flip.
    </p>

    <div class="grid-2">
      <div class="panel">{plate_overlay_div}</div>
      <div class="panel">
        {velocity_overlay_div}
        {spin_overlay_div}
      </div>
    </div>

    <div class="panel">
      <h3>Movement break: horizontal vs vertical</h3>
      <p class="caption">Where the ball ends up (above) is downstream of how it moved (below). Realistic clouds look real on both.</p>
      {movement_overlay_div}
    </div>

    <h2>The honest scoreboard</h2>
    <p class="lede">
      Single-split AUC is the ceiling. Rolling validation across many future-game windows is the
      truth test. Right now Skubal FF clears the single split &mdash; and reveals exactly where
      the next milestone lives.
    </p>

    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Result</th>
            <th>Detail</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Single 70/30 temporal split (Skubal 2025 FF)</td>
            <td><span class="pill green">Validated</span></td>
            <td><strong>0.533 AUC</strong> · 100% pass</td>
            <td>classifier barely beats a coin flip</td>
          </tr>
          <tr>
            <td>Rolling temporal stress test (10 future-game folds)</td>
            <td><span class="pill red">Diagnostic</span></td>
            <td><strong>{rolling_value} mean AUC</strong></td>
            <td>goal ≤ 0.620 · best fold 0.593 · worst 0.929</td>
          </tr>
          <tr>
            <td>Cross-pitcher generalization</td>
            <td><span class="pill gold">Mixed</span></td>
            <td>4 pitchers · 3 pitch types</td>
            <td>Skubal FF validates; SI/CH/Mattson/Peralta/Bradley remain diagnostic</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="grid-2">
      <div class="panel">
        <h3>Rolling fold-by-fold AUC</h3>
        <p class="caption">Each point is one future-game window. The dashed green line is the per-fold success target. The model can match some windows (best 0.593) but drifts hard on others (worst 0.929).</p>
        {rolling_div}
      </div>
      <div class="panel">
        <h3>Cross-pitcher board</h3>
        <p class="caption">Each marker is one pitcher/pitch type. Y axis is C2ST AUC (lower is better). Below the dashed line is validated.</p>
        {cross_pitcher_div}
      </div>
    </div>

    <h2>How it works</h2>

    <div class="panel">
      <h3>The factorized physics chain</h3>
      <p>
        Instead of modeling all pitch features as one tangled blob, the generator chains them in
        physical order &mdash; release point, then velocity and spin given release, then movement
        residual given velocity/spin, then trajectory and command. A Gaussian mixture at each
        layer captures sub-modes (high-inside, low-away, middle-up fastballs aren't one
        distribution); residual layers absorb the mechanical noise the pitcher didn't intend
        &mdash; the human-error envelope. Every layer is conditioned on game state (count, inning,
        batter handedness, pitch-count fatigue, score) and trend-anchored to capture mid-season
        drift in release point and stuff.
      </p>

      <h3>The validator (classifier two-sample test)</h3>
      <p>
        Train on the earlier 70% of a pitcher's pitches. Generate synthetic samples. Mix them
        with real held-out pitches from the later 30%. Train a logistic-regression classifier
        to tell synthetic from real. Report ROC-AUC. Lower means harder to detect.
        <strong>0.50 means the classifier can't distinguish synthetic from real at all.</strong>
      </p>

      <h3>Rolling temporal validation</h3>
      <pre><code>train games  1-10  → test games 11-12
train games  1-12  → test games 13-14
...
train games  1-28  → test games 29-30</code></pre>
      <p>10 rolling folds, 4 repeats per fold. The hard remaining problem isn't command or
      movement &mdash; it's release/spin geometry across future game windows.</p>
    </div>

    <h2>What's next</h2>
    <p class="lede">
      The detection signal in failed folds is concentrated in <code>release_pos_x</code>,
      <code>release_extension</code>, <code>spin_axis_cos/sin</code>, <code>release_spin_rate</code>,
      and <code>vy0</code>. Hand-tuned circular spin residuals (V4) didn't close the gap.
      The next concrete candidate is a learned conditional release-state mixture with circular
      spin-axis component and velocity/spin covariance per latent state &mdash; instead of
      anchoring release/spin to a global average.
    </p>

    <div class="footer">
      <div>
        <strong>Pitcher Twin</strong> · real-data only · public Statcast · classical stats stack
      </div>
      <a href="https://github.com/Elliot-Sones/probabilistic-pitcher-physics" class="gh-button">View on GitHub →</a>
    </div>

    </main>
    </body>
    </html>
    """
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading data...", flush=True)
    combined = build_overlay_dataframe()

    print("building plate overlay...", flush=True)
    plate_div = figure_to_div(make_plate_overlay(combined), "plate-overlay")

    print("building velocity overlay...", flush=True)
    vel_div = figure_to_div(make_velocity_overlay(combined), "velocity-overlay")

    print("building spin overlay...", flush=True)
    spin_div = figure_to_div(make_spin_overlay(combined), "spin-overlay")

    print("building movement overlay...", flush=True)
    movement_div = figure_to_div(make_movement_overlay(combined), "movement-overlay")

    rolling_value = "0.702"
    rolling_div = ""
    rolling_fig = make_rolling_chart()
    if rolling_fig is not None:
        rolling_div = figure_to_div(rolling_fig, "rolling-chart")
        try:
            board = json.loads(ROLLING_BOARD_PATH.read_text())
            scoreboard = board.get("primary_scoreboard", {})
            current = scoreboard.get("current", {})
            mean_auc = current.get("mean_rolling_physics_core_auc")
            if mean_auc is not None:
                rolling_value = f"{mean_auc:.3f}"
        except Exception:  # noqa: BLE001
            pass

    cross_div = ""
    cross_fig = make_cross_pitcher_chart()
    if cross_fig is not None:
        cross_div = figure_to_div(cross_fig, "cross-pitcher-chart")

    html = HTML_TEMPLATE.format(
        rolling_value=rolling_value,
        plate_overlay_div=plate_div,
        velocity_overlay_div=vel_div,
        spin_overlay_div=spin_div,
        movement_overlay_div=movement_div,
        rolling_div=rolling_div or "<p class='caption'>Rolling board artifact not present.</p>",
        cross_pitcher_div=cross_div or "<p class='caption'>Cross-pitcher board not present.</p>",
    )

    output_path = OUTPUT_DIR / "index.html"
    output_path.write_text(html)
    print(f"wrote {output_path} ({len(html):,} bytes)", flush=True)


if __name__ == "__main__":
    main()
