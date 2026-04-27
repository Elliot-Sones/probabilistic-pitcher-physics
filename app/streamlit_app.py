"""Presentation dashboard for the Pitcher Twin real-data demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitcher_twin.conditional import (
    compare_context_distributions,
    layer_status_from_report,
    make_context_dataframe,
    sample_conditional_distribution,
)
from pitcher_twin.data import load_statcast_cache
from pitcher_twin.features import clean_pitch_features
from pitcher_twin.models import fit_generator_suite
from pitcher_twin.rolling_validation import score_rolling_validation_goals
from pitcher_twin.validator import temporal_train_holdout


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "assets" / "real_demo_validation_report.json"
SESSION_PATH = ROOT / "docs" / "assets" / "final_session.json"
MORNING_REPORT_PATH = ROOT / "docs" / "assets" / "real_demo_morning_report.md"
ROLLING_BOARD_PATH = (
    ROOT / "outputs" / "rolling_validation_skubal_2025_ff" / "rolling_validation_board.json"
)
TARGET_AUC = 0.60
VALIDATION_BOARD_OPTIONS = {
    "Skubal 2025 top pitch types": ROOT / "outputs" / "validation_board_skubal_2025_top3",
    "Skubal 2025 V4 release/spin diagnostic": ROOT
    / "outputs"
    / "validation_board_skubal_2025_top3_v4",
    "Latest Statcast top FF candidates": ROOT / "outputs" / "validation_board_latest_statcast_top3",
    "Latest Statcast V4 release/spin diagnostic": ROOT
    / "outputs"
    / "validation_board_latest_statcast_top3_v4",
}

STATUS_COLORS = {
    "validated": "#1f7a4d",
    "borderline": "#b47f1a",
    "diagnostic": "#a43d32",
    "rolling_validated": "#1f7a4d",
    "rolling_candidate": "#b47f1a",
    "rolling_diagnostic": "#a43d32",
}


@st.cache_data
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_model_name(name: str) -> str:
    return name.replace("player_", "").replace("_", " ")


def layer_status_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for status in ("validated", "borderline", "diagnostic"):
        for layer in report.get(f"{status}_layers", []):
            rows.append(
                {
                    "status": status,
                    "feature_group": layer["feature_group"],
                    "model": layer["model"],
                    "auc": float(layer["auc"]),
                    "features": ", ".join(layer.get("features", [])),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {"validated": 0, "borderline": 1, "diagnostic": 2}
    return frame.sort_values(["status", "auc"], key=lambda col: col.map(order) if col.name == "status" else col)


def robustness_frame(report: dict[str, Any]) -> pd.DataFrame:
    results = report.get("robustness_checks", {}).get("results", {})
    rows = [
        {
            "feature_group": feature_group,
            "model": values["model"],
            "mean_auc": float(values["mean_auc"]),
            "std_auc": float(values["std_auc"]),
            "pass_rate": float(values["pass_rate"]),
        }
        for feature_group, values in results.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_auc") if rows else pd.DataFrame()


def session_frame(session: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for pitch in session.get("pitches", []):
        release = pitch.get("release", {})
        velocity = pitch.get("velocity", {})
        spin = pitch.get("spin", {})
        movement = pitch.get("movement", {})
        plate = pitch.get("plate_target", {})
        rows.append(
            {
                "index": pitch["index"],
                "plate_x": plate.get("x"),
                "plate_z": plate.get("z"),
                "release_speed": velocity.get("release_speed"),
                "spin_rate": spin.get("rate"),
                "pfx_x": movement.get("pfx_x"),
                "pfx_z": movement.get("pfx_z"),
                "release_x": release.get("pos_x"),
                "release_z": release.get("pos_z"),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def load_validation_board(path_value: str) -> tuple[pd.DataFrame, str]:
    board_dir = Path(path_value)
    leaderboard_path = board_dir / "leaderboard.csv"
    markdown_path = board_dir / "validation_board.md"
    if not leaderboard_path.exists():
        return pd.DataFrame(), ""
    markdown = markdown_path.read_text() if markdown_path.exists() else ""
    return pd.read_csv(leaderboard_path), markdown


def _resolve_data_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


@st.cache_data
def load_clean_statcast(path_value: str) -> pd.DataFrame:
    return clean_pitch_features(load_statcast_cache(_resolve_data_path(path_value)), pitch_types=None)


@st.cache_data
def load_real_holdout(path_value: str, pitcher_id: int, pitch_type: str) -> pd.DataFrame:
    """Return the real Statcast holdout pitches the model never trained on."""
    clean = load_clean_statcast(path_value)
    subset = clean[(clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == pitch_type)].copy()
    _, holdout = temporal_train_holdout(subset, train_fraction=0.7)
    keep = [
        "plate_x",
        "plate_z",
        "release_speed",
        "release_spin_rate",
        "pfx_x",
        "pfx_z",
        "release_pos_x",
        "release_pos_z",
    ]
    available = [column for column in keep if column in holdout.columns]
    return holdout[available].copy()


@st.cache_resource
def fit_physics_suite(
    path_value: str,
    pitcher_id: int,
    pitch_type: str,
    pitcher_name: str,
):
    clean = load_clean_statcast(path_value)
    subset = clean[(clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == pitch_type)].copy()
    train, _ = temporal_train_holdout(subset, train_fraction=0.7)
    return fit_generator_suite(
        train,
        clean,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        feature_group="physics_core",
    )


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#444")
    return (
        f"<span style='background:{color};color:white;border-radius:999px;"
        f"padding:.2rem .55rem;font-size:.78rem;font-weight:700;"
        f"letter-spacing:.02em;text-transform:uppercase'>{status}</span>"
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
              linear-gradient(90deg, rgba(31, 122, 77, .08) 1px, transparent 1px),
              linear-gradient(180deg, rgba(31, 122, 77, .08) 1px, transparent 1px),
              #f7f4ec;
            background-size: 28px 28px;
            color: #1c1a17;
        }
        [data-testid="stHeader"] {
            background: rgba(247, 244, 236, .85);
            backdrop-filter: blur(8px);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: #171512;
        }
        h1 {
            font-size: 3.2rem;
            line-height: 1;
            margin-bottom: .25rem;
        }
        section[data-testid="stSidebar"] {
            background: #191815;
            color: #f7f4ec;
        }
        section[data-testid="stSidebar"] * {
            color: #f7f4ec;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, .72);
            border: 1px solid rgba(28, 26, 23, .16);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(31, 27, 20, .07);
        }
        .artifact-note {
            border-left: 5px solid #1f7a4d;
            background: rgba(255, 255, 255, .68);
            padding: .9rem 1rem;
            border-radius: 0 8px 8px 0;
        }
        .truth-line {
            font-size: 1.16rem;
            line-height: 1.55;
            max-width: 980px;
            color: #2a2722;
        }
        .kicker {
            font-size: .78rem;
            letter-spacing: .22em;
            text-transform: uppercase;
            font-weight: 700;
            color: #5b5247;
            margin-bottom: .35rem;
        }
        .score-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            border: 1px solid rgba(28, 26, 23, .10);
            box-shadow: 0 18px 38px rgba(31, 27, 20, .08);
            height: 100%;
            position: relative;
            overflow: hidden;
        }
        .score-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: #1f7a4d;
        }
        .score-card.score-diagnostic::before {
            background: #a43d32;
        }
        .score-card.score-tested::before {
            background: #d7a531;
        }
        .score-label {
            font-size: .76rem;
            letter-spacing: .14em;
            text-transform: uppercase;
            font-weight: 700;
            color: #5b5247;
            margin-bottom: .35rem;
        }
        .score-value {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1;
            color: #171512;
            letter-spacing: -.02em;
        }
        .score-unit {
            font-size: 1.05rem;
            font-weight: 600;
            color: #5b5247;
            margin-left: .25rem;
        }
        .score-meta {
            margin-top: .7rem;
            font-size: .9rem;
            color: #3b362f;
            line-height: 1.45;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .25rem;
            border-bottom: 1px solid rgba(28, 26, 23, .12);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: .55rem 1.05rem;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(31, 122, 77, .12);
            color: #1f7a4d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_layer_table(layers: pd.DataFrame) -> None:
    table = layers.copy()
    table["model"] = table["model"].map(_format_model_name)
    table["auc"] = table["auc"].map(lambda value: f"{value:.3f}")
    st.dataframe(
        table[["status", "feature_group", "auc", "model", "features"]],
        hide_index=True,
        width="stretch",
        column_config={
            "status": "Status",
            "feature_group": "Feature Group",
            "auc": "AUC",
            "model": "Selected Model",
            "features": "Features",
        },
    )


def render_context_controls(label: str, key_prefix: str, defaults: dict[str, int | str]) -> pd.DataFrame:
    st.markdown(f"#### {label}")
    inning = st.slider("Inning", 1, 9, int(defaults["inning"]), key=f"{key_prefix}_inning")
    pitch_count = st.slider(
        "Pitcher game pitch count",
        1,
        115,
        int(defaults["pitch_count"]),
        key=f"{key_prefix}_pitch_count",
    )
    count_cols = st.columns(2)
    balls = count_cols[0].selectbox(
        "Balls",
        [0, 1, 2, 3],
        index=int(defaults["balls"]),
        key=f"{key_prefix}_balls",
    )
    strikes = count_cols[1].selectbox(
        "Strikes",
        [0, 1, 2],
        index=int(defaults["strikes"]),
        key=f"{key_prefix}_strikes",
    )
    batter_hand = st.radio(
        "Batter hand",
        ["R", "L"],
        index=["R", "L"].index(str(defaults["batter_hand"])),
        horizontal=True,
        key=f"{key_prefix}_batter_hand",
    )
    score_diff = st.slider(
        "Pitcher score differential",
        -6,
        6,
        int(defaults["score_diff"]),
        key=f"{key_prefix}_score_diff",
    )
    return make_context_dataframe(
        inning=inning,
        pitcher_game_pitch_count=pitch_count,
        balls=balls,
        strikes=strikes,
        batter_hand=str(batter_hand),
        pitcher_score_diff=score_diff,
        repeat=1,
    )


def render_delta_metrics(comparison: dict[str, Any]) -> None:
    delta = comparison["delta"]
    metric_columns = st.columns(4)
    for index, (column, label) in enumerate(
        [
            ("release_speed", "Velocity"),
            ("release_spin_rate", "Spin"),
            ("pfx_x", "Horizontal movement"),
            ("plate_z", "Plate height"),
        ]
    ):
        value = delta.get(column, {}).get("mean_delta")
        if value is None:
            metric_columns[index].metric(label, "n/a")
        else:
            metric_columns[index].metric(label, f"{value:+.2f}")


def render_validation_board_view() -> None:
    st.subheader("Generalization board")
    available = {
        label: path
        for label, path in VALIDATION_BOARD_OPTIONS.items()
        if (path / "leaderboard.csv").exists()
    }
    if not available:
        st.info(
            "No validation-board artifacts are present yet. Run "
            "`python scripts/run_validation_board.py ...` to generate the leaderboard."
        )
        return

    selected_label = st.selectbox("Board", list(available))
    board_dir = available[selected_label]
    leaderboard, markdown = load_validation_board(str(board_dir))
    if leaderboard.empty:
        st.warning(f"No leaderboard rows found in `{board_dir}`.")
        return

    lead_cols = st.columns(4)
    lead_cols[0].metric("Candidates", f"{len(leaderboard)}")
    lead_cols[1].metric(
        "Validated",
        f"{int((leaderboard['artifact_status'] == 'validated_temporal_success').sum())}",
    )
    lead_cols[2].metric(
        "Candidate",
        f"{int((leaderboard['artifact_status'] == 'physics_core_candidate').sum())}",
    )
    lead_cols[3].metric(
        "Diagnostic",
        f"{int((leaderboard['artifact_status'] == 'physics_core_diagnostic').sum())}",
    )

    fig = px.scatter(
        leaderboard,
        x="holdout_count",
        y="physics_core_mean_auc",
        color="artifact_status",
        size="pitch_count",
        hover_data=["pitcher_name", "pitch_type", "best_physics_core_model"],
        color_discrete_map={
            "validated_temporal_success": STATUS_COLORS["validated"],
            "physics_core_candidate": STATUS_COLORS["borderline"],
            "physics_core_diagnostic": STATUS_COLORS["diagnostic"],
        },
        labels={
            "holdout_count": "Temporal holdout rows",
            "physics_core_mean_auc": "Physics-core C2ST AUC",
            "artifact_status": "Status",
        },
    )
    fig.add_hline(y=TARGET_AUC, line_dash="dash", line_color="#171512")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.75)",
        height=390,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        leaderboard.assign(
            physics_core_mean_auc=leaderboard["physics_core_mean_auc"].map(
                lambda value: f"{value:.3f}"
            ),
            physics_core_pass_rate=leaderboard["physics_core_pass_rate"].map(
                lambda value: f"{value:.2f}"
            ),
        ),
        hide_index=True,
        width="stretch",
    )

    scorecard_dir = board_dir / "scorecards"
    scorecards = sorted(scorecard_dir.glob("*.md")) if scorecard_dir.exists() else []
    if scorecards:
        selected_scorecard = st.selectbox(
            "Scorecard",
            scorecards,
            format_func=lambda path: path.stem.replace("_", " ").title(),
        )
        st.markdown(selected_scorecard.read_text())
    elif markdown:
        st.markdown(markdown)


def render_rolling_scoreboard_view() -> None:
    st.subheader("Primary rolling scoreboard")
    if not ROLLING_BOARD_PATH.exists():
        st.info(
            "No rolling-board artifact is present yet. Run "
            "`python scripts/run_rolling_temporal_board.py ...` to generate it."
        )
        return

    board = load_json(ROLLING_BOARD_PATH)
    scoreboard = board.get("primary_scoreboard") or score_rolling_validation_goals(
        board["consistency"]
    )
    current = scoreboard["current"]
    status = str(scoreboard["status"])

    st.markdown(
        f"""
        <div class="artifact-note">
        <strong>Truth test:</strong> rolling validation is now the main gate.
        Status: {status_badge(status)}
        &nbsp; Goals cleared:
        <strong>{scoreboard['cleared_count']}/{scoreboard['check_count']}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Mean rolling AUC",
        f"{current['mean_rolling_physics_core_auc']:.3f}",
        "goal <= 0.620",
    )
    metric_cols[1].metric(
        "Target hit rate",
        f"{current['target_hit_rate']:.2f}",
        "goal >= 0.40",
    )
    metric_cols[2].metric(
        "Worst fold AUC",
        f"{current['worst_fold_physics_core_auc']:.3f}",
        "goal < 0.800",
    )
    metric_cols[3].metric(
        "Best fold AUC",
        f"{current['best_fold_physics_core_auc']:.3f}",
        "fold context",
    )

    checks = pd.DataFrame(scoreboard["checks"])
    checks["result"] = checks["passed"].map({True: "clear", False: "miss"})
    checks["gap_to_goal"] = checks["gap_to_goal"].map(lambda value: f"{float(value):.3f}")
    st.dataframe(
        checks[["metric", "current_display", "goal", "result", "gap_to_goal"]],
        hide_index=True,
        width="stretch",
        column_config={
            "metric": "Metric",
            "current_display": "Current",
            "goal": "Goal",
            "result": "Result",
            "gap_to_goal": "Gap",
        },
    )

    folds = pd.DataFrame(board["folds"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=folds["fold_index"],
            y=folds["physics_core_mean_auc"],
            mode="lines+markers",
            marker={"size": 11, "color": "#a43d32", "line": {"width": 1, "color": "#171512"}},
            line={"color": "#171512", "width": 2},
            customdata=folds[
                [
                    "train_game_range",
                    "test_game_range",
                    "best_physics_core_model",
                    "physics_core_pass_rate",
                    "failure_count",
                ]
            ],
            hovertemplate=(
                "<b>Fold %{x}</b><br>"
                "Physics AUC=%{y:.3f}<br>"
                "Train games=%{customdata[0]}<br>"
                "Test games=%{customdata[1]}<br>"
                "Model=%{customdata[2]}<br>"
                "Pass rate=%{customdata[3]:.2f}<br>"
                "Failures=%{customdata[4]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=float(board["target_auc"]),
        line_dash="dash",
        line_color="#1f7a4d",
        annotation_text="fold success target",
    )
    fig.add_hline(
        y=0.800,
        line_dash="dot",
        line_color="#a43d32",
        annotation_text="worst-fold ceiling",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.75)",
        xaxis_title="Rolling fold",
        yaxis_title="Physics-core C2ST AUC",
        height=430,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Lower AUC is better. The hit rate is the fraction of rolling folds at or below "
        f"the `{board['target_auc']:.2f}` per-fold target."
    )


REAL_COLOR = "#171512"
SIM_COLOR = "#1f7a4d"
SOURCE_PALETTE = {"Real holdout": REAL_COLOR, "Simulated": SIM_COLOR}


def _combine_real_and_simulated(holdout: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    real_columns = [
        column
        for column in ["plate_x", "plate_z", "release_speed", "release_spin_rate", "pfx_x", "pfx_z"]
        if column in holdout.columns
    ]
    real = holdout[real_columns].copy()
    real["source"] = "Real holdout"

    sim = simulated.copy()
    if "spin_rate" in sim.columns and "release_spin_rate" not in sim.columns:
        sim = sim.rename(columns={"spin_rate": "release_spin_rate"})
    sim_columns = [column for column in real_columns if column in sim.columns]
    sim = sim[sim_columns].copy()
    sim["source"] = "Simulated"

    return pd.concat([real, sim], ignore_index=True)


def _styled_plotly_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.78)",
        height=height,
        margin={"l": 12, "r": 12, "t": 36, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        legend_title_text="",
        font={"color": "#171512"},
    )
    return fig


def render_pitch_cloud_overlay(combined: pd.DataFrame) -> None:
    fig = px.scatter(
        combined,
        x="plate_x",
        y="plate_z",
        color="source",
        color_discrete_map=SOURCE_PALETTE,
        opacity=0.62,
        hover_data={
            "release_speed": ":.1f",
            "release_spin_rate": ":.0f",
            "source": True,
            "plate_x": False,
            "plate_z": False,
        },
        labels={"plate_x": "Plate X (ft)", "plate_z": "Plate Z (ft)"},
        title="Plate location · real vs simulated",
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 0.6, "color": "#171512"}})
    fig.add_shape(
        type="rect",
        x0=-0.83,
        x1=0.83,
        y0=1.5,
        y1=3.5,
        line={"color": "#171512", "width": 2},
    )
    fig.update_xaxes(range=[-2.2, 2.2], zeroline=False)
    fig.update_yaxes(range=[0.4, 4.6], zeroline=False, scaleanchor="x", scaleratio=1)
    _styled_plotly_layout(fig, height=420)
    st.plotly_chart(fig, width="stretch")


def render_marginal_overlay(combined: pd.DataFrame, *, column: str, label: str, unit: str) -> None:
    if column not in combined.columns or combined[column].dropna().empty:
        st.info(f"`{column}` is not available in the held-out sample yet.")
        return
    fig = px.histogram(
        combined,
        x=column,
        color="source",
        color_discrete_map=SOURCE_PALETTE,
        barmode="overlay",
        opacity=0.65,
        nbins=24,
        labels={column: f"{label} ({unit})"},
        title=f"{label} distribution · real vs simulated",
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color="#171512")
    _styled_plotly_layout(fig, height=320)
    st.plotly_chart(fig, width="stretch")


def render_movement_overlay(combined: pd.DataFrame) -> None:
    if "pfx_x" not in combined.columns or "pfx_z" not in combined.columns:
        return
    fig = px.scatter(
        combined,
        x="pfx_x",
        y="pfx_z",
        color="source",
        color_discrete_map=SOURCE_PALETTE,
        opacity=0.62,
        labels={"pfx_x": "Horizontal break (ft)", "pfx_z": "Vertical break (ft)"},
        title="Movement break · real vs simulated",
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 0.5, "color": "#171512"}})
    _styled_plotly_layout(fig, height=380)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Pitcher Twin · live demo",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    report = load_json(REPORT_PATH)
    session = load_json(SESSION_PATH)
    layers = layer_status_frame(report)
    robustness = robustness_frame(report)
    samples = session_frame(session)

    candidate = report["selected_candidate"]
    selected_model = report["selected_model"]
    selected_auc = float(report["selected_auc"])
    physics_auc = float(report["physics_temporal_auc"])
    artifact_status = report["artifact_status"]

    with st.sidebar:
        st.markdown("## Artifacts")
        st.write("`docs/assets/real_demo_morning_report.md`")
        st.write("`docs/assets/real_demo_validation_report.json`")
        st.write("`docs/assets/final_session.json`")
        st.markdown("## Run")
        st.code("streamlit run app/streamlit_app.py", language="bash")

    rolling_summary = None
    if ROLLING_BOARD_PATH.exists():
        rolling_board = load_json(ROLLING_BOARD_PATH)
        rolling_summary = rolling_board.get("primary_scoreboard") or score_rolling_validation_goals(
            rolling_board["consistency"]
        )

    st.markdown(
        '<div class="kicker">Pitcher Twin · real-data hosted demo</div>',
        unsafe_allow_html=True,
    )
    st.title("We model the cloud, not just the pitch.")
    st.markdown(
        """
        <div class="truth-line">
        A real pitcher doesn't throw one fastball &mdash; they throw a cloud of them,
        shaped by count, inning, batter, fatigue, score, and the simple fact that no
        human releases the ball the same way twice. <strong>Pitcher Twin</strong> learns
        that cloud from public Statcast and generates pitches a classifier struggles to
        tell apart from real held-out ones.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    score_cols = st.columns(3)
    with score_cols[0]:
        st.markdown(
            f"""
            <div class="score-card score-validated">
            <div class="score-label">Validated · single 70/30 split</div>
            <div class="score-value">0.533 <span class="score-unit">AUC</span></div>
            <div class="score-meta">100% pass rate &middot; Skubal 2025 FF &middot; <strong>0.50 = coin flip</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with score_cols[1]:
        rolling_value = (
            f"{rolling_summary['current']['mean_rolling_physics_core_auc']:.3f}"
            if rolling_summary
            else f"{physics_auc:.3f}"
        )
        st.markdown(
            f"""
            <div class="score-card score-diagnostic">
            <div class="score-label">Diagnostic · rolling stress test</div>
            <div class="score-value">{rolling_value} <span class="score-unit">mean AUC</span></div>
            <div class="score-meta">10 future-game folds &middot; goal &le; 0.620 &middot; the honest gap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with score_cols[2]:
        st.markdown(
            """
            <div class="score-card score-tested">
            <div class="score-label">Tested across</div>
            <div class="score-value">4 <span class="score-unit">pitchers</span></div>
            <div class="score-meta">3 pitch types &middot; Skubal FF/SI/CH + Mattson, Peralta, Bradley FFs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="artifact-note">
        <strong>Layer status on Skubal FF:</strong>
        &nbsp; {status_badge("validated")} command, movement, trajectory, release, full physics
        &nbsp;&middot;&nbsp; full joint physics still {status_badge("rolling_diagnostic")} on rolling
        validation &mdash; the next milestone.
        </div>
        """,
        unsafe_allow_html=True,
    )

    (
        story_tab,
        try_it_tab,
        rolling_tab,
        board_tab,
        validation_tab,
        session_tab,
        build_tab,
    ) = st.tabs(
        [
            "Demo",
            "Try It",
            "Rolling Truth Test",
            "Cross-Pitcher Board",
            "Layer Validation",
            "Session JSON",
            "Build",
        ]
    )

    with story_tab:
        st.subheader("Real held-out pitches vs. generated samples")
        st.markdown(
            """
            <div class="truth-line">
            Dark dots are <strong>real Skubal fastballs</strong> from the held-out 30% of
            the season the model never saw. Green dots are pitches the model
            <strong>generated</strong> from the earlier 70%. A classifier trained
            specifically to spot the green ones lands at <strong>0.533 AUC</strong>
            &mdash; barely above a coin flip.
            </div>
            """,
            unsafe_allow_html=True,
        )

        data_path_value = str(report.get("data_path", ""))
        data_path = _resolve_data_path(data_path_value) if data_path_value else None
        overlay_combined: pd.DataFrame | None = None
        if data_path is not None and data_path.exists():
            holdout = load_real_holdout(
                data_path_value, int(candidate["pitcher"]), str(candidate["pitch_type"])
            )
            overlay_combined = _combine_real_and_simulated(holdout, samples)

        if overlay_combined is None or overlay_combined.empty:
            st.info(
                "Real-holdout overlay needs `data/processed/skubal_2025.csv` to be present "
                "in the repo. Falling back to the generated-only view."
            )
            cols = st.columns([1.1, 1])
            with cols[0]:
                fig = px.scatter(
                    samples,
                    x="plate_x",
                    y="plate_z",
                    color="release_speed",
                    color_continuous_scale=["#1f7a4d", "#d7a531", "#a43d32"],
                    title="Generated Skubal FF samples",
                )
                fig.update_traces(marker={"size": 11, "line": {"width": 1, "color": "#171512"}})
                fig.add_shape(
                    type="rect",
                    x0=-0.83,
                    x1=0.83,
                    y0=1.5,
                    y1=3.5,
                    line={"color": "#171512", "width": 2},
                )
                _styled_plotly_layout(fig, height=420)
                st.plotly_chart(fig, width="stretch")
            with cols[1]:
                st.markdown(
                    """
                    The session export carries layered validation metadata so consumers know
                    which layers are trusted, candidate, or diagnostic before any pitch leaves
                    the cage.
                    """
                )
        else:
            cols = st.columns([1.1, 1])
            with cols[0]:
                render_pitch_cloud_overlay(overlay_combined)
            with cols[1]:
                render_marginal_overlay(
                    overlay_combined,
                    column="release_speed",
                    label="Release velocity",
                    unit="mph",
                )
                render_marginal_overlay(
                    overlay_combined,
                    column="release_spin_rate",
                    label="Spin rate",
                    unit="rpm",
                )

            with st.expander("Movement break overlay (horizontal vs vertical)"):
                render_movement_overlay(overlay_combined)

        story_cols = st.columns(2)
        with story_cols[0]:
            st.markdown("#### What this tab shows")
            st.markdown(
                """
                The model never saw the held-out games. It samples from the cloud it learned
                from earlier games, conditioned on game state. Visually, the green and dark
                clouds should sit on top of each other &mdash; statistically, they almost do.
                """
            )
        with story_cols[1]:
            st.markdown("#### What's still hard")
            st.markdown(
                """
                Single-split AUC is a ceiling. Under the **Rolling Truth Test** tab, the
                same model is forced to predict pitch-by-pitch into many *future* game
                windows. Mean AUC there is 0.702 &mdash; meaning the release/spin/full-joint
                signature still drifts game-to-game in ways the current model can't yet track.
                """
            )

    with rolling_tab:
        render_rolling_scoreboard_view()

    with board_tab:
        render_validation_board_view()

    with validation_tab:
        st.subheader("Layer results")
        fig = px.bar(
            layers,
            x="auc",
            y="feature_group",
            color="status",
            orientation="h",
            color_discrete_map=STATUS_COLORS,
            hover_data=["model", "features"],
            labels={"auc": "Detectability C2ST AUC", "feature_group": ""},
        )
        fig.add_vline(
            x=TARGET_AUC,
            line_dash="dash",
            line_color="#171512",
            annotation_text="success target",
            annotation_position="top",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.75)",
            height=460,
            margin={"l": 10, "r": 10, "t": 20, "b": 20},
            legend_title_text="",
        )
        st.plotly_chart(fig, width="stretch")
        render_layer_table(layers)

        st.subheader("Repeated-seed robustness")
        robust_fig = go.Figure()
        robust_fig.add_trace(
            go.Bar(
                x=robustness["feature_group"],
                y=robustness["mean_auc"],
                name="Mean AUC",
                marker_color="#1f7a4d",
                error_y={
                    "type": "data",
                    "array": robustness["std_auc"],
                    "visible": True,
                    "color": "#171512",
                },
                customdata=robustness[["pass_rate", "model"]],
                hovertemplate=(
                    "<b>%{x}</b><br>Mean AUC=%{y:.3f}<br>"
                    "Pass rate=%{customdata[0]:.2f}<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
        robust_fig.add_hline(y=TARGET_AUC, line_dash="dash", line_color="#171512")
        robust_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.75)",
            yaxis_title="Mean AUC",
            xaxis_title="",
            height=390,
            margin={"l": 10, "r": 10, "t": 20, "b": 20},
        )
        st.plotly_chart(robust_fig, width="stretch")
        st.dataframe(
            robustness.assign(
                model=robustness["model"].map(_format_model_name),
                mean_auc=robustness["mean_auc"].map(lambda value: f"{value:.3f}"),
                std_auc=robustness["std_auc"].map(lambda value: f"{value:.3f}"),
                pass_rate=robustness["pass_rate"].map(lambda value: f"{value:.2f}"),
            ),
            hide_index=True,
            width="stretch",
        )

    with session_tab:
        st.subheader("Exported pitch session")
        st.dataframe(samples, hide_index=True, width="stretch")
        st.subheader("Trajekt-shaped JSON")
        st.json(session, expanded=False)

    with try_it_tab:
        st.subheader("Try it · pick a context, generate a cloud")
        st.markdown(
            """
            <div class="truth-line">
            Set two different game contexts &mdash; say, a 0-0 first inning vs a 2-2 in the
            7th &mdash; and watch the model's generated pitch cloud shift. Same pitcher,
            different situation, different distribution.
            </div>
            """,
            unsafe_allow_html=True,
        )
        data_path = _resolve_data_path(str(report.get("data_path", "")))
        if not data_path.exists():
            st.info(
                "The tracked JSON artifacts are available, but the real Statcast CSV named "
                f"in the report is not present at `{data_path}`. Rebuild or fetch the real "
                "data to enable live conditional sampling."
            )
        else:
            pitcher_id = int(candidate["pitcher"])
            pitch_type = str(candidate["pitch_type"])
            pitcher_name = str(candidate["pitcher_name"])
            clean = load_clean_statcast(str(report["data_path"]))
            subset = clean[(clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == pitch_type)]
            pitcher_hand = str(subset["p_throws"].dropna().iloc[0]) if "p_throws" in subset.columns else None
            suite = fit_physics_suite(str(report["data_path"]), pitcher_id, pitch_type, pitcher_name)
            sample_count = st.slider("Generated samples per context", 25, 300, 100, step=25)

            context_cols = st.columns(2)
            with context_cols[0]:
                context_a = render_context_controls(
                    "Context A",
                    "context_a",
                    {
                        "inning": 1,
                        "pitch_count": 15,
                        "balls": 0,
                        "strikes": 0,
                        "batter_hand": "R",
                        "score_diff": 0,
                    },
                )
            with context_cols[1]:
                context_b = render_context_controls(
                    "Context B",
                    "context_b",
                    {
                        "inning": 7,
                        "pitch_count": 88,
                        "balls": 2,
                        "strikes": 2,
                        "batter_hand": "L",
                        "score_diff": 1,
                    },
                )

            samples_a, metadata_a = sample_conditional_distribution(
                suite,
                context_a,
                n=sample_count,
                random_state=701,
            )
            samples_b, metadata_b = sample_conditional_distribution(
                suite,
                context_b,
                n=sample_count,
                random_state=1701,
            )
            comparison = compare_context_distributions(
                samples_a,
                samples_b,
                pitcher_hand=pitcher_hand,
            )

            st.markdown("#### Estimated shift from Context A to Context B")
            render_delta_metrics(comparison)
            st.caption(
                "Deltas are model-estimated shifts in generated sample means. They are not "
                "causal claims and do not imply exact next-pitch certainty."
            )

            scatter_cols = st.columns(2)
            for title, frame, container in [
                ("Context A plate cloud", samples_a, scatter_cols[0]),
                ("Context B plate cloud", samples_b, scatter_cols[1]),
            ]:
                fig = px.scatter(
                    frame,
                    x="plate_x",
                    y="plate_z",
                    color="release_speed" if "release_speed" in frame.columns else None,
                    color_continuous_scale=["#1f7a4d", "#d7a531", "#a43d32"],
                    title=title,
                )
                fig.add_shape(
                    type="rect",
                    x0=-0.83,
                    x1=0.83,
                    y0=1.5,
                    y1=3.5,
                    line={"color": "#171512", "width": 2},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,.75)",
                    height=360,
                    margin={"l": 10, "r": 10, "t": 50, "b": 10},
                )
                container.plotly_chart(fig, width="stretch")

            tendency_rows = []
            for label, values in [
                ("Context A", comparison["context_a"]["miss_tendency"]),
                ("Context B", comparison["context_b"]["miss_tendency"]),
            ]:
                tendency_rows.append(
                    {
                        "context": label,
                        "horizontal": values["primary_horizontal"],
                        "vertical": values["primary_vertical"],
                        "zone_rate": f"{values['zone_rate']:.2f}",
                        "chase_rate": f"{values['chase_rate']:.2f}",
                        "spike_risk": f"{values['spike_risk_rate']:.2f}",
                    }
                )
            st.markdown("#### Model-estimated miss tendency")
            st.dataframe(pd.DataFrame(tendency_rows), hide_index=True, width="stretch")

            layer_status = layer_status_from_report(report)
            st.markdown("#### Layer confidence")
            status_rows = [
                {"feature_group": group, "status": status}
                for group, status in sorted(layer_status.items())
            ]
            st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")
            st.caption(
                f"Selected model A: `{metadata_a['selected_model']}`; "
                f"selected model B: `{metadata_b['selected_model']}`. Full physics is generated, "
                "but confidence is reported by validated layer."
            )

    with build_tab:
        st.subheader("Pipeline")
        st.code(
            """
Statcast / Baseball Savant
  -> real-data cleaning and feature groups
  -> candidate ranking
  -> temporal train / holdout split
  -> generator suite
  -> held-out classifier two-sample validation
  -> repeated-seed robustness
  -> Trajekt-shaped JSON export
            """.strip(),
            language="text",
        )
        st.subheader("Selected model")
        st.markdown(
            f"""
            `{selected_model}` estimates a pitcher baseline, computes game-level
            shifts, weights recent games more heavily, adds context effects when
            available, then samples pitch-level residual variation.
            """
        )
        st.subheader("Artifact files")
        st.write(f"`{REPORT_PATH.relative_to(ROOT)}`")
        st.write(f"`{MORNING_REPORT_PATH.relative_to(ROOT)}`")
        st.write(f"`{SESSION_PATH.relative_to(ROOT)}`")


if __name__ == "__main__":
    main()
