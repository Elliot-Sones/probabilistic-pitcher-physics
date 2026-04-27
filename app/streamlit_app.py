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
from pitcher_twin.validator import temporal_train_holdout


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "assets" / "real_demo_validation_report.json"
SESSION_PATH = ROOT / "docs" / "assets" / "final_session.json"
MORNING_REPORT_PATH = ROOT / "docs" / "assets" / "real_demo_morning_report.md"
TARGET_AUC = 0.60

STATUS_COLORS = {
    "validated": "#1f7a4d",
    "borderline": "#b47f1a",
    "diagnostic": "#a43d32",
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


def _resolve_data_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


@st.cache_data
def load_clean_statcast(path_value: str) -> pd.DataFrame:
    return clean_pitch_features(load_statcast_cache(_resolve_data_path(path_value)), pitch_types=None)


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


def main() -> None:
    st.set_page_config(
        page_title="Pitcher Twin Presentation",
        page_icon="PT",
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

    st.caption("Pitcher Twin real-data demo")
    st.title("Replicate the pitcher, not just the pitch.")
    st.markdown(
        f"""
        <div class="truth-line">
        We trained a real-data pitcher variability model on
        <strong>{candidate['pitcher_name']} {candidate['pitch_type']}</strong>,
        then tested generated pitches against later held-out Statcast pitches.
        The component result is validated; full joint physics is still diagnostic.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Real FF pitches", f"{candidate['n']:,}")
    metric_cols[1].metric("Games", f"{candidate['games']}")
    metric_cols[2].metric("Holdout rows", f"{candidate['holdout_n']}")
    metric_cols[3].metric("Selected AUC", f"{selected_auc:.3f}", "target <= 0.60")
    metric_cols[4].metric("Physics-core AUC", f"{physics_auc:.3f}", "diagnostic")

    st.markdown(
        f"""
        <div class="artifact-note">
        <strong>Artifact status:</strong> {artifact_status}
        &nbsp; {status_badge("validated")} command/location and movement
        &nbsp; {status_badge("borderline")} release and trajectory
        &nbsp; {status_badge("diagnostic")} full joint physics
        </div>
        """,
        unsafe_allow_html=True,
    )

    story_tab, validation_tab, session_tab, conditional_tab, build_tab = st.tabs(
        ["Meeting Story", "Validation", "Generated Session", "Conditional Explorer", "Technical Build"]
    )

    with story_tab:
        left, right = st.columns([1.05, 1])
        with left:
            st.subheader("The result")
            st.markdown(
                """
                Static pitcher models work inside random splits, but fail across time.
                The useful result came from modeling recent game drift: the pitcher has
                a baseline, each game shifts that baseline, and pitch-level variation
                lives around that shifted state.

                The clean meeting line:

                > Pitcher Twin validates Skubal's command and movement variability
                > across time, while exposing full-physics consistency as the next
                > target system collaboration problem.
                """
            )
            st.subheader("Why it matters")
            st.markdown(
                """
                This turns pitching-machine-style replication from one target trajectory into a
                validated distribution. Instead of exporting one average fastball, the
                system exports a session of realistic pitch samples with metadata about
                which layers have actually validated.
                """
            )
        with right:
            fig = px.scatter(
                samples,
                x="plate_x",
                y="plate_z",
                color="release_speed",
                color_continuous_scale=["#1f7a4d", "#d7a531", "#a43d32"],
                hover_data=["index", "release_speed", "spin_rate", "pfx_x", "pfx_z"],
                labels={
                    "plate_x": "Plate X",
                    "plate_z": "Plate Z",
                    "release_speed": "Velo",
                },
                title="Generated Skubal FF Command Samples",
            )
            fig.update_traces(marker={"size": 11, "line": {"width": 1, "color": "#171512"}})
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.75)",
                height=430,
                margin={"l": 10, "r": 10, "t": 50, "b": 10},
            )
            fig.add_shape(
                type="rect",
                x0=-0.83,
                x1=0.83,
                y0=1.5,
                y1=3.5,
                line={"color": "#171512", "width": 2},
            )
            st.plotly_chart(fig, width="stretch")

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
        st.subheader("machine-session JSON")
        st.json(session, expanded=False)

    with conditional_tab:
        st.subheader("Side-by-side conditional pitch distribution")
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
  -> machine-session JSON export
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
