"""
CityBrain - Vancouver Pavement Risk Assessment Dashboard
=========================================================
Streamlit dashboard for visualising model predictions,
feature importance, and performance metrics.

Run:  streamlit run app.py
Data: Place exported CSVs in webapp/data/ folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score
import os, pathlib

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="CityBrain - Vancouver Pavement Risk",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / "data"

# ── Color palette ────────────────────────────────────────
BLUE_900 = "#1a365d"
BLUE_700 = "#2b6cb0"
BLUE_600 = "#2c7be5"
BLUE_500 = "#3182ce"
BLUE_400 = "#4299e1"
BLUE_100 = "#ebf8ff"
BLUE_50  = "#f0f7ff"
SLATE_50 = "#f8fafc"
SLATE_600 = "#475569"

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    /* Global background */
    .stApp {
        background: linear-gradient(180deg, #f0f7ff 0%, #f8fafc 100%);
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d0e3f7;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 2px 12px rgba(44,123,229,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a365d !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #2b6cb0, #2c7be5);
        color: white;
        padding: 12px 22px;
        border-radius: 10px;
        margin: 24px 0 14px 0;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 10px rgba(44,123,229,0.15);
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #1a365d 40%, #1e3a5f 100%) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #c9daf0 !important;
    }
    section[data-testid="stSidebar"] h1 {
        color: white !important;
    }
    section[data-testid="stSidebar"] label {
        color: #93b5d6 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #a8c8e8 !important;
        font-weight: 500 !important;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
    }
    /* Main title */
    .main-title {
        color: #1a365d;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 16px;
    }
    /* Cards for content */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.04);
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper: section header ───────────────────────────────
def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all 4 exported CSVs.  Returns dict of DataFrames."""
    files = {
        "main":     DATA_DIR / "citybrain_dashboard_data.csv",
        "shap":     DATA_DIR / "citybrain_shap.csv",
        "versions": DATA_DIR / "citybrain_versions.csv",
        "model_f1": DATA_DIR / "citybrain_model_f1s.csv",
    }
    data = {}
    missing = []
    for key, path in files.items():
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            missing.append(path.name)
    return data, missing

data, missing_files = load_data()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:10px 0 6px 0;'>"
        "<span style='font-size:2.4rem;'>🛣️</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='text-align:center; font-size:1.6rem; margin:0; color:white;'>CityBrain</h1>"
        "<p style='text-align:center; color:#7eb8e0; font-size:0.85rem; margin-top:2px;'>"
        "Vancouver Pavement Risk Assessment</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if missing_files:
        st.warning(f"Missing data files: {', '.join(missing_files)}")
        st.info("Run `export_data.py` in Colab, then place CSVs in `webapp/data/`")

    # ─ Model selector ─
    model_options = {
        "Tuned Ensemble (Best)": "pred_tuned",
        "Stacked Ensemble": "pred_stacked",
        "Fusion Neural Net": "pred_fusion",
        "XGBoost Only": "pred_xgb",
    }
    selected_model_name = st.selectbox(
        "Prediction Model",
        list(model_options.keys()),
        index=0,
        help="Switch between different models to compare predictions on the map",
    )
    pred_col = model_options[selected_model_name]

    # ─ Risk filter ─
    LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
    COLOR_MAP = {"Low": "#38b2ac", "Medium": "#ecc94b", "High": "#e53e3e"}
    risk_filter = st.multiselect(
        "Show Risk Levels",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
    )

    # ─ Neighbourhood filter ─
    if "main" in data and "neighbourhood" in data["main"].columns:
        all_hoods = sorted(data["main"]["neighbourhood"].dropna().unique())
        selected_hoods = st.multiselect(
            "Neighbourhoods",
            all_hoods,
            default=[],
            help="Leave empty to show all",
        )
    else:
        selected_hoods = []

    # ─ Confidence slider ─
    min_confidence = st.slider(
        "Min Prediction Confidence",
        0.0, 1.0, 0.0, 0.01,
        help="Only show predictions where max probability >= this threshold",
    )

    st.divider()
    st.caption("COMP 9130 Final Project")
    st.caption("Savina Cai  |  2026")

# ══════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════

if "main" not in data:
    st.error("Dashboard data not found. Please export from Colab first.")
    st.markdown("""
    ### Setup Instructions
    1. Run the training notebook on Colab (v15)
    2. Run `export_data.py` cell at the end
    3. Download the 4 CSV files
    4. Place them in `webapp/data/` folder
    5. Run `streamlit run app.py`
    """)
    st.stop()

# ── Prepare main DataFrame ───────────────────────────────
df = data["main"].copy()
df["pred_label_active"] = df[pred_col]
df["risk_name"] = df["pred_label_active"].map(LABEL_MAP)
df["true_name"] = df["true_label"].map(LABEL_MAP)
df["max_prob"] = df[["prob_low", "prob_medium", "prob_high"]].max(axis=1)

# Apply filters
mask = df["risk_name"].isin(risk_filter)
if selected_hoods:
    mask &= df["neighbourhood"].isin(selected_hoods)
mask &= df["max_prob"] >= min_confidence
dff = df[mask].copy()

# ── Title row ────────────────────────────────────────────
st.markdown(
    '<div class="main-title">CityBrain — Vancouver Pavement Risk Assessment</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="main-subtitle">Showing <b>{len(dff):,}</b> / {len(df):,} road segments &nbsp;|&nbsp; Model: <b>{selected_model_name}</b></div>',
    unsafe_allow_html=True,
)

# ── KPI Metrics ──────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

n_high = (dff["risk_name"] == "High").sum()
n_med  = (dff["risk_name"] == "Medium").sum()
n_low  = (dff["risk_name"] == "Low").sum()

# Macro F1 on filtered data
if len(dff) > 0 and dff["true_label"].nunique() > 1:
    macro_f1 = f1_score(dff["true_label"], dff["pred_label_active"], average="macro")
else:
    macro_f1 = 0.0

col1.metric("Total Segments", f"{len(dff):,}")
col2.metric("High Risk", f"{n_high:,}", delta=f"{n_high/max(len(dff),1)*100:.1f}%", delta_color="inverse")
col3.metric("Medium Risk", f"{n_med:,}")
col4.metric("Low Risk", f"{n_low:,}")
col5.metric("Macro F1", f"{macro_f1:.4f}")

# ══════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════
tab_map, tab_perf, tab_models, tab_features, tab_history = st.tabs([
    "🗺️ Interactive Map",
    "📊 Performance",
    "🤖 Model Comparison",
    "🔬 Feature Importance",
    "📈 Version History",
])

# ══════════════════════════════════════════════════════════
# TAB 1: Interactive Map
# ══════════════════════════════════════════════════════════
with tab_map:
    section_header("Prediction Map — Vancouver Road Segments")

    # Build hover text
    dff["hover_text"] = (
        "<b>Risk: " + dff["risk_name"] + "</b> (True: " + dff["true_name"] + ")<br>"
        + "Confidence: " + (dff["max_prob"] * 100).round(1).astype(str) + "%<br>"
        + "───────────────<br>"
        + "Traffic Load: " + dff["traffic_load"].round(1).astype(str) + "<br>"
        + "Pavement Age: " + dff["est_pavement_age"].round(1).astype(str) + " yrs<br>"
        + "Length: " + dff["length_m"].round(0).astype(str) + " m<br>"
        + "Neighbourhood: " + dff["neighbourhood"].fillna("N/A").astype(str)
    )

    fig_map = go.Figure()

    # Plot each risk level
    for risk in ["Low", "Medium", "High"]:
        subset = dff[dff["risk_name"] == risk]
        if len(subset) == 0:
            continue
        size = 4 if risk != "High" else 7
        fig_map.add_trace(go.Scattermapbox(
            lat=subset["lat"],
            lon=subset["lon"],
            mode="markers",
            marker=dict(size=size, color=COLOR_MAP[risk], opacity=0.75),
            name=f"{risk} Risk ({len(subset):,})",
            text=subset["hover_text"],
            hovertemplate="%{text}<extra></extra>",
        ))

    # High-risk glow layer
    high_df = dff[dff["risk_name"] == "High"]
    if len(high_df) > 0:
        fig_map.add_trace(go.Scattermapbox(
            lat=high_df["lat"],
            lon=high_df["lon"],
            mode="markers",
            marker=dict(size=20, color="rgba(229, 62, 62, 0.2)", opacity=0.35),
            name="High Risk (glow)",
            hoverinfo="skip",
            hovertemplate=None,
            showlegend=False,
        ))

    fig_map.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=49.255, lon=-123.12),
            zoom=11.3,
        ),
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d0e3f7",
            borderwidth=1,
            font=dict(color="#1a365d", size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Misclassification heatmap toggle
    show_errors = st.checkbox("Highlight misclassifications only", value=False)
    if show_errors:
        errors = dff[dff["pred_label_active"] != dff["true_label"]].copy()
        errors["error_type"] = (
            "Pred: " + errors["risk_name"] + " / True: " + errors["true_name"]
        )
        fig_err = px.scatter_mapbox(
            errors, lat="lat", lon="lon",
            color="error_type",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=["neighbourhood", "traffic_load", "est_pavement_age"],
            zoom=11.3, height=500,
        )
        fig_err.update_layout(
            mapbox_style="carto-positron",
            mapbox_center=dict(lat=49.255, lon=-123.12),
            margin=dict(l=0, r=0, t=30, b=0),
            title="Misclassified Segments",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_err, use_container_width=True)
        st.caption(f"Total misclassifications: **{len(errors):,}** ({len(errors)/max(len(dff),1)*100:.1f}%)")

# ══════════════════════════════════════════════════════════
# TAB 2: Performance
# ══════════════════════════════════════════════════════════
with tab_perf:
    section_header("Classification Performance")

    perf_col1, perf_col2 = st.columns(2)

    # ── Confusion Matrix ──
    with perf_col1:
        st.markdown("#### Confusion Matrix")
        labels = [0, 1, 2]
        cm = confusion_matrix(df["true_label"], df["pred_label_active"], labels=labels)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        # Annotation text
        annot = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)")
            annot.append(row)

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_pct,
            x=["Pred Low", "Pred Medium", "Pred High"],
            y=["True Low", "True Medium", "True High"],
            text=annot,
            texttemplate="%{text}",
            colorscale=[[0, "#ebf8ff"], [0.35, "#90cdf4"], [0.65, "#3182ce"], [1, "#1a365d"]],
            showscale=False,
            textfont=dict(size=13),
        ))
        fig_cm.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
            font=dict(size=13, color="#1a365d"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Per-Class F1 ──
    with perf_col2:
        st.markdown("#### Per-Class F1 Score")
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, sup = precision_recall_fscore_support(
            df["true_label"], df["pred_label_active"], labels=[0, 1, 2]
        )
        class_df = pd.DataFrame({
            "Class": ["Low", "Medium", "High"],
            "Precision": p, "Recall": r, "F1": f1, "Support": sup,
        })

        fig_f1 = go.Figure()
        fig_f1.add_trace(go.Bar(
            x=class_df["Class"], y=class_df["F1"],
            marker_color=[COLOR_MAP[c] for c in class_df["Class"]],
            text=class_df["F1"].round(4),
            textposition="outside",
            name="F1",
        ))
        fig_f1.add_trace(go.Bar(
            x=class_df["Class"], y=class_df["Precision"],
            marker_color=["rgba(56,178,172,0.45)", "rgba(236,201,75,0.45)", "rgba(229,62,62,0.45)"],
            text=class_df["Precision"].round(4),
            textposition="outside",
            name="Precision",
        ))
        fig_f1.add_trace(go.Bar(
            x=class_df["Class"], y=class_df["Recall"],
            marker_color=["rgba(56,178,172,0.25)", "rgba(236,201,75,0.25)", "rgba(229,62,62,0.25)"],
            text=class_df["Recall"].round(4),
            textposition="outside",
            name="Recall",
        ))
        fig_f1.update_layout(
            barmode="group",
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", y=1.12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # ── Classification Report Table ──
    st.markdown("#### Detailed Metrics")
    report_df = class_df.copy()
    report_df["Precision"] = report_df["Precision"].map("{:.4f}".format)
    report_df["Recall"]    = report_df["Recall"].map("{:.4f}".format)
    report_df["F1"]        = report_df["F1"].map("{:.4f}".format)
    report_df["Support"]   = report_df["Support"].astype(int)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # ── Probability Distribution ──
    section_header("Prediction Confidence Distribution")
    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.markdown("#### Max Probability by Predicted Class")
        fig_prob = px.histogram(
            df, x="max_prob", color="risk_name",
            color_discrete_map=COLOR_MAP,
            nbins=50, barmode="overlay", opacity=0.7,
            labels={"max_prob": "Max Probability", "risk_name": "Predicted Risk"},
        )
        fig_prob.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prob, use_container_width=True)

    with prob_col2:
        st.markdown("#### Correct vs Incorrect Predictions")
        df["correct"] = df["pred_label_active"] == df["true_label"]
        fig_correct = px.histogram(
            df, x="max_prob", color="correct",
            color_discrete_map={True: "#38b2ac", False: "#e53e3e"},
            nbins=50, barmode="overlay", opacity=0.7,
            labels={"max_prob": "Max Probability", "correct": "Correct?"},
        )
        fig_correct.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_correct, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3: Model Comparison
# ══════════════════════════════════════════════════════════
with tab_models:
    section_header("Model Comparison")

    # ── Per-model F1 bar chart ──
    if "model_f1" in data:
        mf = data["model_f1"].copy()
        mf = mf.sort_values("f1", ascending=True)

        fig_mf = go.Figure(go.Bar(
            x=mf["f1"], y=mf["model"],
            orientation="h",
            marker_color=[
                "#2c7be5" if f == mf["f1"].max() else "#90cdf4" for f in mf["f1"]
            ],
            text=mf["f1"].round(4),
            textposition="outside",
            textfont=dict(color="#1a365d"),
        ))
        fig_mf.update_layout(
            height=400,
            margin=dict(l=10, r=80, t=30, b=10),
            xaxis=dict(range=[0, max(mf["f1"].max() * 1.15, 0.6)], title="Macro F1"),
            yaxis=dict(title=""),
            title=dict(text="Individual Model F1 Scores", font=dict(color="#1a365d")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mf, use_container_width=True)
    else:
        st.info("Model F1 data not available. Export `citybrain_model_f1s.csv`.")

    # ── Side-by-side confusion matrices for different models ──
    st.markdown("#### Compare Predictions Across Models")
    model_cols = {
        "Tuned Ensemble": "pred_tuned",
        "Stacked Ensemble": "pred_stacked",
        "Fusion NN": "pred_fusion",
        "XGBoost": "pred_xgb",
    }

    # Check which model columns exist
    available_models = {k: v for k, v in model_cols.items() if v in df.columns}

    if len(available_models) >= 2:
        compare_cols = st.columns(len(available_models))
        for idx, (mname, mcol) in enumerate(available_models.items()):
            with compare_cols[idx]:
                st.markdown(f"**{mname}**")
                cm_m = confusion_matrix(df["true_label"], df[mcol], labels=[0, 1, 2])
                cm_m_pct = cm_m / cm_m.sum(axis=1, keepdims=True) * 100
                mf1 = f1_score(df["true_label"], df[mcol], average="macro")
                st.caption(f"Macro F1 = {mf1:.4f}")

                fig_cm_m = go.Figure(data=go.Heatmap(
                    z=cm_m_pct,
                    x=["L", "M", "H"], y=["L", "M", "H"],
                    colorscale=[[0, "#ebf8ff"], [0.35, "#90cdf4"], [0.65, "#3182ce"], [1, "#1a365d"]],
                    showscale=False,
                    text=[[f"{cm_m[i][j]}" for j in range(3)] for i in range(3)],
                    texttemplate="%{text}",
                ))
                fig_cm_m.update_layout(
                    height=250, margin=dict(l=5, r=5, t=5, b=5),
                    xaxis=dict(side="bottom", title="Pred"),
                    yaxis=dict(autorange="reversed", title="True"),
                    font=dict(size=11, color="#1a365d"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cm_m, use_container_width=True)

    # ── Agreement analysis ──
    if len(available_models) >= 2:
        section_header("Model Agreement")
        model_preds = pd.DataFrame({k: df[v] for k, v in available_models.items()})
        df["n_agree"] = model_preds.apply(lambda row: row.value_counts().max(), axis=1)
        df["all_agree"] = df["n_agree"] == len(available_models)

        agree_col1, agree_col2 = st.columns(2)
        with agree_col1:
            agree_pct = df["all_agree"].mean() * 100
            st.metric("All Models Agree", f"{agree_pct:.1f}%")
            st.caption(f"{df['all_agree'].sum():,} / {len(df):,} segments")

        with agree_col2:
            # Agreement by true class
            agree_by_class = df.groupby("true_name")["all_agree"].mean() * 100
            fig_agree = go.Figure(go.Bar(
                x=agree_by_class.index,
                y=agree_by_class.values,
                marker_color=[COLOR_MAP.get(c, "#3182ce") for c in agree_by_class.index],
                text=[f"{v:.1f}%" for v in agree_by_class.values],
                textposition="outside",
            ))
            fig_agree.update_layout(
                height=300, margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Agreement Rate by True Class", font=dict(color="#1a365d")),
                yaxis=dict(range=[0, 100], title="%"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_agree, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 4: Feature Importance
# ══════════════════════════════════════════════════════════
with tab_features:
    section_header("SHAP Feature Importance (XGBoost)")

    if "shap" in data:
        shap_df = data["shap"].sort_values("importance", ascending=True)

        fig_shap = go.Figure(go.Bar(
            x=shap_df["importance"],
            y=shap_df["feature"],
            orientation="h",
            marker=dict(
                color=shap_df["importance"],
                colorscale=[[0, "#bee3f8"], [0.4, "#4299e1"], [0.7, "#2b6cb0"], [1, "#1a365d"]],
            ),
            text=shap_df["importance"].round(4),
            textposition="outside",
            textfont=dict(color="#1a365d"),
        ))
        fig_shap.update_layout(
            height=500,
            margin=dict(l=10, r=80, t=30, b=10),
            xaxis=dict(title="Mean |SHAP value|"),
            yaxis=dict(title=""),
            title=dict(text="Top 15 Features by SHAP Importance", font=dict(color="#1a365d")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Insight callout
        st.info("""
        **Key Insight:** The top 6 features are ALL **spatial lag** variables (`sl_risk_*`, `sl_high_*`).
        This confirms that **pavement degradation is spatially clustered** — neighbouring roads
        share similar construction history, underground infrastructure, and traffic patterns.
        This aligns with the physical mechanism of urban road deterioration.
        """)
    else:
        st.info("SHAP data not available. Export `citybrain_shap.csv`.")

    # ── Feature distributions ──
    section_header("Feature Distributions by Risk Level")
    numeric_features = [
        "traffic_load", "est_pavement_age", "length_m", "water_main_avg_age",
        "tree_count_30m", "sewer_combined_pct", "slope_pct", "elevation_m",
        "complaint_total", "utility_density", "drainage_risk", "ROW_width",
    ]
    available_features = [f for f in numeric_features if f in df.columns]

    if available_features:
        selected_feat = st.selectbox("Select Feature", available_features)
        fig_dist = px.histogram(
            df, x=selected_feat, color="risk_name",
            color_discrete_map=COLOR_MAP,
            barmode="overlay", opacity=0.65, nbins=40,
            labels={"risk_name": "Risk Level"},
        )
        fig_dist.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Box plot
        fig_box = px.box(
            df, x="risk_name", y=selected_feat, color="risk_name",
            color_discrete_map=COLOR_MAP,
            category_orders={"risk_name": ["Low", "Medium", "High"]},
        )
        fig_box.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 5: Version History
# ══════════════════════════════════════════════════════════
with tab_history:
    section_header("Model Evolution: v1 to v15")

    if "versions" in data:
        ver = data["versions"].copy()

        fig_ver = go.Figure()

        # Line + markers
        fig_ver.add_trace(go.Scatter(
            x=ver["version"], y=ver["f1"],
            mode="lines+markers+text",
            line=dict(color="#2c7be5", width=3),
            marker=dict(size=10, color="#2c7be5",
                        line=dict(width=2, color="white")),
            text=ver["f1"].round(4),
            textposition="top center",
            textfont=dict(size=10, color="#1a365d"),
            hovertext=ver["key_change"],
            hovertemplate="<b>%{x}</b><br>F1: %{y:.4f}<br>%{hovertext}<extra></extra>",
        ))

        # Shade improvement phases
        fig_ver.add_vrect(x0=-0.5, x1=3.5,
                          fillcolor="rgba(66,153,225,0.08)", line_width=0,
                          annotation_text="Architecture Exploration",
                          annotation_position="top left",
                          annotation_font=dict(size=10, color="#4a5568"))
        fig_ver.add_vrect(x0=3.5, x1=8.5,
                          fillcolor="rgba(56,178,172,0.08)", line_width=0,
                          annotation_text="Feature Engineering",
                          annotation_position="top left",
                          annotation_font=dict(size=10, color="#4a5568"))
        fig_ver.add_vrect(x0=8.5, x1=13.5,
                          fillcolor="rgba(43,108,176,0.08)", line_width=0,
                          annotation_text="Ensemble & Tuning",
                          annotation_position="top left",
                          annotation_font=dict(size=10, color="#4a5568"))

        fig_ver.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(title="Macro F1 Score", range=[0.35, 0.60]),
            xaxis=dict(title="Model Version"),
            title=dict(text="39% Improvement: v1 (0.3915) to v15", font=dict(color="#1a365d")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ver, use_container_width=True)

        # Version table
        st.markdown("#### Change Log")
        st.dataframe(
            ver[["version", "f1", "key_change"]].rename(columns={
                "version": "Version", "f1": "Macro F1", "key_change": "Key Change"
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Key milestones
        st.markdown("#### Key Milestones")
        mile_col1, mile_col2, mile_col3 = st.columns(3)
        mile_col1.success("**v5**: 3-class rebalancing (+10%)")
        mile_col2.success("**v11**: 18 infrastructure features (+1.2%)")
        mile_col3.success("**v13**: 10-Fold Stacking + Optuna HPO (+2.5%)")

    else:
        st.info("Version history not available. Export `citybrain_versions.csv`.")

    # ── Architecture Overview ──
    section_header("Model Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                    CityBrain v15 Architecture                │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │   Road Features (12d)      Tabular Features (44d)           │
    │         │                          │                         │
    │    ┌────▼────┐              ┌──────▼──────┐                 │
    │    │Road-MLP │              │ Tabular-MLP │                 │
    │    │ 128→64  │              │  256→128    │                 │
    │    └────┬────┘              └──────┬──────┘                 │
    │         │                          │                         │
    │         └──────────┬───────────────┘                         │
    │              ┌─────▼──────┐                                  │
    │              │CrossAttention│                                │
    │              │   Fusion    │                                  │
    │              └─────┬──────┘                                  │
    │                    │                                          │
    │    ┌───────────────┼───────────────────────┐                │
    │    │               │                       │                 │
    │  Fusion    XGBoost+CatBoost    LightGBM+ExtraTrees          │
    │    │          +LightGBM            │                         │
    │    │               │               │                         │
    │    └───────┬───────┴───────────────┘                         │
    │      ┌─────▼──────┐                                          │
    │      │  10-Fold   │                                          │
    │      │  Stacking  │                                          │
    │      │ (XGB meta) │                                          │
    │      └─────┬──────┘                                          │
    │            │                                                  │
    │      ┌─────▼──────┐                                          │
    │      │ Threshold  │                                          │
    │      │  Tuning    │                                          │
    │      └─────┬──────┘                                          │
    │            ▼                                                  │
    │     Final Prediction                                         │
    │     (Low / Medium / High)                                    │
    └──────────────────────────────────────────────────────────────┘
    ```
    """)

# ── Footer ───────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.82rem; padding:8px 0;'>"
    "CityBrain &mdash; Vancouver Pavement Risk Assessment &nbsp;|&nbsp; COMP 9130 Final Project &nbsp;|&nbsp; "
    "Savina Cai &nbsp;|&nbsp; Powered by Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
