
"""
Stress Trajectory Prediction - Mental Health Analysis Platform
Polished app with organized visualizations, correct dataset link,
and notebook-aligned backend.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Stress Trajectory Prediction | Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET_URL = "https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset"

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "Mental Health Dataset.csv"

# ---------------------------------------------------
# STYLING
# ---------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab3 100%);
        padding: 2.4rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 35px rgba(0,0,0,0.14);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    .main-header p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        text-align: center;
        margin-top: 0.55rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.86rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 0.45rem;
    }
    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        padding: 0.95rem 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #2d5a87;
        margin: 2rem 0 1.2rem 0;
    }
    .section-header h2 {
        color: #1e3a5f;
        font-weight: 600;
        margin: 0;
        font-size: 1.45rem;
    }
    .info-card {
        background: linear-gradient(145deg, #e8f4fd 0%, #d4e8f7 100%);
        border-radius: 12px;
        padding: 1.15rem;
        border-left: 4px solid #2d5a87;
        margin: 1rem 0;
        color: #1e3a5f;
    }
    .warn-card {
        background: linear-gradient(145deg, #fff5f5 0%, #ffe8e8 100%);
        border-radius: 12px;
        padding: 1.15rem;
        border-left: 4px solid #e53e3e;
        margin: 1rem 0;
        color: #c53030;
    }
    .success-card {
        background: linear-gradient(145deg, #f0fff4 0%, #dcffe4 100%);
        border-radius: 12px;
        padding: 1.15rem;
        border-left: 4px solid #38a169;
        margin: 1rem 0;
        color: #276749;
    }
    .prediction-at-risk {
        background: linear-gradient(145deg, #fff5f5 0%, #ffe8e8 100%);
        border: 2px solid #e53e3e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-not-at-risk {
        background: linear-gradient(145deg, #f0fff4 0%, #dcffe4 100%);
        border: 2px solid #38a169;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def metric_card(value, label, color="#1e3a5f"):
    st.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:{color};">{value}</div><div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True
    )

def get_encoder_classes(encoders, col_name, default_values):
    if isinstance(encoders, dict) and col_name in encoders:
        enc = encoders[col_name]
        if hasattr(enc, "classes_"):
            return list(enc.classes_)
        if isinstance(enc, dict):
            return list(enc.keys())
    return list(default_values)

def safe_encode_value(encoder, value):
    try:
        if hasattr(encoder, "classes_") and hasattr(encoder, "transform"):
            if value in set(encoder.classes_):
                return encoder.transform([value])[0]
            return np.nan
        if isinstance(encoder, dict):
            return encoder.get(value, np.nan)
    except Exception:
        return np.nan
    return np.nan

def build_model_input(input_data, feature_columns, encoders):
    df_in = pd.DataFrame([{col: input_data.get(col, np.nan) for col in feature_columns}])
    for col in feature_columns:
        if isinstance(encoders, dict) and col in encoders:
            df_in[col] = df_in[col].apply(lambda v: safe_encode_value(encoders[col], v))
    return df_in.fillna(0)[feature_columns]

def pred_to_label(pred, class_labels):
    try:
        return class_labels[int(pred)]
    except Exception:
        return "Unknown"

def prepare_eda_df(df):
    temp = df.copy()
    temp["Stress_Risk"] = temp["Growing_Stress"].apply(
        lambda x: "At Risk" if x == "Yes" else ("Not At Risk" if x == "No" else "Maybe")
    )
    temp = temp[temp["Stress_Risk"] != "Maybe"].copy()
    return temp

def plot_feature_risk(df, feature, section_title, insight, key_prefix, category_order=None):
    st.markdown(f'<div class="section-header"><h2>{section_title}</h2></div>', unsafe_allow_html=True)
    ctab = pd.crosstab(df[feature], df["Stress_Risk"], normalize="index") * 100
    if category_order is not None:
        ctab = ctab.reindex([c for c in category_order if c in ctab.index])

    left, right = st.columns([2.1, 1])

    with left:
        viz_style = st.radio(
            "Select Chart Style:",
            ["Grouped Bars", "Stacked Bars", "Donut Charts"],
            horizontal=True,
            key=f"{key_prefix}_style"
        )

        if viz_style == "Grouped Bars":
            fig = go.Figure()
            if "Not At Risk" in ctab.columns:
                fig.add_trace(go.Bar(
                    name="Not At Risk",
                    x=ctab.index.astype(str),
                    y=ctab["Not At Risk"],
                    marker_color="#38a169",
                    text=[f"{v:.1f}%" for v in ctab["Not At Risk"]],
                    textposition="outside"
                ))
            if "At Risk" in ctab.columns:
                fig.add_trace(go.Bar(
                    name="At Risk",
                    x=ctab.index.astype(str),
                    y=ctab["At Risk"],
                    marker_color="#e53e3e",
                    text=[f"{v:.1f}%" for v in ctab["At Risk"]],
                    textposition="outside"
                ))
            fig.update_layout(
                barmode="group",
                height=410,
                xaxis_title=feature.replace("_", " "),
                yaxis_title="Percentage %",
                title=dict(text=f"<b>{section_title}</b>", x=0.5, font_size=16),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_style == "Stacked Bars":
            fig = go.Figure()
            if "Not At Risk" in ctab.columns:
                fig.add_trace(go.Bar(
                    name="Not At Risk",
                    x=ctab.index.astype(str),
                    y=ctab["Not At Risk"],
                    marker_color="#38a169",
                    text=[f"{v:.1f}%" for v in ctab["Not At Risk"]],
                    textposition="inside"
                ))
            if "At Risk" in ctab.columns:
                fig.add_trace(go.Bar(
                    name="At Risk",
                    x=ctab.index.astype(str),
                    y=ctab["At Risk"],
                    marker_color="#e53e3e",
                    text=[f"{v:.1f}%" for v in ctab["At Risk"]],
                    textposition="inside"
                ))
            fig.update_layout(
                barmode="stack",
                height=410,
                xaxis_title=feature.replace("_", " "),
                yaxis_title="Percentage %",
                title=dict(text=f"<b>{section_title} (Stacked)</b>", x=0.5, font_size=16)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            ordered_categories = list(ctab.index)
            cols = st.columns(max(1, len(ordered_categories)))
            for i, cat in enumerate(ordered_categories):
                with cols[i]:
                    values = [
                        ctab.loc[cat, "At Risk"] if "At Risk" in ctab.columns else 0,
                        ctab.loc[cat, "Not At Risk"] if "Not At Risk" in ctab.columns else 0
                    ]
                    fig = go.Figure(data=[go.Pie(
                        labels=["At Risk", "Not At Risk"],
                        values=values,
                        hole=0.6,
                        marker_colors=["#e53e3e", "#38a169"],
                        textinfo="percent",
                        textfont_size=12
                    )])
                    fig.add_annotation(text=f"<b>{cat}</b>", x=0.5, y=0.5, font_size=12, showarrow=False)
                    fig.update_layout(height=260, showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 📊 Quick Stats")
        ordered_categories = list(ctab.index)
        for cat in ordered_categories:
            at_risk_pct = ctab.loc[cat, "At Risk"] if "At Risk" in ctab.columns else 0
            color = "#e53e3e" if at_risk_pct > 70 else "#f6ad55" if at_risk_pct > 60 else "#38a169"
            metric_card(f"{at_risk_pct:.1f}%", f"{feature.replace('_', ' ')}: {cat}", color=color)

        st.markdown("### 📈 Sample Counts")
        for cat in ordered_categories:
            count = (df[feature] == cat).sum()
            st.markdown(f"**{cat}:** {count:,} ({count/len(df)*100:.1f}%)")

    st.markdown(
        f'<div class="info-card"><strong>Insight:</strong> {insight}</div>',
        unsafe_allow_html=True
    )

def show_prediction_card(title, label):
    risk = label == "At Risk"
    css = "prediction-at-risk" if risk else "prediction-not-at-risk"
    icon = "⚠️" if risk else "✅"
    color = "#e53e3e" if risk else "#38a169"
    st.markdown(
        f'<div class="{css}"><h3 style="color:{color};">{title}</h3>'
        f'<div style="font-size:3.5rem;">{icon}</div>'
        f'<h1 style="color:{color};">{label}</h1></div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# LOADERS
# ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None

@st.cache_resource
def load_models():
    try:
        dt_model = joblib.load(MODELS_DIR / "decision_tree_model.pkl")
        nb_model = joblib.load(MODELS_DIR / "naive_bayes_model.pkl")
        dt_encoders = joblib.load(MODELS_DIR / "encoders.pkl")
        dt_feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
        nb_encoders = joblib.load(MODELS_DIR / "nb_encoders.pkl")
        nb_feature_columns = joblib.load(MODELS_DIR / "nb_feature_columns.pkl")
        results = joblib.load(MODELS_DIR / "model_results.pkl")
        return dt_model, nb_model, dt_encoders, dt_feature_columns, nb_encoders, nb_feature_columns, results
    except Exception:
        return None, None, None, None, None, None, None

df = load_data()
dt_model, nb_model, dt_encoders, dt_feature_columns, nb_encoders, nb_feature_columns, model_results = load_models()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h1 style="font-size:2.5rem; margin:0;">🧠</h1>
        <h3 style="color:#1e3a5f; margin:0.5rem 0;">Mental Health</h3>
        <p style="color:#6c757d; font-size:0.85rem; margin:0;">Stress Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Problem Statement",
            "Dataset Overview",
            "EDA & Insights",
            "Data Cleaning & Preprocessing",
            "Model Training & Tuning",
            "Model Performance & Comparison",
            "Final Model Selection",
            "Prediction Demo"
        ],
        label_visibility="collapsed"
    )

# ---------------------------------------------------
# PAGE: Problem
# ---------------------------------------------------
if page == "Problem Statement":
    st.markdown("""
    <div class="main-header">
        <h1>Stress Trajectory Prediction</h1>
        <p>A Machine Learning Approach to Mental Health Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Dataset + Objective + Target Variable</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>Dataset:</strong> Mental Health Dataset (Kaggle)<br><br>
            <strong>Objective:</strong> Predict whether an individual is at risk of growing stress<br><br>
            <strong>Problem Type:</strong> Binary Classification<br><br>
            <strong>Target Variable:</strong> <code>Stress_Risk</code><br>
            • At Risk<br>
            • Not At Risk
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Open Dataset on Kaggle", DATASET_URL)

    with col2:
        st.markdown("""
        <div class="warn-card">
            <strong>Why Recall matters most</strong><br><br>
            Missing someone who may need support can be more harmful than an extra check-in.
            <br><br>
            <strong>False Negative:</strong> an at-risk person is missed<br>
            <strong>False Positive:</strong> an extra follow-up is triggered
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Binary", "Problem Type")
    with c2: metric_card("2", "ML Algorithms")
    with c3: metric_card("Recall", "Primary Metric")
    with c4: metric_card("8", "Selected Features")

# ---------------------------------------------------
# PAGE: Dataset Overview
# ---------------------------------------------------
elif page == "Dataset Overview":
    st.markdown("""
    <div class="main-header">
        <h1>Dataset Overview</h1>
        <p>Dataset size, selected features, and preview of the input data</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found. Place 'Mental Health Dataset.csv' next to app.py.")
    else:
        selected_features = dt_feature_columns if dt_feature_columns is not None else []
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card(f"{len(df):,}", "Raw Records")
        with c2: metric_card(f"{len(df.columns)}", "Columns")
        with c3: metric_card(f"{len(selected_features)}", "Selected Features")
        with c4: metric_card("Kaggle", "Dataset Source")

        st.markdown('<div class="section-header"><h2>Selected Features Used in Training</h2></div>', unsafe_allow_html=True)
        if selected_features:
            st.dataframe(pd.DataFrame({"Selected Features": selected_features}), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header"><h2>Sample Data Preview</h2></div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        eda_df = prepare_eda_df(df)
        counts = eda_df["Stress_Risk"].value_counts()

        st.markdown('<div class="section-header"><h2>Target Distribution After Binary Conversion</h2></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            text=counts.values,
            textposition="outside",
            marker_color=["#e53e3e", "#38a169"] if len(counts.index) == 2 else None
        ))
        fig.update_layout(height=380, title=dict(text="<b>Stress_Risk Class Distribution</b>", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="info-card"><strong>Insight:</strong> The target is converted into a clean binary classification problem by removing the ambiguous "Maybe" rows.</div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------
# PAGE: EDA
# ---------------------------------------------------
elif page == "EDA & Insights":
    st.markdown("""
    <div class="main-header">
        <h1>Exploratory Data Analysis</h1>
        <p>Organized visualizations with selectable chart styles and written insights</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found.")
    else:
        eda_df = prepare_eda_df(df)

        st.markdown('<div class="section-header"><h2>Stress Risk Distribution</h2></div>', unsafe_allow_html=True)
        counts = eda_df["Stress_Risk"].value_counts()
        c1, c2 = st.columns([1.2, 1.6])
        with c1:
            fig = go.Figure(data=[go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.6,
                marker_colors=["#e53e3e", "#38a169"],
                textinfo="percent+label"
            )])
            fig.update_layout(height=350, title=dict(text="<b>Class Balance</b>", x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("""
            <div class="info-card">
                <strong>Insight:</strong><br><br>
                After removing "Maybe", the problem becomes a clear binary classification task.
                This makes the target easier to interpret and keeps the notebook and app consistent.
            </div>
            """, unsafe_allow_html=True)

        plot_feature_risk(
            eda_df,
            "Mood_Swings",
            "Mood Swings Analysis",
            'Higher mood swing levels tend to show a larger share of "At Risk" individuals, making this feature one of the strongest indicators in the project.',
            "mood",
            category_order=["Low", "Medium", "High"]
        )

        plot_feature_risk(
            eda_df,
            "Days_Indoors",
            "Days Indoors Analysis",
            'Longer periods indoors appear associated with a higher proportion of "At Risk" cases, which may reflect the effects of isolation and reduced routine.',
            "indoors",
            category_order=["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"]
        )

        plot_feature_risk(
            eda_df,
            "Social_Weakness",
            "Social Weakness Analysis",
            'Weaker social support patterns are associated with higher stress risk, reinforcing the importance of social connection in mental health.',
            "social",
            category_order=["No", "Maybe", "Yes"]
        )

        plot_feature_risk(
            eda_df,
            "Occupation",
            "Occupation Analysis",
            'Different occupation groups show different stress-risk patterns, suggesting that work or study context may influence mental health outcomes.',
            "occupation"
        )

        st.markdown('<div class="section-header"><h2>Key EDA Findings</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <ul style="margin-bottom:0;">
                <li>Mood swings show a strong association with stress risk.</li>
                <li>More time indoors appears linked with higher risk.</li>
                <li>Social weakness stands out as an important mental health factor.</li>
                <li>Occupation groups have different risk patterns.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE: PREPROCESSING
# ---------------------------------------------------
elif page == "Data Cleaning & Preprocessing":
    st.markdown("""
    <div class="main-header">
        <h1>Data Cleaning & Preprocessing</h1>
        <p>Simple, clear summary of the pipeline used before training</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Cleaning</h4>
            <ul style="color:#4b5563;">
                <li>Removed Timestamp if present</li>
                <li>Removed duplicate records</li>
                <li>Created binary target variable <code>Stress_Risk</code></li>
                <li>Removed <strong>"Maybe"</strong> rows</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Preprocessing</h4>
            <ul style="color:#4b5563;">
                <li>Applied Label Encoding</li>
                <li>Selected 8 key features</li>
                <li>Performed train-test split</li>
                <li>Saved encoders and feature columns for deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE: TRAINING
# ---------------------------------------------------
elif page == "Model Training & Tuning":
    st.markdown("""
    <div class="main-header">
        <h1>Model Training & Hyperparameter Tuning</h1>
        <p>Two machine learning algorithms were trained and tuned using GridSearchCV</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="metric-card" style="border-top:4px solid #2d5a87;">
            <h3 style="color:#1e3a5f; text-align:center;">🌳 Decision Tree</h3>
            <p style="color:#4b5563;">
                A rule-based classifier that can capture interactions between features and is easy to interpret.
            </p>
            <p style="color:#4b5563;">
                <strong>Tuned:</strong><br>
                • max_depth<br>
                • min_samples_split<br>
                • min_samples_leaf<br>
                • class_weight
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card" style="border-top:4px solid #38a169;">
            <h3 style="color:#1e3a5f; text-align:center;">📊 Categorical Naive Bayes</h3>
            <p style="color:#4b5563;">
                A probabilistic classifier suited for encoded categorical inputs and useful as a strong comparison baseline.
            </p>
            <p style="color:#4b5563;">
                <strong>Tuned:</strong><br>
                • alpha
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <strong>Tuning method:</strong> GridSearchCV<br>
        <strong>Optimization goal:</strong> maximize <strong>Recall</strong>, since this project aims to catch as many at-risk individuals as possible.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE: PERFORMANCE
# ---------------------------------------------------
elif page == "Model Performance & Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>Model Performance & Comparison</h1>
        <p>Evaluation metrics, comparison visuals, and confusion matrices</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_results:
        st.error("Saved model results not found.")
    else:
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Decision Tree": [
                model_results.get("dt_accuracy", 0),
                model_results.get("dt_precision", 0),
                model_results.get("dt_recall", 0),
                model_results.get("dt_f1", 0),
            ],
            "Naive Bayes": [
                model_results.get("nb_accuracy", 0),
                model_results.get("nb_precision", 0),
                model_results.get("nb_recall", 0),
                model_results.get("nb_f1", 0),
            ]
        })

        st.markdown('<div class="section-header"><h2>Detailed Metrics Breakdown</h2></div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Decision Tree 🌳",
            x=metrics_df["Metric"],
            y=metrics_df["Decision Tree"],
            marker_color="#2d5a87",
            text=[f"{v:.1%}" for v in metrics_df["Decision Tree"]],
            textposition="outside"
        ))
        fig.add_trace(go.Bar(
            name="Naive Bayes 📊",
            x=metrics_df["Metric"],
            y=metrics_df["Naive Bayes"],
            marker_color="#38a169",
            text=[f"{v:.1%}" for v in metrics_df["Naive Bayes"]],
            textposition="outside"
        ))
        fig.update_layout(
            barmode="group",
            height=430,
            title=dict(text="<b>All Metrics Comparison</b>", x=0.5),
            yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header"><h2>Confusion Matrix Analysis</h2></div>', unsafe_allow_html=True)
        dt_cm = model_results.get("dt_cm", [[0,0],[0,0]])
        nb_cm = model_results.get("nb_cm", [[0,0],[0,0]])

        c1, c2 = st.columns(2)
        with c1:
            fig_dt = go.Figure(data=go.Heatmap(
                z=dt_cm,
                text=dt_cm,
                texttemplate="%{text}",
                showscale=False,
                colorscale=[[0, '#dbeafe'], [1, '#1e3a5f']]
            ))
            fig_dt.update_layout(height=320, title=dict(text="<b>Decision Tree</b>", x=0.5))
            st.plotly_chart(fig_dt, use_container_width=True)

        with c2:
            fig_nb = go.Figure(data=go.Heatmap(
                z=nb_cm,
                text=nb_cm,
                texttemplate="%{text}",
                showscale=False,
                colorscale=[[0, '#dcfce7'], [1, '#276749']]
            ))
            fig_nb.update_layout(height=320, title=dict(text="<b>Naive Bayes</b>", x=0.5))
            st.plotly_chart(fig_nb, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <strong>Interpretation:</strong><br>
            Accuracy measures overall correctness, Precision measures how reliable positive predictions are,
            Recall measures how many true at-risk individuals are caught, and F1 balances Precision with Recall.
            For this project, <strong>Recall is the most important metric</strong>.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE: FINAL MODEL
# ---------------------------------------------------
elif page == "Final Model Selection":
    st.markdown("""
    <div class="main-header">
        <h1>Final Model Selection</h1>
        <p>Chosen based on project goal and evaluation results</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_results:
        st.error("Saved model results not found.")
    else:
        best_model = model_results.get("best_model", "Decision Tree")
        best_recall = max(model_results.get("dt_recall", 0), model_results.get("nb_recall", 0))

        st.markdown(f"""
        <div class="success-card">
            <strong>Selected Model:</strong> {best_model}<br><br>
            <strong>Best Recall:</strong> {best_recall:.1%}<br><br>
            <strong>Justification:</strong> This model best supports the project goal of reducing missed at-risk individuals.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE: PREDICTION
# ---------------------------------------------------
elif page == "Prediction Demo":
    st.markdown("""
    <div class="main-header">
        <h1>Stress Risk Prediction</h1>
        <p>Live prediction demo using the saved trained models</p>
    </div>
    """, unsafe_allow_html=True)

    if dt_model is None or nb_model is None or dt_encoders is None or dt_feature_columns is None or model_results is None:
        st.error("Models or artifacts not loaded.")
    else:
        st.markdown('<div class="section-header"><h2>Enter Feature Values</h2></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("👤 Gender", get_encoder_classes(dt_encoders, "Gender", ["Male", "Female"]))
            occupation = st.selectbox("💼 Occupation", get_encoder_classes(dt_encoders, "Occupation", ["Student", "Corporate", "Business", "Housewife", "Others"]))
            days_indoors = st.selectbox("🏠 Days Indoors", get_encoder_classes(dt_encoders, "Days_Indoors", ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"]))
        with c2:
            mood_swings = st.selectbox("😰 Mood Swings", get_encoder_classes(dt_encoders, "Mood_Swings", ["Low", "Medium", "High"]))
            social_weakness = st.selectbox("👥 Social Weakness", get_encoder_classes(dt_encoders, "Social_Weakness", ["Yes", "No", "Maybe"]))
            changes_habits = st.selectbox("🔄 Changes in Habits", get_encoder_classes(dt_encoders, "Changes_Habits", ["Yes", "No", "Maybe"]))
        with c3:
            work_interest = st.selectbox("📋 Work Interest", get_encoder_classes(dt_encoders, "Work_Interest", ["Yes", "No", "Maybe"]))
            mental_health_history = st.selectbox("🧠 Mental Health History", get_encoder_classes(dt_encoders, "Mental_Health_History", ["Yes", "No", "Maybe"]))
            model_choice = st.selectbox("Model Choice", ["Compare Both", "Decision Tree", "Naive Bayes"])

        if st.button("🔮 Predict", use_container_width=True):
            input_data = {
                "Mood_Swings": mood_swings,
                "Days_Indoors": days_indoors,
                "Occupation": occupation,
                "Social_Weakness": social_weakness,
                "Changes_Habits": changes_habits,
                "Gender": gender,
                "Work_Interest": work_interest,
                "Mental_Health_History": mental_health_history
            }

            class_labels = model_results.get("class_labels", ["At Risk", "Not At Risk"])
            dt_input = build_model_input(input_data, dt_feature_columns, dt_encoders)
            nb_input = build_model_input(input_data, nb_feature_columns, nb_encoders)

            dt_label = pred_to_label(dt_model.predict(dt_input)[0], class_labels)
            nb_label = pred_to_label(nb_model.predict(nb_input)[0], class_labels)

            st.markdown('<div class="section-header"><h2>Prediction Result</h2></div>', unsafe_allow_html=True)

            if model_choice == "Decision Tree":
                x1, x2, x3 = st.columns([1, 2, 1])
                with x2:
                    show_prediction_card("🌳 Decision Tree", dt_label)
            elif model_choice == "Naive Bayes":
                x1, x2, x3 = st.columns([1, 2, 1])
                with x2:
                    show_prediction_card("📊 Naive Bayes", nb_label)
            else:
                x1, x2 = st.columns(2)
                with x1:
                    show_prediction_card("🌳 Decision Tree", dt_label)
                with x2:
                    show_prediction_card("📊 Naive Bayes", nb_label)

            st.markdown("""
            <div class="info-card">
                <strong>Note:</strong> This is an educational screening demo and not a medical diagnosis.
            </div>
            """, unsafe_allow_html=True)
