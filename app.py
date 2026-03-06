
"""
Polished Streamlit App - Stress Trajectory Prediction
Keeps organized visualizations, includes the correct dataset link,
and aligns with the actual notebook/model workflow.

Requirements covered:
- Dataset link + clear problem statement
- EDA with meaningful visualizations and insights
- Data cleaning and preprocessing
- Minimum 2 ML algorithms + comparison
- Hyperparameter tuning summary
- Evaluation metrics + interpretation
- Final model selection + justification
- Live prediction demo
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
        padding: 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.3rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.92);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    .section-header {
        background: linear-gradient(90deg, #f3f7fb 0%, #ffffff 100%);
        border-left: 5px solid #2d5a87;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin: 1.6rem 0 1rem 0;
    }
    .section-header h2 {
        color: #1e3a5f;
        margin: 0;
        font-size: 1.35rem;
    }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid #edf2f7;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        line-height: 1;
    }
    .metric-label {
        color: #718096;
        margin-top: 0.4rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .info-card {
        background: linear-gradient(145deg, #e8f4fd 0%, #d9edf9 100%);
        border-left: 4px solid #2d5a87;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0;
    }
    .warn-card {
        background: linear-gradient(145deg, #fff5f5 0%, #ffe6e6 100%);
        border-left: 4px solid #e53e3e;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0;
    }
    .success-card {
        background: linear-gradient(145deg, #f0fff4 0%, #dcffe8 100%);
        border-left: 4px solid #38a169;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0;
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
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def metric_card(value, label):
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
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

def show_prediction_card(title, label):
    risk = label == "At Risk"
    css = "prediction-at-risk" if risk else "prediction-not-at-risk"
    icon = "⚠️" if risk else "✅"
    color = "#e53e3e" if risk else "#38a169"
    st.markdown(
        f'<div class="{css}"><h3 style="color:{color}; margin-top:0;">{title}</h3>'
        f'<div style="font-size:3.5rem;">{icon}</div>'
        f'<h1 style="color:{color}; margin-bottom:0;">{label}</h1></div>',
        unsafe_allow_html=True
    )

def prepare_eda_target(dataframe):
    temp = dataframe.copy()
    temp["Stress_Risk"] = temp["Growing_Stress"].apply(
        lambda x: "At Risk" if x == "Yes" else ("Not At Risk" if x == "No" else "Maybe")
    )
    temp = temp[temp["Stress_Risk"] != "Maybe"].copy()
    return temp

def stacked_risk_chart(dataframe, feature, title, insight, category_order=None):
    if feature not in dataframe.columns or "Stress_Risk" not in dataframe.columns:
        st.warning(f"{feature} not found in dataset.")
        return

    ctab = pd.crosstab(dataframe[feature], dataframe["Stress_Risk"], normalize="index") * 100
    if category_order is not None:
        ctab = ctab.reindex([c for c in category_order if c in ctab.index])

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
        height=400,
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis_title=feature.replace("_", " "),
        yaxis_title="Percentage %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="info-card"><strong>Insight:</strong> {insight}</div>', unsafe_allow_html=True)

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
    <div style="text-align:center; padding:0.8rem 0 0.5rem 0;">
        <div style="font-size:2.4rem;">🧠</div>
        <h3 style="margin:0.3rem 0; color:#1e3a5f;">Mental Health</h3>
        <p style="color:#6b7280; margin:0;">Stress Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

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
            "Prediction Demo",
        ],
        label_visibility="collapsed"
    )

# ---------------------------------------------------
# PAGE 1
# ---------------------------------------------------
if page == "Problem Statement":
    st.markdown("""
    <div class="main-header">
        <h1>Stress Trajectory Prediction</h1>
        <p>A machine learning approach to identifying individuals at risk of growing stress</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Dataset + Objective</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>Dataset:</strong> Mental Health Dataset (Kaggle)<br><br>
            <strong>Objective:</strong> Predict whether a person is at risk of growing stress<br><br>
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
            <strong>Why Recall is prioritized</strong><br><br>
            In a mental health screening setting, missing someone who may need support is more harmful than creating an extra check-in.
            <br><br>
            <strong>False Negative:</strong> an at-risk person is missed<br>
            <strong>False Positive:</strong> an extra follow-up is triggered
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Why this project matters</h2></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("1 in 5", "Mental health challenges")
    with c2: metric_card("Binary", "Classification task")
    with c3: metric_card("Recall", "Priority metric")
    with c4: metric_card("ML", "Decision support")

# ---------------------------------------------------
# PAGE 2
# ---------------------------------------------------
elif page == "Dataset Overview":
    st.markdown("""
    <div class="main-header">
        <h1>Dataset Overview</h1>
        <p>Understanding the dataset structure and selected training features</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found. Place 'Mental Health Dataset.csv' next to app.py.")
    else:
        selected_features = dt_feature_columns if dt_feature_columns is not None else []
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card(f"{len(df):,}", "Raw records")
        with c2: metric_card(f"{len(df.columns)}", "Columns")
        with c3: metric_card(f"{len(selected_features)}", "Selected features")
        with c4: metric_card("2", "ML models")

        st.markdown('<div class="section-header"><h2>Selected features used in training</h2></div>', unsafe_allow_html=True)
        if selected_features:
            st.dataframe(pd.DataFrame({"Selected Features": selected_features}), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header"><h2>Sample data preview</h2></div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        eda_df = prepare_eda_target(df)
        if "Stress_Risk" in eda_df.columns:
            counts = eda_df["Stress_Risk"].value_counts()
            fig = go.Figure(go.Bar(
                x=counts.index,
                y=counts.values,
                text=counts.values,
                textposition="outside"
            ))
            fig.update_layout(
                height=380,
                title=dict(text="<b>Target Distribution After Binary Conversion</b>", x=0.5),
                xaxis_title="Stress_Risk",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="info-card">Rows with <strong>"Maybe"</strong> were removed from the target so the notebook and app both use the same binary classification setup.</div>',
                unsafe_allow_html=True
            )

# ---------------------------------------------------
# PAGE 3
# ---------------------------------------------------
elif page == "EDA & Insights":
    st.markdown("""
    <div class="main-header">
        <h1>Exploratory Data Analysis</h1>
        <p>Organized visualizations with clear insights from the dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found.")
    else:
        eda_df = prepare_eda_target(df)

        st.markdown('<div class="section-header"><h2>Stress Risk Distribution</h2></div>', unsafe_allow_html=True)
        counts = eda_df["Stress_Risk"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            textinfo="percent+label"
        )])
        fig.update_layout(height=400, title=dict(text="<b>Stress_Risk Class Balance</b>", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="info-card"><strong>Insight:</strong> After removing "Maybe", the task becomes a clean binary classification problem with two clearly defined classes.</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-header"><h2>Mood Swings vs Stress Risk</h2></div>', unsafe_allow_html=True)
        stacked_risk_chart(
            eda_df,
            "Mood_Swings",
            "Stress Risk by Mood Swings",
            "Higher mood swing levels tend to show a larger proportion of at-risk individuals, making this one of the strongest features in the project.",
            category_order=["Low", "Medium", "High"]
        )

        st.markdown('<div class="section-header"><h2>Days Indoors vs Stress Risk</h2></div>', unsafe_allow_html=True)
        stacked_risk_chart(
            eda_df,
            "Days_Indoors",
            "Stress Risk by Days Indoors",
            "Longer periods indoors appear associated with a larger at-risk share, suggesting that isolation may be linked with higher stress patterns.",
            category_order=["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"]
        )

        st.markdown('<div class="section-header"><h2>Social Weakness vs Stress Risk</h2></div>', unsafe_allow_html=True)
        stacked_risk_chart(
            eda_df,
            "Social_Weakness",
            "Stress Risk by Social Weakness",
            "Weaker social support patterns are associated with higher stress risk, which supports the importance of social factors in mental health screening.",
            category_order=["No", "Maybe", "Yes"]
        )

        st.markdown('<div class="section-header"><h2>Occupation vs Stress Risk</h2></div>', unsafe_allow_html=True)
        stacked_risk_chart(
            eda_df,
            "Occupation",
            "Stress Risk by Occupation",
            "Different occupation groups show different stress-risk patterns, suggesting daily environment and lifestyle may influence mental health outcomes."
        )

        st.markdown('<div class="section-header"><h2>Key EDA findings</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <ul style="margin-bottom:0;">
                <li>Mood swings stand out as a strong stress-related signal.</li>
                <li>More days indoors appear linked with higher risk.</li>
                <li>Social weakness is meaningfully associated with risk level.</li>
                <li>Occupation groups show different stress patterns.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE 4
# ---------------------------------------------------
elif page == "Data Cleaning & Preprocessing":
    st.markdown("""
    <div class="main-header">
        <h1>Data Cleaning & Preprocessing</h1>
        <p>Simple explanation of the workflow used before model training</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Cleaning steps</h4>
            <ul style="color:#4b5563;">
                <li>Removed <code>Timestamp</code> if present</li>
                <li>Removed duplicate rows</li>
                <li>Created binary target variable <code>Stress_Risk</code></li>
                <li>Removed <strong>"Maybe"</strong> rows from the target</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Preprocessing steps</h4>
            <ul style="color:#4b5563;">
                <li>Applied Label Encoding to categorical features</li>
                <li>Selected the final 8 training features</li>
                <li>Performed train-test split</li>
                <li>Saved encoders and feature columns for Streamlit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Selected features</h2></div>', unsafe_allow_html=True)
    if dt_feature_columns:
        st.code("\n".join(dt_feature_columns))
    else:
        st.info("Feature columns not found in saved artifacts.")

# ---------------------------------------------------
# PAGE 5
# ---------------------------------------------------
elif page == "Model Training & Tuning":
    st.markdown("""
    <div class="main-header">
        <h1>Model Training & Hyperparameter Tuning</h1>
        <p>Two machine learning algorithms were trained and compared</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card" style="border-top:4px solid #2d5a87;">
            <h3 style="color:#1e3a5f; text-align:center;">🌳 Decision Tree</h3>
            <p style="color:#4b5563;">
                A rule-based model that can capture interactions between features and is easy to interpret.
            </p>
            <p style="color:#4b5563;">
                <strong>Tuned parameters:</strong><br>
                • max_depth<br>
                • min_samples_split<br>
                • min_samples_leaf<br>
                • class_weight
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="border-top:4px solid #38a169;">
            <h3 style="color:#1e3a5f; text-align:center;">📊 Categorical Naive Bayes</h3>
            <p style="color:#4b5563;">
                A probabilistic model suited to encoded categorical inputs and useful as a strong baseline model.
            </p>
            <p style="color:#4b5563;">
                <strong>Tuned parameter:</strong><br>
                • alpha (smoothing)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Tuning strategy</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <strong>Method used:</strong> GridSearchCV<br>
        <strong>Goal:</strong> optimize for <strong>Recall</strong> so the final system catches as many at-risk individuals as possible.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE 6
# ---------------------------------------------------
elif page == "Model Performance & Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>Model Performance & Comparison</h1>
        <p>Evaluation metrics, confusion matrices, and interpretation</p>
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

        st.markdown('<div class="section-header"><h2>Metric comparison</h2></div>', unsafe_allow_html=True)
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
            title=dict(text="<b>Model Comparison Across Metrics</b>", x=0.5),
            yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Score")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header"><h2>Confusion matrices</h2></div>', unsafe_allow_html=True)
        dt_cm = model_results.get("dt_cm", [[0,0],[0,0]])
        nb_cm = model_results.get("nb_cm", [[0,0],[0,0]])

        c1, c2 = st.columns(2)
        with c1:
            fig_dt = go.Figure(data=go.Heatmap(
                z=dt_cm,
                text=dt_cm,
                texttemplate="%{text}",
                showscale=False
            ))
            fig_dt.update_layout(height=320, title=dict(text="<b>Decision Tree</b>", x=0.5))
            st.plotly_chart(fig_dt, use_container_width=True)

        with c2:
            fig_nb = go.Figure(data=go.Heatmap(
                z=nb_cm,
                text=nb_cm,
                texttemplate="%{text}",
                showscale=False
            ))
            fig_nb.update_layout(height=320, title=dict(text="<b>Naive Bayes</b>", x=0.5))
            st.plotly_chart(fig_nb, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <strong>Interpretation:</strong> Accuracy measures overall correctness, Precision measures how reliable positive flags are,
            Recall measures how many true at-risk individuals are caught, and F1 balances Precision with Recall.
            For this project, <strong>Recall is the most important metric</strong>.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# PAGE 7
# ---------------------------------------------------
elif page == "Final Model Selection":
    st.markdown("""
    <div class="main-header">
        <h1>Final Model Selection</h1>
        <p>The best model is chosen with clear justification</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_results:
        st.error("Saved model results not found.")
    else:
        best_model = model_results.get("best_model", "Decision Tree")
        recall_dt = model_results.get("dt_recall", 0)
        recall_nb = model_results.get("nb_recall", 0)
        best_recall = max(recall_dt, recall_nb)

        st.markdown(f"""
        <div class="success-card">
            <strong>Selected Model:</strong> {best_model}<br><br>
            <strong>Why selected:</strong> It achieved the best Recall score, which is the most important metric for this screening problem.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <strong>Justification:</strong><br>
            In a mental health context, missing an at-risk individual is more harmful than a false alarm.
            Because of that, the final model is selected based on its ability to catch more true positive at-risk cases.
        </div>
        """, unsafe_allow_html=True)

        st.metric("Best Recall", f"{best_recall:.1%}")

# ---------------------------------------------------
# PAGE 8
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
        st.markdown('<div class="section-header"><h2>Enter feature values</h2></div>', unsafe_allow_html=True)

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
            mode = st.selectbox("Model choice", ["Compare Both", "Decision Tree", "Naive Bayes"])

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

            st.markdown('<div class="section-header"><h2>Prediction result</h2></div>', unsafe_allow_html=True)

            if mode == "Decision Tree":
                c1, c2, c3 = st.columns([1,2,1])
                with c2:
                    show_prediction_card("🌳 Decision Tree", dt_label)
            elif mode == "Naive Bayes":
                c1, c2, c3 = st.columns([1,2,1])
                with c2:
                    show_prediction_card("📊 Naive Bayes", nb_label)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    show_prediction_card("🌳 Decision Tree", dt_label)
                with c2:
                    show_prediction_card("📊 Naive Bayes", nb_label)

            st.markdown("""
            <div class="info-card">
                <strong>Note:</strong> This is an educational screening demo. It is not a medical diagnosis.
            </div>
            """, unsafe_allow_html=True)
