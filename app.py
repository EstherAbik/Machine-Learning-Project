"""
Full Streamlit App: Stress Trajectory Prediction - Mental Health Analysis Platform

What this app includes:
- Dataset link + problem statement
- Dataset overview
- EDA with insights
- Data cleaning & preprocessing summary
- Model training & hyperparameter tuning summary
- Model performance comparison
- Final model selection with justification
- Live risk prediction demo

Important assumptions aligned with your notebook:
- Target is binary: "At Risk" vs "Not At Risk"
- "Maybe" rows are removed from the target during training
- Saved feature set is notebook-aligned (expected 8 features)
- class_labels are loaded from model_results.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Stress Trajectory Prediction | Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace with your real dataset link if you want
DATASET_URL = "https://www.kaggle.com/datasets"

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "Mental Health Dataset.csv"

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab3 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }

    .main-header p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        text-align: center;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.35rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.88rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 0.5rem;
    }

    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, white 100%);
        padding: 1rem 1.3rem;
        border-radius: 12px;
        border-left: 4px solid #2d5a87;
        margin: 2rem 0 1.2rem 0;
    }

    .section-header h2 {
        color: #1e3a5f;
        font-weight: 600;
        margin: 0;
        font-size: 1.4rem;
    }

    .info-card {
        background: linear-gradient(145deg, #e8f4fd 0%, #d4e8f7 100%);
        border-radius: 12px;
        padding: 1.15rem;
        border-left: 4px solid #2d5a87;
        margin: 1rem 0;
    }

    .warn-card {
        background: linear-gradient(145deg, #fff5f5 0%, #ffe8e8 100%);
        border-radius: 12px;
        padding: 1.15rem;
        border-left: 4px solid #e53e3e;
        margin: 1rem 0;
    }

    .success-card {
        background: linear-gradient(145deg, #f0fff4 0%, #dcffe4 100%);
        border-radius: 12px;
        padding: 1.15rem;
        border-left: 4px solid #38a169;
        margin: 1rem 0;
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

    .stButton > button {
        background: linear-gradient(135deg, #2d5a87 0%, #1e3a5f 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3d7ab3 0%, #2d5a87 100%);
        box-shadow: 0 4px 15px rgba(45, 90, 135, 0.4);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
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
    is_risk = (label == "At Risk")
    css_class = "prediction-at-risk" if is_risk else "prediction-not-at-risk"
    icon = "⚠️" if is_risk else "✅"
    color = "#e53e3e" if is_risk else "#38a169"
    st.markdown(
        f'<div class="{css_class}"><h3 style="color: {color};">{title}</h3>'
        f'<div style="font-size: 4rem;">{icon}</div>'
        f'<h1 style="color: {color};">{label}</h1></div>',
        unsafe_allow_html=True
    )

def show_metric_cards(items):
    cols = st.columns(len(items))
    for col, (value, label) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

def make_stress_risk_for_eda(df):
    temp = df.copy()
    if "Stress_Risk" not in temp.columns and "Growing_Stress" in temp.columns:
        temp["Stress_Risk"] = temp["Growing_Stress"].apply(
            lambda x: "At Risk" if x == "Yes" else ("Not At Risk" if x == "No" else "Maybe")
        )
    if "Stress_Risk" in temp.columns:
        temp = temp[temp["Stress_Risk"] != "Maybe"]
    return temp

# -----------------------------
# LOADERS
# -----------------------------
@st.cache_resource
def load_models():
    try:
        dt_model = joblib.load(MODELS_DIR / "decision_tree_model.pkl")
        nb_model = joblib.load(MODELS_DIR / "naive_bayes_model.pkl")
        dt_encoders = joblib.load(MODELS_DIR / "encoders.pkl")
        dt_feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
        nb_encoders = joblib.load(MODELS_DIR / "nb_encoders.pkl")
        nb_feature_columns = joblib.load(MODELS_DIR / "nb_feature_columns.pkl")
        raw_results = joblib.load(MODELS_DIR / "model_results.pkl")
        model_results = {
            "Decision Tree": {
                "Accuracy": raw_results.get("dt_accuracy", 0),
                "Precision": raw_results.get("dt_precision", 0),
                "Recall": raw_results.get("dt_recall", 0),
                "F1": raw_results.get("dt_f1", 0),
                "Confusion Matrix": raw_results.get("dt_cm", [[0, 0], [0, 0]])
            },
            "Naive Bayes": {
                "Accuracy": raw_results.get("nb_accuracy", 0),
                "Precision": raw_results.get("nb_precision", 0),
                "Recall": raw_results.get("nb_recall", 0),
                "F1": raw_results.get("nb_f1", 0),
                "Confusion Matrix": raw_results.get("nb_cm", [[0, 0], [0, 0]])
            },
            "best_model": raw_results.get("best_model", "Decision Tree"),
            "class_labels": raw_results.get("class_labels", ["At Risk", "Not At Risk"]),
            "maybe_removed": raw_results.get("maybe_removed", True),
            "features_used": raw_results.get("features_used", [])
        }
        return dt_model, nb_model, dt_encoders, dt_feature_columns, nb_encoders, nb_feature_columns, model_results
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None

@st.cache_data
def load_data():
    try:
        if DATA_PATH.exists():
            return pd.read_csv(DATA_PATH)
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# -----------------------------
# LOAD
# -----------------------------
df = load_data()
dt_model, nb_model, dt_encoders, dt_feature_columns, nb_encoders, nb_feature_columns, model_results = load_models()
encoders = dt_encoders if dt_encoders else {}
saved_features = model_results.get("features_used", []) if model_results else []
if not saved_features and dt_feature_columns:
    saved_features = list(dt_feature_columns)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.5rem; margin: 0;">🧠</h1>
        <h3 style="color: #1e3a5f; margin: 0.5rem 0;">Mental Health</h3>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">Stress Analysis Platform</p>
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
            "Risk Prediction"
        ],
        label_visibility="collapsed"
    )

# =====================================================
# PAGE 1: PROBLEM STATEMENT
# =====================================================
if page == "Problem Statement":
    st.markdown("""
    <div class="main-header">
        <h1>Stress Trajectory Prediction</h1>
        <p>A Machine Learning Approach to Mental Health Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>🎯 Dataset, Objective, and Problem Type</h2></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">Project Summary</h4>
            <p>
                <strong>Dataset:</strong> Mental Health Dataset<br><br>
                <strong>Objective:</strong> Predict whether an individual is at risk of growing stress<br><br>
                <strong>Problem Type:</strong> Binary Classification<br><br>
                <strong>Target Variable:</strong> <code>Stress_Risk</code><br>
                • <strong>At Risk</strong><br>
                • <strong>Not At Risk</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Open Dataset Link", DATASET_URL)
    with col2:
        st.markdown("""
        <div class="warn-card">
            <h4 style="color: #c53030; margin-top: 0;">Why Recall Matters Most</h4>
            <p style="color: #c53030;">
                In mental health screening, missing a person who may need support is more harmful than triggering an extra check-in.
                <br><br>
                <strong>False Negative:</strong> At-risk individual missed<br><br>
                <strong>False Positive:</strong> Extra screening or check-in
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>💡 Why This Project Matters</h2></div>', unsafe_allow_html=True)
    show_metric_cards([
        ("1 in 5", "People with Mental Health Challenges"),
        ("Binary", "Classification Problem"),
        ("Recall", "Primary Metric"),
        ("AI", "Scalable Screening Support")
    ])

# PAGE 2
elif page == "Dataset Overview":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dataset Overview</h1>
        <p>Understanding the data used for stress risk prediction</p>
    </div>
    """, unsafe_allow_html=True)
    if df is None:
        st.error("Dataset file not found. Put `Mental Health Dataset.csv` next to app.py.")
    else:
        eda_df = make_stress_risk_for_eda(df)
        show_metric_cards([
            (f"{len(df):,}", "Raw Records"),
            (f"{len(df.columns)}", "Raw Columns"),
            (f"{len(saved_features)}", "Selected Features"),
            ("2", "ML Algorithms")
        ])
        st.markdown('<div class="section-header"><h2>🧾 Selected Features Used in Training</h2></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Selected Features": saved_features}), use_container_width=True, hide_index=True)
        st.markdown('<div class="section-header"><h2>📄 Sample Data Preview</h2></div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('<div class="section-header"><h2>📌 Target Distribution After Binary Conversion</h2></div>', unsafe_allow_html=True)
        if "Stress_Risk" in eda_df.columns:
            target_counts = eda_df["Stress_Risk"].value_counts()
            fig = go.Figure(go.Bar(x=target_counts.index, y=target_counts.values, text=target_counts.values, textposition="outside"))
            fig.update_layout(height=380, title=dict(text="<b>Stress_Risk Class Distribution</b>", x=0.5), xaxis_title="Class", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# PAGE 3
elif page == "EDA & Insights":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Exploratory Data Analysis</h1>
        <p>Meaningful visualizations and insights from the dataset</p>
    </div>
    """, unsafe_allow_html=True)
    if df is None:
        st.error("Dataset file not found.")
    else:
        eda_df = make_stress_risk_for_eda(df)
        def risk_chart(feature, title, insight):
            if feature in eda_df.columns and "Stress_Risk" in eda_df.columns:
                ctab = pd.crosstab(eda_df[feature], eda_df["Stress_Risk"], normalize="index") * 100
                fig = go.Figure()
                if "Not At Risk" in ctab.columns:
                    fig.add_trace(go.Bar(name="Not At Risk", x=ctab.index.astype(str), y=ctab["Not At Risk"], marker_color="#38a169", text=[f"{v:.1f}%" for v in ctab["Not At Risk"]], textposition="inside"))
                if "At Risk" in ctab.columns:
                    fig.add_trace(go.Bar(name="At Risk", x=ctab.index.astype(str), y=ctab["At Risk"], marker_color="#e53e3e", text=[f"{v:.1f}%" for v in ctab["At Risk"]], textposition="inside"))
                fig.update_layout(barmode="stack", height=380, title=dict(text=f"<b>{title}</b>", x=0.5), xaxis_title=feature, yaxis_title="Percentage %")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'<div class="info-card"><p><strong>Insight:</strong> {insight}</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>1️⃣ Target Distribution</h2></div>', unsafe_allow_html=True)
        counts = eda_df["Stress_Risk"].value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, textinfo="percent+label"))
        fig.update_layout(height=380, title=dict(text="<b>Class Balance</b>", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="section-header"><h2>2️⃣ Mood Swings vs Stress Risk</h2></div>', unsafe_allow_html=True)
        risk_chart("Mood_Swings", "Stress Risk by Mood Swings", "Higher mood swing levels tend to show a larger proportion of at-risk individuals, making this an important predictive feature.")
        st.markdown('<div class="section-header"><h2>3️⃣ Days Indoors vs Stress Risk</h2></div>', unsafe_allow_html=True)
        risk_chart("Days_Indoors", "Stress Risk by Days Indoors", "Longer indoor duration appears associated with higher stress risk, suggesting isolation may be an important factor.")
        st.markdown('<div class="section-header"><h2>4️⃣ Social Weakness vs Stress Risk</h2></div>', unsafe_allow_html=True)
        risk_chart("Social_Weakness", "Stress Risk by Social Weakness", "Weaker social support patterns are associated with higher rates of stress risk.")
        st.markdown('<div class="section-header"><h2>5️⃣ Occupation vs Stress Risk</h2></div>', unsafe_allow_html=True)
        risk_chart("Occupation", "Stress Risk by Occupation", "Different occupation groups show different stress-risk patterns, indicating lifestyle and work context may matter.")

# PAGE 4
elif page == "Data Cleaning & Preprocessing":
    st.markdown("""
    <div class="main-header">
        <h1>🧹 Data Cleaning & Preprocessing</h1>
        <p>Preparing the data for machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Data Cleaning</h4>
            <ul style="color:#6c757d;">
                <li>Removed non-predictive columns such as Timestamp if present</li>
                <li>Checked and handled missing values</li>
                <li>Removed duplicate records</li>
                <li>Validated categorical feature values</li>
            </ul>
        </div>
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Target Engineering</h4>
            <ul style="color:#6c757d;">
                <li>Created <code>Stress_Risk</code> from the original stress-related field</li>
                <li>Mapped Yes → At Risk and No → Not At Risk</li>
                <li>Removed Maybe rows for binary classification consistency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Feature Preprocessing</h4>
            <ul style="color:#6c757d;">
                <li>Applied label encoding to categorical variables</li>
                <li>Saved encoders for reuse in the app</li>
                <li>Selected the final feature subset used during training</li>
            </ul>
        </div>
        <div class="metric-card">
            <h4 style="color:#1e3a5f; margin-top:0;">Splitting & Balancing</h4>
            <ul style="color:#6c757d;">
                <li>Used train-test split for evaluation</li>
                <li>Used stratification when appropriate</li>
                <li>Focused on improving Recall for the at-risk class</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f'<div class="info-card"><p><strong>Final Selected Features:</strong> {", ".join(saved_features)}</p></div>', unsafe_allow_html=True)

# PAGE 5
elif page == "Model Training & Tuning":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Model Training & Hyperparameter Tuning</h1>
        <p>Algorithms used, tuning strategy, and training workflow</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-top:4px solid #2d5a87;">
            <h3 style="color:#1e3a5f; text-align:center;">🌳 Decision Tree</h3>
            <p style="color:#6c757d;">Interpretable, rule-based model that can capture non-linear patterns in encoded features.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-top:4px solid #38a169;">
            <h3 style="color:#1e3a5f; text-align:center;">📊 Naive Bayes</h3>
            <p style="color:#6c757d;">Fast probabilistic model used as a strong baseline and comparison model.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="info-card"><p><strong>Method Used:</strong> GridSearchCV with cross-validation, optimized mainly for <strong>Recall</strong>.</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.code("""param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}""")
    with col2:
        st.code("""param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 2.0]
}""")

# PAGE 6
elif page == "Model Performance & Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Model Performance & Comparison</h1>
        <p>Evaluation metrics, confusion matrices, and model comparison</p>
    </div>
    """, unsafe_allow_html=True)
    dt = model_results["Decision Tree"]
    nb = model_results["Naive Bayes"]
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Decision Tree": [dt["Accuracy"], dt["Precision"], dt["Recall"], dt["F1"]],
        "Naive Bayes": [nb["Accuracy"], nb["Precision"], nb["Recall"], nb["F1"]]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Decision Tree 🌳", x=metrics_df["Metric"], y=metrics_df["Decision Tree"], text=[f"{v:.1%}" for v in metrics_df["Decision Tree"]], textposition="outside", marker_color="#2d5a87"))
    fig.add_trace(go.Bar(name="Naive Bayes 📊", x=metrics_df["Metric"], y=metrics_df["Naive Bayes"], text=[f"{v:.1%}" for v in metrics_df["Naive Bayes"]], textposition="outside", marker_color="#38a169"))
    fig.update_layout(barmode="group", height=420, title=dict(text="<b>Model Comparison Across Metrics</b>", x=0.5), yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Score"))
    st.plotly_chart(fig, use_container_width=True)

# PAGE 7
elif page == "Final Model Selection":
    st.markdown("""
    <div class="main-header">
        <h1>🏆 Final Model Selection</h1>
        <p>Choosing the best model with clear justification</p>
    </div>
    """, unsafe_allow_html=True)
    best = model_results.get("best_model", "Decision Tree")
    dt = model_results["Decision Tree"]
    nb = model_results["Naive Bayes"]
    best_recall = max(dt["Recall"], nb["Recall"])
    st.markdown(f'<div class="success-card"><p><strong>Selected Model:</strong> {best}<br><strong>Main Reason:</strong> Highest Recall for catching at-risk individuals.<br><strong>Recall:</strong> {best_recall:.1%}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><p>This final model was chosen because the project prioritizes minimizing missed at-risk individuals over maximizing overall accuracy alone.</p></div>', unsafe_allow_html=True)

# PAGE 8
elif page == "Risk Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Stress Risk Prediction</h1>
        <p>Live prediction demo using the saved trained models</p>
    </div>
    """, unsafe_allow_html=True)
    mode = st.selectbox("Prediction mode", ["🔄 Both Models (Compare)", "🌳 Decision Tree", "📊 Naive Bayes"], index=0)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("👤 Gender", get_encoder_classes(encoders, "Gender", ["Male", "Female"]))
        occupation = st.selectbox("💼 Occupation", get_encoder_classes(encoders, "Occupation", ["Student", "Corporate", "Business", "Housewife", "Others"]))
        days_indoors = st.selectbox("🏠 Days Indoors", get_encoder_classes(encoders, "Days_Indoors", ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"]))
    with col2:
        mood_swings = st.selectbox("😰 Mood Swings", get_encoder_classes(encoders, "Mood_Swings", ["Low", "Medium", "High"]))
        social_weakness = st.selectbox("👥 Social Weakness", get_encoder_classes(encoders, "Social_Weakness", ["Yes", "No", "Maybe"]))
        changes_habits = st.selectbox("🔄 Changes in Habits", get_encoder_classes(encoders, "Changes_Habits", ["Yes", "No", "Maybe"]))
    with col3:
        work_interest = st.selectbox("📋 Work Interest", get_encoder_classes(encoders, "Work_Interest", ["Yes", "No", "Maybe"]))
        mental_health_history = st.selectbox("🧠 Mental Health History", get_encoder_classes(encoders, "Mental_Health_History", ["Yes", "No", "Maybe"]))
    if st.button("🔮 Analyze Risk", use_container_width=True):
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
        dt_input = build_model_input(input_data, dt_feature_columns, dt_encoders)
        nb_input = build_model_input(input_data, nb_feature_columns, nb_encoders)
        class_labels = model_results.get("class_labels", ["At Risk", "Not At Risk"])
        dt_label = pred_to_label(dt_model.predict(dt_input)[0], class_labels)
        nb_label = pred_to_label(nb_model.predict(nb_input)[0], class_labels)
        if mode == "🌳 Decision Tree":
            show_prediction_card("Decision Tree Prediction", dt_label)
        elif mode == "📊 Naive Bayes":
            show_prediction_card("Naive Bayes Prediction", nb_label)
        else:
            col1, col2 = st.columns(2)
            with col1:
                show_prediction_card("🌳 Decision Tree", dt_label)
            with col2:
                show_prediction_card("📊 Naive Bayes", nb_label)
        st.markdown('<div class="info-card"><p><strong>Note:</strong> This tool is for educational screening and demonstration purposes. It is not a medical diagnosis.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="warn-card"><p>If someone feels unsafe or overwhelmed, contacting a trusted adult, counselor, or crisis support service can help. In the US/Canada, call or text <strong>988</strong>.</p></div>', unsafe_allow_html=True)
