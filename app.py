"""
Stress Trajectory Prediction - Mental Health Analysis Platform

Features:
- Introduction page
- Dataset sample in Introduction
- Why This Matters box in Introduction
- EDA dashboard with tabs
- Decision Tree visualization under EDA
- Why Recall Matters under Model Performance
- KPI row + confusion matrices in Model Performance
- Model choice only in Prediction Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
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
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 35px rgba(0,0,0,0.14);
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
        font-size: 1rem;
        text-align: center;
        margin-top: 0.55rem;
    }
    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        padding: 0.9rem 1.15rem;
        border-radius: 10px;
        border-left: 4px solid #2d5a87;
        margin: 1.4rem 0 1rem 0;
    }
    .section-header h2 {
        color: #1e3a5f;
        margin: 0;
        font-size: 1.3rem;
    }
    .info-card {
        background: linear-gradient(145deg, #e8f4fd 0%, #d4e8f7 100%);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        border-left: 4px solid #2d5a87;
        margin: 0.8rem 0;
    }
    .prediction-at-risk {
        background: #fff5f5;
        border: 2px solid #e53e3e;
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
    }
    .prediction-not-at-risk {
        background: #f0fff4;
        border: 2px solid #38a169;
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPERS
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
        model_results = joblib.load(MODELS_DIR / "model_results.pkl")
        return dt_model, nb_model, dt_encoders, dt_feature_columns, nb_encoders, nb_feature_columns, model_results
    except Exception:
        return None, None, None, None, None, None, None

def prepare_eda_df(df):
    temp = df.copy()
    temp["Stress_Risk"] = temp["Growing_Stress"].map({"Yes": "At Risk", "No": "Not At Risk"})
    temp = temp.dropna(subset=["Stress_Risk"]).copy()
    return temp

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
        f'<div class="{css}"><h3 style="color:{color};">{title}</h3>'
        f'<div style="font-size:3rem;">{icon}</div><h2 style="color:{color};">{label}</h2></div>',
        unsafe_allow_html=True
    )

def stacked_plot(eda_df, feature, title, order=None):
    ct = pd.crosstab(eda_df[feature], eda_df["Stress_Risk"], normalize="index") * 100
    if order is not None:
        ct = ct.reindex([x for x in order if x in ct.index])

    fig = go.Figure()
    if "Not At Risk" in ct.columns:
        fig.add_trace(go.Bar(
            name="Not At Risk",
            x=ct.index.astype(str),
            y=ct["Not At Risk"],
            marker_color="#2A9D8F",
            text=[f"{v:.1f}%" for v in ct["Not At Risk"]],
            textposition="inside"
        ))
    if "At Risk" in ct.columns:
        fig.add_trace(go.Bar(
            name="At Risk",
            x=ct.index.astype(str),
            y=ct["At Risk"],
            marker_color="#E63946",
            text=[f"{v:.1f}%" for v in ct["At Risk"]],
            textposition="inside"
        ))
    fig.update_layout(
        barmode="stack",
        height=420,
        title=title,
        yaxis_title="Percentage %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# LOAD
# ---------------------------------------------------
df = load_data()
dt_model, nb_model, dt_encoders, dt_feature_columns, nb_encoders, nb_feature_columns, model_results = load_models()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-bottom:10px;">
        <h1 style="font-size:40px;">🧠</h1>
        <h2 style="margin-bottom:0;">Stress Risk App</h2>
        <p style="color:gray; font-size:13px;">Mental Health Screening Tool</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Introduction",
            "EDA & Insights",
            "Preprocessing",
            "Model Performance",
            "Prediction Demo"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Machine Learning Project")

# ---------------------------------------------------
# PAGE: Introduction
# ---------------------------------------------------
if page == "Introduction":
    st.markdown('<div class="main-header"><h1>Introduction</h1><p>Stress Trajectory Prediction – A Machine Learning Approach to Mental Health Risk Assessment</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 1])

    with col1:
        st.write("**Goal:** Predict whether a person is **At Risk** of growing stress.")
        st.write("**Problem Type:** Binary Classification")
        st.write("**Target Variable:** `Stress_Risk`")
        st.link_button("Open Dataset on Kaggle", DATASET_URL)

        st.markdown("""
        <div class="info-card">
            <strong>Project overview:</strong><br><br>
            This project uses behavioral and lifestyle factors such as mood swings, time spent indoors,
            social weakness, and work interest to estimate stress risk.
            The task is treated as a binary classification problem after removing ambiguous
            <strong>"Maybe"</strong> responses from the target.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <strong>💡 Why This Matters for Mental Health</strong><br><br>
            🔹 <strong>Stress</strong> is a major risk factor linked to anxiety and depression.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if df is not None:
            st.markdown('<div class="section-header"><h2>Dataset Sample</h2></div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------
# PAGE: EDA
# ---------------------------------------------------
elif page == "EDA & Insights":
    st.markdown('<div class="main-header"><h1>Exploratory Data Analysis</h1><p>Interactive dashboard for data insights and model visualization</p></div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found.")
    else:
        eda_df = prepare_eda_df(df)

        tabs = st.tabs([
            "Stress Risk Distribution",
            "Mood Swings",
            "Days Indoors",
            "Social Weakness",
            "Occupation",
            "Correlation Heatmap",
            "Decision Tree Visualization"
        ])

        with tabs[0]:
            st.markdown('<div class="section-header"><h2>Stress Risk Distribution</h2></div>', unsafe_allow_html=True)
            counts = eda_df["Stress_Risk"].value_counts()
            fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.55, textinfo="percent+label")])
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.write("Insight: After removing `Maybe`, the problem becomes a clean binary classification task.")

        with tabs[1]:
            st.markdown('<div class="section-header"><h2>Mood Swings Analysis</h2></div>', unsafe_allow_html=True)
            stacked_plot(eda_df, "Mood_Swings", "Stress Risk by Mood Swings", ["Low", "Medium", "High"])
            st.write("Insight: Higher mood swings are associated with higher stress risk.")

        with tabs[2]:
            st.markdown('<div class="section-header"><h2>Days Indoors Analysis</h2></div>', unsafe_allow_html=True)
            stacked_plot(eda_df, "Days_Indoors", "Stress Risk by Days Indoors", ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"])
            st.write("Insight: More time indoors appears linked with higher stress risk.")

        with tabs[3]:
            st.markdown('<div class="section-header"><h2>Social Weakness Analysis</h2></div>', unsafe_allow_html=True)
            stacked_plot(eda_df, "Social_Weakness", "Stress Risk by Social Weakness", ["No", "Maybe", "Yes"])
            st.write("Insight: Social weakness is associated with a higher share of at-risk cases.")

        with tabs[4]:
            st.markdown('<div class="section-header"><h2>Occupation Analysis</h2></div>', unsafe_allow_html=True)
            stacked_plot(eda_df, "Occupation", "Stress Risk by Occupation")
            st.write("Insight: Different occupation groups show different stress-risk patterns.")

        with tabs[5]:
            st.markdown('<div class="section-header"><h2>Correlation Heatmap</h2></div>', unsafe_allow_html=True)
            corr_df = eda_df.drop(columns=["Timestamp", "Growing_Stress"], errors="ignore").copy()
            for col in corr_df.columns:
                if corr_df[col].dtype == "object":
                    corr_df[col] = corr_df[col].astype("category").cat.codes
            corr = corr_df.corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
            st.write("Insight: Mood_Swings, Days_Indoors, and Social_Weakness show meaningful association patterns with stress risk.")

        with tabs[6]:
            st.markdown('<div class="section-header"><h2>Decision Tree Visualization</h2></div>', unsafe_allow_html=True)

            if dt_model is None:
                st.error("Decision Tree model not loaded.")
            else:
                fi = getattr(dt_model, "feature_importances_", None)
                if fi is not None and dt_feature_columns is not None:
                    st.markdown('<div class="section-header"><h2>Feature Importance</h2></div>', unsafe_allow_html=True)
                    fi_df = pd.DataFrame({"Feature": dt_feature_columns, "Importance": fi}).sort_values("Importance", ascending=True)
                    fig = go.Figure(go.Bar(
                        x=fi_df["Importance"],
                        y=fi_df["Feature"],
                        orientation="h",
                        text=[f"{v:.3f}" for v in fi_df["Importance"]],
                        textposition="outside"
                    ))
                    fig.update_layout(height=420, title="Decision Tree Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

                depth = st.slider("Tree depth to display", 1, 6, 3)
                fig, ax = plt.subplots(figsize=(20, 8))
                plot_tree(
                    dt_model,
                    feature_names=dt_feature_columns,
                    class_names=model_results.get("class_labels", ["At Risk", "Not At Risk"]) if model_results else ["At Risk", "Not At Risk"],
                    filled=True,
                    rounded=True,
                    max_depth=depth,
                    fontsize=9,
                    ax=ax,
                    impurity=False
                )
                st.pyplot(fig)

# ---------------------------------------------------
# PAGE: Preprocessing
# ---------------------------------------------------
elif page == "Preprocessing":
    st.markdown('<div class="main-header"><h1>Data Cleaning & Preprocessing</h1></div>', unsafe_allow_html=True)
    st.write("""
- Removed `Maybe` responses from the target
- Dropped unnecessary columns like `Timestamp`, `Growing_Stress`, and `Country` if present
- Filled missing values with mode
- Removed duplicates
- Label encoded categorical variables
- Selected 8 final features
- Split the data into train and test sets
""")

# ---------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="main-header"><h1>Model Performance</h1></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <strong>Why Recall Matters</strong><br><br>
        This project prioritizes <strong>Recall</strong> because a false negative means an at-risk person may be missed.
        In a mental health setting, that is more serious than a false positive, where someone simply receives an extra follow-up.
        For that reason, the final model is selected based primarily on its ability to catch as many at-risk individuals as possible.
    </div>
    """, unsafe_allow_html=True)

    if model_results is None:
        st.error("Model results not found.")
    else:
        best_model = model_results.get("best_model", "Decision Tree")
        if best_model == "Decision Tree":
            best_recall = model_results.get("dt_recall", 0)
            best_f1 = model_results.get("dt_f1", 0)
        else:
            best_recall = model_results.get("nb_recall", 0)
            best_f1 = model_results.get("nb_f1", 0)

        k1, k2, k3 = st.columns(3)
        k1.metric("Selected Model", best_model)
        k2.metric("Best Recall", f"{best_recall:.1%}")
        k3.metric("Best F1", f"{best_f1:.1%}")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Decision Tree": [
                model_results.get("dt_accuracy", 0),
                model_results.get("dt_precision", 0),
                model_results.get("dt_recall", 0),
                model_results.get("dt_f1", 0)
            ],
            "Naive Bayes": [
                model_results.get("nb_accuracy", 0),
                model_results.get("nb_precision", 0),
                model_results.get("nb_recall", 0),
                model_results.get("nb_f1", 0)
            ]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics_df["Metric"], y=metrics_df["Decision Tree"], name="Decision Tree", marker_color="#2d5a87"))
        fig.add_trace(go.Bar(x=metrics_df["Metric"], y=metrics_df["Naive Bayes"], name="Naive Bayes", marker_color="#38a169"))
        fig.update_layout(barmode="group", height=420, yaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <strong>Model selection summary:</strong><br><br>
            The final model is chosen based on the highest <strong>Recall</strong>.
            This ensures the app prioritizes catching more truly at-risk individuals, even if that means allowing some extra false positives.
        </div>
        """, unsafe_allow_html=True)

        st.write("Best model based on Recall:", best_model)

        st.markdown('<div class="section-header"><h2>Confusion Matrices</h2></div>', unsafe_allow_html=True)
        dt_cm = model_results.get("dt_cm", [[0, 0], [0, 0]])
        nb_cm = model_results.get("nb_cm", [[0, 0], [0, 0]])

        c1, c2 = st.columns(2)
        with c1:
            fig_dt, ax_dt = plt.subplots(figsize=(5, 4))
            sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_dt)
            ax_dt.set_title("Decision Tree")
            st.pyplot(fig_dt)

        with c2:
            fig_nb, ax_nb = plt.subplots(figsize=(5, 4))
            sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_nb)
            ax_nb.set_title("Naive Bayes")
            st.pyplot(fig_nb)

# ---------------------------------------------------
# PAGE: Prediction Demo
# ---------------------------------------------------
elif page == "Prediction Demo":
    st.markdown('<div class="main-header"><h1>Prediction Demo</h1></div>', unsafe_allow_html=True)

    if dt_model is None or nb_model is None or dt_encoders is None:
        st.error("Models or encoders not loaded.")
    else:
        st.markdown("""
        <div class="info-card">
        <strong>Stress Risk Screening</strong><br><br>
        Answer the questions below about mood, habits, and daily life.
        The model will estimate whether the responses indicate a higher risk of stress.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header"><h2>Choose model for prediction</h2></div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Model choice",
            ["Decision Tree", "Naive Bayes", "Compare Both"],
            index=2
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", get_encoder_classes(dt_encoders, "Gender", ["Male", "Female"]))
            occupation = st.selectbox("Occupation", get_encoder_classes(dt_encoders, "Occupation", ["Student", "Corporate", "Business", "Housewife", "Others"]))
            days_indoors = st.selectbox("Days Indoors", get_encoder_classes(dt_encoders, "Days_Indoors", ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"]))
        with c2:
            mood = st.selectbox("Mood Swings", get_encoder_classes(dt_encoders, "Mood_Swings", ["Low", "Medium", "High"]))
            social = st.selectbox("Social Weakness", get_encoder_classes(dt_encoders, "Social_Weakness", ["Yes", "No", "Maybe"]))
            changes = st.selectbox("Changes in Habits", get_encoder_classes(dt_encoders, "Changes_Habits", ["Yes", "No", "Maybe"]))
        with c3:
            work_interest = st.selectbox("Work Interest", get_encoder_classes(dt_encoders, "Work_Interest", ["Yes", "No", "Maybe"]))
            history = st.selectbox("Mental Health History", get_encoder_classes(dt_encoders, "Mental_Health_History", ["Yes", "No", "Maybe"]))

        st.divider()
        predict_button = st.button("Run Stress Risk Prediction", use_container_width=True)

        if predict_button:
            input_data = {
                "Mood_Swings": mood,
                "Days_Indoors": days_indoors,
                "Occupation": occupation,
                "Social_Weakness": social,
                "Changes_Habits": changes,
                "Gender": gender,
                "Work_Interest": work_interest,
                "Mental_Health_History": history
            }

            class_labels = model_results.get("class_labels", ["At Risk", "Not At Risk"]) if model_results else ["At Risk", "Not At Risk"]
            dt_input = build_model_input(input_data, dt_feature_columns, dt_encoders)
            nb_input = build_model_input(input_data, nb_feature_columns, nb_encoders)

            with st.spinner("Analyzing responses..."):
                dt_label = pred_to_label(dt_model.predict(dt_input)[0], class_labels)
                nb_label = pred_to_label(nb_model.predict(nb_input)[0], class_labels)

            if model_choice == "Decision Tree":
                show_prediction_card("Decision Tree", dt_label)
            elif model_choice == "Naive Bayes":
                show_prediction_card("Naive Bayes", nb_label)
            else:
                a, b = st.columns(2)
                with a:
                    show_prediction_card("Decision Tree", dt_label)
                with b:
                    show_prediction_card("Naive Bayes", nb_label)

            st.info("This is an educational screening demo and not a medical diagnosis.")
