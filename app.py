
"""
Stress Trajectory Prediction - Mental Health Analysis Platform

This app includes:
• Dataset link + clear problem statement
• Dataset overview
• EDA with organized visualizations and insights
• Data cleaning & preprocessing explanation
• Model training + hyperparameter tuning summary
• Model comparison
• Final model selection
• Live prediction demo
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
    page_title="Stress Trajectory Prediction",
    page_icon="🧠",
    layout="wide"
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
    background: linear-gradient(135deg,#1e3a5f,#3d7ab3);
    padding:2rem;
    border-radius:14px;
    margin-bottom:2rem;
}
.main-header h1 {
    color:white;
    text-align:center;
}
.section-header {
    background:#f2f5f9;
    padding:0.7rem 1rem;
    border-left:5px solid #3d7ab3;
    border-radius:6px;
    margin-top:2rem;
}
.metric-box {
    background:white;
    padding:1rem;
    border-radius:10px;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return None

@st.cache_resource
def load_models():
    try:
        dt = joblib.load(MODELS_DIR / "decision_tree_model.pkl")
        nb = joblib.load(MODELS_DIR / "naive_bayes_model.pkl")
        enc = joblib.load(MODELS_DIR / "encoders.pkl")
        features = joblib.load(MODELS_DIR / "feature_columns.pkl")
        results = joblib.load(MODELS_DIR / "model_results.pkl")
        return dt, nb, enc, features, results
    except:
        return None,None,None,None,None

df = load_data()
dt_model, nb_model, encoders, feature_columns, model_results = load_models()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
page = st.sidebar.radio(
"Navigation",
[
"Problem Statement",
"Dataset Overview",
"EDA & Insights",
"Preprocessing",
"Model Training",
"Model Comparison",
"Final Model",
"Prediction Demo"
]
)

# ---------------------------------------------------
# PAGE: PROBLEM
# ---------------------------------------------------
if page == "Problem Statement":

    st.markdown('<div class="main-header"><h1>Stress Trajectory Prediction</h1></div>', unsafe_allow_html=True)

    st.markdown("### Dataset")
    st.write("Mental Health Dataset from Kaggle.")
    st.link_button("Open Dataset", DATASET_URL)

    st.markdown("### Objective")
    st.write("Predict whether a person is **At Risk** of growing stress.")

    st.markdown("### Problem Type")
    st.write("Binary Classification")

    st.markdown("### Target Variable")
    st.write("Stress_Risk → At Risk / Not At Risk")

    st.markdown("### Why This Matters")
    st.write(
    """
    Early detection of stress risk allows earlier support and intervention.
    Machine learning can help identify patterns linked with higher stress levels.
    """
    )

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------
elif page == "Dataset Overview":

    st.markdown('<div class="main-header"><h1>Dataset Overview</h1></div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found.")
    else:

        c1,c2,c3 = st.columns(3)

        with c1:
            st.metric("Rows", len(df))

        with c2:
            st.metric("Columns", len(df.columns))

        with c3:
            st.metric("Selected Features", 8)

        st.dataframe(df.head())

# ---------------------------------------------------
# EDA
# ---------------------------------------------------
elif page == "EDA & Insights":

    st.markdown('<div class="main-header"><h1>Exploratory Data Analysis</h1></div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset missing")
    else:

        df["Stress_Risk"] = df["Growing_Stress"].apply(
            lambda x: "At Risk" if x=="Yes" else ("Not At Risk" if x=="No" else "Maybe")
        )

        df = df[df["Stress_Risk"]!="Maybe"]

        st.markdown('<div class="section-header"><b>Stress Risk Distribution</b></div>', unsafe_allow_html=True)

        counts = df["Stress_Risk"].value_counts()

        fig = go.Figure(data=[go.Bar(
            x=counts.index,
            y=counts.values
        )])

        st.plotly_chart(fig,use_container_width=True)

        st.markdown("Insight: Class distribution shows the proportion of individuals at stress risk.")

        # Mood swings
        st.markdown('<div class="section-header"><b>Mood Swings vs Stress</b></div>', unsafe_allow_html=True)

        ct = pd.crosstab(df["Mood_Swings"],df["Stress_Risk"],normalize="index")

        fig = go.Figure()
        fig.add_bar(x=ct.index,y=ct["At Risk"],name="At Risk")
        fig.add_bar(x=ct.index,y=ct["Not At Risk"],name="Not At Risk")

        st.plotly_chart(fig,use_container_width=True)

        st.markdown("Insight: Higher mood swings correlate with higher stress risk.")

        # days indoors
        st.markdown('<div class="section-header"><b>Days Indoors vs Stress</b></div>', unsafe_allow_html=True)

        ct = pd.crosstab(df["Days_Indoors"],df["Stress_Risk"],normalize="index")

        fig = go.Figure()
        fig.add_bar(x=ct.index,y=ct["At Risk"],name="At Risk")
        fig.add_bar(x=ct.index,y=ct["Not At Risk"],name="Not At Risk")

        st.plotly_chart(fig,use_container_width=True)

        st.markdown("Insight: Spending more days indoors is associated with increased stress risk.")

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
elif page == "Preprocessing":

    st.markdown('<div class="main-header"><h1>Data Cleaning & Preprocessing</h1></div>', unsafe_allow_html=True)

    st.write("""
    Steps performed:

    • Removed **Timestamp** column  
    • Removed duplicate rows  
    • Created binary target variable **Stress_Risk**  
    • Removed **Maybe** rows  
    • Applied **Label Encoding** to categorical variables  
    • Selected 8 key predictive features  
    • Performed **Train/Test split**
    """)

# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------
elif page == "Model Training":

    st.markdown('<div class="main-header"><h1>Model Training & Hyperparameter Tuning</h1></div>', unsafe_allow_html=True)

    st.write("Two machine learning models were trained:")

    st.write("• Decision Tree")
    st.write("• Naive Bayes")

    st.write("Hyperparameter tuning performed using GridSearchCV.")

# ---------------------------------------------------
# COMPARISON
# ---------------------------------------------------
elif page == "Model Comparison":

    st.markdown('<div class="main-header"><h1>Model Performance Comparison</h1></div>', unsafe_allow_html=True)

    if model_results:

        data = {
        "Metric":["Accuracy","Precision","Recall","F1"],
        "Decision Tree":[
        model_results["dt_accuracy"],
        model_results["dt_precision"],
        model_results["dt_recall"],
        model_results["dt_f1"]
        ],
        "Naive Bayes":[
        model_results["nb_accuracy"],
        model_results["nb_precision"],
        model_results["nb_recall"],
        model_results["nb_f1"]
        ]
        }

        df_metrics = pd.DataFrame(data)

        fig = go.Figure()

        fig.add_bar(
        x=df_metrics["Metric"],
        y=df_metrics["Decision Tree"],
        name="Decision Tree"
        )

        fig.add_bar(
        x=df_metrics["Metric"],
        y=df_metrics["Naive Bayes"],
        name="Naive Bayes"
        )

        st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------
# FINAL MODEL
# ---------------------------------------------------
elif page == "Final Model":

    st.markdown('<div class="main-header"><h1>Final Model Selection</h1></div>', unsafe_allow_html=True)

    if model_results:
        st.write("Selected Model:",model_results["best_model"])
        st.write("Reason: Highest Recall, which is critical for identifying at-risk individuals.")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
elif page == "Prediction Demo":

    st.markdown('<div class="main-header"><h1>Stress Risk Prediction</h1></div>', unsafe_allow_html=True)

    if not dt_model:
        st.error("Models not loaded.")
    else:

        gender = st.selectbox("Gender",["Male","Female"])
        mood = st.selectbox("Mood Swings",["Low","Medium","High"])
        indoors = st.selectbox("Days Indoors",[
        "Go out Every day","1-14 days","15-30 days","31-60 days","More than 2 months"
        ])
        occupation = st.selectbox("Occupation",["Student","Corporate","Business","Housewife","Others"])
        social = st.selectbox("Social Weakness",["Yes","No","Maybe"])
        habits = st.selectbox("Changes Habits",["Yes","No","Maybe"])
        work = st.selectbox("Work Interest",["Yes","No","Maybe"])
        history = st.selectbox("Mental Health History",["Yes","No","Maybe"])

        if st.button("Predict"):

            input_data = pd.DataFrame([{
            "Mood_Swings":mood,
            "Days_Indoors":indoors,
            "Occupation":occupation,
            "Social_Weakness":social,
            "Changes_Habits":habits,
            "Gender":gender,
            "Work_Interest":work,
            "Mental_Health_History":history
            }])

            for col in input_data.columns:
                if col in encoders:
                    le = encoders[col]
                    input_data[col] = le.transform(input_data[col])

            pred = dt_model.predict(input_data)[0]

            label = model_results["class_labels"][pred]

            st.success(f"Prediction: {label}")
