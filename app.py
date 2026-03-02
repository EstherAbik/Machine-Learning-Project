import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Stress Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# Load models and objects
@st.cache_resource
def load_models():
    dt_model = joblib.load('models/decision_tree_model.pkl')
    nb_model = joblib.load('models/naive_bayes_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    minmax_scaler = joblib.load('models/minmax_scaler.pkl')  # For Naive Bayes
    encoders = joblib.load('models/encoders.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    results = joblib.load('models/model_results.pkl')
    return dt_model, nb_model, scaler, minmax_scaler, encoders, feature_columns, results

# Try to load models
try:
    dt_model, nb_model, scaler, minmax_scaler, encoders, feature_columns, results = load_models()
    models_loaded = True
except:
    models_loaded = False
    dt_model = nb_model = scaler = minmax_scaler = encoders = feature_columns = results = None

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to:", [
    "1. Introduction",
    "2. EDA Visualizations",
    "3. Feature Selection",
    "4. Model Performance",
    "5. Make Prediction"
])

# ============== SECTION 1: INTRODUCTION ==============
if section == "1. Introduction":
    st.title("🧠 Stress Risk Prediction")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📌 Project Overview")
        st.write("""
        This app predicts if someone is **at risk of growing stress**.
        
        **Task:** Binary Classification  
        **Target:** Stress Risk (At Risk / Not At Risk)
        """)
        
        st.header("🎯 Goal")
        st.write("Identify at-risk individuals early for intervention.")
    
    with col2:
        st.header("⚠️ Why Recall Matters")
        st.warning("""
        **Missing at-risk people is costly!**
        
        - False Negative = missed someone who needs help
        - We prioritize **RECALL** to catch all at-risk cases
        """)
        
        st.header("📊 Evaluation Priority")
        st.info("""
        1. **Recall** (catch all at-risk cases)
        2. **F1-Score** (balance)
        3. **Accuracy** (overall)
        """)

# ============== SECTION 2: EDA VISUALIZATIONS ==============
elif section == "2. EDA Visualizations":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    # Display saved plots if available
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        try:
            st.image('class_distribution.png')
        except:
            st.info("Run the notebook to generate visualizations")
    
    with col2:
        st.subheader("Correlation Heatmap")
        try:
            st.image('correlation_heatmap.png')
        except:
            st.info("Run the notebook to generate visualizations")
    
    st.subheader("Feature Relationships")
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image('eda_mood_swings.png')
            st.image('eda_days_indoors.png')
        except:
            st.info("Run the notebook to generate visualizations")
    with col2:
        try:
            st.image('eda_coping_struggles.png')
            st.image('eda_social_weakness.png')
        except:
            st.info("Run the notebook to generate visualizations")

# ============== SECTION 3: FEATURE SELECTION ==============
elif section == "3. Feature Selection":
    st.title("🎯 Feature Selection")
    st.markdown("---")
    
    if models_loaded:
        st.header("Selected Features")
        st.write(f"**{len(feature_columns)} features used:**")
        st.write(feature_columns)
    else:
        st.warning("Run notebook first to see features")
    
    st.subheader("Feature Importance")
    try:
        st.image('feature_importance.png')
    except:
        st.info("Run notebook first")
    
    st.header("Selection Method")
    st.write("""
    1. Dropped Timestamp and Country (not useful)
    2. Chi-Square tests for significance
    3. Decision Tree importance scores
    """)

# ============== SECTION 4: MODEL PERFORMANCE ==============
elif section == "4. Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")
    
    if not models_loaded:
        st.error("⚠️ Models not found! Run the notebook first.")
    else:
        # Metrics table
        st.header("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall ⬅️', 'F1-Score'],
            'Decision Tree': [
                f"{results['dt_accuracy']:.4f}",
                f"{results['dt_precision']:.4f}",
                f"{results['dt_recall']:.4f}",
                f"{results['dt_f1']:.4f}"
            ],
            'Naive Bayes': [
                f"{results['nb_accuracy']:.4f}",
                f"{results['nb_precision']:.4f}",
                f"{results['nb_recall']:.4f}",
                f"{results['nb_f1']:.4f}"
            ]
        })
        st.table(metrics_df)
        
        # Confusion Matrices
        st.header("Confusion Matrices")
        col1, col2 = st.columns(2)
        
        # Get class labels from saved results
        class_labels = results.get('class_labels', ['At Risk', 'Not At Risk'])
        
        with col1:
            st.subheader("Decision Tree")
            cm_dt = np.array(results['dt_cm'])
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax1)
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            st.pyplot(fig1)
            
            # Binary: use ravel() for 2x2 matrix
            tn, fp, fn, tp = cm_dt.ravel()
            recall_dt = tp / (tp + fn) * 100
            st.success(f"✅ Caught {recall_dt:.1f}% of at-risk individuals")
            st.error(f"⚠️ Missed {fn:,} cases (False Negatives)")
        
        with col2:
            st.subheader("Naive Bayes")
            cm_nb = np.array(results['nb_cm'])
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax2)
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            st.pyplot(fig2)
            
            tn, fp, fn, tp = cm_nb.ravel()
            recall_nb = tp / (tp + fn) * 100
            st.success(f"✅ Caught {recall_nb:.1f}% of at-risk individuals")
            st.error(f"⚠️ Missed {fn:,} cases (False Negatives)")
        
        # Best model
        st.header("🏆 Selected Model")
        st.info(f"**{results['best_model']}** - Higher Recall (catches more at-risk cases)")

# ============== SECTION 5: MAKE PREDICTION ==============
elif section == "5. Make Prediction":
    st.title("🔮 Predict Stress Risk")
    st.markdown("---")
    
    if not models_loaded:
        st.error("⚠️ Models not found! Run the notebook first.")
    else:
        st.write("**Answer these questions to predict if someone is at risk of stress:**")
        st.markdown("")
        
        model_choice = st.selectbox("Choose a model:", ["Decision Tree", "Naive Bayes"])
        
        st.markdown("### 📋 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("What is your gender?", ["Male", "Female"])
            occupation = st.selectbox("What is your occupation?", ["Corporate", "Student", "Business", "Housewife", "Others"])
            self_employed = st.selectbox("Are you self-employed?", ["No", "Yes"])
            family_history = st.selectbox("Does your family have mental health history?", ["No", "Yes"])
            treatment = st.selectbox("Are you currently receiving treatment?", ["No", "Yes"])
            days_indoors = st.selectbox("How long do you stay indoors?", ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"])
            changes_habits = st.selectbox("Have your habits changed recently?", ["No", "Yes"])
        
        with col2:
            mental_health_history = st.selectbox("Do you have mental health history?", ["No", "Yes"])
            mood_swings = st.selectbox("How often do you have mood swings?", ["Low", "Medium", "High"])
            coping_struggles = st.selectbox("Do you struggle to cope with problems?", ["No", "Yes"])
            work_interest = st.selectbox("Do you have interest in work?", ["Yes", "No"])
            social_weakness = st.selectbox("Do you feel socially weak?", ["No", "Yes"])
            mental_health_interview = st.selectbox("Would you discuss mental health in interview?", ["Yes", "No", "Maybe"])
            care_options = st.selectbox("Are you aware of care options?", ["Yes", "No", "Not sure"])
        
        st.markdown("---")
        
        if st.button("🔮 Get My Prediction", type="primary", use_container_width=True):
            # Create input dict
            input_data = {
                'Gender': gender,
                'Occupation': occupation,
                'self_employed': self_employed,
                'family_history': family_history,
                'treatment': treatment,
                'Days_Indoors': days_indoors,
                'Changes_Habits': changes_habits,
                'Mental_Health_History': mental_health_history,
                'Mood_Swings': mood_swings,
                'Coping_Struggles': coping_struggles,
                'Work_Interest': work_interest,
                'Social_Weakness': social_weakness,
                'mental_health_interview': mental_health_interview,
                'care_options': care_options
            }
            
            # Encode input
            input_encoded = []
            for col in feature_columns:
                if col in encoders and col in input_data:
                    try:
                        val = encoders[col].transform([input_data[col]])[0]
                    except:
                        val = 0
                    input_encoded.append(val)
                else:
                    input_encoded.append(0)
            
            input_array = np.array(input_encoded).reshape(1, -1)
            
            # Predict
            if model_choice == "Decision Tree":
                prediction = dt_model.predict(input_array)[0]
                proba = dt_model.predict_proba(input_array)[0]
            else:
                # Naive Bayes uses MinMax scaler (for ComplementNB)
                input_scaled = minmax_scaler.transform(input_array)
                prediction = nb_model.predict(input_scaled)[0]
                proba = nb_model.predict_proba(input_scaled)[0]
            
            # Get result
            class_labels = results.get('class_labels', ['At Risk', 'Not At Risk'])
            pred_label = class_labels[prediction]
            confidence = max(proba) * 100
            
            # Display result - SIMPLE AND CLEAR
            st.markdown("---")
            st.markdown("## 📊 Your Result")
            
            if pred_label == "At Risk":
                st.error("## ⚠️ AT RISK")
                st.markdown("""
                **What this means:**  
                Based on your answers, you may be at risk of growing stress.
                
                **What to do:**
                - Consider talking to a mental health professional
                - Practice stress management techniques
                - Reach out to supportive friends or family
                """)
            else:
                st.success("## ✅ NOT AT RISK")
                st.markdown("""
                **What this means:**  
                Based on your answers, you appear to have low stress risk.
                
                **Keep it up:**
                - Continue healthy habits
                - Maintain work-life balance
                - Stay connected with others
                """)
            
            # Simple confidence display
            st.metric(label="Confidence Level", value=f"{confidence:.0f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Stress Risk Prediction App")
