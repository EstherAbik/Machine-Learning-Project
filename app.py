import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Stress Trajectory Prediction | Mental Health",
    page_icon="🧠💚",
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
st.sidebar.markdown("## 🧠💚 Stress Trajectory")
st.sidebar.markdown("*Mental Health Risk Assessment*")
st.sidebar.markdown("---")
section = st.sidebar.radio("Navigate:", [
    "1. Introduction",
    "2. EDA Visualizations",
    "3. Feature Selection",
    "4. Model Performance",
    "5. Make Prediction"
])

# ============== SECTION 1: INTRODUCTION ==============
if section == "1. Introduction":
    st.markdown("""
    <div style="text-align: center;">
        <h1>🧠💚 STRESS TRAJECTORY PREDICTION</h1>
        <h3><em>A Machine Learning Approach to Mental Health Risk Assessment</em></h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mental health awareness banner
    st.success("💚 **Mental Health Matters** | 🤖 **AI for Good** | 🛡️ **Early Intervention**")
    
    st.markdown("""
    > *"Mental health is not a destination, but a process. It's about how you drive, not where you're going."* — Noam Shpancer
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎯 Project Mission")
        st.write("""
        This project uses **Machine Learning** to identify individuals who may be 
        at risk of developing chronic stress, enabling **early intervention** and 
        support for better mental health outcomes.
        
        | Aspect | Description |
        |--------|-------------|
        | **Goal** | Predict stress trajectory risk |
        | **Target** | At Risk 🔴 vs Not At Risk 🟢 |
        | **Metric** | **Recall** - Catch all at-risk cases |
        """)
        
    with col2:
        st.header("💡 Why This Matters")
        st.info("""
        🔹 **1 in 5** adults experience mental health challenges yearly  
        🔹 **Stress** is the #1 trigger for anxiety & depression  
        🔹 **Early detection** prevents escalation to severe conditions  
        🔹 **AI screening** enables scalable mental health support
        """)
        
        st.header("⚠️ Why We Prioritize RECALL")
        st.warning("""
        **Missing someone who needs help is far worse than a false alarm.**
        
        | Error | Impact |
        |-------|--------|
        | **False Negative** ❌ | Missed at-risk → No help → Crisis |
        | **False Positive** ⚡ | Extra check-in → Still helpful |
        """)

# ============== SECTION 2: EDA VISUALIZATIONS ==============
elif section == "2. EDA Visualizations":
    st.markdown("""
    <div style="text-align: center;">
        <h1>📊 Exploratory Data Analysis</h1>
        <h4><em>Understanding the Mental Health Landscape</em></h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.info("🔍 Exploring how different factors relate to stress risk in our mental health dataset.")
    
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
    st.markdown("""
    <div style="text-align: center;">
        <h1>🎯 Feature Selection</h1>
        <h4><em>Identifying Key Mental Health Indicators</em></h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.info("🔬 Statistical analysis to find which factors best predict stress risk.")
    
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
    st.markdown("""
    <div style="text-align: center;">
        <h1>🏆 Model Performance</h1>
        <h4><em>Selecting the Best Mental Health Screening Model</em></h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.info("🥇 Comparing models to find the best approach for identifying at-risk individuals.")
    
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
    st.markdown("""
    <div style="text-align: center;">
        <h1>🔮 Predict Stress Trajectory</h1>
        <h4><em>Mental Health Risk Assessment Tool</em></h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    if not models_loaded:
        st.error("⚠️ Models not found! Run the notebook first.")
    else:
        st.info("💚 Answer these questions to assess stress risk. This is a screening tool, not a diagnosis.")
        st.markdown("")
        
        model_choice = st.selectbox("Choose a model:", ["Decision Tree", "Naive Bayes"])
        
        st.markdown("### 📋 Mental Health Assessment Questions")
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
            
            # Mental health resources
            st.markdown("---")
            st.markdown("### 💚 Mental Health Resources")
            st.markdown("""
            *If you or someone you know is struggling, help is available.*
            
            🇺🇸 **National Suicide Prevention Lifeline:** 988  
            🌍 **Crisis Text Line:** Text HOME to 741741  
            
            *"It's okay to not be okay. What matters is that you seek help."* 💚
            """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 💚 Mental Health Matters")
st.sidebar.caption("Stress Trajectory Prediction App")
st.sidebar.markdown("*Early intervention saves lives.*")
