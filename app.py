"""
Stress Trajectory Prediction - Mental Health Analysis Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Stress Trajectory Prediction | Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL CSS STYLING
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
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, white 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2d5a87;
        margin: 2rem 0 1.5rem 0;
    }
    
    .section-header h2 {
        color: #1e3a5f;
        font-weight: 600;
        margin: 0;
        font-size: 1.5rem;
    }
    
    .info-card {
        background: linear-gradient(145deg, #e8f4fd 0%, #d4e8f7 100%);
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid #2d5a87;
        margin: 1rem 0;
    }
    
    .info-card p { margin: 0; color: #1e3a5f; font-size: 0.95rem; line-height: 1.6; }
    
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


# LOAD MODELS AND DATA
@st.cache_resource
def load_models():
    try:
        dt_model = joblib.load('models/decision_tree_model.pkl')
        nb_model = joblib.load('models/naive_bayes_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        minmax_scaler = joblib.load('models/minmax_scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        model_results = joblib.load('models/model_results.pkl')
        return dt_model, nb_model, scaler, minmax_scaler, encoders, feature_columns, model_results
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None


@st.cache_data
def load_data():
    try:
        return pd.read_csv('Mental Health Dataset.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# SIDEBAR NAVIGATION
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
        ["🏠 Problem Statement", "📊 EDA & Insights", "🔧 Methodology", "🎯 Model Performance", "🏆 Final Results", "🔮 Risk Prediction"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(145deg, #e8f4fd 0%, #d4e8f7 100%); 
                padding: 1rem; border-radius: 12px; margin-top: 1rem;">
        <p style="font-size: 0.85rem; color: #1e3a5f; margin: 0; text-align: center;">
            <strong>📋 Project Components</strong><br>
            ✅ Problem Statement<br>
            ✅ EDA & Visualizations<br>
            ✅ Data Preprocessing<br>
            ✅ 2 ML Algorithms<br>
            ✅ Hyperparameter Tuning<br>
            ✅ Model Evaluation<br>
            ✅ Final Model Selection
        </p>
    </div>
    """, unsafe_allow_html=True)


# Load data and models
df = load_data()
dt_model, nb_model, scaler, minmax_scaler, encoders, feature_columns, model_results = load_models()


# =====================================================
# PAGE 1: PROBLEM STATEMENT
# =====================================================
if page == "🏠 Problem Statement":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Stress Trajectory Prediction</h1>
        <p>A Machine Learning Approach to Mental Health Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem Definition
    st.markdown('<div class="section-header"><h2>📌 Problem Definition</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">🎯 Classification Problem</h4>
            <p>
                <strong>Goal:</strong> Predict whether an individual is at risk of developing growing stress<br><br>
                <strong>Type:</strong> Binary Classification<br><br>
                <strong>Target Variable:</strong> <code>Stress_Risk</code><br>
                • Class 1: "At Risk" 🔴<br>
                • Class 0: "Not At Risk" 🟢
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #fff5f5 0%, #ffe8e8 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid #e53e3e; margin: 1rem 0;">
            <h4 style="color: #c53030; margin-top: 0;">⚠️ Why We Prioritize RECALL</h4>
            <p style="color: #c53030;">
                In mental health screening, <strong>missing someone who needs help is far worse than a false alarm</strong>.<br><br>
                <strong>False Negative:</strong> Missed at-risk person → No intervention → Potential crisis ❌<br><br>
                <strong>False Positive:</strong> Extra check-in → Minor inconvenience → Still helpful ⚡
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Why This Matters
    st.markdown('<div class="section-header"><h2>� Why Mental Health Screening Matters</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">1 in 5</div><div class="metric-label">Adults with Mental Health Issues</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">#1</div><div class="metric-label">Stress as Depression Trigger</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">70%</div><div class="metric-label">Preventable with Early Detection</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">AI</div><div class="metric-label">Enables Scalable Screening</div></div>', unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown('<div class="section-header"><h2>📊 Dataset Overview</h2></div>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">ML Models</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">Recall</div><div class="metric-label">Primary Metric</div></div>', unsafe_allow_html=True)
        
        # Show sample data
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(5), use_container_width=True)


# =====================================================
# PAGE 2: EDA & INSIGHTS
# =====================================================
elif page == "📊 EDA & Insights":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Understanding Patterns in Mental Health Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        st.markdown('<div class="section-header"><h2>📋 Dataset Overview</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Records</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.isnull().sum().sum()}</div><div class="metric-label">Missing</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h2>📈 Visualizations</h2></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🎯 Target", "😰 Mood & Stress", "👥 Social Factors"])
        
        with tab1:
            if 'Stress_Risk' not in df.columns and 'Growing_Stress' in df.columns:
                df['Stress_Risk'] = df['Growing_Stress'].map({'Yes': 'At Risk', 'No': 'Not At Risk'})
            
            if 'Stress_Risk' in df.columns:
                risk_counts = df['Stress_Risk'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index, values=risk_counts.values,
                    hole=0.6, marker_colors=['#e53e3e', '#38a169'],
                    textinfo='percent+label'
                )])
                fig.update_layout(title="Stress Risk Distribution", height=400,
                                  annotations=[dict(text='Risk', x=0.5, y=0.5, font_size=16, showarrow=False)])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                if 'Mood_Swings' in df.columns and 'Stress_Risk' in df.columns:
                    cross_tab = pd.crosstab(df['Mood_Swings'], df['Stress_Risk'], normalize='index') * 100
                    fig = go.Figure()
                    for col in cross_tab.columns:
                        color = '#e53e3e' if col == 'At Risk' else '#38a169'
                        fig.add_trace(go.Bar(name=col, x=cross_tab.index, y=cross_tab[col], marker_color=color,
                                           text=[f'{v:.1f}%' for v in cross_tab[col]], textposition='inside'))
                    fig.update_layout(title="Mood Swings vs Stress Risk", barmode='stack', height=350, yaxis=dict(range=[0,100]))
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'Coping_Struggles' in df.columns and 'Stress_Risk' in df.columns:
                    cross_tab = pd.crosstab(df['Coping_Struggles'], df['Stress_Risk'], normalize='index') * 100
                    fig = go.Figure()
                    for col in cross_tab.columns:
                        color = '#e53e3e' if col == 'At Risk' else '#38a169'
                        fig.add_trace(go.Bar(name=col, x=cross_tab.index, y=cross_tab[col], marker_color=color,
                                           text=[f'{v:.1f}%' for v in cross_tab[col]], textposition='inside'))
                    fig.update_layout(title="Coping Struggles vs Stress Risk", barmode='stack', height=350, showlegend=False, yaxis=dict(range=[0,100]))
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                if 'social_weakness' in df.columns and 'Stress_Risk' in df.columns:
                    cross_tab = pd.crosstab(df['social_weakness'], df['Stress_Risk'], normalize='index') * 100
                    fig = go.Figure()
                    for col in cross_tab.columns:
                        color = '#e53e3e' if col == 'At Risk' else '#38a169'
                        fig.add_trace(go.Bar(name=col, x=cross_tab.index, y=cross_tab[col], marker_color=color,
                                           text=[f'{v:.1f}%' for v in cross_tab[col]], textposition='inside'))
                    fig.update_layout(title="Social Weakness vs Stress Risk", barmode='stack', height=350, yaxis=dict(range=[0,100]))
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'family_history' in df.columns and 'Stress_Risk' in df.columns:
                    cross_tab = pd.crosstab(df['family_history'], df['Stress_Risk'], normalize='index') * 100
                    fig = go.Figure()
                    for col in cross_tab.columns:
                        color = '#e53e3e' if col == 'At Risk' else '#38a169'
                        fig.add_trace(go.Bar(name=col, x=cross_tab.index, y=cross_tab[col], marker_color=color,
                                           text=[f'{v:.1f}%' for v in cross_tab[col]], textposition='inside'))
                    fig.update_layout(title="Family History vs Stress Risk", barmode='stack', height=350, showlegend=False, yaxis=dict(range=[0,100]))
                    st.plotly_chart(fig, use_container_width=True)


# MODEL PERFORMANCE PAGE
elif page == "🎯 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Model Performance</h1>
        <p>Comparing Machine Learning Model Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model_results:
        st.markdown('<div class="section-header"><h2>📊 Performance Comparison</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card" style="border-top: 4px solid #2d5a87;"><h3 style="color: #1e3a5f; text-align: center;">🌳 Decision Tree</h3></div>', unsafe_allow_html=True)
            dt = model_results['Decision Tree']
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{dt['Accuracy']:.2%}")
            c1.metric("Recall", f"{dt['Recall']:.2%}")
            c2.metric("Precision", f"{dt['Precision']:.2%}")
            c2.metric("F1 Score", f"{dt['F1']:.2%}")
        
        with col2:
            st.markdown('<div class="metric-card" style="border-top: 4px solid #38a169;"><h3 style="color: #1e3a5f; text-align: center;">📊 Naive Bayes</h3></div>', unsafe_allow_html=True)
            nb = model_results['Naive Bayes']
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{nb['Accuracy']:.2%}")
            c1.metric("Recall", f"{nb['Recall']:.2%}")
            c2.metric("Precision", f"{nb['Precision']:.2%}")
            c2.metric("F1 Score", f"{nb['F1']:.2%}")
        
        st.markdown('<div class="section-header"><h2>📈 Visual Comparison</h2></div>', unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Decision Tree': [dt['Accuracy'], dt['Precision'], dt['Recall'], dt['F1']],
            'Naive Bayes': [nb['Accuracy'], nb['Precision'], nb['Recall'], nb['F1']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Decision Tree', x=metrics_df['Metric'], y=metrics_df['Decision Tree'],
                            marker_color='#2d5a87', text=[f'{v:.2%}' for v in metrics_df['Decision Tree']], textposition='outside'))
        fig.add_trace(go.Bar(name='Naive Bayes', x=metrics_df['Metric'], y=metrics_df['Naive Bayes'],
                            marker_color='#38a169', text=[f'{v:.2%}' for v in metrics_df['Naive Bayes']], textposition='outside'))
        fig.update_layout(barmode='group', title='Model Comparison', yaxis=dict(range=[0, 1.1], tickformat='.0%'), height=450)
        st.plotly_chart(fig, use_container_width=True)


# RISK PREDICTION PAGE
elif page == "🔮 Risk Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Stress Risk Prediction</h1>
        <p>Get Personalized Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if dt_model and nb_model and encoders and feature_columns:
        st.markdown("""
        <div class="info-card">
            <p><strong>How it works:</strong> Enter characteristics below for instant stress risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h2>📝 Enter Information</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("👤 Gender", list(encoders.get('Gender', {'Male': 0, 'Female': 1}).keys()))
            occupation = st.selectbox("💼 Occupation", list(encoders.get('Occupation', {'Student': 0}).keys()))
            self_employed = st.selectbox("🏢 Self Employed", list(encoders.get('self_employed', {'Yes': 0, 'No': 1}).keys()))
            days_indoors = st.selectbox("🏠 Days Indoors", list(encoders.get('Days_Indoors', {'Go Out Every Day': 0}).keys()))
        
        with col2:
            mood_swings = st.selectbox("😰 Mood Swings", list(encoders.get('Mood_Swings', {'Low': 0, 'Medium': 1, 'High': 2}).keys()))
            coping_struggles = st.selectbox("💪 Coping Struggles", list(encoders.get('Coping_Struggles', {'Yes': 0, 'No': 1}).keys()))
            work_interest = st.selectbox("📋 Work Interest", list(encoders.get('Work_Interest', {'Yes': 0, 'No': 1}).keys()))
            social_weakness = st.selectbox("👥 Social Weakness", list(encoders.get('social_weakness', {'Yes': 0, 'No': 1}).keys()))
        
        with col3:
            family_history = st.selectbox("👨‍👩‍👧 Family History", list(encoders.get('family_history', {'Yes': 0, 'No': 1}).keys()))
            treatment = st.selectbox("💊 Treatment", list(encoders.get('treatment', {'Yes': 0, 'No': 1}).keys()))
            mental_health_history = st.selectbox("🧠 Mental Health History", list(encoders.get('mental_health_history', {'Yes': 0}).keys()))
            changes_habits = st.selectbox("🔄 Changes in Habits", list(encoders.get('Changes_Habits', {'Yes': 0, 'No': 1}).keys()))
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔮 Analyze Risk", use_container_width=True)
        
        if predict_button:
            input_data = {
                'Gender': gender, 'Occupation': occupation, 'self_employed': self_employed,
                'Days_Indoors': days_indoors, 'Mood_Swings': mood_swings, 'Coping_Struggles': coping_struggles,
                'Work_Interest': work_interest, 'social_weakness': social_weakness, 'family_history': family_history,
                'treatment': treatment, 'mental_health_history': mental_health_history, 'Changes_Habits': changes_habits
            }
            
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if col in encoders:
                    try:
                        input_df[col] = input_df[col].map(encoders[col])
                    except:
                        input_df[col] = 0
            
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_columns]
            
            input_scaled_dt = scaler.transform(input_df)
            input_scaled_nb = minmax_scaler.transform(input_df)
            
            dt_pred = dt_model.predict(input_scaled_dt)[0]
            nb_pred = nb_model.predict(input_scaled_nb)[0]
            
            dt_label = "At Risk" if dt_pred == 1 else "Not At Risk"
            nb_label = "At Risk" if nb_pred == 1 else "Not At Risk"
            
            st.markdown('<div class="section-header"><h2>📊 Results</h2></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                css_class = "prediction-at-risk" if dt_label == "At Risk" else "prediction-not-at-risk"
                icon = "⚠️" if dt_label == "At Risk" else "✅"
                color = "#e53e3e" if dt_label == "At Risk" else "#38a169"
                st.markdown(f'<div class="{css_class}"><h3 style="color: {color};">🌳 Decision Tree</h3><div style="font-size: 3rem;">{icon}</div><h2 style="color: {color};">{dt_label}</h2></div>', unsafe_allow_html=True)
            
            with col2:
                css_class = "prediction-at-risk" if nb_label == "At Risk" else "prediction-not-at-risk"
                icon = "⚠️" if nb_label == "At Risk" else "✅"
                color = "#e53e3e" if nb_label == "At Risk" else "#38a169"
                st.markdown(f'<div class="{css_class}"><h3 style="color: {color};">📊 Naive Bayes</h3><div style="font-size: 3rem;">{icon}</div><h2 style="color: {color};">{nb_label}</h2></div>', unsafe_allow_html=True)
            
            if dt_label == nb_label:
                if dt_label == "At Risk":
                    st.markdown('<div style="background: #fff5f5; padding: 1.5rem; border-radius: 12px; border: 2px solid #e53e3e; text-align: center; margin-top: 1rem;"><h3 style="color: #e53e3e;">⚠️ Both Models: At Risk</h3><p>Consider consulting a mental health professional.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background: #f0fff4; padding: 1.5rem; border-radius: 12px; border: 2px solid #38a169; text-align: center; margin-top: 1rem;"><h3 style="color: #38a169;">✅ Both Models: Not At Risk</h3><p>Keep maintaining your mental wellness!</p></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card" style="margin-top: 2rem;">
            <h4 style="color: #1e3a5f;">🆘 Mental Health Resources</h4>
            <ul style="color: #1e3a5f;">
                <li><strong>National Suicide Prevention:</strong> 988</li>
                <li><strong>Crisis Text Line:</strong> Text HOME to 741741</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Models not loaded. Ensure model files are in 'models/' folder.")


# =====================================================
# PAGE 5: METHODOLOGY
# =====================================================
elif page == "🔧 Methodology":
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Methodology</h1>
        <p>Data Preprocessing & Machine Learning Pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Preprocessing Section
    st.markdown('<div class="section-header"><h2>🧹 Data Preprocessing</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">1️⃣ Data Cleaning</h4>
            <ul style="color: #6c757d; margin-bottom: 0;">
                <li>Removed <strong>Timestamp</strong> column (not predictive)</li>
                <li>Handled missing values</li>
                <li>Removed duplicate records</li>
                <li>Validated data types</li>
            </ul>
        </div>
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">3️⃣ Feature Scaling</h4>
            <ul style="color: #6c757d; margin-bottom: 0;">
                <li><strong>StandardScaler</strong> for Decision Tree</li>
                <li><strong>MinMaxScaler</strong> for Naive Bayes (non-negative values required)</li>
                <li>Ensures features contribute equally</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">2️⃣ Encoding Categorical Features</h4>
            <ul style="color: #6c757d; margin-bottom: 0;">
                <li><strong>Label Encoding</strong> for binary features</li>
                <li>Gender, Occupation, Self-Employed, etc.</li>
                <li>Converted text labels to numerical values</li>
            </ul>
        </div>
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">4️⃣ Train-Test Split</h4>
            <ul style="color: #6c757d; margin-bottom: 0;">
                <li><strong>80% Training</strong> / <strong>20% Testing</strong></li>
                <li>Stratified sampling to preserve class balance</li>
                <li><code>random_state=42</code> for reproducibility</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Class Imbalance Section
    st.markdown('<div class="section-header"><h2>⚖️ Handling Class Imbalance</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #fff5f5 0%, #ffe8e8 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid #e53e3e; margin: 1rem 0;">
            <h4 style="color: #c53030; margin-top: 0;">⚠️ The Problem</h4>
            <p style="color: #c53030;">
                Original dataset had <strong>imbalanced classes</strong>:<br>
                • More "Not At Risk" samples<br>
                • Models tend to predict majority class<br>
                • Poor recall on minority class (At Risk)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #f0fff4 0%, #c6f6d5 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid #38a169; margin: 1rem 0;">
            <h4 style="color: #276749; margin-top: 0;">✅ Our Solutions</h4>
            <p style="color: #276749;">
                <strong>Decision Tree:</strong> <code>class_weight='balanced'</code><br>
                • Automatically adjusts weights<br><br>
                <strong>Naive Bayes:</strong> <code>SMOTE</code> oversampling<br>
                • Synthetic Minority Over-sampling Technique
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ML Algorithms Section
    st.markdown('<div class="section-header"><h2>🤖 Machine Learning Algorithms</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #2d5a87;">
            <h3 style="color: #1e3a5f; text-align: center;">🌳 Decision Tree Classifier</h3>
            <hr style="border: 1px solid #e0e0e0;">
            <p style="color: #6c757d;">
                <strong>How it works:</strong><br>
                Creates a tree-like model of decisions based on feature values. 
                Splits data at each node to maximize information gain.
            </p>
            <p style="color: #6c757d;">
                <strong>Advantages:</strong><br>
                ✓ Easy to interpret and visualize<br>
                ✓ Handles non-linear relationships<br>
                ✓ No feature scaling required (but used for consistency)
            </p>
            <p style="color: #6c757d;">
                <strong>Key Hyperparameters Tuned:</strong><br>
                • <code>max_depth</code>: Tree depth limit<br>
                • <code>min_samples_split</code>: Min samples to split<br>
                • <code>class_weight='balanced'</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-top: 4px solid #38a169;">
            <h3 style="color: #1e3a5f; text-align: center;">📊 Complement Naive Bayes</h3>
            <hr style="border: 1px solid #e0e0e0;">
            <p style="color: #6c757d;">
                <strong>How it works:</strong><br>
                Probabilistic classifier using Bayes' theorem. 
                Complement NB is designed specifically for imbalanced datasets.
            </p>
            <p style="color: #6c757d;">
                <strong>Advantages:</strong><br>
                ✓ Fast training and prediction<br>
                ✓ Works well with small datasets<br>
                ✓ Better for imbalanced classes
            </p>
            <p style="color: #6c757d;">
                <strong>Key Hyperparameters Tuned:</strong><br>
                • <code>alpha</code>: Smoothing parameter<br>
                • <code>norm</code>: Weight normalization<br>
                • Combined with <code>SMOTE</code> oversampling
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Hyperparameter Tuning
    st.markdown('<div class="section-header"><h2>🎛️ Hyperparameter Tuning</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #1e3a5f; margin-top: 0;">GridSearchCV with Cross-Validation</h4>
        <p style="color: #6c757d;">
            We used <strong>GridSearchCV</strong> with <strong>5-fold cross-validation</strong> to find optimal hyperparameters.
            The search was optimized for <strong>Recall</strong> (our primary metric) to maximize detection of at-risk individuals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #2d5a87;">🌳 Decision Tree Grid</h4>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 8px; font-size: 0.9rem;">
{
  'max_depth': [3, 5, 7, 10, None],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4],
  'class_weight': ['balanced']
}
            </pre>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #38a169;">📊 Naive Bayes Grid</h4>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 8px; font-size: 0.9rem;">
{
  'classifier__alpha': [0.1, 0.5, 1.0],
  'classifier__norm': [True, False],
  'smote__k_neighbors': [3, 5, 7]
}
            </pre>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# PAGE 6: FINAL RESULTS
# =====================================================
elif page == "🏆 Final Results":
    st.markdown("""
    <div class="main-header">
        <h1>🏆 Final Results & Model Selection</h1>
        <p>Comparing Models and Justifying Our Choice</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model_results:
        # Model Comparison Table
        st.markdown('<div class="section-header"><h2>📊 Model Comparison</h2></div>', unsafe_allow_html=True)
        
        dt = model_results['Decision Tree']
        nb = model_results['Naive Bayes']
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Decision Tree': [f"{dt['Accuracy']:.2%}", f"{dt['Precision']:.2%}", f"{dt['Recall']:.2%}", f"{dt['F1']:.2%}"],
            'Naive Bayes': [f"{nb['Accuracy']:.2%}", f"{nb['Precision']:.2%}", f"{nb['Recall']:.2%}", f"{nb['F1']:.2%}"],
            'Winner': ['', '', '', '']
        })
        
        # Determine winners
        comparison_df.loc[0, 'Winner'] = '🌳' if dt['Accuracy'] >= nb['Accuracy'] else '📊'
        comparison_df.loc[1, 'Winner'] = '🌳' if dt['Precision'] >= nb['Precision'] else '📊'
        comparison_df.loc[2, 'Winner'] = '🌳' if dt['Recall'] >= nb['Recall'] else '📊'
        comparison_df.loc[3, 'Winner'] = '🌳' if dt['F1'] >= nb['F1'] else '📊'
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visual Comparison
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        dt_values = [dt['Accuracy'], dt['Precision'], dt['Recall'], dt['F1']]
        nb_values = [nb['Accuracy'], nb['Precision'], nb['Recall'], nb['F1']]
        
        fig.add_trace(go.Bar(name='Decision Tree 🌳', x=metrics, y=dt_values,
                            marker_color='#2d5a87', text=[f'{v:.1%}' for v in dt_values], textposition='outside'))
        fig.add_trace(go.Bar(name='Naive Bayes 📊', x=metrics, y=nb_values,
                            marker_color='#38a169', text=[f'{v:.1%}' for v in nb_values], textposition='outside'))
        fig.update_layout(barmode='group', title='Side-by-Side Performance Comparison',
                         yaxis=dict(range=[0, 1.15], tickformat='.0%'), height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        # Evaluation Metrics Explanation
        st.markdown('<div class="section-header"><h2>📖 Understanding the Metrics</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1e3a5f;">🎯 Accuracy</h4>
                <p style="color: #6c757d;">Overall correct predictions / Total predictions.<br>
                <em>Limitation: Misleading with imbalanced data.</em></p>
            </div>
            <div class="metric-card">
                <h4 style="color: #1e3a5f;">🔴 Recall (Sensitivity)</h4>
                <p style="color: #6c757d;">Of all actual positives, how many did we catch?<br>
                <strong style="color: #e53e3e;">OUR PRIMARY METRIC - minimizes missed at-risk individuals.</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1e3a5f;">🎯 Precision</h4>
                <p style="color: #6c757d;">Of all predicted positives, how many were correct?<br>
                <em>Important but secondary to Recall here.</em></p>
            </div>
            <div class="metric-card">
                <h4 style="color: #1e3a5f;">⚖️ F1 Score</h4>
                <p style="color: #6c757d;">Harmonic mean of Precision and Recall.<br>
                <em>Good overall balance metric.</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Final Model Selection
        st.markdown('<div class="section-header"><h2>🏆 Final Model Selection</h2></div>', unsafe_allow_html=True)
        
        # Determine best model based on Recall
        best_model = "Decision Tree" if dt['Recall'] >= nb['Recall'] else "Naive Bayes"
        best_icon = "🌳" if best_model == "Decision Tree" else "📊"
        best_recall = max(dt['Recall'], nb['Recall'])
        
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #f0fff4 0%, #c6f6d5 100%); border-radius: 16px; padding: 2rem; border: 2px solid #38a169; text-align: center; margin: 1rem 0;">
            <h2 style="color: #276749; margin-top: 0;">{best_icon} Selected Model: {best_model}</h2>
            <div style="font-size: 3rem; margin: 1rem 0;">🏆</div>
            <h3 style="color: #276749;">Recall Score: {best_recall:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Justification
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">📝 Justification for Model Selection</h4>
            <p style="color: #6c757d;">
                <strong>Why we prioritize Recall:</strong><br>
                In mental health screening, our goal is to identify as many at-risk individuals as possible. 
                A <strong>False Negative</strong> (missing someone who needs help) has far more severe consequences 
                than a <strong>False Positive</strong> (flagging someone who is actually fine for a follow-up).
            </p>
            <p style="color: #6c757d;">
                <strong>Trade-off accepted:</strong><br>
                We accept slightly lower precision (some false alarms) in exchange for higher recall 
                (catching more at-risk individuals). The cost of a missed diagnosis far outweighs the 
                cost of an extra check-in.
            </p>
            <p style="color: #6c757d;">
                <strong>Techniques used to maximize Recall:</strong><br>
                • Decision Tree with <code>class_weight='balanced'</code><br>
                • Naive Bayes with <code>SMOTE</code> oversampling<br>
                • Hyperparameter tuning optimized for Recall
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Takeaways
        st.markdown('<div class="section-header"><h2>📌 Key Takeaways</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem;">✅</div>
                <h4 style="color: #1e3a5f;">Binary Classification</h4>
                <p style="color: #6c757d;">At Risk vs Not At Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem;">🤖</div>
                <h4 style="color: #1e3a5f;">2 ML Models</h4>
                <p style="color: #6c757d;">Decision Tree & Naive Bayes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem;">🔴</div>
                <h4 style="color: #1e3a5f;">Recall Priority</h4>
                <p style="color: #6c757d;">Minimize missed diagnoses</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model results not available. Please ensure models are trained and saved.")
