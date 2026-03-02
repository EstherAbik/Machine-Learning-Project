"""
Stress Trajectory Prediction - Mental Health Analysis Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
        with open('models/decision_tree_model.pkl', 'rb') as f:
            dt_model = pickle.load(f)
        with open('models/naive_bayes_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/minmax_scaler.pkl', 'rb') as f:
            minmax_scaler = pickle.load(f)
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        with open('models/model_results.pkl', 'rb') as f:
            model_results = pickle.load(f)
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
        ["🏠 Home", "📊 Data Insights", "🎯 Model Performance", "🔮 Risk Prediction"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(145deg, #e8f4fd 0%, #d4e8f7 100%); 
                padding: 1rem; border-radius: 12px; margin-top: 1rem;">
        <p style="font-size: 0.85rem; color: #1e3a5f; margin: 0; text-align: center;">
            <strong>💡 Quick Tip</strong><br>
            Early detection of stress patterns can help prevent burnout.
        </p>
    </div>
    """, unsafe_allow_html=True)


# Load data and models
df = load_data()
dt_model, nb_model, scaler, minmax_scaler, encoders, feature_columns, model_results = load_models()


# HOME PAGE
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>�� Stress Trajectory Prediction</h1>
        <p>Leveraging Machine Learning for Early Mental Health Risk Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p>
            <strong>Welcome to the Stress Trajectory Prediction Platform</strong> — an AI-powered tool designed 
            to analyze behavioral and lifestyle patterns to identify individuals who may be at risk of 
            stress-related mental health challenges.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>📈 Platform Overview</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    if df is not None:
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">ML Models</div></div>', unsafe_allow_html=True)
        with col4:
            if model_results:
                best_recall = max(model_results['Decision Tree']['Recall'], model_results['Naive Bayes']['Recall'])
                st.markdown(f'<div class="metric-card"><div class="metric-value">{best_recall:.1%}</div><div class="metric-label">Best Recall</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><h2>🎯 Key Features</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">📊 Data Analysis</h4>
            <p style="color: #6c757d;">Explore patterns through interactive visualizations.</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">🔮 Risk Prediction</h4>
            <p style="color: #6c757d;">Get instant stress risk assessments using ML models.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">🤖 Dual Model Approach</h4>
            <p style="color: #6c757d;">Compare Decision Tree and Naive Bayes predictions.</p>
        </div>
        <div class="metric-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">📈 Performance Metrics</h4>
            <p style="color: #6c757d;">View detailed accuracy, precision, recall scores.</p>
        </div>
        """, unsafe_allow_html=True)


# DATA INSIGHTS PAGE
elif page == "📊 Data Insights":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Insights</h1>
        <p>Exploring Patterns in Mental Health Data</p>
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
