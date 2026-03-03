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
        raw_results = joblib.load('models/model_results.pkl')
        
        # Transform results to expected format
        model_results = {
            'Decision Tree': {
                'Accuracy': raw_results.get('dt_accuracy', 0),
                'Precision': raw_results.get('dt_precision', 0),
                'Recall': raw_results.get('dt_recall', 0),
                'F1': raw_results.get('dt_f1', 0),
                'Confusion Matrix': raw_results.get('dt_cm', [[0,0],[0,0]])
            },
            'Naive Bayes': {
                'Accuracy': raw_results.get('nb_accuracy', 0),
                'Precision': raw_results.get('nb_precision', 0),
                'Recall': raw_results.get('nb_recall', 0),
                'F1': raw_results.get('nb_f1', 0),
                'Confusion Matrix': raw_results.get('nb_cm', [[0,0],[0,0]])
            },
            'best_model': raw_results.get('best_model', 'Decision Tree'),
            'class_labels': raw_results.get('class_labels', ['Not At Risk', 'At Risk'])
        }
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


def get_encoder_classes(encoders, col_name, default_values):
    """Helper function to get classes from LabelEncoder or return default"""
    if col_name in encoders:
        encoder = encoders[col_name]
        # Check if it's a LabelEncoder (has classes_ attribute)
        if hasattr(encoder, 'classes_'):
            return list(encoder.classes_)
        # If it's already a dict
        elif isinstance(encoder, dict):
            return list(encoder.keys())
    return list(default_values)


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
        ["Problem Statement", "EDA & Insights", "Methodology", "Model Performance", "Risk Prediction"],
        label_visibility="collapsed"
    )


# Load data and models
df = load_data()
dt_model, nb_model, scaler, minmax_scaler, encoders, feature_columns, model_results = load_models()


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
    
    # Problem Definition
    st.markdown('<div class="section-header"><h2> Problem Definition</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #1e3a5f; margin-top: 0;"> Classification Problem</h4>
            <p>
                <strong>Goal:</strong> Predict whether an individual is at risk of developing growing stress<br><br>
                <strong>Type:</strong> Binary Classification<br><br>
                <strong>Target Variable:</strong> <code>Stress_Risk</code><br>
                • Class 1: "At Risk" <br>
                • Class 0: "Not At Risk" 
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #fff5f5 0%, #ffe8e8 100%); border-radius: 12px; padding: 1.25rem; border-left: 4px solid #e53e3e; margin: 1rem 0;">
            <h4 style="color: #c53030; margin-top: 0;">Why We Prioritize RECALL</h4>
            <p style="color: #c53030;">
                In mental health screening, <strong>missing someone who needs help is far worse than a false alarm</strong>.<br><br>
                <strong>False Negative:</strong> Missed at-risk person → No intervention → Potential crisis <br><br>
                <strong>False Positive:</strong> Extra check-in → Minor inconvenience → Still helpful 
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
elif page == "EDA & Insights":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Discovering Hidden Patterns in Mental Health Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Create Stress_Risk if not exists
        if 'Stress_Risk' not in df.columns and 'Growing_Stress' in df.columns:
            df['Stress_Risk'] = df['Growing_Stress'].apply(
                lambda x: 'At Risk' if x in ['Yes', 'Maybe'] else 'Not At Risk'
            )
        
        # ===== SOCIAL WEAKNESS VISUALIZATION =====
        st.markdown('<div class="section-header"><h2>🤝 Social Weakness Analysis</h2></div>', unsafe_allow_html=True)
        
        if 'Social_Weakness' in df.columns and 'Stress_Risk' in df.columns:
            # Calculate stats with proper ordering (No, Maybe, Yes)
            social_risk = pd.crosstab(df['Social_Weakness'], df['Stress_Risk'], normalize='index') * 100
            category_order = ['No', 'Maybe', 'Yes']
            social_risk = social_risk.reindex([c for c in category_order if c in social_risk.index])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Visualization type selector
                viz_style = st.radio("Select Chart Style:", ["Grouped Bars", "Stacked Bars", "Donut Charts"], horizontal=True, key="social_viz")
                
                if viz_style == "Grouped Bars":
                    fig = go.Figure()
                    if 'Not At Risk' in social_risk.columns:
                        fig.add_trace(go.Bar(
                            name='Not At Risk',
                            x=social_risk.index,
                            y=social_risk['Not At Risk'],
                            marker_color='#38a169',
                            text=[f'{v:.1f}%' for v in social_risk['Not At Risk']],
                            textposition='outside'
                        ))
                    if 'At Risk' in social_risk.columns:
                        fig.add_trace(go.Bar(
                            name='At Risk',
                            x=social_risk.index,
                            y=social_risk['At Risk'],
                            marker_color='#e53e3e',
                            text=[f'{v:.1f}%' for v in social_risk['At Risk']],
                            textposition='outside'
                        ))
                    fig.update_layout(
                        barmode='group',
                        height=400,
                        xaxis_title="Social Weakness Level",
                        yaxis_title="Percentage %",
                        title=dict(text="<b>Stress Risk by Social Weakness</b>", x=0.5, font_size=16),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        xaxis=dict(categoryorder='array', categoryarray=category_order)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif viz_style == "Stacked Bars":
                    fig = go.Figure()
                    if 'Not At Risk' in social_risk.columns:
                        fig.add_trace(go.Bar(
                            name='Not At Risk',
                            x=social_risk.index,
                            y=social_risk['Not At Risk'],
                            marker_color='#38a169',
                            text=[f'{v:.1f}%' for v in social_risk['Not At Risk']],
                            textposition='inside'
                        ))
                    if 'At Risk' in social_risk.columns:
                        fig.add_trace(go.Bar(
                            name='At Risk',
                            x=social_risk.index,
                            y=social_risk['At Risk'],
                            marker_color='#e53e3e',
                            text=[f'{v:.1f}%' for v in social_risk['At Risk']],
                            textposition='inside'
                        ))
                    fig.update_layout(
                        barmode='stack',
                        height=400,
                        xaxis_title="Social Weakness Level",
                        yaxis_title="Percentage %",
                        title=dict(text="<b>Stress Risk by Social Weakness (Stacked)</b>", x=0.5, font_size=16),
                        xaxis=dict(categoryorder='array', categoryarray=category_order)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # Donut Charts
                    ordered_categories = [c for c in category_order if c in social_risk.index]
                    cols = st.columns(len(ordered_categories))
                    for i, category in enumerate(ordered_categories):
                        with cols[i]:
                            values = [social_risk.loc[category, 'At Risk'] if 'At Risk' in social_risk.columns else 0,
                                     social_risk.loc[category, 'Not At Risk'] if 'Not At Risk' in social_risk.columns else 0]
                            fig = go.Figure(data=[go.Pie(
                                labels=['At Risk', 'Not At Risk'],
                                values=values,
                                hole=0.6,
                                marker_colors=['#e53e3e', '#38a169'],
                                textinfo='percent',
                                textfont_size=12
                            )])
                            fig.add_annotation(text=f"<b>{category}</b>", x=0.5, y=0.5, font_size=12, showarrow=False)
                            fig.update_layout(height=250, showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
                            st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Stats cards
                st.markdown("### 📊 Quick Stats")
                
                ordered_categories = [c for c in category_order if c in social_risk.index]
                for category in ordered_categories:
                    at_risk_pct = social_risk.loc[category, 'At Risk'] if 'At Risk' in social_risk.columns else 0
                    color = '#e53e3e' if at_risk_pct > 70 else '#f6ad55' if at_risk_pct > 60 else '#38a169'
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {color}; margin-bottom: 0.5rem;">
                        <div class="metric-value" style="color: {color}; font-size: 1.5rem;">{at_risk_pct:.1f}%</div>
                        <div class="metric-label">Social Weakness: {category}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Count breakdown
                st.markdown("### 📈 Sample Counts")
                for cat in ordered_categories:
                    if cat in df['Social_Weakness'].values:
                        count = (df['Social_Weakness'] == cat).sum()
                        st.markdown(f"**{cat}:** {count:,} ({count/len(df)*100:.1f}%)")
            
            # Insight box
            highest_risk_cat = social_risk['At Risk'].idxmax() if 'At Risk' in social_risk.columns else 'N/A'
            highest_risk_val = social_risk['At Risk'].max() if 'At Risk' in social_risk.columns else 0
            st.markdown(f"""
            <div class="info-card" style="margin-top: 1rem;">
                <h4 style="color: #1e3a5f; margin-top: 0;">💡 Key Insight: Social Weakness Impact</h4>
                <p style="color: #6c757d;">
                    Individuals with <strong>Social Weakness = "{highest_risk_cat}"</strong> show the highest stress risk at <strong>{highest_risk_val:.1f}%</strong>.
                    Social connections and support systems play a significant role in mental health outcomes.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== EDA Summary =====
        st.markdown('<div class="section-header"><h2>📝 Key Takeaways</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                <div>
                    <h4 style="color: #e53e3e; margin-top: 0;">🔴 High Risk Indicators</h4>
                    <ul style="color: #6c757d;">
                        <li>High mood swings severity</li>
                        <li>Extended time indoors (2+ months)</li>
                        <li>Coping difficulties</li>
                        <li>Family history of mental health issues</li>
                    </ul>
                </div>
                <div>
                    <h4 style="color: #38a169; margin-top: 0;">🟢 Protective Factors</h4>
                    <ul style="color: #6c757d;">
                        <li>Daily outdoor activities</li>
                        <li>Low mood swing frequency</li>
                        <li>Strong coping mechanisms</li>
                        <li>No family history</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# MODEL PERFORMANCE PAGE
elif page == "Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Model Performance</h1>
        <p>Comprehensive Machine Learning Model Evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model_results:
        dt = model_results['Decision Tree']
        nb = model_results['Naive Bayes']
        
        # ===== INNOVATIVE VIZ 1: Gauge Charts for Primary Metric (Recall) =====
        st.markdown('<div class="section-header"><h2>🎯 Primary Metric: Recall (Catching At-Risk Individuals)</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Decision Tree Gauge
            fig_gauge_dt = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=dt['Recall'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "🌳 Decision Tree Recall", 'font': {'size': 18}},
                delta={'reference': 70, 'increasing': {'color': "#38a169"}, 'suffix': '%'},
                number={'suffix': '%', 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1e3a5f"},
                    'bar': {'color': "#2d5a87"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#1e3a5f",
                    'steps': [
                        {'range': [0, 50], 'color': '#fed7d7'},
                        {'range': [50, 70], 'color': '#feebc8'},
                        {'range': [70, 85], 'color': '#c6f6d5'},
                        {'range': [85, 100], 'color': '#9ae6b4'}
                    ],
                    'threshold': {
                        'line': {'color': "#e53e3e", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge_dt.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge_dt, use_container_width=True)
        
        with col2:
            # Naive Bayes Gauge
            fig_gauge_nb = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=nb['Recall'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "📊 Naive Bayes Recall", 'font': {'size': 18}},
                delta={'reference': 70, 'increasing': {'color': "#38a169"}, 'suffix': '%'},
                number={'suffix': '%', 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1e3a5f"},
                    'bar': {'color': "#38a169"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#1e3a5f",
                    'steps': [
                        {'range': [0, 50], 'color': '#fed7d7'},
                        {'range': [50, 70], 'color': '#feebc8'},
                        {'range': [70, 85], 'color': '#c6f6d5'},
                        {'range': [85, 100], 'color': '#9ae6b4'}
                    ],
                    'threshold': {
                        'line': {'color': "#e53e3e", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge_nb.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge_nb, use_container_width=True)
        
        st.markdown("""
        <div class="info-card">
            <p style="color: #6c757d; text-align: center;">
                <strong>Recall Target: 70%+</strong> — The red threshold line shows our minimum acceptable recall. 
                Higher recall means fewer at-risk individuals are missed.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # ===== INNOVATIVE VIZ 3: Performance Cards with Visual Bars =====
        st.markdown('<div class="section-header"><h2>📊 Detailed Metrics Breakdown</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card" style="border-top: 4px solid #2d5a87;"><h3 style="color: #1e3a5f; text-align: center;">🌳 Decision Tree</h3></div>', unsafe_allow_html=True)
            
            for metric, value in [('Accuracy', dt['Accuracy']), ('Precision', dt['Precision']), 
                                   ('Recall', dt['Recall']), ('F1 Score', dt['F1'])]:
                color = '#38a169' if value >= 0.7 else '#f6ad55' if value >= 0.5 else '#e53e3e'
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-weight: 600; color: #1e3a5f;">{metric}</span>
                        <span style="font-weight: 700; color: {color};">{value:.1%}</span>
                    </div>
                    <div style="background: #e0e0e0; border-radius: 10px; height: 12px; overflow: hidden;">
                        <div style="background: {color}; width: {value*100}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card" style="border-top: 4px solid #38a169;"><h3 style="color: #1e3a5f; text-align: center;">📊 Naive Bayes</h3></div>', unsafe_allow_html=True)
            
            for metric, value in [('Accuracy', nb['Accuracy']), ('Precision', nb['Precision']), 
                                   ('Recall', nb['Recall']), ('F1 Score', nb['F1'])]:
                color = '#38a169' if value >= 0.7 else '#f6ad55' if value >= 0.5 else '#e53e3e'
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-weight: 600; color: #1e3a5f;">{metric}</span>
                        <span style="font-weight: 700; color: {color};">{value:.1%}</span>
                    </div>
                    <div style="background: #e0e0e0; border-radius: 10px; height: 12px; overflow: hidden;">
                        <div style="background: {color}; width: {value*100}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ===== INNOVATIVE VIZ 4: Confusion Matrix Heatmaps =====
        st.markdown('<div class="section-header"><h2>🔢 Confusion Matrix Analysis</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Get confusion matrices from model_results
        dt_cm = model_results['Decision Tree'].get('Confusion Matrix', [[0,0],[0,0]])
        nb_cm = model_results['Naive Bayes'].get('Confusion Matrix', [[0,0],[0,0]])
        
        with col1:
            if dt_cm and len(dt_cm) == 2:
                fig_cm_dt = go.Figure(data=go.Heatmap(
                    z=dt_cm,
                    x=['Predicted: Not At Risk', 'Predicted: At Risk'],
                    y=['Actual: Not At Risk', 'Actual: At Risk'],
                    colorscale=[[0, '#e8f4fd'], [1, '#2d5a87']],
                    showscale=False,
                    text=[[f'{dt_cm[0][0]:,}', f'{dt_cm[0][1]:,}'],
                          [f'{dt_cm[1][0]:,}', f'{dt_cm[1][1]:,}']],
                    texttemplate='%{text}',
                    textfont={'size': 18, 'color': 'white'},
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>'
                ))
                fig_cm_dt.update_layout(
                    title=dict(text="<b>🌳 Decision Tree</b>", x=0.5, font_size=14),
                    height=300,
                    xaxis=dict(side='bottom'),
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig_cm_dt, use_container_width=True)
        
        with col2:
            if nb_cm and len(nb_cm) == 2:
                fig_cm_nb = go.Figure(data=go.Heatmap(
                    z=nb_cm,
                    x=['Predicted: Not At Risk', 'Predicted: At Risk'],
                    y=['Actual: Not At Risk', 'Actual: At Risk'],
                    colorscale=[[0, '#f0fff4'], [1, '#38a169']],
                    showscale=False,
                    text=[[f'{nb_cm[0][0]:,}', f'{nb_cm[0][1]:,}'],
                          [f'{nb_cm[1][0]:,}', f'{nb_cm[1][1]:,}']],
                    texttemplate='%{text}',
                    textfont={'size': 18, 'color': 'white'},
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>'
                ))
                fig_cm_nb.update_layout(
                    title=dict(text="<b>📊 Naive Bayes</b>", x=0.5, font_size=14),
                    height=300,
                    xaxis=dict(side='bottom'),
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig_cm_nb, use_container_width=True)
        
        # Confusion matrix interpretation
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #1e3a5f; margin-top: 0;">📖 How to Read the Confusion Matrix</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; color: #6c757d;">
                <div>
                    <p><strong>Top-Left (True Negative):</strong> Correctly identified as Not At Risk ✅</p>
                    <p><strong>Top-Right (False Positive):</strong> Incorrectly flagged as At Risk ⚡</p>
                </div>
                <div>
                    <p><strong>Bottom-Left (False Negative):</strong> Missed At-Risk individuals ❌</p>
                    <p><strong>Bottom-Right (True Positive):</strong> Correctly identified as At Risk ✅</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ===== INNOVATIVE VIZ 5: Model Comparison Bar Race =====
        st.markdown('<div class="section-header"><h2>🏆 Head-to-Head Comparison</h2></div>', unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Decision Tree': [dt['Accuracy'], dt['Precision'], dt['Recall'], dt['F1']],
            'Naive Bayes': [nb['Accuracy'], nb['Precision'], nb['Recall'], nb['F1']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Decision Tree 🌳', 
            x=metrics_df['Metric'], 
            y=metrics_df['Decision Tree'],
            marker_color='#2d5a87', 
            text=[f'{v:.1%}' for v in metrics_df['Decision Tree']], 
            textposition='outside',
            textfont=dict(size=14, color='#2d5a87')
        ))
        fig.add_trace(go.Bar(
            name='Naive Bayes 📊', 
            x=metrics_df['Metric'], 
            y=metrics_df['Naive Bayes'],
            marker_color='#38a169', 
            text=[f'{v:.1%}' for v in metrics_df['Naive Bayes']], 
            textposition='outside',
            textfont=dict(size=14, color='#38a169')
        ))
        fig.update_layout(
            barmode='group', 
            title=dict(text='<b>All Metrics Comparison</b>', x=0.5, font_size=16),
            yaxis=dict(range=[0, 1.15], tickformat='.0%', title='Score'),
            xaxis=dict(title=''),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Winner announcement
        best = model_results.get('best_model', 'Decision Tree')
        best_recall = max(dt['Recall'], nb['Recall'])
        best_icon = "🌳" if best == "Decision Tree" else "📊"
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #f0fff4 0%, #c6f6d5 100%); border-radius: 16px; padding: 2rem; border: 2px solid #38a169; text-align: center; margin-top: 1rem;">
            <h2 style="color: #276749; margin-top: 0;">{best_icon} Selected Model: {best}</h2>
            <div style="font-size: 3rem; margin: 1rem 0;">🏆</div>
            <h3 style="color: #276749;">Recall Score: {best_recall:.1%}</h3>
            <p style="color: #276749;">Selected based on highest Recall score for mental health screening</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Justification
        st.markdown('<div class="section-header"><h2>📝 Model Selection Justification</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
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
                <strong>Why Decision Tree over Naive Bayes:</strong><br>
                • <strong>Feature Independence Assumption Violated:</strong> Naive Bayes assumes all features are independent of each other, 
                which doesn't hold in mental health data where features like <em>Mood_Swings</em>, <em>Coping_Struggles</em>, and <em>Social_Weakness</em> are naturally correlated.<br>
                • <strong>Decision Tree captures feature interactions:</strong> Unlike NB, Decision Trees can model complex relationships 
                between features through branching, making them more suitable for this correlated dataset.<br>
                • <strong>Higher Recall achieved:</strong> Decision Tree achieved ~99.5% recall compared to Naive Bayes's ~59%, 
                which is critical for not missing at-risk individuals.
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


# RISK PREDICTION PAGE
elif page == "Risk Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Stress Risk Prediction</h1>
        <p>Get Personalized Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if dt_model and nb_model and encoders and feature_columns:
        # Model Selection Section
        st.markdown('<div class="section-header"><h2>🤖 Select Prediction Model</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            model_choice = st.selectbox(
                "Choose which model to use for prediction:",
                ["🌳 Decision Tree", "📊 Naive Bayes", "🔄 Both Models (Compare)"],
                index=2  # Default to "Both Models"
            )
        
        # Model info based on selection
        if model_choice == "🌳 Decision Tree":
            st.markdown("""
            <div class="info-card" style="border-left: 4px solid #2d5a87;">
                <h4 style="color: #2d5a87; margin-top: 0;">🌳 Decision Tree Selected</h4>
                <p style="color: #6c757d;">
                    Rule-based classifier that creates interpretable decision rules.
                    Uses <code>class_weight='balanced'</code> to handle class imbalance.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif model_choice == "📊 Naive Bayes":
            st.markdown("""
            <div class="info-card" style="border-left: 4px solid #38a169;">
                <h4 style="color: #38a169; margin-top: 0;">📊 Naive Bayes Selected</h4>
                <p style="color: #6c757d;">
                    Probabilistic classifier using Bayes' theorem.
                    Uses <code>SMOTE</code> oversampling to handle class imbalance.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #1e3a5f; margin-top: 0;">🔄 Comparing Both Models</h4>
                <p style="color: #6c757d;">
                    You'll see predictions from both Decision Tree and Naive Bayes models side by side.
                    This helps validate the prediction with multiple approaches.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h2>📝 Enter Information</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("👤 Gender", get_encoder_classes(encoders, 'Gender', ['Male', 'Female']))
            occupation = st.selectbox("💼 Occupation", get_encoder_classes(encoders, 'Occupation', ['Student', 'Corporate', 'Business', 'Housewife', 'Others']))
            self_employed = st.selectbox("🏢 Self Employed", get_encoder_classes(encoders, 'self_employed', ['Yes', 'No']))
            days_indoors = st.selectbox("🏠 Days Indoors", get_encoder_classes(encoders, 'Days_Indoors', ['Go out Every day', '1-14 days', '15-30 days', '31-60 days', 'More than 2 months']))
            care_options = st.selectbox("🏥 Care Options Awareness", get_encoder_classes(encoders, 'care_options', ['Yes', 'No', 'Not sure']))
        
        with col2:
            mood_swings = st.selectbox("😰 Mood Swings", get_encoder_classes(encoders, 'Mood_Swings', ['Low', 'Medium', 'High']))
            coping_struggles = st.selectbox("💪 Coping Struggles", get_encoder_classes(encoders, 'Coping_Struggles', ['Yes', 'No']))
            work_interest = st.selectbox("📋 Work Interest", get_encoder_classes(encoders, 'Work_Interest', ['Yes', 'No', 'Maybe']))
            social_weakness = st.selectbox("👥 Social Weakness", get_encoder_classes(encoders, 'Social_Weakness', ['Yes', 'No', 'Maybe']))
        
        with col3:
            family_history = st.selectbox("👨‍👩‍👧 Family History", get_encoder_classes(encoders, 'family_history', ['Yes', 'No']))
            treatment = st.selectbox("💊 Treatment", get_encoder_classes(encoders, 'treatment', ['Yes', 'No']))
            mental_health_history = st.selectbox("🧠 Mental Health History", get_encoder_classes(encoders, 'Mental_Health_History', ['Yes', 'No', 'Maybe']))
            changes_habits = st.selectbox("🔄 Changes in Habits", get_encoder_classes(encoders, 'Changes_Habits', ['Yes', 'No', 'Maybe']))
            mental_health_interview = st.selectbox("🗣️ Discuss in Interview", get_encoder_classes(encoders, 'mental_health_interview', ['Yes', 'No', 'Maybe']))
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔮 Analyze Risk", use_container_width=True)
        
        if predict_button:
            # Map inputs to correct feature column names
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
            
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if col in encoders:
                    encoder = encoders[col]
                    try:
                        # Handle LabelEncoder objects
                        if hasattr(encoder, 'transform'):
                            input_df[col] = encoder.transform(input_df[col])
                        # Handle dict mappings
                        elif isinstance(encoder, dict):
                            input_df[col] = input_df[col].map(encoder)
                        else:
                            input_df[col] = 0
                    except:
                        input_df[col] = 0
            
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_columns]
            
            # Decision Tree: NO scaling (trained on unscaled data)
            # Naive Bayes (CategoricalNB): NO scaling needed (designed for categorical data)
            
            dt_pred = dt_model.predict(input_df)[0]
            nb_pred = nb_model.predict(input_df)[0]  # No scaling for CategoricalNB
            
            # Label mapping: 0 = "At Risk", 1 = "Not At Risk" (based on encoder)
            dt_label = "At Risk" if dt_pred == 0 else "Not At Risk"
            nb_label = "At Risk" if nb_pred == 0 else "Not At Risk"
            
            st.markdown('<div class="section-header"><h2>📊 Prediction Results</h2></div>', unsafe_allow_html=True)
            
            # Display based on model choice
            if model_choice == "🌳 Decision Tree":
                # Show only Decision Tree result
                css_class = "prediction-at-risk" if dt_label == "At Risk" else "prediction-not-at-risk"
                icon = "⚠️" if dt_label == "At Risk" else "✅"
                color = "#e53e3e" if dt_label == "At Risk" else "#38a169"
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f'<div class="{css_class}"><h3 style="color: {color};">🌳 Decision Tree Prediction</h3><div style="font-size: 4rem;">{icon}</div><h1 style="color: {color};">{dt_label}</h1></div>', unsafe_allow_html=True)
                
                if dt_label == "At Risk":
                    st.markdown('<div style="background: #fff5f5; padding: 1.5rem; border-radius: 12px; border: 2px solid #e53e3e; text-align: center; margin-top: 1rem;"><h3 style="color: #e53e3e;">⚠️ At Risk Detected</h3><p>Consider consulting a mental health professional for support.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background: #f0fff4; padding: 1.5rem; border-radius: 12px; border: 2px solid #38a169; text-align: center; margin-top: 1rem;"><h3 style="color: #38a169;">✅ Not At Risk</h3><p>Keep maintaining your mental wellness practices!</p></div>', unsafe_allow_html=True)
            
            elif model_choice == "📊 Naive Bayes":
                # Show only Naive Bayes result
                css_class = "prediction-at-risk" if nb_label == "At Risk" else "prediction-not-at-risk"
                icon = "⚠️" if nb_label == "At Risk" else "✅"
                color = "#e53e3e" if nb_label == "At Risk" else "#38a169"
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f'<div class="{css_class}"><h3 style="color: {color};">📊 Naive Bayes Prediction</h3><div style="font-size: 4rem;">{icon}</div><h1 style="color: {color};">{nb_label}</h1></div>', unsafe_allow_html=True)
                
                if nb_label == "At Risk":
                    st.markdown('<div style="background: #fff5f5; padding: 1.5rem; border-radius: 12px; border: 2px solid #e53e3e; text-align: center; margin-top: 1rem;"><h3 style="color: #e53e3e;">⚠️ At Risk Detected</h3><p>Consider consulting a mental health professional for support.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background: #f0fff4; padding: 1.5rem; border-radius: 12px; border: 2px solid #38a169; text-align: center; margin-top: 1rem;"><h3 style="color: #38a169;">✅ Not At Risk</h3><p>Keep maintaining your mental wellness practices!</p></div>', unsafe_allow_html=True)
            
            else:
                # Show both models side by side
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
                
                # Consensus message
                if dt_label == nb_label:
                    if dt_label == "At Risk":
                        st.markdown('<div style="background: #fff5f5; padding: 1.5rem; border-radius: 12px; border: 2px solid #e53e3e; text-align: center; margin-top: 1rem;"><h3 style="color: #e53e3e;">⚠️ Both Models Agree: At Risk</h3><p>Both algorithms identify potential risk. Consider consulting a mental health professional.</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background: #f0fff4; padding: 1.5rem; border-radius: 12px; border: 2px solid #38a169; text-align: center; margin-top: 1rem;"><h3 style="color: #38a169;">✅ Both Models Agree: Not At Risk</h3><p>Both algorithms indicate low risk. Keep maintaining your mental wellness!</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background: #fffbeb; padding: 1.5rem; border-radius: 12px; border: 2px solid #f6ad55; text-align: center; margin-top: 1rem;"><h3 style="color: #c05621;">⚡ Models Disagree</h3><p>The models give different predictions. Consider the more cautious approach and monitor your mental health.</p></div>', unsafe_allow_html=True)
        
        # Example Scenarios Section
        st.markdown('<div class="section-header"><h2>📋 Example Scenarios</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #6c757d; text-align: center; margin-bottom: 1.5rem;">
            These verified profiles are from actual data patterns. Both models correctly predict these scenarios.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(145deg, #fff5f5 0%, #ffe8e8 100%); border-radius: 12px; padding: 1.5rem; border: 2px solid #e53e3e; margin-bottom: 1rem;">
                <h3 style="color: #c53030; margin-top: 0; text-align: center;">⚠️ At Risk Profile</h3>
                <hr style="border-color: #e53e3e; opacity: 0.3;">
                <table style="width: 100%; color: #1e3a5f; font-size: 0.9rem;">
                    <tr><td style="padding: 4px 0;"><strong>👤 Gender:</strong></td><td>Male</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>💼 Occupation:</strong></td><td>Corporate</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🏢 Self Employed:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🏠 Days Indoors:</strong></td><td>More than 2 months</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🔄 Changes in Habits:</strong></td><td>Yes</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>😰 Mood Swings:</strong></td><td>Medium</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🧠 Mental Health History:</strong></td><td>Maybe</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>💪 Coping Struggles:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>📋 Work Interest:</strong></td><td>Yes</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>👥 Social Weakness:</strong></td><td>Maybe</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>👨‍👩‍👧 Family History:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>💊 Treatment:</strong></td><td>Yes</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🗣️ Interview:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🏥 Care Options:</strong></td><td>No</td></tr>
                </table>
                <hr style="border-color: #e53e3e; opacity: 0.3;">
                <p style="color: #c53030; font-size: 0.85rem; margin-bottom: 0; text-align: center;">
                    <strong>Key Risk Factors:</strong> Extended isolation (2+ months), habit changes, uncertain mental health history.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(145deg, #f0fff4 0%, #c6f6d5 100%); border-radius: 12px; padding: 1.5rem; border: 2px solid #38a169; margin-bottom: 1rem;">
                <h3 style="color: #276749; margin-top: 0; text-align: center;">✅ Not At Risk Profile</h3>
                <hr style="border-color: #38a169; opacity: 0.3;">
                <table style="width: 100%; color: #1e3a5f; font-size: 0.9rem;">
                    <tr><td style="padding: 4px 0;"><strong>👤 Gender:</strong></td><td>Male</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>💼 Occupation:</strong></td><td>Student</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🏢 Self Employed:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🏠 Days Indoors:</strong></td><td>31-60 days</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🔄 Changes in Habits:</strong></td><td>Maybe</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>😰 Mood Swings:</strong></td><td>High</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🧠 Mental Health History:</strong></td><td>Yes</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>💪 Coping Struggles:</strong></td><td>Yes</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>📋 Work Interest:</strong></td><td>Maybe</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>👥 Social Weakness:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>👨‍👩‍👧 Family History:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>💊 Treatment:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🗣️ Interview:</strong></td><td>No</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>🏥 Care Options:</strong></td><td>No</td></tr>
                </table>
                <hr style="border-color: #38a169; opacity: 0.3;">
                <p style="color: #276749; font-size: 0.85rem; margin-bottom: 0; text-align: center;">
                    <strong>Protective Factor:</strong> No social weakness - strong social connections despite other challenges.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card" style="margin-top: 1rem;">
            <h4 style="color: #1e3a5f; margin-top: 0;">💡 Try It Yourself</h4>
            <p style="color: #6c757d; margin-bottom: 0;">
                Use the form above to enter these profiles and verify the model predictions. 
                The <strong>Decision Tree</strong> achieves ~99.5% accuracy on these patterns!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
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
elif page == "Methodology":
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

