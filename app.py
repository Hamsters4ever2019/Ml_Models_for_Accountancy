import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import numpy as np

# Import our modules
from data_generator import generate_accounting_data, generate_balanced_accounting_data
from model_utils import (prepare_supervised_data, prepare_unsupervised_data, 
                         display_metrics, plot_confusion_matrix, plot_feature_importance,
                         get_model_description)
from supervised_models import train_random_forest, train_logistic_regression, train_xgboost
from unsupervised_models import run_kmeans, run_dbscan, run_isolation_forest, visualize_clusters
from model_comparison import run_model_comparison

# ============================================
# 🛡️ SECURITY: HIDE STREAMLIT UI
# ============================================

st.set_page_config(
    page_title="Accountancy ML Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# Comprehensive CSS to hide everything
hide_streamlit_ui = """
    <style>
        /* Hide EVERYTHING in the top section */
        header {display: none !important;}
        footer {display: none !important;}
        .stAppHeader {display: none !important;}
        .stAppFooter {display: none !important;}
        .stAppMenu {display: none !important;}
        .stAppToolbar {display: none !important;}
        .stToolbar {display: none !important;}
        .stDeployButton {display: none !important;}
        .stAppDeployButton {display: none !important;}
        #MainMenu {visibility: hidden !important;}
        .main-header {margin-top: -50px;}
        
        /* Hide GitHub button and any external links */
        .e1f1d6gn2 {display: none !important;}
        .css-1jc7ptx {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stHeader"] {display: none !important;}
        
        /* Remove padding to reclaim space */
        .main > div {padding-top: 0rem;}
        .block-container {padding-top: 1rem;}
        
        /* Hide any Streamlit branding */
        .st-emotion-cache-16txtl3 {display: none !important;}
        .st-emotion-cache-1avcm0n {display: none !important;}
        .st-emotion-cache-1v0mbdj {display: none !important;}
        .st-emotion-cache-11wj3xh {display: none !important;}
        .st-emotion-cache-1aak7d9 {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_ui, unsafe_allow_html=True)

# Optional: JavaScript to remove elements after page load
hide_js = """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const elements = document.querySelectorAll('.stDeployButton, .stAppDeployButton');
            elements.forEach(el => el.style.display = 'none');
            const githubElements = document.querySelectorAll('a[href*="github.com"]');
            githubElements.forEach(el => {
                if (el.parentElement) {
                    el.parentElement.style.display = 'none';
                }
            });
        });
    </script>
"""
st.markdown(hide_js, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
    }
    .comparison-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .best-model-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Accountancy ML Lab</p>', unsafe_allow_html=True)
st.markdown("*Interactive Machine Learning Demonstrations for Accounting Professionals*")

# ============================================
# 🎯 DATA BALANCE CHECK FUNCTION
# ============================================

def check_data_balance(df):
    """Check if data has enough fraud cases for training"""
    if df is None:
        return False, "No data loaded"
    
    if 'is_fraud' not in df.columns:
        return False, "No 'is_fraud' column found"
    
    fraud_count = df['is_fraud'].sum()
    total = len(df)
    
    if fraud_count < 5:
        return False, f"Only {fraud_count} fraud cases (need at least 5-10)"
    elif fraud_count < 10:
        return False, f"Only {fraud_count} fraud cases (recommend 10+ for better training)"
    else:
        return True, f"{fraud_count} fraud cases ({fraud_count/total:.1%})"

# ============================================
# SIDEBAR CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Use Sample Data", "Upload CSV"],
        index=0
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload accounting data (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
        else:
            st.info("Please upload a CSV file or use sample data")
            df = None
    else:
        # Generate or load sample data
        if st.button("🔄 Generate Fresh Sample Data", use_container_width=True):
            with st.spinner("Generating synthetic accounting data..."):
                df = generate_accounting_data(2000)
                st.session_state.df = df
            st.success("✅ Sample data generated!")
        elif 'df' in st.session_state:
            df = st.session_state.df
            st.info(f"📊 Using sample data ({len(df)} rows)")
        else:
            df = generate_accounting_data(2000)
            st.session_state.df = df
            st.info(f"📊 Using sample data ({len(df)} rows)")
    
    # ============================================
    # ⚖️ DATA BALANCE CONTROLS
    # ============================================
    
    st.divider()
    st.subheader("⚖️ Data Balance Settings")
    
    fraud_rate = st.slider(
        "🎯 Target Fraud Rate",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        format="%.0f%%",
        help="Higher fraud rate = more examples for the model to learn from"
    )
    
    if st.button("🔄 Generate Balanced Data", use_container_width=True):
        with st.spinner(f"Generating {fraud_rate:.0%} fraud rate data..."):
            df = generate_balanced_accounting_data(2000, fraud_rate=fraud_rate)
            st.session_state.df = df
            st.success(f"✅ Generated {df['is_fraud'].sum()} fraud cases ({fraud_rate:.0%})")
            st.rerun()
    
    # ============================================
    # COMPARISON MODE TOGGLE
    # ============================================
    
    st.divider()
    comparison_mode = st.checkbox("🔬 Model Comparison Mode", value=False)
    
    # ============================================
    # SINGLE MODEL CONFIGURATION
    # ============================================
    
    if not comparison_mode:
        st.divider()
        model_type = st.radio(
            "🧠 Select Model Type",
            ["Supervised", "Unsupervised"]
        )
        
        # Model selection based on type
        if model_type == "Supervised":
            model_choice = st.selectbox(
                "Choose Supervised Model",
                ["Random Forest Classifier", "Logistic Regression", "XGBoost"]
            )
            target_col = st.selectbox(
                "Target Variable",
                ["is_fraud", "is_high_value"] if df is not None else []
            )
        else:
            model_choice = st.selectbox(
                "Choose Unsupervised Model",
                ["K-Means Clustering", "DBSCAN", "Isolation Forest"]
            )
        
        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            if "Random Forest" in model_choice:
                n_estimators = st.slider("Number of Trees", 50, 200, 100)
            elif "K-Means" in model_choice:
                n_clusters = st.slider("Number of Clusters (K)", 2, 8, 3)
            elif "DBSCAN" in model_choice:
                eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("Min Samples", 2, 20, 5)
            elif "Isolation Forest" in model_choice:
                contamination = st.slider("Contamination Rate", 0.01, 0.3, 0.1, 0.01)
            
            # ============================================
            # ⚖️ BALANCING OPTIONS FOR SUPERVISED MODELS
            # ============================================
            
            if model_type == "Supervised":
                st.divider()
                st.subheader("⚖️ Balance Configuration")
                
                use_smote = st.checkbox(
                    "🔄 Use SMOTE",
                    value=True,
                    help="Synthetic Minority Oversampling - creates synthetic fraud cases"
                )
                
                use_class_weight = st.checkbox(
                    "⚖️ Use Class Weights",
                    value=True,
                    help="Adjusts model to pay more attention to fraud cases"
                )
                
                if use_smote and use_class_weight:
                    st.success("✅ Both SMOTE and Class Weights enabled!")
                elif use_smote:
                    st.info("🔄 SMOTE enabled - synthetic data will be created")
                elif use_class_weight:
                    st.info("⚖️ Class weights enabled - model will focus on fraud")
                else:
                    st.warning("⚠️ No balancing techniques enabled - model may be biased")
        
        # Run button for single model
        st.divider()
        run_button = st.button("🚀 Run Model", type="primary")
    
    else:
        # Comparison mode parameters
        st.divider()
        target_col = st.selectbox(
            "Target Variable for Comparison",
            ["is_fraud", "is_high_value"] if df is not None else []
        )
        st.divider()
        run_comparison = st.button("🔬 Run Comparison", type="primary")

# ============================================
# MAIN CONTENT AREA
# ============================================

# Check if data is loaded
if df is not None:
    # ============================================
    # 📊 DATA QUALITY METRICS
    # ============================================
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Records", len(df))
    with col2:
        fraud_count = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
        fraud_rate_val = (fraud_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric("🚨 Fraud Cases", f"{fraud_count} ({fraud_rate_val:.1f}%)")
    with col3:
        legit_count = len(df) - fraud_count if 'is_fraud' in df.columns else len(df)
        legit_rate = (legit_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric("✅ Legit Cases", f"{legit_count} ({legit_rate:.1f}%)")
    
    # ============================================
    # ⚠️ DATA BALANCE WARNINGS
    # ============================================
    
    if 'is_fraud' in df.columns:
        is_balanced, msg = check_data_balance(df)
        
        if not is_balanced:
            st.markdown(f"""
            <div class="error-box">
                <h4>🚨 Insufficient Fraud Cases for Training!</h4>
                <p><b>Status:</b> {msg}</p>
                <p><b>Recommendations:</b></p>
                <ul>
                    <li>🔄 Use <b>"Generate Balanced Data"</b> in the sidebar</li>
                    <li>📈 Increase the fraud rate to at least 5-10%</li>
                    <li>⚖️ Enable SMOTE and Class Weights (recommended)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Emergency one-click fix
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Generate 10% Fraud Data", use_container_width=True):
                    with st.spinner("Generating balanced data..."):
                        df = generate_balanced_accounting_data(2000, fraud_rate=0.10)
                        st.session_state.df = df
                        st.rerun()
            
            with col2:
                if st.button("🔄 Generate 20% Fraud Data", use_container_width=True):
                    with st.spinner("Generating balanced data..."):
                        df = generate_balanced_accounting_data(2000, fraud_rate=0.20)
                        st.session_state.df = df
                        st.rerun()
        elif fraud_rate_val < 2:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Very Low Fraud Rate</h4>
                <p>Fraud rate is only {fraud_rate_val:.1f}%. For better model training:</p>
                <ul>
                    <li>🔄 Use <b>"Generate Balanced Data"</b> with higher fraud rate</li>
                    <li>⚖️ Enable SMOTE and Class Weights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Data Balance OK</h4>
                <p>{msg}</p>
                <p>📊 Ready for model training!</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# 🎯 COMPARISON MODE
# ============================================

if comparison_mode:
    if df is not None and run_comparison:
        st.markdown("""
        <div class="comparison-header">
            <h3 style="margin: 0;">🔬 Model Comparison Mode</h3>
            <p style="margin: 5px 0 0 0;">Training and comparing multiple models simultaneously...</p>
        </div>
        """, unsafe_allow_html=True)
        
        if target_col:
            run_model_comparison(df, target_col)
        else:
            st.warning("⚠️ Please select a target variable")
    
    elif df is None:
        st.info("👈 Please generate or upload data in the sidebar first.")
    
    else:
        st.info("👈 Configure your comparison settings in the sidebar and click 'Run Comparison'")

# ============================================
# 🎯 SINGLE MODEL MODE
# ============================================

else:
    if df is not None and run_button:
        try:
            # Check if data is balanced enough
            is_balanced, msg = check_data_balance(df) if 'is_fraud' in df.columns else (True, "")
            
            if not is_balanced and model_type == "Supervised":
                st.error(f"""
                ❌ **Cannot train model with insufficient fraud cases!**
                
                {msg}
                
                Please use the **"Generate Balanced Data"** button in the sidebar.
                """)
            else:
                with st.spinner(f"Running {model_choice}..."):
                    # Display data preview
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader("📋 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                    with col2:
                        if 'is_fraud' in df.columns:
                            st.metric("Total Records", len(df))
                            st.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")
                    
                    # Model execution
                    if model_type == "Supervised":
                        # Prepare data
                        X_train, X_test, y_train, y_test, scaler, le = prepare_supervised_data(df, target_col)
                        feature_names = ['client_age', 'annual_income', 'transaction_amount', 
                                        'days_overdue', 'expense_category']
                        
                        # Train model with balancing options
                        if model_choice == "Random Forest Classifier":
                            model, metrics, cm = train_random_forest(
                                X_train, X_test, y_train, y_test, 
                                feature_names, 
                                use_smote=use_smote, 
                                use_class_weight=use_class_weight
                            )
                        elif model_choice == "Logistic Regression":
                            model, metrics, cm = train_logistic_regression(
                                X_train, X_test, y_train, y_test,
                                use_smote=use_smote,
                                use_class_weight=use_class_weight
                            )
                        else:  # XGBoost
                            model, metrics, cm = train_xgboost(
                                X_train, X_test, y_train, y_test,
                                use_smote=use_smote,
                                use_class_weight=use_class_weight
                            )
                        
                        # Display results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("📈 Model Performance")
                            display_metrics(metrics)
                            
                            # Feature importance
                            if hasattr(model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='Feature Importance',
                                    color='Importance',
                                    color_continuous_scale='Viridis'
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("🎯 Confusion Matrix")
                            st.plotly_chart(plot_confusion_matrix(cm, ['Legitimate', 'Fraud']), use_container_width=True)
                            
                            # Educational tooltip
                            with st.expander("📚 Learn About This Model"):
                                st.markdown(get_model_description(model_choice))
                        
                        # Add summary section
                        st.divider()
                        with st.expander("📝 Model Interpretation for Accountants"):
                            st.markdown("""
                            **What this means for your accounting practice:**
                            - The model predicts whether a transaction might be fraudulent
                            - Higher **Precision** means fewer false alarms (less wasted time)
                            - Higher **Recall** means catching more actual frauds
                            - **AUC-ROC** above 0.8 indicates good discriminative power
                            - Feature importance shows which factors most influence fraud likelihood
                            """)
                    
                    else:  # Unsupervised
                        # Prepare data
                        X_scaled, scaler, df_clean = prepare_unsupervised_data(df)
                        
                        if model_choice == "K-Means Clustering":
                            n_clusters = st.session_state.get('n_clusters', 3)
                            model, labels, metrics = run_kmeans(X_scaled, n_clusters)
                            
                        elif model_choice == "DBSCAN":
                            eps = st.session_state.get('eps', 0.5)
                            min_samples = st.session_state.get('min_samples', 5)
                            model, labels, metrics = run_dbscan(X_scaled, eps, min_samples)
                            
                        else:  # Isolation Forest
                            contamination = st.session_state.get('contamination', 0.1)
                            model, labels, metrics = run_isolation_forest(X_scaled, contamination)
                        
                        # Add labels to dataframe
                        df_clean['Cluster'] = labels
                        
                        # Display results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("📊 Cluster Analysis")
                            display_metrics(metrics)
                            
                            # Cluster distribution
                            cluster_counts = df_clean['Cluster'].value_counts().sort_index()
                            fig_dist = px.bar(
                                x=cluster_counts.index.astype(str), 
                                y=cluster_counts.values,
                                labels={'x': 'Cluster', 'y': 'Count'},
                                title="Cluster Distribution",
                                color=cluster_counts.index.astype(str),
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with col2:
                            # Visualization
                            fig_viz = visualize_clusters(X_scaled, labels, f"{model_choice} Results")
                            st.plotly_chart(fig_viz, use_container_width=True)
                            
                            with st.expander("📚 Learn About This Model"):
                                st.markdown(get_model_description(model_choice))
                        
                        # Add summary section
                        st.divider()
                        with st.expander("📝 Model Interpretation for Accountants"):
                            st.markdown("""
                            **What this means for your accounting practice:**
                            - **K-Means/DBSCAN** helps segment clients or transactions into groups
                            - Use clusters to identify high-risk vs low-risk clients
                            - **Isolation Forest** flags transactions that deviate from normal patterns
                            - Investigate outliers flagged by unsupervised methods for potential fraud
                            - These techniques can automate risk assessment and anomaly detection
                            """)
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your data format and try again.")

    elif df is None:
        st.info("👈 Please generate or upload data in the sidebar first.")
    
    else:
        st.info("👈 Configure your model in the sidebar and click 'Run Model'")

# ============================================
# 📊 DATA EXPLORATION SECTION (Always Visible)
# ============================================

if df is not None:
    st.divider()
    with st.expander("📊 Data Explorer - Click to View Full Dataset"):
        st.dataframe(df, use_container_width=True)
        
        # Download data option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"accounting_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================
# FOOTER
# ============================================

st.divider()
st.caption("🔬 Accountancy ML Lab | Built with Streamlit | For Educational and Demonstration Purposes")
