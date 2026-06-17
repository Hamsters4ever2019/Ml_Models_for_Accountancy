import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Import our modules
from data_generator import generate_accounting_data
from model_utils import (prepare_supervised_data, prepare_unsupervised_data, 
                         display_metrics, plot_confusion_matrix, plot_feature_importance,
                         get_model_description)
from supervised_models import train_random_forest, train_logistic_regression, train_xgboost
from unsupervised_models import run_kmeans, run_dbscan, run_isolation_forest, visualize_clusters

# Page configuration
st.set_page_config(
    page_title="Accountancy ML Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565C0;
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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Accountancy ML Lab</p>', unsafe_allow_html=True)
st.markdown("*Interactive Machine Learning Demonstrations for Accounting Professionals*")

# Sidebar
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
        if st.button("🔄 Generate Fresh Sample Data"):
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
    
    # Model type selection
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
    
    # Run button
    st.divider()
    run_button = st.button("🚀 Run Model", type="primary")

# Main content
if df is not None and run_button:
    try:
        with st.spinner(f"Running {model_choice}..."):
            # Display data preview
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
            with col2:
                st.metric("Total Records", len(df))
                if 'is_fraud' in df.columns:
                    fraud_rate = df['is_fraud'].mean()
                    st.metric("Fraud Rate", f"{fraud_rate:.2%}")
            
            # Model execution
            if model_type == "Supervised":
                # Prepare data
                X_train, X_test, y_train, y_test, scaler, le = prepare_supervised_data(df, target_col)
                feature_names = ['client_age', 'annual_income', 'transaction_amount', 
                                'days_overdue', 'expense_category']
                
                # Train model
                if model_choice == "Random Forest Classifier":
                    model, metrics, cm = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
                elif model_choice == "Logistic Regression":
                    model, metrics, cm = train_logistic_regression(X_train, X_test, y_train, y_test)
                else:  # XGBoost
                    model, metrics, cm = train_xgboost(X_train, X_test, y_train, y_test)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📈 Model Performance")
                    display_metrics(metrics)
                    
                    # Feature importance
                    if feature_importance := plot_feature_importance(model, feature_names):
                        st.plotly_chart(feature_importance, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Confusion Matrix")
                    st.plotly_chart(plot_confusion_matrix(cm, ['Legitimate', 'Fraud']), use_container_width=True)
                    
                    # Educational tooltip
                    with st.expander("📚 Learn About This Model"):
                        st.markdown(get_model_description(model_choice))
                
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
                    fig_dist = px.bar(x=cluster_counts.index.astype(str), y=cluster_counts.values,
                                     labels={'x': 'Cluster', 'y': 'Count'},
                                     title="Cluster Distribution",
                                     color=cluster_counts.index.astype(str),
                                     color_discrete_sequence=px.colors.qualitative.Set3)
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
                if model_type == "Supervised":
                    st.markdown("""
                    **What this means for your accounting practice:**
                    - The model predicts whether a transaction might be fraudulent
                    - Higher **Precision** means fewer false alarms (less wasted time)
                    - Higher **Recall** means catching more actual frauds
                    - **AUC-ROC** above 0.8 indicates good discriminative power
                    - Feature importance shows which factors most influence fraud likelihood
                    """)
                else:
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

# Footer
st.divider()
st.caption("🔬 Accountancy ML Lab | Built with Streamlit | For Educational and Demonstration Purposes")
