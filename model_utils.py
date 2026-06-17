import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, mean_squared_error, 
                            r2_score, silhouette_score, davies_bouldin_score)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# ENHANCED DATA PREPARATION WITH SMART BALANCING
# ============================================

def prepare_supervised_data(df, target_col='is_fraud'):
    """
    Prepare data for supervised learning with automatic imbalance handling.
    Shows clear warnings and statistics about class distribution.
    """
    df_clean = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    df_clean['expense_category_encoded'] = le.fit_transform(df_clean['expense_category'])
    
    # Select features
    feature_cols = ['client_age', 'annual_income', 'transaction_amount', 
                    'days_overdue', 'expense_category_encoded']
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # ============================================
    # 📊 CLASS DISTRIBUTION ANALYSIS
    # ============================================
    class_counts = y.value_counts()
    total_samples = len(y)
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
    
    # Display class distribution
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;">
        <h4 style="margin: 0;">📊 Data Distribution Analysis</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create distribution display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Samples", total_samples)
    with col2:
        # Show fraud count and percentage
        if target_col in df.columns:
            fraud_count = df[target_col].sum()
            fraud_pct = (fraud_count / len(df)) * 100
            st.metric("🚨 Fraud Cases", f"{fraud_count} ({fraud_pct:.1f}%)")
    with col3:
        legit_count = len(df) - fraud_count if target_col in df.columns else len(df)
        legit_pct = (legit_count / len(df)) * 100
        st.metric("✅ Legit Cases", f"{legit_count} ({legit_pct:.1f}%)")
    
    # Show imbalance warning
    if imbalance_ratio > 3:
        st.warning(f"""
        ⚠️ **Class Imbalance Detected!**
        
        - **Imbalance Ratio**: {imbalance_ratio:.1f}:1
        - **Minority Class**: Only {min_class_count} samples
        - **Majority Class**: {max_class_count} samples
        
        **Impact on Model**: The model may be biased toward the majority class.
        **Solution**: Using random split (without stratification) to handle this.
        """)
    else:
        st.success(f"✅ Dataset is relatively balanced (Ratio: {imbalance_ratio:.1f}:1)")
    
    # ============================================
    # 🔄 SMART DATA SPLITTING
    # ============================================
    
    # Try stratified split first (if there are enough samples in each class)
    try:
        # Check if each class has at least 2 samples (minimum for stratification)
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            st.success("✅ Using **Stratified Split** (preserves class distribution)")
        else:
            raise ValueError("Not enough samples in minority class for stratification")
            
    except (ValueError, TypeError) as e:
        # Fallback to non-stratified split
        st.info("🔄 Using **Random Split** (stratification skipped due to limited data)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # ============================================
    # 📊 SPLIT DISTRIBUTION ANALYSIS
    # ============================================
    
    # Show split sizes
    st.markdown("""
    <div style="background: #f0f2f6;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;">
        <h5 style="margin: 0;">📋 Data Split Summary</h5>
    </div>
    """, unsafe_allow_html=True)
    
    split_col1, split_col2 = st.columns(2)
    with split_col1:
        st.metric("🔄 Training Set", f"{len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        # Show training set fraud rate
        train_fraud_rate = y_train.sum() / len(y_train) if y_train.sum() > 0 else 0
        st.metric("📊 Train Fraud Rate", f"{train_fraud_rate:.2%}")
    
    with split_col2:
        st.metric("🧪 Test Set", f"{len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        # Show test set fraud rate
        test_fraud_rate = y_test.sum() / len(y_test) if y_test.sum() > 0 else 0
        st.metric("📊 Test Fraud Rate", f"{test_fraud_rate:.2%}")
    
    # ============================================
    # 📊 BALANCING OPTIONS (if needed)
    # ============================================
    
    # Show balancing options if imbalance is severe
    if imbalance_ratio > 5:
        st.markdown("""
        <div style="background: #fff3cd;
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #ffc107;
                    margin: 10px 0;">
            <h5 style="margin: 0; color: #856404;">💡 Balancing Options</h5>
            <p style="margin: 5px 0 0 0; color: #856404;">
                Consider these techniques for better performance:
                • 🔄 SMOTE (Synthetic Minority Oversampling)
                • 📊 Class weights in model training
                • 🎯 Use F1-Score as evaluation metric (not accuracy)
                • 📈 Collect more minority class samples
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # 📊 VISUALIZE CLASS DISTRIBUTION
    # ============================================
    
    # Plot class distribution before split
    fig_dist = go.Figure()
    
    # Training set distribution
    train_dist = y_train.value_counts()
    fig_dist.add_trace(go.Bar(
        name='Training Set',
        x=['Legit', 'Fraud'] if len(train_dist) == 2 else ['Class 0', 'Class 1'],
        y=train_dist.values,
        text=train_dist.values,
        textposition='auto',
        marker_color=['#2ecc71', '#e74c3c']
    ))
    
    # Test set distribution
    test_dist = y_test.value_counts()
    fig_dist.add_trace(go.Bar(
        name='Test Set',
        x=['Legit', 'Fraud'] if len(test_dist) == 2 else ['Class 0', 'Class 1'],
        y=test_dist.values,
        text=test_dist.values,
        textposition='auto',
        marker_color=['#27ae60', '#c0392b']
    ))
    
    fig_dist.update_layout(
        title="Class Distribution: Training vs Test Sets",
        barmode='group',
        height=350,
        yaxis_title="Number of Samples",
        xaxis_title="Class",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # ============================================
    # 📊 FEATURE STATISTICS
    # ============================================
    
    with st.expander("📊 Feature Statistics"):
        st.dataframe(X.describe(), use_container_width=True)
    
    # ============================================
    # 🎯 SCALING
    # ============================================
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Show scaling info
    st.success("✅ Data scaled successfully (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

# ============================================
# LEGACY FUNCTIONS (Keep for compatibility)
# ============================================

def prepare_unsupervised_data(df):
    """Prepare data for unsupervised learning"""
    df_clean = df.copy()
    
    # Encode categorical
    le = LabelEncoder()
    df_clean['expense_category_encoded'] = le.fit_transform(df_clean['expense_category'])
    
    feature_cols = ['client_age', 'annual_income', 'transaction_amount', 
                    'days_overdue', 'expense_category_encoded']
    X = df_clean[feature_cols]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, df_clean

def display_metrics(metrics_dict, title="Model Performance"):
    """Display metrics in a clean Streamlit layout"""
    st.subheader(f"📊 {title}")
    
    cols = st.columns(len(metrics_dict))
    for col, (metric_name, value) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(label=metric_name, value=f"{value:.4f}" if isinstance(value, float) else value)

def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix using Plotly"""
    fig = px.imshow(cm, 
                    text_auto=True, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels,
                    y=labels,
                    color_continuous_scale="Blues",
                    title="Confusion Matrix")
    fig.update_layout(height=500)
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        fig = px.bar(x=importance, y=feature_names, 
                     orientation='h',
                     title="Feature Importance",
                     labels={'x': 'Importance', 'y': 'Feature'},
                     color=importance,
                     color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        return fig
    return None

def get_model_description(model_name):
    """Educational tooltips for each model"""
    descriptions = {
        'Random Forest Classifier': """
        **Random Forest** is an ensemble learning method that combines multiple decision trees.
        
        **How it works:** It creates many decision trees on different subsets of data and averages their predictions.
        
        **Accounting Use Case:** Detecting fraudulent transactions by learning patterns from historical fraud cases.
        
        **Strengths:** Handles non-linear relationships, works well with mixed data types, provides feature importance.
        
        **Limitations:** Can be slower to train, less interpretable than single decision trees.
        """,
        
        'Logistic Regression': """
        **Logistic Regression** is a statistical model for binary classification.
        
        **How it works:** It estimates the probability that a transaction is fraudulent using a logistic function.
        
        **Accounting Use Case:** Predicting client payment default or classifying expense categories.
        
        **Strengths:** Highly interpretable, fast to train, provides probability scores.
        
        **Limitations:** Assumes linear relationships, sensitive to outliers.
        """,
        
        'XGBoost': """
        **XGBoost** (Extreme Gradient Boosting) is a powerful gradient boosting algorithm.
        
        **How it works:** Builds models sequentially, each correcting errors of previous ones.
        
        **Accounting Use Case:** Forecasting cash flow or predicting financial risk.
        
        **Strengths:** State-of-the-art performance, handles missing data, prevents overfitting.
        
        **Limitations:** Can be overkill for small datasets, many hyperparameters to tune.
        """,
        
        'K-Means Clustering': """
        **K-Means** is a partitioning clustering algorithm.
        
        **How it works:** Groups data into K clusters based on similarity, minimizing within-cluster variance.
        
        **Accounting Use Case:** Segmenting clients by spending behavior or risk profiles.
        
        **Strengths:** Simple, fast, scalable to large datasets.
        
        **Limitations:** Requires specifying K, assumes spherical clusters, sensitive to outliers.
        """,
        
        'DBSCAN': """
        **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).
        
        **How it works:** Groups dense regions of points together, identifies outliers as noise.
        
        **Accounting Use Case:** Detecting anomalous transactions or expense outliers.
        
        **Strengths:** Finds arbitrary shaped clusters, robust to outliers, doesn't need K.
        
        **Limitations:** Struggles with varying density clusters, parameter sensitive.
        """,
        
        'Isolation Forest': """
        **Isolation Forest** is an anomaly detection algorithm.
        
        **How it works:** It isolates outliers by randomly selecting features and split values.
        
        **Accounting Use Case:** Detecting unusual transactions that could indicate fraud.
        
        **Strengths:** Efficient, works well with high-dimensional data, identifies global outliers.
        
        **Limitations:** Can struggle with local outliers, less interpretable.
        """
    }
    return descriptions.get(model_name, "Model description not available.")
