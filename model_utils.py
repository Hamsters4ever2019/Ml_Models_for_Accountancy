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

def prepare_supervised_data(df, target_col='is_fraud'):
    """Prepare data for supervised learning with automatic imbalance handling"""
    df_clean = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    df_clean['expense_category_encoded'] = le.fit_transform(df_clean['expense_category'])
    
    # Select features
    feature_cols = ['client_age', 'annual_income', 'transaction_amount', 
                    'days_overdue', 'expense_category_encoded']
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Check class distribution
    class_counts = y.value_counts()
    
    # Try stratified split first
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback to non-stratified split
        st.warning("⚠️ Using random split (stratification failed due to class imbalance)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

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
