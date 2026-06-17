"""
Supervised Models with SMOTE and Class Weight Support
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from model_utils import display_metrics, plot_confusion_matrix, plot_feature_importance

# ============================================
# TRAINERS WITH SMOTE & CLASS WEIGHTS
# ============================================

def train_random_forest(X_train, X_test, y_train, y_test, feature_names, use_smote=True, use_class_weight=True):
    """
    Train Random Forest with SMOTE and Class Weight options
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
                padding: 15px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;">
        <h4 style="margin: 0;">🌲 Training Random Forest with Smart Balancing</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Show configuration
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 SMOTE: {'✅ Enabled' if use_smote else '❌ Disabled'}")
    with col2:
        st.info(f"⚖️ Class Weights: {'✅ Enabled' if use_class_weight else '❌ Disabled'}")
    
    # Apply SMOTE if enabled
    if use_smote:
        try:
            # Count samples before SMOTE
            before_count = len(y_train)
            fraud_before = y_train.sum()
            
            # Check if we have enough samples for SMOTE
            min_class_count = y_train.value_counts().min()
            if min_class_count >= 2:
                # Apply SMOTE
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                
                # Show SMOTE results
                after_count = len(y_train_resampled)
                fraud_after = y_train_resampled.sum()
                
                st.success(f"""
                ✅ **SMOTE Applied Successfully!**
                - Training samples: {before_count} → {after_count} (+{after_count - before_count})
                - Fraud cases: {fraud_before} → {fraud_after} (+{fraud_after - fraud_before})
                - Balancing ratio: 1:1
                """)
                
                X_train_balanced = X_train_resampled
                y_train_balanced = y_train_resampled
            else:
                st.warning(f"⚠️ Not enough samples for SMOTE (need at least 2 per class). Using original data.")
                X_train_balanced = X_train
                y_train_balanced = y_train
                
        except Exception as e:
            st.warning(f"⚠️ SMOTE failed: {str(e)}. Using original data.")
            X_train_balanced = X_train
            y_train_balanced = y_train
    else:
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    # Create model with class weights if enabled
    if use_class_weight:
        # Calculate class weights automatically
        classes = np.unique(y_train_balanced)
        if len(classes) >= 2:
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
            class_weight_dict = dict(zip(classes, class_weights))
            
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight=class_weight_dict,
                n_jobs=-1
            )
            st.info(f"⚖️ Using balanced class weights: {class_weight_dict}")
        else:
            st.warning("⚠️ Only one class detected. Using default weights.")
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            )
    else:
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        )
    
    # Train model
    with st.spinner("Training Random Forest..."):
        model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Show detailed classification report
    with st.expander("📊 Detailed Classification Report"):
        report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])
        st.code(report)
    
    return model, metrics, cm

def train_logistic_regression(X_train, X_test, y_train, y_test, use_smote=True, use_class_weight=True):
    """
    Train Logistic Regression with SMOTE and Class Weight options
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;">
        <h4 style="margin: 0;">📈 Training Logistic Regression with Smart Balancing</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Show configuration
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 SMOTE: {'✅ Enabled' if use_smote else '❌ Disabled'}")
    with col2:
        st.info(f"⚖️ Class Weights: {'✅ Enabled' if use_class_weight else '❌ Disabled'}")
    
    # Apply SMOTE if enabled
    if use_smote:
        try:
            before_count = len(y_train)
            fraud_before = y_train.sum()
            
            min_class_count = y_train.value_counts().min()
            if min_class_count >= 2:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                
                after_count = len(y_train_resampled)
                fraud_after = y_train_resampled.sum()
                
                st.success(f"""
                ✅ **SMOTE Applied Successfully!**
                - Training samples: {before_count} → {after_count} (+{after_count - before_count})
                - Fraud cases: {fraud_before} → {fraud_after} (+{fraud_after - fraud_before})
                - Balancing ratio: 1:1
                """)
                
                X_train_balanced = X_train_resampled
                y_train_balanced = y_train_resampled
            else:
                st.warning(f"⚠️ Not enough samples for SMOTE. Using original data.")
                X_train_balanced = X_train
                y_train_balanced = y_train
                
        except Exception as e:
            st.warning(f"⚠️ SMOTE failed: {str(e)}. Using original data.")
            X_train_balanced = X_train
            y_train_balanced = y_train
    else:
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    # Create model with class weights if enabled
    if use_class_weight:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        st.info("⚖️ Using balanced class weights")
    else:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    
    # Train model
    with st.spinner("Training Logistic Regression..."):
        model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Show detailed classification report
    with st.expander("📊 Detailed Classification Report"):
        report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])
        st.code(report)
    
    # Show coefficients
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': ['client_age', 'annual_income', 'transaction_amount', 
                       'days_overdue', 'expense_category'],
            'Coefficient': model.coef_[0]
        })
        fig = px.bar(
            coef_df,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Feature Coefficients',
            color='Coefficient',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    return model, metrics, cm

def train_xgboost(X_train, X_test, y_train, y_test, use_smote=True, use_class_weight=True):
    """
    Train XGBoost with SMOTE and Class Weight options
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 15px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;">
        <h4 style="margin: 0;">⚡ Training XGBoost with Smart Balancing</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Show configuration
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 SMOTE: {'✅ Enabled' if use_smote else '❌ Disabled'}")
    with col2:
        st.info(f"⚖️ Scale Pos Weight: {'✅ Enabled' if use_class_weight else '❌ Disabled'}")
    
    # Calculate class weights for XGBoost
    if use_class_weight:
        # Calculate scale_pos_weight (ratio of negative to positive)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
        st.info(f"⚖️ Scale Pos Weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1
    
    # Apply SMOTE if enabled
    if use_smote:
        try:
            before_count = len(y_train)
            fraud_before = y_train.sum()
            
            min_class_count = y_train.value_counts().min()
            if min_class_count >= 2:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                
                after_count = len(y_train_resampled)
                fraud_after = y_train_resampled.sum()
                
                st.success(f"""
                ✅ **SMOTE Applied Successfully!**
                - Training samples: {before_count} → {after_count} (+{after_count - before_count})
                - Fraud cases: {fraud_before} → {fraud_after} (+{fraud_after - fraud_before})
                - Balancing ratio: 1:1
                """)
                
                X_train_balanced = X_train_resampled
                y_train_balanced = y_train_resampled
            else:
                st.warning(f"⚠️ Not enough samples for SMOTE. Using original data.")
                X_train_balanced = X_train
                y_train_balanced = y_train
                
        except Exception as e:
            st.warning(f"⚠️ SMOTE failed: {str(e)}. Using original data.")
            X_train_balanced = X_train
            y_train_balanced = y_train
    else:
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    # Create XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
    
    # Train model
    with st.spinner("Training XGBoost..."):
        model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Show detailed classification report
    with st.expander("📊 Detailed Classification Report"):
        report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])
        st.code(report)
    
    # Show feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': ['client_age', 'annual_income', 'transaction_amount', 
                       'days_overdue', 'expense_category'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (Balanced Model)',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    return model, metrics, cm
