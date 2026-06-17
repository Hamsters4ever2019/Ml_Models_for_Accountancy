"""
Model Comparison Module
Side-by-side comparison of multiple ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import time

from model_utils import prepare_supervised_data, get_model_description

# ============================================
# MODEL TRAINERS
# ============================================

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model and return performance metrics"""
    
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        y_proba = None
        auc = None
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': auc,
        'Training Time (s)': training_time,
        'Prediction Time (s)': prediction_time,
    }
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        metrics['CV F1-Mean'] = cv_scores.mean()
        metrics['CV F1-Std'] = cv_scores.std()
    except:
        metrics['CV F1-Mean'] = None
        metrics['CV F1-Std'] = None
    
    return model, metrics, y_pred, y_proba

# ============================================
# COMPARISON VISUALIZATIONS
# ============================================

def create_comparison_table(metrics_df):
    """Create a styled comparison table"""
    # Select relevant columns
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    display_df = metrics_df[display_cols].copy()
    
    # Format numbers
    for col in display_cols[1:]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    # Add ranking
    if 'F1-Score' in metrics_df.columns:
        metrics_df['Rank'] = metrics_df['F1-Score'].rank(ascending=False).astype(int)
        # Add emoji medals
        display_df['🏆'] = metrics_df['Rank'].map({1: '🥇', 2: '🥈', 3: '🥉', 4: '4️⃣'})
    
    return display_df

def create_performance_radar(metrics_df):
    """Create radar chart for model comparison"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for idx, row in metrics_df.iterrows():
        values = [row[m] for m in metrics]
        values.append(values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=row['Model'],
            fill='toself',
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="📊 Model Performance Radar",
        height=500
    )
    
    return fig

def create_performance_bar_chart(metrics_df):
    """Create grouped bar chart"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            text=metrics_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        barmode='group',
        title="📈 Model Performance Comparison",
        yaxis_title="Score",
        xaxis_title="Model",
        height=500,
        legend_title="Metrics"
    )
    
    return fig

def create_heatmap(metrics_df):
    """Create performance heatmap"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Prepare data
    heatmap_data = metrics_df[metrics].values
    model_names = metrics_df['Model'].tolist()
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=metrics,
        y=model_names,
        colorscale='Viridis',
        text=heatmap_data.round(3),
        texttemplate='%{text}',
        textfont={"size": 12},
    ))
    
    fig.update_layout(
        title="🔍 Performance Heatmap",
        height=400,
        xaxis_title="Metrics",
        yaxis_title="Model"
    )
    
    return fig

def create_best_model_card(metrics_df):
    """Create a card highlighting the best model"""
    if 'F1-Score' not in metrics_df.columns:
        return None
    
    best_idx = metrics_df['F1-Score'].idxmax()
    best_model = metrics_df.loc[best_idx]
    
    # Create metrics display
    metrics_display = {
        'F1-Score': f"{best_model['F1-Score']:.3f}",
        'Accuracy': f"{best_model['Accuracy']:.3f}",
        'Precision': f"{best_model['Precision']:.3f}",
        'Recall': f"{best_model['Recall']:.3f}",
    }
    
    return best_model['Model'], metrics_display

def create_scatter_plot(metrics_df):
    """Create scatter plot of Precision vs Recall with size as F1"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metrics_df['Precision'],
        y=metrics_df['Recall'],
        mode='markers+text',
        marker=dict(
            size=metrics_df['F1-Score'] * 50,
            color=metrics_df['F1-Score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="F1-Score")
        ),
        text=metrics_df['Model'],
        textposition="top center",
        hovertemplate=
        "<b>%{text}</b><br>" +
        "Precision: %{x:.3f}<br>" +
        "Recall: %{y:.3f}<br>" +
        "F1-Score: %{marker.size:.1f}<br>" +
        "<extra></extra>"
    ))
    
    fig.update_layout(
        title="🎯 Precision vs Recall Trade-off",
        xaxis_title="Precision",
        yaxis_title="Recall",
        height=500,
        showlegend=False
    )
    
    return fig

def create_confusion_matrix_comparison(models_data, y_test):
    """Compare confusion matrices side by side"""
    n_models = len(models_data)
    cols = min(3, n_models)  # Max 3 per row
    
    fig = make_subplots(
        rows=(n_models + cols - 1) // cols,
        cols=cols,
        subplot_titles=[name for name, _ in models_data],
        specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range((n_models + cols - 1) // cols)]
    )
    
    row, col = 1, 1
    for model_name, y_pred in models_data:
        cm = confusion_matrix(y_test, y_pred)
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=False
            ),
            row=row, col=col
        )
        
        col += 1
        if col > cols:
            col = 1
            row += 1
    
    fig.update_layout(
        title="📋 Confusion Matrix Comparison",
        height=300 * ((n_models + cols - 1) // cols)
    )
    
    return fig

def create_detailed_comparison(metrics_df):
    """Create detailed comparison with all metrics"""
    # Create multiple columns
    cols = st.columns([2, 1])
    
    with cols[0]:
        st.subheader("📊 Performance Comparison")
        
        # Radar chart
        st.plotly_chart(create_performance_radar(metrics_df), use_container_width=True)
        
        # Bar chart
        st.plotly_chart(create_performance_bar_chart(metrics_df), use_container_width=True)
    
    with cols[1]:
        st.subheader("🏆 Model Ranking")
        
        # Best model card
        best_model, metrics = create_best_model_card(metrics_df)
        if best_model:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 10px;
                        color: white;
                        margin-bottom: 20px;">
                <h3 style="margin: 0;">🥇 Best Model</h3>
                <h2 style="margin: 10px 0;">{best_model}</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>📈 F1-Score<br><strong>{metrics['F1-Score']}</strong></div>
                    <div>🎯 Accuracy<br><strong>{metrics['Accuracy']}</strong></div>
                    <div>⚡ Precision<br><strong>{metrics['Precision']}</strong></div>
                    <div>📊 Recall<br><strong>{metrics['Recall']}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Heatmap
        st.plotly_chart(create_heatmap(metrics_df), use_container_width=True)
        
        # Scatter plot
        st.plotly_chart(create_scatter_plot(metrics_df), use_container_width=True)

# ============================================
# COMPARISON EXPLANATION
# ============================================

def get_model_recommendation(metrics_df):
    """Generate a recommendation based on model performance"""
    if 'F1-Score' not in metrics_df.columns:
        return "Insufficient data for recommendation."
    
    best_idx = metrics_df['F1-Score'].idxmax()
    best_model = metrics_df.loc[best_idx]
    
    # Determine use case
    if best_model['Precision'] > 0.9:
        precision_note = "🔒 **Low False Positives**: Great for cases where false alarms are costly."
    else:
        precision_note = "⚠️ **Moderate False Positives**: Consider threshold tuning for production."
    
    if best_model['Recall'] > 0.9:
        recall_note = "🎯 **High Detection Rate**: Excellent at catching most positive cases."
    else:
        recall_note = "📋 **Moderate Detection**: May miss some positive cases."
    
    # Determine speed vs accuracy tradeoff
    if best_model['Training Time (s)'] < 1:
        speed_note = "⚡ **Fast Training**: Suitable for frequent retraining."
    else:
        speed_note = "🔄 **Slower Training**: Consider periodic retraining."
    
    recommendation = f"""
    ### 🎯 Best Model: {best_model['Model']}
    
    **Performance Summary:**
    - F1-Score: {best_model['F1-Score']:.3f}
    - Accuracy: {best_model['Accuracy']:.3f}
    - Precision: {best_model['Precision']:.3f}
    - Recall: {best_model['Recall']:.3f}
    
    **Key Strengths:**
    - {precision_note}
    - {recall_note}
    - {speed_note}
    
    **Recommendations:**
    1. ✅ Use {best_model['Model']} as your primary model
    2. 📈 Monitor performance regularly with new data
    3. 🔧 Fine-tune threshold to balance Precision/Recall
    4. 📊 Consider ensemble methods for even better performance
    """
    
    return recommendation

# ============================================
# MAIN COMPARISON FUNCTION
# ============================================

def run_model_comparison(df, target_col='is_fraud'):
    """Main function to run model comparison"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin-bottom: 20px;">
        <h2 style="margin: 0;">🤖 Model Comparison Studio</h2>
        <p style="margin: 5px 0 0 0;">Compare multiple ML models side by side to find the best performer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    with st.spinner("Preparing data..."):
        X_train, X_test, y_train, y_test, scaler, le = prepare_supervised_data(df, target_col)
        feature_names = ['client_age', 'annual_income', 'transaction_amount', 
                        'days_overdue', 'expense_category']
    
    # Define models to compare
    models_to_compare = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
    }
    
    # Add option to select models
    st.subheader("⚙️ Select Models to Compare")
    
    cols = st.columns(len(models_to_compare))
    selected_models = {}
    
    for col, (name, model) in zip(cols, models_to_compare.items()):
        with col:
            if st.checkbox(f"✅ {name}", value=True):
                selected_models[name] = model
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model to compare")
        return
    
    # Train and evaluate selected models
    st.subheader("🔄 Training Models...")
    
    results = []
    models_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (model_name, model) in enumerate(selected_models.items()):
        status_text.text(f"Training {model_name}...")
        
        try:
            trained_model, metrics, y_pred, y_proba = train_and_evaluate(
                model, model_name, X_train, X_test, y_train, y_test
            )
            results.append(metrics)
            models_data.append((model_name, y_pred))
        except Exception as e:
            st.error(f"❌ Failed to train {model_name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(selected_models))
    
    status_text.text("✅ All models trained!")
    progress_bar.empty()
    
    if not results:
        st.error("❌ No models were successfully trained.")
        return
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Display comparison
    st.divider()
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Models Compared", len(metrics_df))
    with col2:
        best_f1 = metrics_df['F1-Score'].max()
        st.metric("🏆 Best F1-Score", f"{best_f1:.3f}")
    with col3:
        best_model = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
        st.metric("🥇 Best Model", best_model)
    
    # Detailed comparison table
    st.subheader("📋 Performance Comparison Table")
    display_df = create_comparison_table(metrics_df)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Visual comparisons
    create_detailed_comparison(metrics_df)
    
    # Confusion matrix comparison
    st.subheader("📊 Confusion Matrix Comparison")
    if len(models_data) > 1:
        st.plotly_chart(create_confusion_matrix_comparison(models_data, y_test), use_container_width=True)
    
    # Model descriptions
    with st.expander("📚 Learn About Each Model"):
        for model_name in metrics_df['Model']:
            st.markdown(f"### {model_name}")
            st.markdown(get_model_description(model_name))
            st.divider()
    
    # Recommendation
    st.divider()
    st.subheader("💡 Model Recommendation")
    st.markdown(get_model_recommendation(metrics_df))
    
    # Download results
    csv = metrics_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison Results (CSV)",
        data=csv,
        file_name="model_comparison_results.csv",
        mime="text/csv",
        use_container_width=True
    )
