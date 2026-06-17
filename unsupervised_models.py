from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def run_kmeans(X_scaled, n_clusters=3):
    """Run K-Means clustering"""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    
    # Metrics
    silhouette = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    
    return model, labels, {'Silhouette Score': silhouette, 'Davies-Bouldin Score': db_score}

def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    """Run DBSCAN clustering"""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        'Number of Clusters': n_clusters,
        'Noise Points': n_noise,
        'Noise Ratio': f"{n_noise/len(labels):.2%}"
    }
    
    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
        metrics['Silhouette Score'] = silhouette
    
    return model, labels, metrics

def run_isolation_forest(X_scaled, contamination=0.1):
    """Run Isolation Forest for anomaly detection"""
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X_scaled)
    
    # Convert to binary (1 for inliers, -1 for outliers)
    n_outliers = (labels == -1).sum()
    metrics = {
        'Outliers Detected': n_outliers,
        'Outlier Ratio': f"{n_outliers/len(labels):.2%}",
        'Inliers': (labels == 1).sum()
    }
    
    return model, labels, metrics

def visualize_clusters(X_scaled, labels, title="Cluster Visualization", n_components=2):
    """Visualize clusters using PCA"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    df_plot['Cluster'] = labels.astype(str)
    
    if n_components == 2:
        fig = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster', 
                         title=title, 
                         color_discrete_sequence=px.colors.qualitative.Set3)
    else:
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='Cluster',
                           title=title, color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(height=600)
    return fig
