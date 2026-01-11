"""
Evaluation metrics for clustering.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from typing import Optional


def silhouette_score_metric(labels: np.ndarray, features: np.ndarray) -> float:
    """Compute Silhouette Score."""
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(features, labels)


def calinski_harabasz_metric(labels: np.ndarray, features: np.ndarray) -> float:
    """Compute Calinski-Harabasz Index."""
    if len(np.unique(labels)) < 2:
        return 0.0
    return calinski_harabasz_score(features, labels)


def davies_bouldin_metric(labels: np.ndarray, features: np.ndarray) -> float:
    """Compute Davies-Bouldin Index (lower is better)."""
    if len(np.unique(labels)) < 2:
        return float('inf')
    return davies_bouldin_score(features, labels)


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Rand Index."""
    return adjusted_rand_score(labels_true, labels_pred)


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Normalized Mutual Information."""
    return normalized_mutual_info_score(labels_true, labels_pred)


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Cluster Purity.
    
    Purity = (1/n) * sum_k max_j |c_k ∩ t_j|
    where c_k is cluster k and t_j is true class j.
    """
    n = len(labels_true)
    clusters = np.unique(labels_pred)
    classes = np.unique(labels_true)
    
    purity_sum = 0.0
    for cluster in clusters:
        cluster_mask = labels_pred == cluster
        cluster_labels = labels_true[cluster_mask]
        
        if len(cluster_labels) == 0:
            continue
        
        # Count occurrences of each class in this cluster
        class_counts = {}
        for class_label in classes:
            class_counts[class_label] = np.sum(cluster_labels == class_label)
        
        # Find the dominant class
        max_count = max(class_counts.values())
        purity_sum += max_count
    
    return purity_sum / n


def evaluate_clustering(
    labels: np.ndarray,
    features: np.ndarray,
    labels_true: Optional[np.ndarray] = None
) -> dict:
    """
    Evaluate clustering with multiple metrics.
    
    Args:
        labels: Predicted cluster labels
        features: Feature matrix
        labels_true: True labels (optional, for supervised metrics)
    
    Returns:
        Dictionary of metric scores
    """
    metrics = {}
    
    # Unsupervised metrics
    metrics['silhouette_score'] = silhouette_score_metric(labels, features)
    metrics['calinski_harabasz'] = calinski_harabasz_metric(labels, features)
    metrics['davies_bouldin'] = davies_bouldin_metric(labels, features)
    
    # Supervised metrics (if true labels available)
    if labels_true is not None:
        metrics['adjusted_rand_index'] = adjusted_rand_index(labels_true, labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info(labels_true, labels)
        metrics['cluster_purity'] = cluster_purity(labels_true, labels)
    
    return metrics
