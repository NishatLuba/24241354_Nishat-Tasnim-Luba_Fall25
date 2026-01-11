"""
Clustering algorithms for music data.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from typing import Tuple, Optional


def kmeans_clustering(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-Means clustering.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Cluster labels and fitted KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels, kmeans


def agglomerative_clustering(
    features: np.ndarray,
    n_clusters: int,
    linkage: str = 'ward'
) -> Tuple[np.ndarray, AgglomerativeClustering]:
    """
    Perform Agglomerative Clustering.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average')
    
    Returns:
        Cluster labels and fitted AgglomerativeClustering model
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = clustering.fit_predict(features)
    return labels, clustering


def dbscan_clustering(
    features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> Tuple[np.ndarray, DBSCAN]:
    """
    Perform DBSCAN clustering.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood
    
    Returns:
        Cluster labels and fitted DBSCAN model
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(features)
    return labels, clustering


def find_optimal_clusters_kmeans(
    features: np.ndarray,
    max_clusters: int = 10,
    random_state: int = 42
) -> int:
    """
    Find optimal number of clusters using elbow method (inertia).
    
    Args:
        features: Feature matrix
        max_clusters: Maximum number of clusters to try
        random_state: Random seed
    
    Returns:
        Optimal number of clusters (based on elbow method)
    """
    inertias = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection (can be improved)
    # For now, return a reasonable default
    return 5
