"""
Visualization utilities for clustering results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from typing import Optional
import os

sns.set_style("whitegrid")


def plot_latent_space(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'umap',
    title: str = 'Latent Space Visualization',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot latent space using t-SNE or UMAP.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        method: 'umap' or 'tsne'
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings = reducer.fit_transform(features)
    elif method == 'tsne':
        # Ensure perplexity is valid (must be < n_samples)
        max_perplexity = min(30, max(5, len(features) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=max_perplexity)
        embeddings = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return embeddings


def plot_cluster_distribution(
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    title: str = 'Cluster Distribution',
    save_path: Optional[str] = None
):
    """
    Plot distribution of clusters.
    
    Args:
        labels: Cluster labels
        true_labels: True labels (optional, for comparison)
        title: Plot title
        save_path: Path to save figure
    """
    if true_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predicted clusters
        unique, counts = np.unique(labels, return_counts=True)
        axes[0].bar(unique, counts, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Predicted Clusters')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # True labels
        unique_true, counts_true = np.unique(true_labels, return_counts=True)
        axes[1].bar(unique_true, counts_true, color='coral', alpha=0.7)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('True Labels')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
    else:
        plt.figure(figsize=(10, 6))
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts, color='steelblue', alpha=0.7)
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.tight_layout()


def plot_reconstruction_examples(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_examples: int = 8,
    title: str = 'Reconstruction Examples',
    save_path: Optional[str] = None
):
    """
    Plot original vs reconstructed features.
    
    Args:
        original: Original features (n_samples, n_features)
        reconstructed: Reconstructed features (n_samples, n_features)
        n_examples: Number of examples to plot
        title: Plot title
        save_path: Path to save figure
    """
    n_examples = min(n_examples, len(original))
    if n_examples == 0:
        print("Warning: No examples to plot for reconstruction")
        return
    
    indices = np.random.choice(len(original), n_examples, replace=False)
    
    # Handle single example case
    if n_examples == 1:
        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        axes = axes.reshape(2, 1)
    else:
        fig, axes = plt.subplots(2, n_examples, figsize=(2 * n_examples, 4))
    
    for i, idx in enumerate(indices):
        # Original
        if n_examples == 1:
            ax_orig = axes[0, 0]
            ax_recon = axes[1, 0]
        else:
            ax_orig = axes[0, i]
            ax_recon = axes[1, i]
        
        ax_orig.plot(original[idx], color='blue', alpha=0.7)
        ax_orig.set_title(f'Original {idx+1}')
        ax_orig.grid(True, alpha=0.3)
        
        # Reconstructed
        ax_recon.plot(reconstructed[idx], color='red', alpha=0.7)
        ax_recon.set_title(f'Reconstructed {idx+1}')
        ax_recon.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")


def plot_metrics_comparison(
    metrics_dict: dict,
    title: str = 'Metrics Comparison',
    save_path: Optional[str] = None
):
    """
    Plot comparison of metrics across different methods.
    
    Args:
        metrics_dict: Dictionary with method names as keys and metric dicts as values
        title: Plot title
        save_path: Path to save figure
    """
    methods = list(metrics_dict.keys())
    metric_names = list(metrics_dict[methods[0]].keys())
    
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(metric_names):
        values = [metrics_dict[method][metric_name] for method in methods]
        axes[i].bar(methods, values, alpha=0.7)
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(metric_name)
        axes[i].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
