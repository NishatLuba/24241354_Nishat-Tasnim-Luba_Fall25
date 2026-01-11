"""
Easy Task: Basic VAE for music clustering.
- Implement basic VAE
- Extract latent features
- K-Means clustering
- Visualization with t-SNE/UMAP
- Compare with PCA + K-Means baseline
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
from tqdm import tqdm
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vae import BasicVAE, vae_loss
from src.dataset import MusicDataset
from src.clustering import kmeans_clustering
from src.evaluation import evaluate_clustering
from src.preprocessing import split_dataset, create_scaled_datasets

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def train_vae(
    model: BasicVAE,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> list:
    """Train VAE model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        for batch_features, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_features)
            total_loss, recon_loss, kl_loss = vae_loss(recon, batch_features, mu, logvar)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)
        
        losses.append({
            'total': avg_loss,
            'reconstruction': avg_recon,
            'kl': avg_kl
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
    
    return losses


def extract_latent_features(model: BasicVAE, dataloader: DataLoader, device: str = 'cpu') -> np.ndarray:
    """Extract latent features from trained VAE."""
    model = model.to(device)
    model.eval()
    
    latent_features = []
    file_names = []
    
    with torch.no_grad():
        for batch_features, batch_names in dataloader:
            batch_features = batch_features.to(device)
            latent = model.get_latent(batch_features)
            latent_features.append(latent.cpu().numpy())
            file_names.extend(batch_names)
    
    return np.vstack(latent_features), file_names


def extract_genre_labels(file_names: list) -> tuple:
    """
    Extract genre labels from filenames.
    Assumes format: 'genre_genre.XXXXX.wav' or 'genre.XXXXX.wav'
    Returns: (genre_labels_numeric, unique_genres)
    """
    genres = []
    for name in file_names:
        # Extract genre from filename (before first underscore or dot)
        base_name = os.path.splitext(name)[0]  # Remove extension
        # Try to extract genre (e.g., "blues" from "blues_blues.00000")
        parts = base_name.split('_')
        if len(parts) > 0:
            genre = parts[0].lower()
            genres.append(genre)
        else:
            genres.append('unknown')
    
    unique_genres = sorted(list(set(genres)))
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    genre_labels_numeric = np.array([genre_to_idx[g] for g in genres])
    
    return genre_labels_numeric, unique_genres


def visualize_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'umap',
    title: str = 'Cluster Visualization',
    save_path: Optional[str] = None
):
    """Visualize clusters using t-SNE or UMAP."""
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings = reducer.fit_transform(features)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()  # Close figure instead of showing (better for batch processing)


def main():
    """Main function for Easy Task."""
    print("=" * 60)
    print("Easy Task: Basic VAE for Music Clustering")
    print("=" * 60)
    
    # Configuration
    audio_dir = 'data/Audio/audio'
    results_dir = 'results'
    latent_viz_dir = 'results/latent_visualization'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if data directory exists
    if not os.path.exists(audio_dir) or len(os.listdir(audio_dir)) == 0:
        print(f"\n⚠️  Warning: No audio files found in {audio_dir}")
        print("Please add audio files (.wav, .mp3, .flac) to the data/audio directory.")
        print("\nYou can download datasets like:")
        print("- GTZAN Genre Collection: http://marsyas.info/downloads/datasets.html")
        print("- Or use any music dataset with audio files")
        return
    
    # Load dataset
    print("\n1. Loading dataset...")
    full_dataset = MusicDataset(audio_dir, feature_type='mfcc', n_mfcc=13, normalize=False)  # We'll scale properly
    print(f"   Found {len(full_dataset)} audio files")
    
    if len(full_dataset) == 0:
        print("No audio files found. Exiting.")
        return
    
    # Extract genre labels for stratified splitting
    print("\n1.5. Extracting genre labels for stratified splitting...")
    file_names_full = [full_dataset.file_names[i] for i in range(len(full_dataset))]
    genre_labels_full, unique_genres = extract_genre_labels(file_names_full)
    print(f"   Found {len(unique_genres)} genres: {unique_genres}")
    
    # Split into train and test (80/20)
    print("\n1.6. Splitting dataset into train (80%) and test (20%)...")
    train_dataset, test_dataset, train_indices, test_indices = split_dataset(
        full_dataset,
        test_size=0.2,
        random_state=42,
        stratify=genre_labels_full if len(unique_genres) > 1 else None
    )
    print(f"   Train set: {len(train_dataset)} samples ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"   Test set: {len(test_dataset)} samples ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
    
    # Apply scaling (fit on train, apply to both)
    print("\n1.7. Applying StandardScaler (fit on train, apply to both)...")
    train_dataset_scaled, test_dataset_scaled, scaler = create_scaled_datasets(
        train_dataset, test_dataset, full_dataset, train_indices, test_indices, is_hybrid=False
    )
    
    # Get input dimension
    sample_features, _ = train_dataset_scaled[0]
    input_dim = sample_features.shape[0]
    print(f"   Input feature dimension: {input_dim}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset_scaled, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset_scaled, batch_size=32, shuffle=False)
    dataloader = train_dataloader  # Use train for training
    
    # Initialize VAE
    print("\n2. Initializing VAE...")
    latent_dim = 32
    model = BasicVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[128, 64])
    print(f"   Latent dimension: {latent_dim}")
    
    # Train VAE
    print("\n3. Training VAE...")
    losses = train_vae(model, dataloader, epochs=50, lr=1e-3, device=device)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(losses) + 1)
    plt.plot(epochs_range, [l['total'] for l in losses], label='Total Loss')
    plt.plot(epochs_range, [l['reconstruction'] for l in losses], label='Reconstruction Loss')
    plt.plot(epochs_range, [l['kl'] for l in losses], label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved training loss plot to {results_dir}/training_loss.png")
    plt.close()
    
    # Extract latent features (on both train and test)
    print("\n4. Extracting latent features from training set...")
    train_latent_features, train_file_names = extract_latent_features(model, train_dataloader, device=device)
    print(f"   Train latent features shape: {train_latent_features.shape}")
    
    print("\n4.5. Extracting latent features from test set...")
    test_latent_features, test_file_names = extract_latent_features(model, test_dataloader, device=device)
    print(f"   Test latent features shape: {test_latent_features.shape}")
    
    # Use train set for clustering (as per standard practice)
    latent_features = train_latent_features
    file_names = train_file_names
    
    # Extract genre labels from filenames
    print("\n4.6. Extracting genre labels from filenames...")
    genre_labels, unique_genres = extract_genre_labels(file_names)
    print(f"   Found {len(unique_genres)} genres: {unique_genres}")
    print(f"   Genre distribution: {dict(zip(*np.unique(genre_labels, return_counts=True)))}")
    
    # Determine number of clusters (use number of genres if available, otherwise heuristic)
    if len(unique_genres) > 1:
        n_clusters = len(unique_genres)
        print(f"\n5. Performing K-Means clustering (n_clusters={n_clusters}, matching number of genres)...")
    else:
        n_clusters = max(2, int(np.sqrt(len(train_dataset_scaled) / 2)))
        print(f"\n5. Performing K-Means clustering (n_clusters={n_clusters})...")
    vae_labels, kmeans_model = kmeans_clustering(latent_features, n_clusters=n_clusters)
    
    # Evaluate VAE + K-Means
    print("\n6. Evaluating VAE + K-Means clustering...")
    vae_metrics = evaluate_clustering(vae_labels, latent_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    print(f"   Silhouette Score: {vae_metrics['silhouette_score']:.4f}")
    print(f"   Calinski-Harabasz Index: {vae_metrics['calinski_harabasz']:.4f}")
    print(f"   Davies-Bouldin Index: {vae_metrics['davies_bouldin']:.4f}")
    if len(unique_genres) > 1:
        print(f"   [ACCURACY] Adjusted Rand Index (ARI): {vae_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
        print(f"   [ACCURACY] Normalized Mutual Info (NMI): {vae_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"   [ACCURACY] Cluster Purity: {vae_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Baseline: PCA + K-Means
    print("\n7. Baseline: PCA + K-Means...")
    # Get original features from training set (scaled)
    train_features_list = []
    for i in range(len(train_dataset_scaled)):
        features, _ = train_dataset_scaled[i]
        train_features_list.append(features.numpy())
    original_features = np.array(train_features_list)
    
    # Apply PCA (fit on train)
    # PCA n_components cannot exceed min(n_samples, n_features)
    pca_n_components = min(latent_dim, original_features.shape[1], original_features.shape[0] - 1)
    if pca_n_components < 1:
        pca_n_components = 1
    pca = PCA(n_components=pca_n_components)
    pca_features = pca.fit_transform(original_features)
    print(f"   PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # K-Means on PCA features
    pca_labels, _ = kmeans_clustering(pca_features, n_clusters=n_clusters)
    
    # Evaluate PCA + K-Means
    pca_metrics = evaluate_clustering(pca_labels, pca_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    print(f"   Silhouette Score: {pca_metrics['silhouette_score']:.4f}")
    print(f"   Calinski-Harabasz Index: {pca_metrics['calinski_harabasz']:.4f}")
    print(f"   Davies-Bouldin Index: {pca_metrics['davies_bouldin']:.4f}")
    if len(unique_genres) > 1:
        print(f"   [ACCURACY] Adjusted Rand Index (ARI): {pca_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
        print(f"   [ACCURACY] Normalized Mutual Info (NMI): {pca_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"   [ACCURACY] Cluster Purity: {pca_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Compare results
    print("\n8. Comparison:")
    print(f"   VAE Silhouette Score: {vae_metrics['silhouette_score']:.4f} vs PCA: {pca_metrics['silhouette_score']:.4f}")
    print(f"   VAE Calinski-Harabasz: {vae_metrics['calinski_harabasz']:.4f} vs PCA: {pca_metrics['calinski_harabasz']:.4f}")
    if len(unique_genres) > 1:
        print(f"\n   [ACCURACY METRICS] (vs True Genres):")
        print(f"   VAE ARI: {vae_metrics.get('adjusted_rand_index', 'N/A'):.4f} vs PCA ARI: {pca_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
        print(f"   VAE NMI: {vae_metrics.get('normalized_mutual_info', 'N/A'):.4f} vs PCA NMI: {pca_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"   VAE Purity: {vae_metrics.get('cluster_purity', 'N/A'):.4f} vs PCA Purity: {pca_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Evaluate on test set as well
    print("\n7.5. Evaluating on test set...")
    test_genre_labels, _ = extract_genre_labels(test_file_names)
    
    # Extract test latent features and evaluate
    test_vae_labels, _ = kmeans_clustering(test_latent_features, n_clusters=n_clusters)
    test_vae_metrics = evaluate_clustering(test_vae_labels, test_latent_features, labels_true=test_genre_labels if len(unique_genres) > 1 else None)
    
    # Apply PCA to test set
    test_features_list = []
    for i in range(len(test_dataset_scaled)):
        features, _ = test_dataset_scaled[i]
        test_features_list.append(features.numpy())
    test_original_features = np.array(test_features_list)
    test_pca_features = pca.transform(test_original_features)  # Use fitted PCA
    test_pca_labels, _ = kmeans_clustering(test_pca_features, n_clusters=n_clusters)
    test_pca_metrics = evaluate_clustering(test_pca_labels, test_pca_features, labels_true=test_genre_labels if len(unique_genres) > 1 else None)
    
    print(f"   Test Set - VAE Silhouette: {test_vae_metrics['silhouette_score']:.4f}, PCA: {test_pca_metrics['silhouette_score']:.4f}")
    if len(unique_genres) > 1:
        print(f"   Test Set - VAE ARI: {test_vae_metrics.get('adjusted_rand_index', 'N/A'):.4f}, PCA: {test_pca_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
    
    # Save metrics (train set results)
    metrics_dict = {
        'Method': ['VAE + K-Means (Train)', 'PCA + K-Means (Train)', 'VAE + K-Means (Test)', 'PCA + K-Means (Test)'],
        'Silhouette Score': [
            vae_metrics['silhouette_score'], 
            pca_metrics['silhouette_score'],
            test_vae_metrics['silhouette_score'],
            test_pca_metrics['silhouette_score']
        ],
        'Calinski-Harabasz': [
            vae_metrics['calinski_harabasz'], 
            pca_metrics['calinski_harabasz'],
            test_vae_metrics['calinski_harabasz'],
            test_pca_metrics['calinski_harabasz']
        ],
        'Davies-Bouldin': [
            vae_metrics['davies_bouldin'], 
            pca_metrics['davies_bouldin'],
            test_vae_metrics['davies_bouldin'],
            test_pca_metrics['davies_bouldin']
        ]
    }
    if len(unique_genres) > 1:
        metrics_dict['Adjusted Rand Index (ARI)'] = [
            vae_metrics.get('adjusted_rand_index', 0.0),
            pca_metrics.get('adjusted_rand_index', 0.0),
            test_vae_metrics.get('adjusted_rand_index', 0.0),
            test_pca_metrics.get('adjusted_rand_index', 0.0)
        ]
        metrics_dict['Normalized Mutual Info (NMI)'] = [
            vae_metrics.get('normalized_mutual_info', 0.0),
            pca_metrics.get('normalized_mutual_info', 0.0),
            test_vae_metrics.get('normalized_mutual_info', 0.0),
            test_pca_metrics.get('normalized_mutual_info', 0.0)
        ]
        metrics_dict['Cluster Purity'] = [
            vae_metrics.get('cluster_purity', 0.0),
            pca_metrics.get('cluster_purity', 0.0),
            test_vae_metrics.get('cluster_purity', 0.0),
            test_pca_metrics.get('cluster_purity', 0.0)
        ]
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_path = os.path.join(results_dir, 'clustering_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n   Saved metrics to {metrics_path}")
    
    # Visualizations
    print("\n9. Creating visualizations...")
    
    # VAE latent space visualization
    visualize_clusters(
        latent_features,
        vae_labels,
        method='umap',
        title='VAE Latent Space Clustering (UMAP)',
        save_path=os.path.join(latent_viz_dir, 'vae_umap.png')
    )
    
    visualize_clusters(
        latent_features,
        vae_labels,
        method='tsne',
        title='VAE Latent Space Clustering (t-SNE)',
        save_path=os.path.join(latent_viz_dir, 'vae_tsne.png')
    )
    
    # PCA visualization
    visualize_clusters(
        pca_features,
        pca_labels,
        method='umap',
        title='PCA + K-Means Clustering (UMAP)',
        save_path=os.path.join(latent_viz_dir, 'pca_umap.png')
    )
    
    # Save model
    model_path = os.path.join(results_dir, 'vae_model_easy.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n10. Saved VAE model to {model_path}")
    
    print("\n" + "=" * 60)
    print("Easy Task completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
