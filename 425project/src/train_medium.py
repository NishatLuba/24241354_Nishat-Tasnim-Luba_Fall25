"""
Medium Task: Enhanced VAE with convolutional architecture and hybrid features.
- Convolutional VAE for spectrograms/MFCC
- Hybrid feature representation (audio + lyrics)
- Multiple clustering algorithms (K-Means, Agglomerative, DBSCAN)
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
from src.dataset import MusicDataset, HybridMusicDataset
from src.clustering import kmeans_clustering, agglomerative_clustering, dbscan_clustering
from src.evaluation import evaluate_clustering
from src.visualization import plot_latent_space, plot_cluster_distribution, plot_metrics_comparison
from src.preprocessing import split_dataset, create_scaled_datasets

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def train_hybrid_vae(
    model: BasicVAE,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> list:
    """Train VAE on hybrid features (audio + lyrics)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        for audio_feat, lyrics_feat, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Concatenate audio and lyrics features
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(hybrid_feat)
            total_loss, recon_loss, kl_loss = vae_loss(recon, hybrid_feat, mu, logvar)
            
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


def extract_latent_hybrid(model: BasicVAE, dataloader: DataLoader, device: str = 'cpu') -> np.ndarray:
    """Extract latent features from hybrid VAE."""
    model = model.to(device)
    model.eval()
    
    latent_features = []
    
    with torch.no_grad():
        for audio_feat, lyrics_feat, _ in dataloader:
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            latent = model.get_latent(hybrid_feat)
            latent_features.append(latent.cpu().numpy())
    
    return np.vstack(latent_features)


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


def main():
    """Main function for Medium Task."""
    print("=" * 60)
    print("Medium Task: Enhanced VAE with Hybrid Features")
    print("=" * 60)
    
    # Configuration
    audio_dir = 'data/Audio/audio'
    lyrics_dir = 'data/Lyrics'
    lyrics_csv = None  # Dataset uses TXT files, not CSV
    results_dir = 'results'
    latent_viz_dir = 'results/latent_visualization'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if data directory exists
    if not os.path.exists(audio_dir) or len(os.listdir(audio_dir)) == 0:
        print(f"\n⚠️  Warning: No audio files found in {audio_dir}")
        print("Please add audio files to the data/audio directory.")
        return
    
    # Hybrid Features (Audio + Lyrics)
    print("\n" + "=" * 60)
    print("Part 2: Hybrid Features (Audio + Lyrics)")
    print("=" * 60)
    
    print("\n1. Loading hybrid dataset...")
    # Check if CSV file exists, otherwise use directory
    use_csv = lyrics_csv is not None and os.path.exists(lyrics_csv)
    use_dir = lyrics_dir is not None and os.path.exists(lyrics_dir) and len(os.listdir(lyrics_dir)) > 0
    
    if use_csv:
        print(f"   Using CSV file: {lyrics_csv}")
    elif use_dir:
        print(f"   Using lyrics directory: {lyrics_dir}")
    else:
        print("   No lyrics found (CSV or directory). Using audio-only features.")
    
    full_hybrid_dataset = HybridMusicDataset(
        audio_dir=audio_dir,
        lyrics_dir=lyrics_dir if use_dir else None,
        lyrics_csv=lyrics_csv if use_csv else None,
        feature_type='mfcc',
        n_mfcc=13,
        lyrics_embedding_dim=64,
        normalize=False  # We'll scale properly
    )
    
    if len(full_hybrid_dataset.lyrics_dict) > 0:
        print(f"   Found {len(full_hybrid_dataset.lyrics_dict)} lyrics files")
    else:
        print("   No lyrics files found. Using audio-only features.")
    
    print(f"   Total samples: {len(full_hybrid_dataset)}")
    
    # Extract genre labels for stratified splitting
    print("\n1.5. Extracting genre labels for stratified splitting...")
    file_names_full = [full_hybrid_dataset.file_names[i] for i in range(len(full_hybrid_dataset))]
    genre_labels_full, unique_genres = extract_genre_labels(file_names_full)
    print(f"   Found {len(unique_genres)} genres: {unique_genres}")
    
    # Split into train and test (80/20)
    print("\n1.6. Splitting dataset into train (80%) and test (20%)...")
    train_hybrid_dataset, test_hybrid_dataset, train_indices, test_indices = split_dataset(
        full_hybrid_dataset,
        test_size=0.2,
        random_state=42,
        stratify=genre_labels_full if len(unique_genres) > 1 else None
    )
    print(f"   Train set: {len(train_hybrid_dataset)} samples ({len(train_hybrid_dataset)/len(full_hybrid_dataset)*100:.1f}%)")
    print(f"   Test set: {len(test_hybrid_dataset)} samples ({len(test_hybrid_dataset)/len(full_hybrid_dataset)*100:.1f}%)")
    
    # Apply scaling (fit on train, apply to both)
    print("\n1.7. Applying StandardScaler (fit on train, apply to both)...")
    train_hybrid_dataset_scaled, test_hybrid_dataset_scaled, scaler = create_scaled_datasets(
        train_hybrid_dataset, test_hybrid_dataset, full_hybrid_dataset, 
        train_indices, test_indices, is_hybrid=True
    )
    
    # Get hybrid feature dimension
    sample_audio, sample_lyrics, _ = train_hybrid_dataset_scaled[0]
    hybrid_dim = sample_audio.shape[0] + sample_lyrics.shape[0]
    print(f"   Hybrid feature dimension: {hybrid_dim} (audio: {sample_audio.shape[0]}, lyrics: {sample_lyrics.shape[0]})")
    
    # Create dataloaders
    train_hybrid_dataloader = DataLoader(train_hybrid_dataset_scaled, batch_size=32, shuffle=True)
    test_hybrid_dataloader = DataLoader(test_hybrid_dataset_scaled, batch_size=32, shuffle=False)
    hybrid_dataloader = train_hybrid_dataloader  # Use train for training
    
    # Initialize and train hybrid VAE
    print("\n2. Initializing Hybrid VAE...")
    latent_dim = 32
    hybrid_vae = BasicVAE(input_dim=hybrid_dim, latent_dim=latent_dim, hidden_dims=[256, 128, 64])
    print(f"   Latent dimension: {latent_dim}")
    
    print("\n3. Training Hybrid VAE...")
    hybrid_losses = train_hybrid_vae(hybrid_vae, hybrid_dataloader, epochs=50, lr=1e-3, device=device)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(hybrid_losses) + 1)
    plt.plot(epochs_range, [l['total'] for l in hybrid_losses], label='Total Loss')
    plt.plot(epochs_range, [l['reconstruction'] for l in hybrid_losses], label='Reconstruction Loss')
    plt.plot(epochs_range, [l['kl'] for l in hybrid_losses], label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Hybrid VAE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'hybrid_vae_training_loss.png'), dpi=300, bbox_inches='tight')
    print(f"   Saved training loss plot to {results_dir}/hybrid_vae_training_loss.png")
    plt.close()
    
    # Extract latent features (on both train and test)
    print("\n4. Extracting latent features from training set...")
    train_hybrid_latent_features = extract_latent_hybrid(hybrid_vae, train_hybrid_dataloader, device=device)
    print(f"   Train latent features shape: {train_hybrid_latent_features.shape}")
    
    print("\n4.5. Extracting latent features from test set...")
    test_hybrid_latent_features = extract_latent_hybrid(hybrid_vae, test_hybrid_dataloader, device=device)
    print(f"   Test latent features shape: {test_hybrid_latent_features.shape}")
    
    # Use train set for clustering (as per standard practice)
    hybrid_latent_features = train_hybrid_latent_features
    
    # Extract genre labels from filenames (train set)
    print("\n4.6. Extracting genre labels from filenames...")
    train_file_names = [full_hybrid_dataset.file_names[train_indices[i]] for i in range(len(train_hybrid_dataset_scaled))]
    genre_labels, unique_genres = extract_genre_labels(train_file_names)
    print(f"   Found {len(unique_genres)} genres: {unique_genres}")
    print(f"   Genre distribution: {dict(zip(*np.unique(genre_labels, return_counts=True)))}")
    
    # Multiple Clustering Algorithms
    print("\n" + "=" * 60)
    print("Multiple Clustering Algorithms")
    print("=" * 60)
    
    # Use number of genres if available, otherwise heuristic
    if len(unique_genres) > 1:
        n_clusters = len(unique_genres)
        print(f"\n5. Performing clustering with n_clusters={n_clusters} (matching number of genres)...")
    else:
        n_clusters = max(2, int(np.sqrt(len(train_hybrid_dataset_scaled) / 2)))
        print(f"\n5. Performing clustering with n_clusters={n_clusters}...")
    
    all_results = {}
    
    # K-Means
    print("\n   a) K-Means Clustering...")
    kmeans_labels, _ = kmeans_clustering(hybrid_latent_features, n_clusters=n_clusters)
    kmeans_metrics = evaluate_clustering(kmeans_labels, hybrid_latent_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    all_results['K-Means'] = kmeans_metrics
    print(f"      Silhouette Score: {kmeans_metrics['silhouette_score']:.4f}")
    print(f"      Calinski-Harabasz: {kmeans_metrics['calinski_harabasz']:.4f}")
    print(f"      Davies-Bouldin: {kmeans_metrics['davies_bouldin']:.4f}")
    if len(unique_genres) > 1:
        print(f"      [ACCURACY] Adjusted Rand Index (ARI): {kmeans_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
        print(f"      [ACCURACY] Normalized Mutual Info (NMI): {kmeans_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"      [ACCURACY] Cluster Purity: {kmeans_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Agglomerative Clustering
    print("\n   b) Agglomerative Clustering...")
    agg_labels, _ = agglomerative_clustering(hybrid_latent_features, n_clusters=n_clusters, linkage='ward')
    agg_metrics = evaluate_clustering(agg_labels, hybrid_latent_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    all_results['Agglomerative'] = agg_metrics
    print(f"      Silhouette Score: {agg_metrics['silhouette_score']:.4f}")
    print(f"      Calinski-Harabasz: {agg_metrics['calinski_harabasz']:.4f}")
    print(f"      Davies-Bouldin: {agg_metrics['davies_bouldin']:.4f}")
    if len(unique_genres) > 1:
        print(f"      [ACCURACY] Adjusted Rand Index (ARI): {agg_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
        print(f"      [ACCURACY] Normalized Mutual Info (NMI): {agg_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"      [ACCURACY] Cluster Purity: {agg_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # DBSCAN
    print("\n   c) DBSCAN Clustering...")
    # DBSCAN doesn't require n_clusters, but we need to set eps and min_samples
    eps = 0.5
    min_samples = max(3, len(train_hybrid_dataset_scaled) // 20)
    dbscan_labels, dbscan_model = dbscan_clustering(hybrid_latent_features, eps=eps, min_samples=min_samples)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"      Found {n_clusters_dbscan} clusters (eps={eps}, min_samples={min_samples})")
    if n_clusters_dbscan > 1:
        dbscan_metrics = evaluate_clustering(dbscan_labels, hybrid_latent_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
        all_results['DBSCAN'] = dbscan_metrics
        print(f"      Silhouette Score: {dbscan_metrics['silhouette_score']:.4f}")
        print(f"      Calinski-Harabasz: {dbscan_metrics['calinski_harabasz']:.4f}")
        print(f"      Davies-Bouldin: {dbscan_metrics['davies_bouldin']:.4f}")
        if len(unique_genres) > 1:
            print(f"      [ACCURACY] Adjusted Rand Index (ARI): {dbscan_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
            print(f"      [ACCURACY] Normalized Mutual Info (NMI): {dbscan_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
            print(f"      [ACCURACY] Cluster Purity: {dbscan_metrics.get('cluster_purity', 'N/A'):.4f}")
    else:
        print("      DBSCAN found too few clusters for evaluation")
    
    # Visualizations
    print("\n6. Creating visualizations...")
    
    # K-Means visualization
    plot_latent_space(
        hybrid_latent_features,
        kmeans_labels,
        method='umap',
        title='Hybrid VAE + K-Means Clustering (UMAP)',
        save_path=os.path.join(latent_viz_dir, 'hybrid_kmeans_umap.png')
    )
    
    plot_latent_space(
        hybrid_latent_features,
        kmeans_labels,
        method='tsne',
        title='Hybrid VAE + K-Means Clustering (t-SNE)',
        save_path=os.path.join(latent_viz_dir, 'hybrid_kmeans_tsne.png')
    )
    
    # Agglomerative visualization
    plot_latent_space(
        hybrid_latent_features,
        agg_labels,
        method='umap',
        title='Hybrid VAE + Agglomerative Clustering (UMAP)',
        save_path=os.path.join(latent_viz_dir, 'hybrid_agg_umap.png')
    )
    
    # Metrics comparison
    plot_metrics_comparison(
        all_results,
        title='Clustering Algorithms Comparison',
        save_path=os.path.join(results_dir, 'clustering_comparison_medium.png')
    )
    
    # Save metrics
    metrics_data = []
    for method, metrics in all_results.items():
        row = {
            'Method': method,
            'Silhouette Score': metrics['silhouette_score'],
            'Calinski-Harabasz': metrics['calinski_harabasz'],
            'Davies-Bouldin': metrics['davies_bouldin']
        }
        if len(unique_genres) > 1:
            row['Adjusted Rand Index (ARI)'] = metrics.get('adjusted_rand_index', 0.0)
            row['Normalized Mutual Info (NMI)'] = metrics.get('normalized_mutual_info', 0.0)
            row['Cluster Purity'] = metrics.get('cluster_purity', 0.0)
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(results_dir, 'clustering_metrics_medium.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n   Saved metrics to {metrics_path}")
    
    # Print accuracy summary
    if len(unique_genres) > 1:
        print("\n" + "=" * 60)
        print("[ACCURACY METRICS SUMMARY] (vs True Genres)")
        print("=" * 60)
        for method, metrics in all_results.items():
            print(f"\n{method}:")
            print(f"  ARI: {metrics.get('adjusted_rand_index', 'N/A'):.4f}")
            print(f"  NMI: {metrics.get('normalized_mutual_info', 'N/A'):.4f}")
            print(f"  Purity: {metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Save model
    model_path = os.path.join(results_dir, 'hybrid_vae_model_medium.pth')
    torch.save(hybrid_vae.state_dict(), model_path)
    print(f"\n7. Saved Hybrid VAE model to {model_path}")
    
    print("\n" + "=" * 60)
    print("Medium Task completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
