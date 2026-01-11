"""
Hard Task: Advanced VAE architectures with comprehensive evaluation.
- CVAE or Beta-VAE for disentangled representations
- Multi-modal clustering (audio + lyrics + genre)
- Comprehensive evaluation (all metrics)
- Detailed visualizations
- Comparison with all baseline methods
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
from typing import Optional, Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vae import BasicVAE, ConditionalVAE, vae_loss
from src.dataset import MusicDataset, HybridMusicDataset
from src.clustering import kmeans_clustering, agglomerative_clustering, dbscan_clustering
from src.evaluation import evaluate_clustering
from src.visualization import (
    plot_latent_space, plot_cluster_distribution, 
    plot_reconstruction_examples, plot_metrics_comparison
)
from src.preprocessing import split_dataset, create_scaled_datasets

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def train_beta_vae(
    model: BasicVAE,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 4.0,
    device: str = 'cpu'
) -> list:
    """Train Beta-VAE model."""
    model = model.to(device)
    model.beta = beta  # Set beta parameter
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        for audio_feat, lyrics_feat, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            batch_features = torch.cat([audio_feat, lyrics_feat], dim=1)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_features)
            total_loss, recon_loss, kl_loss = vae_loss(recon, batch_features, mu, logvar, beta=beta)
            
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


def train_cvae(
    model: ConditionalVAE,
    dataloader: DataLoader,
    conditions: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = 'cpu'
) -> list:
    """Train Conditional VAE model."""
    model = model.to(device)
    model.beta = beta
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        for idx, (audio_feat, lyrics_feat, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            batch_features = torch.cat([audio_feat, lyrics_feat], dim=1)
            # Get conditions for current batch (handle last batch which might be smaller)
            batch_start = idx * dataloader.batch_size
            batch_end = min(batch_start + len(audio_feat), len(conditions))
            batch_conditions = conditions[batch_start:batch_end].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_features, batch_conditions)
            total_loss, recon_loss, kl_loss = vae_loss(recon, batch_features, mu, logvar, beta=beta)
            
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


def create_genre_conditions(file_names: List[str], genres: Optional[Dict[str, str]] = None, audio_paths: Optional[List[str]] = None) -> np.ndarray:
    """
    Create genre condition vectors (one-hot encoding).
    If genres not provided, infer from file names or folder structure.
    
    Args:
        file_names: List of audio file names
        genres: Optional dictionary mapping file names to genres
        audio_paths: Optional list of full audio file paths (to extract genre from folder structure)
    """
    if genres is None:
        genres = {}
        
        # If audio_paths provided, extract genre from folder structure
        if audio_paths and len(audio_paths) == len(file_names):
            for i, path in enumerate(audio_paths):
                name = file_names[i] if i < len(file_names) else os.path.basename(path)
                # Extract genre from path (e.g., 'data/Audio/audio/spanish/reggaeton/song.mp3')
                path_parts = path.replace('\\', '/').split('/')
                
                # Check for Spanish subgenres first
                if 'spanish' in path_parts:
                    spanish_idx = path_parts.index('spanish')
                    if spanish_idx + 1 < len(path_parts):
                        subgenre = path_parts[spanish_idx + 1]
                        # Skip if it's the filename itself
                        if subgenre.endswith('.mp3') or subgenre.endswith('.wav'):
                            # No subgenre, just use 'latin'
                            genres[name] = 'latin'
                        else:
                            # Map Spanish subgenres to main genres
                            subgenre_map = {
                                'reggaeton': 'latin', 'latin_pop': 'latin', 'latin_rock': 'rock',
                                'latin_trap': 'hiphop', 'latin_indie': 'indie', 'latin_folk': 'folk',
                                'salsa': 'latin', 'bachata': 'latin', 'mariachi': 'latin',
                                'flamenco': 'latin', 'flamenco_pop': 'latin', 'corrido': 'latin',
                                'norteno': 'latin', 'tejano': 'latin', 'vallenato': 'latin',
                                'banda': 'latin', 'bolero': 'latin'
                            }
                            genres[name] = subgenre_map.get(subgenre, 'latin')
                    else:
                        genres[name] = 'latin'
                    continue
                
                # Extract language/genre from folder structure
                if 'audio' in path_parts:
                    audio_idx = path_parts.index('audio')
                    if audio_idx + 1 < len(path_parts):
                        lang_or_genre = path_parts[audio_idx + 1]
                        # Skip if it's the filename itself
                        if lang_or_genre.endswith('.mp3') or lang_or_genre.endswith('.wav'):
                            continue
                        # Map languages to genres (or use language as genre for multi-language clustering)
                        lang_map = {
                            'arabic': 'world', 'bangla': 'world', 'hindi': 'world',
                            'english': 'pop', 'spanish': 'latin'
                        }
                        genres[name] = lang_map.get(lang_or_genre, lang_or_genre)
                        continue
        
        # Fallback: Infer from file names (improved heuristic)
        for name in file_names:
            if name not in genres:
                name_lower = name.lower()
                if any(kw in name_lower for kw in ['rock', 'metal', 'punk']):
                    genres[name] = 'rock'
                elif any(kw in name_lower for kw in ['pop', 'dance', 'electronic', 'edm']):
                    genres[name] = 'pop'
                elif any(kw in name_lower for kw in ['jazz', 'blues', 'swing']):
                    genres[name] = 'jazz'
                elif any(kw in name_lower for kw in ['classical', 'orchestra', 'symphony']):
                    genres[name] = 'classical'
                elif any(kw in name_lower for kw in ['hip', 'hop', 'rap', 'r&b', 'rb']):
                    genres[name] = 'hiphop'
                elif any(kw in name_lower for kw in ['country', 'folk', 'bluegrass']):
                    genres[name] = 'country'
                elif any(kw in name_lower for kw in ['reggae', 'ska']):
                    genres[name] = 'reggae'
                elif any(kw in name_lower for kw in ['disco', 'funk']):
                    genres[name] = 'disco'
                elif any(kw in name_lower for kw in ['latin', 'spanish', 'es_']):
                    genres[name] = 'latin'
                elif any(kw in name_lower for kw in ['arabic', 'ar_', 'bangla', 'bn_', 'hindi', 'hi_']):
                    genres[name] = 'world'
                else:
                    genres[name] = 'other'
    else:
        # If genres provided (e.g., from CSV), fill in missing ones with filename inference
        for name in file_names:
            if name not in genres:
                # Infer from filename if not in provided genres
                name_lower = name.lower()
                if any(kw in name_lower for kw in ['rock', 'metal', 'punk']):
                    genres[name] = 'rock'
                elif any(kw in name_lower for kw in ['pop', 'dance', 'electronic', 'edm']):
                    genres[name] = 'pop'
                elif any(kw in name_lower for kw in ['jazz', 'blues', 'swing']):
                    genres[name] = 'jazz'
                elif any(kw in name_lower for kw in ['classical', 'orchestra', 'symphony']):
                    genres[name] = 'classical'
                elif any(kw in name_lower for kw in ['hip', 'hop', 'rap', 'r&b', 'rb']):
                    genres[name] = 'hiphop'
                elif any(kw in name_lower for kw in ['country', 'folk', 'bluegrass']):
                    genres[name] = 'country'
                elif any(kw in name_lower for kw in ['reggae', 'ska']):
                    genres[name] = 'reggae'
                elif any(kw in name_lower for kw in ['disco', 'funk']):
                    genres[name] = 'disco'
                else:
                    genres[name] = 'other'
    
    # Normalize genre names (handle variations like 'r&b' vs 'rb', 'hip hop' vs 'hiphop')
    genre_normalization = {
        'r&b': 'rb', 'rnb': 'rb', 'r and b': 'rb',
        'hip hop': 'hiphop', 'hip-hop': 'hiphop',
        'edm': 'pop', 'electronic': 'pop',
        'blues': 'jazz',
        'punk': 'rock', 'metal': 'rock'
    }
    
    normalized_genres = {}
    for name, genre in genres.items():
        genre_lower = genre.lower().strip()
        normalized_genres[name] = genre_normalization.get(genre_lower, genre_lower)
    
    genres = normalized_genres
    
    # Get unique genres - ensure 'other' is always included
    unique_genres = sorted(set(genres.values()))
    if 'other' not in unique_genres:
        unique_genres.append('other')
        unique_genres = sorted(unique_genres)  # Re-sort after adding 'other'
    n_genres = len(unique_genres)
    
    # Create one-hot encoding
    conditions = []
    for name in file_names:
        genre = genres.get(name, 'other')
        # Ensure genre is in unique_genres list (fallback to 'other' if not found)
        if genre not in unique_genres:
            genre = 'other'
        one_hot = np.zeros(n_genres)
        one_hot[unique_genres.index(genre)] = 1.0
        conditions.append(one_hot)
    
    return np.array(conditions), unique_genres


def extract_latent_cvae(model: ConditionalVAE, dataloader: DataLoader, 
                       conditions: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """Extract latent features from CVAE."""
    model = model.to(device)
    model.eval()
    
    latent_features = []
    
    with torch.no_grad():
        for idx, (audio_feat, lyrics_feat, _) in enumerate(dataloader):
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            batch_features = torch.cat([audio_feat, lyrics_feat], dim=1)
            # Get conditions for current batch (handle last batch which might be smaller)
            batch_start = idx * dataloader.batch_size
            batch_end = min(batch_start + len(audio_feat), len(conditions))
            batch_conditions = conditions[batch_start:batch_end].to(device)
            latent = model.get_latent(batch_features, batch_conditions)
            latent_features.append(latent.cpu().numpy())
    
    return np.vstack(latent_features)


def main():
    """Main function for Hard Task."""
    print("=" * 60)
    print("Hard Task: Advanced VAE with Comprehensive Evaluation")
    print("=" * 60)
    
    # Configuration
    audio_dir = 'data/Audio/audio'
    lyrics_dir = 'data/Lyrics'
    lyrics_csv = None  # Dataset uses TXT files, not CSV
    results_dir = 'results'
    latent_viz_dir = 'results/latent_visualization'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if data directory exists (support nested structure)
    if not os.path.exists(audio_dir):
        print(f"\nWarning: Audio directory not found: {audio_dir}")
        print("Please check the path and ensure audio files are available.")
        return
    
    # Check if directory has files (support nested structure)
    audio_files_check = []
    for root, dirs, files in os.walk(audio_dir):
        audio_files_check.extend([f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))])
    
    if len(audio_files_check) == 0:
        print(f"\nWarning: No audio files found in {audio_dir}")
        print("Please add audio files to the directory.")
        return
    
    # Load hybrid dataset
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
    
    # Store audio paths for genre extraction
    audio_paths_full = full_hybrid_dataset.audio_files if hasattr(full_hybrid_dataset, 'audio_files') else None
    
    print(f"   Found {len(full_hybrid_dataset)} audio files")
    if len(full_hybrid_dataset.lyrics_dict) > 0:
        print(f"   Found {len(full_hybrid_dataset.lyrics_dict)} lyrics files")
    
    if len(full_hybrid_dataset) == 0:
        print("No audio files found. Exiting.")
        return
    
    # Get audio paths for genre extraction from folder structure
    audio_paths_full = full_hybrid_dataset.audio_files if hasattr(full_hybrid_dataset, 'audio_files') else None
    
    # Create genre conditions (before splitting)
    print("\n1.5. Creating genre conditions...")
    file_names_full = [full_hybrid_dataset.file_names[i] for i in range(len(full_hybrid_dataset))]
    
    # Try to get genres from CSV metadata if available
    genres_from_csv = None
    if use_csv:
        try:
            from src.dataset import load_metadata_from_csv
            metadata = load_metadata_from_csv(lyrics_csv)
            if metadata:
                genres_from_csv = {}
                # Try to match by track_id or filename
                for name in file_names_full:
                    name_base = os.path.splitext(name)[0]
                    matched = False
                    # First try exact track_id match
                    for key, meta in metadata.items():
                        if str(key).lower() == name_base.lower() or name_base.lower() in str(key).lower():
                            if 'genre' in meta and pd.notna(meta['genre']):
                                genres_from_csv[name] = str(meta['genre']).lower().strip()
                                matched = True
                                break
                            elif 'playlist_genre' in meta and pd.notna(meta['playlist_genre']):
                                genres_from_csv[name] = str(meta['playlist_genre']).lower().strip()
                                matched = True
                                break
                    # If no match found, will use filename-based inference
                if genres_from_csv:
                    print(f"   Found genres for {len(genres_from_csv)}/{len(file_names_full)} files from CSV metadata")
                else:
                    print(f"   No genre matches found in CSV metadata, using filename-based inference")
        except Exception as e:
            print(f"   Could not load genres from CSV: {e}")
            import traceback
            traceback.print_exc()
    
    genre_conditions_full, unique_genres = create_genre_conditions(file_names_full, genres=genres_from_csv)
    condition_dim = genre_conditions_full.shape[1]
    print(f"   Found {len(unique_genres)} genres: {unique_genres}")
    print(f"   Condition dimension: {condition_dim}")
    
    # Extract genre labels for stratified splitting
    genre_labels_full = np.array([np.argmax(genre_conditions_full[i]) for i in range(len(file_names_full))])
    print(f"   Genre distribution: {dict(zip(*np.unique(genre_labels_full, return_counts=True)))}")
    
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
    
    # Get hybrid features
    sample_audio, sample_lyrics, _ = train_hybrid_dataset_scaled[0]
    hybrid_dim = sample_audio.shape[0] + sample_lyrics.shape[0]
    print(f"   Hybrid feature dimension: {hybrid_dim}")
    
    # Use genre conditions from full dataset (subset for train/test)
    print("\n2. Creating genre conditions for train set...")
    # Use the same genre_conditions_full and subset it to maintain consistent dimensions
    genre_conditions = genre_conditions_full[train_indices]
    train_file_names = [full_hybrid_dataset.file_names[train_indices[i]] for i in range(len(train_hybrid_dataset_scaled))]
    
    # Extract genre labels for accuracy metrics (train set)
    genre_labels = np.array([np.argmax(genre_conditions[i]) for i in range(len(train_file_names))])
    
    # Create dataloaders
    train_hybrid_dataloader = DataLoader(train_hybrid_dataset_scaled, batch_size=32, shuffle=True)
    test_hybrid_dataloader = DataLoader(test_hybrid_dataset_scaled, batch_size=32, shuffle=False)
    hybrid_dataloader = train_hybrid_dataloader  # Use train for training
    
    # Part 1: Beta-VAE
    print("\n" + "=" * 60)
    print("Part 1: Beta-VAE (Beta=4.0 for disentangled representations)")
    print("=" * 60)
    
    print("\n3. Training Beta-VAE...")
    # Reduced beta from 4.0 to 2.0 for better balance between disentanglement and reconstruction
    # Higher beta can cause over-regularization leading to poor clustering
    beta_value = 2.0
    beta_vae = BasicVAE(input_dim=hybrid_dim, latent_dim=32, hidden_dims=[256, 128, 64], beta=beta_value)
    
    # Train Beta-VAE
    beta_losses = train_beta_vae(beta_vae, train_hybrid_dataloader, epochs=50, lr=1e-3, beta=beta_value, device=device)
    
    # Extract latent features (on both train and test)
    print("\n4. Extracting latent features from Beta-VAE (train set)...")
    train_beta_latent_features = []
    beta_vae = beta_vae.to(device)
    beta_vae.eval()
    with torch.no_grad():
        for audio_feat, lyrics_feat, _ in train_hybrid_dataloader:
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            latent = beta_vae.get_latent(hybrid_feat)
            train_beta_latent_features.append(latent.cpu().numpy())
    train_beta_latent_features = np.vstack(train_beta_latent_features)
    print(f"   Train latent features shape: {train_beta_latent_features.shape}")
    
    print("\n4.5. Extracting latent features from Beta-VAE (test set)...")
    test_beta_latent_features = []
    with torch.no_grad():
        for audio_feat, lyrics_feat, _ in test_hybrid_dataloader:
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            latent = beta_vae.get_latent(hybrid_feat)
            test_beta_latent_features.append(latent.cpu().numpy())
    test_beta_latent_features = np.vstack(test_beta_latent_features)
    print(f"   Test latent features shape: {test_beta_latent_features.shape}")
    
    # Use train set for clustering
    beta_latent_features = train_beta_latent_features
    
    # Part 2: CVAE
    print("\n" + "=" * 60)
    print("Part 2: Conditional VAE (CVAE)")
    print("=" * 60)
    
    print("\n5. Training CVAE...")
    cvae = ConditionalVAE(input_dim=hybrid_dim, condition_dim=condition_dim, latent_dim=32, beta=1.0)
    genre_conditions_tensor = torch.tensor(genre_conditions, dtype=torch.float32)
    
    # Train CVAE
    cvae_losses = train_cvae(cvae, train_hybrid_dataloader, genre_conditions_tensor, epochs=50, lr=1e-3, device=device)
    
    # Extract latent features (on both train and test)
    print("\n6. Extracting latent features from CVAE (train set)...")
    train_cvae_latent_features = extract_latent_cvae(cvae, train_hybrid_dataloader, genre_conditions_tensor, device=device)
    print(f"   Train latent features shape: {train_cvae_latent_features.shape}")
    
    # Use test genre conditions from full dataset (subset to maintain consistent dimensions)
    test_genre_conditions = genre_conditions_full[test_indices]
    test_genre_conditions_tensor = torch.tensor(test_genre_conditions, dtype=torch.float32)
    
    print("\n6.5. Extracting latent features from CVAE (test set)...")
    test_cvae_latent_features = extract_latent_cvae(cvae, test_hybrid_dataloader, test_genre_conditions_tensor, device=device)
    print(f"   Test latent features shape: {test_cvae_latent_features.shape}")
    
    # Use train set for clustering
    cvae_latent_features = train_cvae_latent_features
    
    # Part 3: Multi-modal Clustering
    print("\n" + "=" * 60)
    print("Part 3: Multi-modal Clustering")
    print("=" * 60)
    
    # Use number of genres if available, otherwise heuristic
    if len(unique_genres) > 1:
        n_clusters = len(unique_genres)
        print(f"\n7. Performing clustering with n_clusters={n_clusters} (matching number of genres)...")
    else:
        n_clusters = max(2, int(np.sqrt(len(train_hybrid_dataset_scaled) / 2)))
        print(f"\n7. Performing clustering with n_clusters={n_clusters}...")
    
    all_results = {}
    
    # Beta-VAE + K-Means
    print("\n   a) Beta-VAE + K-Means...")
    beta_kmeans_labels, _ = kmeans_clustering(beta_latent_features, n_clusters=n_clusters)
    beta_kmeans_metrics = evaluate_clustering(beta_kmeans_labels, beta_latent_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    all_results['Beta-VAE + K-Means'] = beta_kmeans_metrics
    print(f"      Silhouette: {beta_kmeans_metrics['silhouette_score']:.4f}")
    print(f"      Calinski-Harabasz: {beta_kmeans_metrics['calinski_harabasz']:.4f}")
    if len(unique_genres) > 1:
        print(f"      [ACCURACY] Adjusted Rand Index (ARI): {beta_kmeans_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
        print(f"      [ACCURACY] Normalized Mutual Info (NMI): {beta_kmeans_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"      [ACCURACY] Cluster Purity: {beta_kmeans_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # CVAE + K-Means
    print("\n   b) CVAE + K-Means...")
    cvae_kmeans_labels, _ = kmeans_clustering(cvae_latent_features, n_clusters=n_clusters)
    cvae_kmeans_metrics = evaluate_clustering(cvae_kmeans_labels, cvae_latent_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    all_results['CVAE + K-Means'] = cvae_kmeans_metrics
    print(f"      Silhouette: {cvae_kmeans_metrics['silhouette_score']:.4f}")
    print(f"      Calinski-Harabasz: {cvae_kmeans_metrics['calinski_harabasz']:.4f}")
    if len(unique_genres) > 1:
        print(f"      [ACCURACY] Adjusted Rand Index (ARI): {cvae_kmeans_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
        print(f"      [ACCURACY] Normalized Mutual Info (NMI): {cvae_kmeans_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
        print(f"      [ACCURACY] Cluster Purity: {cvae_kmeans_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Part 4: Baseline Comparisons
    print("\n" + "=" * 60)
    print("Part 4: Baseline Comparisons")
    print("=" * 60)
    
    print("\n8. Computing baseline methods...")
    
    # Get original hybrid features (from train set, scaled)
    train_features_list = []
    for i in range(len(train_hybrid_dataset_scaled)):
        audio_feat, lyrics_feat, _ = train_hybrid_dataset_scaled[i]
        hybrid_feat = np.concatenate([audio_feat.numpy(), lyrics_feat.numpy()])
        train_features_list.append(hybrid_feat)
    original_hybrid_features = np.array(train_features_list)
    
    # PCA + K-Means
    print("\n   a) PCA + K-Means...")
    # PCA n_components cannot exceed min(n_samples, n_features)
    pca_n_components = min(32, original_hybrid_features.shape[1], original_hybrid_features.shape[0] - 1)
    if pca_n_components < 1:
        pca_n_components = 1
    pca = PCA(n_components=pca_n_components)
    pca_features = pca.fit_transform(original_hybrid_features)
    pca_labels, _ = kmeans_clustering(pca_features, n_clusters=n_clusters)
    pca_metrics = evaluate_clustering(pca_labels, pca_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    all_results['PCA + K-Means'] = pca_metrics
    print(f"      Silhouette: {pca_metrics['silhouette_score']:.4f}")
    if len(unique_genres) > 1:
        print(f"      [ACCURACY] ARI: {pca_metrics.get('adjusted_rand_index', 'N/A'):.4f}, NMI: {pca_metrics.get('normalized_mutual_info', 'N/A'):.4f}, Purity: {pca_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Autoencoder + K-Means
    print("\n   b) Autoencoder + K-Means...")
    # Simple autoencoder (VAE without KL term, beta=0)
    ae_model = BasicVAE(input_dim=hybrid_dim, latent_dim=32, hidden_dims=[256, 128, 64], beta=0.0)
    ae_model = ae_model.to(device)
    
    # Quick training (fewer epochs for baseline)
    optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)
    ae_model.train()
    for epoch in range(20):  # Fewer epochs for baseline
        for audio_feat, lyrics_feat, _ in train_hybrid_dataloader:
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            
            optimizer.zero_grad()
            recon, mu, logvar = ae_model(hybrid_feat)
            total_loss, _, _ = vae_loss(recon, hybrid_feat, mu, logvar, beta=0.0)
            total_loss.backward()
            optimizer.step()
    
    # Extract features
    ae_latent_features = []
    ae_model.eval()
    with torch.no_grad():
        for audio_feat, lyrics_feat, _ in train_hybrid_dataloader:
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            latent = ae_model.get_latent(hybrid_feat)
            ae_latent_features.append(latent.cpu().numpy())
    ae_latent_features = np.vstack(ae_latent_features)
    
    ae_labels, _ = kmeans_clustering(ae_latent_features, n_clusters=n_clusters)
    ae_metrics = evaluate_clustering(ae_labels, ae_latent_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    all_results['Autoencoder + K-Means'] = ae_metrics
    print(f"      Silhouette: {ae_metrics['silhouette_score']:.4f}")
    if len(unique_genres) > 1:
        print(f"      [ACCURACY] ARI: {ae_metrics.get('adjusted_rand_index', 'N/A'):.4f}, NMI: {ae_metrics.get('normalized_mutual_info', 'N/A'):.4f}, Purity: {ae_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Direct spectral feature clustering
    print("\n   c) Direct Spectral Feature Clustering...")
    # Use only audio features (from train set, scaled)
    train_audio_features_list = []
    for i in range(len(train_hybrid_dataset_scaled)):
        audio_feat, _, _ = train_hybrid_dataset_scaled[i]
        train_audio_features_list.append(audio_feat.numpy())
    audio_features = np.array(train_audio_features_list)
    direct_labels, _ = kmeans_clustering(audio_features, n_clusters=n_clusters)
    direct_metrics = evaluate_clustering(direct_labels, audio_features, labels_true=genre_labels if len(unique_genres) > 1 else None)
    all_results['Direct Spectral + K-Means'] = direct_metrics
    print(f"      Silhouette: {direct_metrics['silhouette_score']:.4f}")
    if len(unique_genres) > 1:
        print(f"      [ACCURACY] ARI: {direct_metrics.get('adjusted_rand_index', 'N/A'):.4f}, NMI: {direct_metrics.get('normalized_mutual_info', 'N/A'):.4f}, Purity: {direct_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Part 5: Comprehensive Evaluation
    print("\n" + "=" * 60)
    print("Part 5: Comprehensive Evaluation")
    print("=" * 60)
    
    if len(unique_genres) > 1:
        print("\n9. [ACCURACY METRICS SUMMARY] (vs True Genres):")
        print("-" * 60)
        for method_name, metrics in all_results.items():
            if 'adjusted_rand_index' in metrics:
                print(f"\n{method_name}:")
                print(f"  [ACCURACY] Adjusted Rand Index (ARI): {metrics['adjusted_rand_index']:.4f}")
                print(f"  [ACCURACY] Normalized Mutual Info (NMI): {metrics['normalized_mutual_info']:.4f}")
                print(f"  [ACCURACY] Cluster Purity: {metrics['cluster_purity']:.4f}")
    
    # Save comprehensive metrics
    metrics_data = []
    for method, metrics in all_results.items():
        row = {'Method': method}
        row.update(metrics)
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(results_dir, 'clustering_metrics_hard.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n   Saved comprehensive metrics to {metrics_path}")
    
    # Part 6: Detailed Visualizations
    print("\n" + "=" * 60)
    print("Part 6: Detailed Visualizations")
    print("=" * 60)
    
    print("\n10. Creating visualizations...")
    
    # Beta-VAE visualizations
    plot_latent_space(
        beta_latent_features,
        beta_kmeans_labels,
        method='umap',
        title='Beta-VAE Latent Space (UMAP)',
        save_path=os.path.join(latent_viz_dir, 'beta_vae_umap.png')
    )
    
    plot_latent_space(
        beta_latent_features,
        beta_kmeans_labels,
        method='tsne',
        title='Beta-VAE Latent Space (t-SNE)',
        save_path=os.path.join(latent_viz_dir, 'beta_vae_tsne.png')
    )
    
    # CVAE visualizations
    plot_latent_space(
        cvae_latent_features,
        cvae_kmeans_labels,
        method='umap',
        title='CVAE Latent Space (UMAP)',
        save_path=os.path.join(latent_viz_dir, 'cvae_umap.png')
    )
    
    # Cluster distribution over genres
    if len(unique_genres) > 1:
        plot_cluster_distribution(
            beta_kmeans_labels,
            true_labels=genre_labels,
            title='Beta-VAE Cluster Distribution vs True Genres',
            save_path=os.path.join(latent_viz_dir, 'beta_vae_cluster_dist.png')
        )
        
        # CVAE cluster distribution
        plot_cluster_distribution(
            cvae_kmeans_labels,
            true_labels=genre_labels,
            title='CVAE Cluster Distribution vs True Genres',
            save_path=os.path.join(latent_viz_dir, 'cvae_cluster_dist.png')
        )
    
    # Reconstruction examples
    print("\n11. Generating reconstruction examples...")
    beta_vae = beta_vae.to(device)
    beta_vae.eval()
    
    # Get a batch of original and reconstructed features
    original_features_list = []
    reconstructed_features_list = []
    
    with torch.no_grad():
        for audio_feat, lyrics_feat, _ in list(train_hybrid_dataloader)[:1]:  # Just first batch
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            
            recon, _, _ = beta_vae(hybrid_feat)
            original_features_list.append(hybrid_feat.cpu().numpy())
            reconstructed_features_list.append(recon.cpu().numpy())
    
    if len(original_features_list) > 0:
        original_features = np.vstack(original_features_list)
        reconstructed_features = np.vstack(reconstructed_features_list)
        
        plot_reconstruction_examples(
            original_features,
            reconstructed_features,
            n_examples=min(8, len(original_features)),
            title='Beta-VAE Reconstruction Examples',
            save_path=os.path.join(latent_viz_dir, 'beta_vae_reconstructions.png')
        )
    
    # CVAE reconstruction examples
    cvae = cvae.to(device)
    cvae.eval()
    
    original_cvae_list = []
    reconstructed_cvae_list = []
    
    with torch.no_grad():
        for idx, (audio_feat, lyrics_feat, _) in enumerate(list(train_hybrid_dataloader)[:1]):
            audio_feat = audio_feat.to(device)
            lyrics_feat = lyrics_feat.to(device)
            hybrid_feat = torch.cat([audio_feat, lyrics_feat], dim=1)
            batch_conditions = genre_conditions_tensor[idx * train_hybrid_dataloader.batch_size:
                                                       (idx + 1) * train_hybrid_dataloader.batch_size].to(device)
            
            recon, _, _ = cvae(hybrid_feat, batch_conditions)
            original_cvae_list.append(hybrid_feat.cpu().numpy())
            reconstructed_cvae_list.append(recon.cpu().numpy())
    
    if len(original_cvae_list) > 0:
        original_cvae = np.vstack(original_cvae_list)
        reconstructed_cvae = np.vstack(reconstructed_cvae_list)
        
        plot_reconstruction_examples(
            original_cvae,
            reconstructed_cvae,
            n_examples=min(8, len(original_cvae)),
            title='CVAE Reconstruction Examples',
            save_path=os.path.join(latent_viz_dir, 'cvae_reconstructions.png')
        )
    
    # Cluster distribution over languages (if we can infer language)
    print("\n12. Analyzing cluster distribution over languages...")
    # Infer language from file names (simple heuristic)
    language_labels = []
    for name in train_file_names:
        name_lower = name.lower()
        # Simple heuristic: check for common language indicators
        if any(kw in name_lower for kw in ['bangla', 'bengali', 'bn_', '_bn']):
            language_labels.append('Bangla')
        elif any(kw in name_lower for kw in ['english', 'en_', '_en']):
            language_labels.append('English')
        else:
            language_labels.append('Unknown')
    
    if len(set(language_labels)) > 1:
        # Create language label array
        unique_languages = sorted(set(language_labels))
        language_label_array = np.array([unique_languages.index(lang) for lang in language_labels])
        
        # Plot cluster distribution over languages
        plot_cluster_distribution(
            beta_kmeans_labels,
            true_labels=language_label_array,
            title='Beta-VAE Cluster Distribution vs Languages',
            save_path=os.path.join(latent_viz_dir, 'beta_vae_language_dist.png')
        )
        
        # Evaluate clustering with language labels
        if len(unique_languages) > 1:
            beta_lang_metrics = evaluate_clustering(beta_kmeans_labels, beta_latent_features, 
                                                    labels_true=language_label_array)
            cvae_lang_metrics = evaluate_clustering(cvae_kmeans_labels, cvae_latent_features,
                                                     labels_true=language_label_array)
            
            print(f"   Beta-VAE Language Clustering:")
            print(f"     ARI: {beta_lang_metrics.get('adjusted_rand_index', 'N/A'):.4f}")
            print(f"     NMI: {beta_lang_metrics.get('normalized_mutual_info', 'N/A'):.4f}")
            print(f"     Purity: {beta_lang_metrics.get('cluster_purity', 'N/A'):.4f}")
    
    # Metrics comparison
    plot_metrics_comparison(
        all_results,
        title='Comprehensive Clustering Methods Comparison',
        save_path=os.path.join(results_dir, 'clustering_comparison_hard.png')
    )
    
    # Additional visualization: Latent space colored by genre
    if len(unique_genres) > 1:
        print("\n13. Creating genre-colored latent space visualizations...")
        
        # Beta-VAE with genre colors
        plt.figure(figsize=(10, 8))
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        beta_embeddings = reducer.fit_transform(beta_latent_features)
        
        scatter = plt.scatter(beta_embeddings[:, 0], beta_embeddings[:, 1], 
                            c=genre_labels, cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Genre')
        plt.title('Beta-VAE Latent Space Colored by Genre (UMAP)', fontsize=14, fontweight='bold')
        plt.xlabel('UMAP Component 1', fontsize=12)
        plt.ylabel('UMAP Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(latent_viz_dir, 'beta_vae_genre_colored.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # CVAE with genre colors
        cvae_embeddings = reducer.fit_transform(cvae_latent_features)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(cvae_embeddings[:, 0], cvae_embeddings[:, 1], 
                            c=genre_labels, cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Genre')
        plt.title('CVAE Latent Space Colored by Genre (UMAP)', fontsize=14, fontweight='bold')
        plt.xlabel('UMAP Component 1', fontsize=12)
        plt.ylabel('UMAP Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(latent_viz_dir, 'cvae_genre_colored.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved genre-colored latent space visualizations")
    
    # Save models
    beta_model_path = os.path.join(results_dir, 'beta_vae_model_hard.pth')
    cvae_model_path = os.path.join(results_dir, 'cvae_model_hard.pth')
    torch.save(beta_vae.state_dict(), beta_model_path)
    torch.save(cvae.state_dict(), cvae_model_path)
    print(f"\n14. Saved models:")
    print(f"   Beta-VAE: {beta_model_path}")
    print(f"   CVAE: {cvae_model_path}")
    
    print("\n" + "=" * 60)
    print("Hard Task completed successfully!")
    print("=" * 60)
    print("\n[FINAL RESULTS SUMMARY]:")
    print("=" * 60)
    for method, metrics in all_results.items():
        print(f"\n{method}:")
        print(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
        print(f"  Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A'):.4f}")
        if 'adjusted_rand_index' in metrics:
            print(f"  [ACCURACY] ARI: {metrics['adjusted_rand_index']:.4f}")
            print(f"  [ACCURACY] NMI: {metrics['normalized_mutual_info']:.4f}")
            print(f"  [ACCURACY] Purity: {metrics['cluster_purity']:.4f}")
        print()


if __name__ == '__main__':
    main()
