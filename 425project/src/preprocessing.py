"""
Data preprocessing utilities: scaling and train/test splitting.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, Subset


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Tuple[Subset, Subset, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.
    
    Args:
        dataset: PyTorch Dataset
        test_size: Proportion of data for test set (default: 0.2 for 80/20 split)
        random_state: Random seed for reproducibility
        stratify: Optional array for stratified splitting (e.g., genre labels)
    
    Returns:
        Tuple of (train_dataset, test_dataset, train_indices, test_indices)
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if stratify is not None:
        # Check if stratified splitting is possible (each class needs at least 2 samples)
        unique_labels, counts = np.unique(stratify, return_counts=True)
        min_samples_per_class = min(counts)
        
        # Stratified split requires at least 2 samples per class
        # (1 for train, 1 for test with 80/20 split)
        if min_samples_per_class < 2:
            print(f"   Warning: Some classes have too few samples ({min_samples_per_class} < 2). Using non-stratified split.")
            print(f"   Class distribution: {dict(zip(unique_labels, counts))}")
            train_indices, test_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state
            )
        else:
            try:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify
                )
            except ValueError as e:
                # Fallback to non-stratified if stratified fails
                print(f"   Warning: Stratified split failed ({str(e)}). Using non-stratified split.")
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=test_size,
                    random_state=random_state
                )
    else:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state
        )
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset, train_indices, test_indices


def fit_and_scale_features(
    dataset: Dataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    is_hybrid: bool = False
) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
    """
    Fit StandardScaler on training data and scale both train and test features.
    
    Args:
        dataset: PyTorch Dataset
        train_indices: Indices for training set
        test_indices: Indices for test set
        is_hybrid: Whether dataset returns 3 values (audio, lyrics, filename)
    
    Returns:
        Tuple of (fitted_scaler, scaled_train_features, scaled_test_features)
    """
    scaler = StandardScaler()
    
    # Collect training features
    train_features = []
    for idx in train_indices:
        item = dataset[idx]
        if is_hybrid:
            audio_feat, lyrics_feat, _ = item
            # For hybrid, we'll scale audio and lyrics separately
            train_features.append(audio_feat.numpy())
        else:
            features, _ = item
            train_features.append(features.numpy())
    
    train_features = np.array(train_features)
    
    # Fit scaler on training data
    scaler.fit(train_features)
    
    # Transform both train and test
    train_scaled = scaler.transform(train_features)
    
    test_features = []
    for idx in test_indices:
        item = dataset[idx]
        if is_hybrid:
            audio_feat, lyrics_feat, _ = item
            test_features.append(audio_feat.numpy())
        else:
            features, _ = item
            test_features.append(features.numpy())
    
    test_features = np.array(test_features)
    test_scaled = scaler.transform(test_features)
    
    return scaler, train_scaled, test_scaled


class ScaledDataset(Dataset):
    """
    Wrapper dataset that applies pre-fitted scaling to features.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        scaler: StandardScaler,
        indices: Optional[np.ndarray] = None,
        is_hybrid: bool = False
    ):
        """
        Args:
            base_dataset: Base dataset to wrap
            scaler: Pre-fitted StandardScaler
            indices: Optional subset of indices to use (for train/test split)
            is_hybrid: Whether dataset returns 3 values (audio, lyrics, filename)
        """
        self.base_dataset = base_dataset
        self.scaler = scaler
        self.indices = indices if indices is not None else np.arange(len(base_dataset))
        self.is_hybrid = is_hybrid
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        """Get item with scaled features."""
        actual_idx = self.indices[idx]
        item = self.base_dataset[actual_idx]
        
        if self.is_hybrid:
            # HybridMusicDataset: (audio, lyrics, filename)
            audio_feat, lyrics_feat, filename = item
            audio_np = audio_feat.numpy()
            
            # Scale audio features
            if audio_np.ndim == 1:
                audio_scaled = self.scaler.transform(audio_np.reshape(1, -1))[0]
            else:
                audio_scaled = self.scaler.transform(audio_np)
            
            # Lyrics are already normalized in get_lyrics_embedding, keep as-is
            return torch.tensor(audio_scaled, dtype=torch.float32), lyrics_feat, filename
        else:
            # MusicDataset: (features, filename)
            features, filename = item
            features_np = features.numpy()
            
            # Scale features
            if features_np.ndim == 1:
                features_scaled = self.scaler.transform(features_np.reshape(1, -1))[0]
            else:
                features_scaled = self.scaler.transform(features_np)
            
            return torch.tensor(features_scaled, dtype=torch.float32), filename


def create_scaled_datasets(
    train_dataset: Subset,
    test_dataset: Subset,
    base_dataset: Dataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    is_hybrid: bool = False
) -> Tuple[ScaledDataset, ScaledDataset, StandardScaler]:
    """
    Create scaled versions of train and test datasets.
    Fits scaler on training data, applies to both.
    
    Args:
        train_dataset: Training dataset (Subset)
        test_dataset: Test dataset (Subset)
        base_dataset: Original full dataset
        train_indices: Training indices
        test_indices: Test indices
        is_hybrid: Whether dataset is hybrid (returns 3 values)
    
    Returns:
        Tuple of (scaled_train_dataset, scaled_test_dataset, fitted_scaler)
    """
    # Fit scaler on training data
    scaler = StandardScaler()
    
    # Collect training features to fit scaler
    train_features = []
    for idx in train_indices:
        item = base_dataset[idx]
        if is_hybrid:
            audio_feat, _, _ = item
            train_features.append(audio_feat.numpy())
        else:
            features, _ = item
            train_features.append(features.numpy())
    
    train_features = np.array(train_features)
    scaler.fit(train_features)
    print(f"   Scaler fitted on {len(train_features)} training samples, feature shape: {train_features.shape}")
    
    # Create scaled datasets
    scaled_train = ScaledDataset(base_dataset, scaler, train_indices, is_hybrid)
    scaled_test = ScaledDataset(base_dataset, scaler, test_indices, is_hybrid)
    
    return scaled_train, scaled_test, scaler
