"""
Dataset loading and preprocessing for music data.
Supports audio feature extraction (MFCC, spectrograms) and lyrics loading.
"""

import os
import numpy as np
import librosa
import warnings
import pandas as pd
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset
import torch

# Suppress librosa/soundfile warnings about Xing stream size (harmless MP3 metadata warnings)
# These warnings occur with VBR MP3 files and don't affect functionality
warnings.filterwarnings('ignore', category=UserWarning, module='soundfile')
warnings.filterwarnings('ignore', message='.*Xing.*')


class MusicDataset(Dataset):
    """Dataset class for loading music audio files and extracting features."""
    
    def __init__(
        self,
        audio_dir: str,
        feature_type: str = 'mfcc',
        n_mfcc: int = 13,
        sr: int = 22050,
        duration: float = 30.0,
        hop_length: int = 512,
        n_fft: int = 2048,
        normalize: bool = True
    ):
        """
        Args:
            audio_dir: Directory containing audio files
            feature_type: Type of feature to extract ('mfcc', 'spectrogram', 'melspectrogram')
            n_mfcc: Number of MFCC coefficients
            sr: Sample rate for audio loading
            duration: Maximum duration of audio to load (seconds)
            hop_length: Hop length for STFT
            n_fft: FFT window size
            normalize: Whether to normalize features
        """
        self.audio_dir = audio_dir
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.duration = duration
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalize = normalize
        
        # Get list of audio files
        self.audio_files = self._get_audio_files()
        self.file_names = [os.path.basename(f) for f in self.audio_files]
        
        # Filter out problematic files during initialization
        print(f"Found {len(self.audio_files)} audio files. Testing file validity...")
        valid_files = []
        valid_names = []
        # Suppress warnings during file validation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='soundfile')
            warnings.filterwarnings('ignore', message='.*Xing.*')
            for i, audio_file in enumerate(self.audio_files):
                try:
                    # Quick test load - try multiple methods
                    try:
                        test_y, _ = librosa.load(audio_file, sr=self.sr, duration=1.0, res_type='kaiser_fast', mono=True)
                    except:
                        # Fallback: try without res_type
                        test_y, _ = librosa.load(audio_file, sr=self.sr, duration=1.0, mono=True)
                    
                    if len(test_y) > 1000:  # Valid audio
                        valid_files.append(audio_file)
                        valid_names.append(self.file_names[i])
                except Exception as e:
                    print(f"  Skipping problematic file: {os.path.basename(audio_file)} ({str(e)[:50]})")
                    continue
        
        self.audio_files = valid_files
        self.file_names = valid_names
        print(f"  Using {len(self.audio_files)} valid audio files")
        
    def _get_audio_files(self) -> List[str]:
        """Get list of audio files from directory (supports recursive search)."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        if not os.path.exists(self.audio_dir):
            return audio_files
        
        # Support both flat structure and nested structure (e.g., by language/genre)
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        return sorted(audio_files)
    
    def _extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file."""
        try:
            # Suppress warnings for this specific load operation
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='soundfile')
                warnings.filterwarnings('ignore', message='.*Xing.*')
                # Load audio with faster resampling and error handling
                # Use mono=True and res_type='kaiser_fast' for faster loading
                try:
                    y, sr = librosa.load(
                        audio_path, 
                        sr=self.sr, 
                        duration=self.duration, 
                        res_type='kaiser_fast',
                        mono=True
                    )
                except Exception as load_error:
                    # If kaiser_fast fails, try without res_type
                    try:
                        y, sr = librosa.load(
                            audio_path, 
                            sr=self.sr, 
                            duration=self.duration,
                            mono=True
                        )
                    except:
                        raise load_error  # Re-raise original error
            
            # Check if audio is valid (not empty or too short)
            if len(y) < 1000:  # Less than ~0.05 seconds at 22050 Hz
                raise ValueError(f"Audio file too short: {len(y)} samples")
            
            if self.feature_type == 'mfcc':
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                )
                # Take mean over time to get fixed-size feature vector
                features = np.mean(mfcc, axis=1)
                
            elif self.feature_type == 'spectrogram':
                # Extract spectrogram
                stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
                magnitude = np.abs(stft)
                # Take mean over time
                features = np.mean(magnitude, axis=1)
                
            elif self.feature_type == 'melspectrogram':
                # Extract mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                )
                # Take mean over time
                features = np.mean(mel_spec, axis=1)
                
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
            
            # Normalize if requested
            if self.normalize:
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            print(f"  Skipping this file and using zero vector as fallback")
            # Return zero vector as fallback
            if self.feature_type == 'mfcc':
                return np.zeros(self.n_mfcc, dtype=np.float32)
            else:
                return np.zeros(self.n_fft // 2 + 1, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get item from dataset."""
        audio_path = self.audio_files[idx]
        features = self._extract_features(audio_path)
        file_name = self.file_names[idx]
        
        return torch.tensor(features), file_name
    
    def get_all_features(self) -> np.ndarray:
        """Get all features as numpy array for clustering."""
        features_list = []
        for i in range(len(self)):
            item = self[i]
            # Handle both MusicDataset (2 values) and HybridMusicDataset (3 values)
            if len(item) == 3:
                audio_features, _, _ = item  # (audio, lyrics, filename)
                features_list.append(audio_features.numpy())
            else:
                features, _ = item  # (features, filename)
                features_list.append(features.numpy())
        return np.array(features_list)
    
    def _extract_spectrogram_full(self, audio_path: str) -> np.ndarray:
        """Extract full spectrogram (2D) for ConvVAE."""
        try:
            # Suppress warnings for this specific load operation
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='soundfile')
                warnings.filterwarnings('ignore', message='.*Xing.*')
                y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)
            
            if self.feature_type == 'melspectrogram':
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft,
                    n_mels=128
                )
                # Convert to log scale
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                return mel_spec_db.astype(np.float32)
            else:
                # Default to mel spectrogram
                stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
                magnitude = np.abs(stft)
                # Convert to mel scale
                mel_spec = librosa.feature.melspectrogram(
                    S=magnitude**2,
                    sr=sr,
                    n_mels=128
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                return mel_spec_db.astype(np.float32)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros((128, 100), dtype=np.float32)  # Default shape


def load_lyrics(lyrics_dir: str, file_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Load lyrics from directory (supports recursive search).
    
    Args:
        lyrics_dir: Directory containing lyrics files
        file_mapping: Optional mapping from audio file names to lyrics file names
    
    Returns:
        Dictionary mapping file names to lyrics text
    """
    lyrics_dict = {}
    
    if not os.path.exists(lyrics_dir):
        return lyrics_dict
    
    # Support both flat structure and nested structure (e.g., by language)
    for root, dirs, files in os.walk(lyrics_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lyrics_dict[file] = f.read()
                except Exception as e:
                    print(f"Error loading lyrics from {file_path}: {e}")
    
    return lyrics_dict


def load_lyrics_from_csv(
    csv_path: str,
    audio_file_column: str = 'filename',
    lyrics_column: str = 'lyrics',
    audio_file_column_alt: Optional[str] = None
) -> Dict[str, str]:
    """
    Load lyrics from CSV file.
    
    Args:
        csv_path: Path to CSV file containing lyrics
        audio_file_column: Column name in CSV that contains audio file names
        lyrics_column: Column name in CSV that contains lyrics text
        audio_file_column_alt: Alternative column names to try (e.g., 'song', 'track', 'file')
    
    Returns:
        Dictionary mapping audio file names (without extension) to lyrics text
    """
    lyrics_dict = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return lyrics_dict
    
    try:
        df = pd.read_csv(csv_path)
        
        # Try to find the audio file column
        audio_col = None
        possible_names = [audio_file_column]
        if audio_file_column_alt:
            possible_names.append(audio_file_column_alt)
        # Support common column names including track identifiers
        possible_names.extend([
            'filename', 'file', 'song', 'track', 'song_name', 'track_name', 
            'audio_file', 'track_id', 'track_nan', 'id'
        ])
        
        for name in possible_names:
            if name in df.columns:
                audio_col = name
                break
        
        if audio_col is None:
            print(f"Warning: Could not find audio file column in CSV. Available columns: {df.columns.tolist()}")
            print(f"Using first column as audio file identifier.")
            audio_col = df.columns[0]
        
        # Try to find the lyrics column
        lyrics_col = None
        possible_lyrics = [lyrics_column, 'lyrics', 'lyric', 'text', 'song_lyrics', 'lyrics_text']
        
        for name in possible_lyrics:
            if name in df.columns:
                lyrics_col = name
                break
        
        if lyrics_col is None:
            print(f"Warning: Could not find lyrics column in CSV. Available columns: {df.columns.tolist()}")
            # Try to find a text-like column
            for col in df.columns:
                if col != audio_col and df[col].dtype == 'object':
                    lyrics_col = col
                    print(f"Using '{col}' as lyrics column.")
                    break
        
        if lyrics_col is None:
            print(f"Error: No suitable lyrics column found in CSV.")
            return lyrics_dict
        
        # Create mapping: audio filename (without extension) -> lyrics
        for idx, row in df.iterrows():
            audio_file = str(row[audio_col]).strip()
            lyrics_text = str(row[lyrics_col]) if pd.notna(row[lyrics_col]) else ""
            
            # Remove file extension from audio file name for matching
            audio_base = os.path.splitext(audio_file)[0]
            lyrics_dict[audio_base] = lyrics_text
        
        print(f"Loaded {len(lyrics_dict)} lyrics from CSV file: {csv_path}")
        
    except Exception as e:
        print(f"Error loading lyrics from CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return lyrics_dict


def load_metadata_from_csv(
    csv_path: str,
    track_id_column: str = 'track_id',
    genre_column: str = 'playlist_genre',  # Updated to match your CSV
    language_column: str = 'language'
) -> Dict[str, Dict[str, str]]:
    """
    Load metadata (genre, language) from CSV file.
    
    Args:
        csv_path: Path to CSV file
        track_id_column: Column name for track identifier
        genre_column: Column name for genre
        language_column: Column name for language
    
    Returns:
        Dictionary mapping track_id to metadata dict
    """
    metadata_dict = {}
    
    if not os.path.exists(csv_path):
        return metadata_dict
    
    try:
        df = pd.read_csv(csv_path)
        
        # Find columns (try multiple possible names)
        track_col = None
        for col in [track_id_column, 'track_id', 'id']:
            if col in df.columns:
                track_col = col
                break
        
        genre_col = None
        for col in [genre_column, 'playlist_genre', 'playlist_g', 'genre', 'genres']:
            if col in df.columns:
                genre_col = col
                break
        
        lang_col = None
        for col in [language_column, 'language', 'lang', 'lang_code']:
            if col in df.columns:
                lang_col = col
                break
        
        if track_col:
            for idx, row in df.iterrows():
                track_id = str(row[track_col]).strip()
                metadata = {}
                
                if genre_col and pd.notna(row[genre_col]):
                    metadata['genre'] = str(row[genre_col]).strip()
                if lang_col and pd.notna(row[lang_col]):
                    metadata['language'] = str(row[lang_col]).strip()
                
                if metadata:
                    metadata_dict[track_id] = metadata
        
        print(f"Loaded metadata for {len(metadata_dict)} tracks from CSV")
        
    except Exception as e:
        print(f"Error loading metadata from CSV: {e}")
    
    return metadata_dict


def get_lyrics_embedding(lyrics_text: str, embedding_dim: int = 64) -> np.ndarray:
    """
    Create simple lyrics embedding using TF-IDF-like features.
    For a more sophisticated approach, use pre-trained word embeddings.
    
    Args:
        lyrics_text: Lyrics text
        embedding_dim: Dimension of embedding
    
    Returns:
        Embedding vector
    """
    if not lyrics_text or len(lyrics_text.strip()) == 0:
        return np.zeros(embedding_dim, dtype=np.float32)
    
    # Simple character and word-based features
    text = lyrics_text.lower()
    
    # Character n-gram features
    char_features = []
    for i in range(min(embedding_dim // 2, len(text) - 1)):
        if i < len(text) - 1:
            char_features.append(ord(text[i]) + ord(text[i+1]))
        else:
            char_features.append(0)
    
    # Word count and length features
    words = text.split()
    word_features = [
        len(words),
        sum(len(w) for w in words) / max(len(words), 1),
        len(set(words)),
    ]
    
    # Combine features
    features = char_features[:embedding_dim - len(word_features)] + word_features
    features = features[:embedding_dim]
    
    # Pad or truncate to embedding_dim
    if len(features) < embedding_dim:
        features = features + [0] * (embedding_dim - len(features))
    else:
        features = features[:embedding_dim]
    
    # Normalize
    features = np.array(features, dtype=np.float32)
    if np.std(features) > 0:
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    return features


class HybridMusicDataset(MusicDataset):
    """Dataset that combines audio and lyrics features."""
    
    def __init__(
        self,
        audio_dir: str,
        lyrics_dir: Optional[str] = None,
        lyrics_csv: Optional[str] = None,
        lyrics_csv_audio_column: str = 'track_id',  # Default to track_id for your CSV
        lyrics_csv_lyrics_column: str = 'lyrics',
        lyrics_embedding_dim: int = 64,
        feature_type: str = 'mfcc',
        **kwargs
    ):
        """
        Args:
            audio_dir: Directory containing audio files
            lyrics_dir: Directory containing lyrics text files (optional)
            lyrics_csv: Path to CSV file containing lyrics (optional)
            lyrics_csv_audio_column: Column name in CSV for audio file names (e.g., 'track_id', 'track_nan', 'filename')
            lyrics_csv_lyrics_column: Column name in CSV for lyrics text
            lyrics_embedding_dim: Dimension of lyrics embedding
            feature_type: Type of audio feature
            **kwargs: Additional arguments for MusicDataset
        """
        super().__init__(audio_dir, feature_type=feature_type, **kwargs)
        self.lyrics_dir = lyrics_dir
        self.lyrics_embedding_dim = lyrics_embedding_dim
        
        # Load lyrics from CSV if provided, otherwise from directory
        self.lyrics_dict = {}
        self.metadata_dict = {}
        
        if lyrics_csv:
            # Try track_id first, then track_name, then filename
            self.lyrics_dict = load_lyrics_from_csv(
                lyrics_csv,
                audio_file_column=lyrics_csv_audio_column,
                lyrics_column=lyrics_csv_lyrics_column,
                audio_file_column_alt='track_name'  # Alternative column name (matches your CSV)
            )
            # Also load metadata (genre, language) if available
            self.metadata_dict = load_metadata_from_csv(lyrics_csv)
        elif lyrics_dir:
            # Load from text files in directory (supports recursive search)
            txt_dict = load_lyrics(lyrics_dir)
            # Convert to base name format (without .txt extension)
            # Also handle matching with audio files that have additional suffixes
            # (e.g., 'ar_0001.txt' matches 'ar_0001_SongName.mp3')
            for filename, lyrics_text in txt_dict.items():
                base_name = os.path.splitext(filename)[0]
                # Store with base name (language_code + number, e.g., 'ar_0001')
                self.lyrics_dict[base_name] = lyrics_text
                # Also store variations for matching (e.g., if audio is 'ar_0001_SongName')
                # The matching will be done in __getitem__ method
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get item with both audio and lyrics features."""
        audio_features, file_name = super().__getitem__(idx)
        
        # Get lyrics embedding
        # Try matching by base name (without extension)
        base_name = os.path.splitext(file_name)[0]
        lyrics_text = self.lyrics_dict.get(base_name, "")
        
        # If not found, try extracting prefix (e.g., 'ar_0001' from 'ar_0001_SongName')
        # This handles cases where audio filename has additional info but lyrics use just prefix
        if not lyrics_text:
            # Extract language_code + number pattern (e.g., 'ar_0001' from 'ar_0001_SongName')
            parts = base_name.split('_')
            if len(parts) >= 2:
                prefix = f"{parts[0]}_{parts[1]}"  # language_code + number
                lyrics_text = self.lyrics_dict.get(prefix, "")
        
        # If still not found, try with .txt extension (for backward compatibility)
        if not lyrics_text:
            lyrics_text = self.lyrics_dict.get(f"{base_name}.txt", "")
            if not lyrics_text:
                parts = base_name.split('_')
                if len(parts) >= 2:
                    prefix = f"{parts[0]}_{parts[1]}"
                    lyrics_text = self.lyrics_dict.get(f"{prefix}.txt", "")
        
        lyrics_embedding = get_lyrics_embedding(lyrics_text, self.lyrics_embedding_dim)
        
        return audio_features, torch.tensor(lyrics_embedding), file_name
    
    def get_all_features(self) -> np.ndarray:
        """Get all audio features as numpy array for clustering (audio-only, not hybrid)."""
        features_list = []
        for i in range(len(self)):
            audio_feat, _, _ = self[i]  # Unpack 3 values: (audio, lyrics, filename)
            features_list.append(audio_feat.numpy())
        return np.array(features_list)
    
    def get_hybrid_features(self) -> np.ndarray:
        """Get combined audio + lyrics features."""
        hybrid_features = []
        for i in range(len(self)):
            audio_feat, lyrics_feat, _ = self[i]
            combined = np.concatenate([audio_feat.numpy(), lyrics_feat.numpy()])
            hybrid_features.append(combined)
        return np.array(hybrid_features)
