# VAE for Hybrid Language Music Clustering

This project implements Variational Autoencoders (VAE) for clustering music tracks, with support for multiple languages. The project is structured into three difficulty levels: Easy, Medium, and Hard tasks.

**Note:** While the project supports hybrid language datasets (e.g., English + Bangla), you can also use English-only datasets. The code works with any audio dataset regardless of language.

## Project Structure

```
425project/
├── data/
│   ├── Audio/audio/    # Audio files (MP3, WAV, etc.)
│   └── Lyrics/         # Lyrics files (TXT)
├── src/
│   ├── vae.py          # VAE implementations (BasicVAE, Beta-VAE, CVAE)
│   ├── dataset.py      # Dataset loading and preprocessing
│   ├── clustering.py   # Clustering algorithms (K-Means, Agglomerative, DBSCAN)
│   ├── evaluation.py   # Evaluation metrics
│   ├── preprocessing.py # Data preprocessing utilities
│   ├── visualization.py # Visualization utilities
│   ├── train_easy.py   # Easy Task: Basic VAE + K-Means
│   ├── train_medium.py # Medium Task: Hybrid VAE + Multiple clustering
│   └── train_hard.py   # Hard Task: Advanced VAEs + Comprehensive evaluation
├── results/            # Generated results (excluded from git)
│   └── latent_visualization/
├── generate_results_summary.py   # Generate results summary
├── run_all_tasks.py    # Run all tasks sequentially
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare your dataset in the `data/` directory.
   - **Note**: Dataset files are excluded from git via `.gitignore` due to their large size.
   - Place audio files in `data/Audio/audio/` and lyrics files in `data/Lyrics/`.

## Usage

### Quick Start (Run All Tasks)
```bash
python run_all_tasks.py
```
This will run Easy, Medium, and Hard tasks sequentially.

### Individual Tasks

#### Easy Task
```bash
python src/train_easy.py
```

#### Medium Task
```bash
python src/train_medium.py
```

#### Hard Task
```bash
python src/train_hard.py
```

### Generate Results Summary
```bash
python generate_results_summary.py
```
Creates a comprehensive markdown summary of all results.

## Dataset

This project uses a multi-language music dataset organized by language/genre:
- **Audio files**: 1119 MP3 files organized by language (Arabic, Bangla, English, Hindi, Spanish)
  - Located in `data/Audio/audio/`
  - Supports: .wav, .mp3, .flac, .m4a
  - **30-second clips are acceptable** (default duration)
- **Lyrics**: 677 TXT files with 100% match rate to audio files
  - Located in `data/Lyrics/`
  - Organized by language matching audio structure
- **Genre information**: Extracted from folder structure (e.g., Spanish subgenres: reggaeton, latin_pop, etc.)

**Language Priority:** English is recommended first, then Bangla, then any other language. However, **English-only datasets are acceptable** if Bangla data is unavailable.

**Recommended datasets:**
- **GTZAN Genre Collection** ⭐ BEST START - English, 1000 songs, 10 genres, 30-second clips
  - URL: http://marsyas.info/downloads/datasets.html
- **Kaggle Lyrics Datasets** - For lyrics (search: "lyrics")
  - URL: https://www.kaggle.com/datasets?search=lyrics
- **MIR-1K Dataset** - Multi-language (Mandarin + English)
- **BanglaBeats Dataset** - Bangla, 30-second clips (if available)


## Results

Results and visualizations are saved in the `results/` directory.
