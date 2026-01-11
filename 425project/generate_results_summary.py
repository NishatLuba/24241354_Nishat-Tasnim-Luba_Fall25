"""
Generate a comprehensive summary of all results from the three tasks.
This creates a markdown report with all metrics and visualizations.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def dataframe_to_markdown(df):
    """Convert DataFrame to markdown table, with fallback if tabulate is not available."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # Fallback: create simple markdown table without tabulate
        lines = []
        # Header
        lines.append("| " + " | ".join(df.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        # Rows
        for _, row in df.iterrows():
            lines.append("| " + " | ".join([str(val) for val in row]) + " |")
        return "\n".join(lines)

def load_metrics(file_path):
    """Load metrics from CSV file."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def generate_summary():
    """Generate comprehensive results summary."""
    
    results_dir = 'results'
    summary_lines = []
    
    summary_lines.append("# Results Summary: VAE for Hybrid Language Music Clustering\n")
    summary_lines.append("Generated automatically from experiment results.\n")
    summary_lines.append("=" * 80 + "\n")
    
    # Easy Task Results
    summary_lines.append("\n## Easy Task Results\n")
    easy_metrics = load_metrics(os.path.join(results_dir, 'clustering_metrics.csv'))
    if easy_metrics is not None:
        summary_lines.append("### Clustering Metrics\n")
        summary_lines.append(dataframe_to_markdown(easy_metrics))
        summary_lines.append("\n")
    else:
        summary_lines.append("⚠️  Metrics file not found. Run `python src/train_easy.py` first.\n")
    
    # Medium Task Results
    summary_lines.append("\n## Medium Task Results\n")
    medium_metrics = load_metrics(os.path.join(results_dir, 'clustering_metrics_medium.csv'))
    if medium_metrics is not None:
        summary_lines.append("### Clustering Metrics\n")
        summary_lines.append(dataframe_to_markdown(medium_metrics))
        summary_lines.append("\n")
    else:
        summary_lines.append("⚠️  Metrics file not found. Run `python src/train_medium.py` first.\n")
    
    # Hard Task Results
    summary_lines.append("\n## Hard Task Results\n")
    hard_metrics = load_metrics(os.path.join(results_dir, 'clustering_metrics_hard.csv'))
    if hard_metrics is not None:
        summary_lines.append("### Comprehensive Clustering Metrics\n")
        summary_lines.append(dataframe_to_markdown(hard_metrics))
        summary_lines.append("\n")
    else:
        summary_lines.append("⚠️  Metrics file not found. Run `python src/train_hard.py` first.\n")
    
    # Visualizations
    summary_lines.append("\n## Generated Visualizations\n")
    viz_dir = os.path.join(results_dir, 'latent_visualization')
    
    if os.path.exists(viz_dir):
        viz_files = glob.glob(os.path.join(viz_dir, '*.png'))
        if viz_files:
            summary_lines.append("The following visualizations have been generated:\n")
            for viz_file in sorted(viz_files):
                viz_name = os.path.basename(viz_file)
                summary_lines.append(f"- `{viz_name}`\n")
        else:
            summary_lines.append("⚠️  No visualizations found. Run the training scripts to generate them.\n")
    else:
        summary_lines.append("⚠️  Visualization directory not found.\n")
    
    # Models
    summary_lines.append("\n## Saved Models\n")
    model_files = glob.glob(os.path.join(results_dir, '*.pth'))
    if model_files:
        summary_lines.append("The following trained models have been saved:\n")
        for model_file in sorted(model_files):
            model_name = os.path.basename(model_file)
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            summary_lines.append(f"- `{model_name}` ({file_size:.2f} MB)\n")
    else:
        summary_lines.append("⚠️  No model files found. Run the training scripts to save models.\n")
    
    # Training Loss Plots
    summary_lines.append("\n## Training Loss Plots\n")
    loss_files = glob.glob(os.path.join(results_dir, '*_training_loss.png'))
    loss_files.extend(glob.glob(os.path.join(results_dir, '*_loss.png')))
    if loss_files:
        summary_lines.append("Training loss curves have been generated:\n")
        for loss_file in sorted(loss_files):
            loss_name = os.path.basename(loss_file)
            summary_lines.append(f"- `{loss_name}`\n")
    else:
        summary_lines.append("⚠️  No training loss plots found.\n")
    
    # Write summary
    summary_path = os.path.join(results_dir, 'RESULTS_SUMMARY.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✅ Results summary generated: {summary_path}")
    print("\nSummary includes:")
    print("  - Easy Task metrics")
    print("  - Medium Task metrics")
    print("  - Hard Task metrics")
    print("  - List of all visualizations")
    print("  - List of saved models")
    print("  - List of training loss plots")

if __name__ == '__main__':
    generate_summary()
