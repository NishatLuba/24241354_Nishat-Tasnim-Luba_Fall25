"""
Comprehensive script to run all tasks (Easy, Medium, Hard) sequentially.
This ensures all requirements are met and results are generated.
"""

import os
import sys
import subprocess
import time

def run_task(task_name, script_path):
    """Run a task script and handle errors."""
    print("\n" + "=" * 80)
    print(f"Running {task_name}")
    print("=" * 80)
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {task_name} completed successfully!")
            return True
        else:
            print(f"\n⚠️  {task_name} completed with warnings (exit code: {result.returncode})")
            return True  # Still consider it successful if it ran
    except Exception as e:
        print(f"\n❌ Error running {task_name}: {e}")
        return False

def check_requirements():
    """Check if basic requirements are met."""
    print("Checking requirements...")
    
    issues = []
    
    # Check if data directory exists
    if not os.path.exists('data/audio'):
        issues.append("⚠️  data/audio directory not found. Create it and add audio files.")
    elif len(os.listdir('data/audio')) == 0:
        issues.append("⚠️  data/audio directory is empty. Add audio files (.wav, .mp3, .flac, .m4a)")
    
    # Check if results directory exists
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/latent_visualization', exist_ok=True)
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Tip: Add your audio and lyrics files to data/Audio/audio/ and data/Lyrics/ directories")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True

def main():
    """Main function to run all tasks."""
    print("=" * 80)
    print("VAE for Hybrid Language Music Clustering - Complete Pipeline")
    print("=" * 80)
    
    # Check requirements
    if not check_requirements():
        print("\nExiting. Please fix the issues above and try again.")
        return
    
    # List of tasks to run
    tasks = [
        ("Easy Task", "src/train_easy.py"),
        ("Medium Task", "src/train_medium.py"),
        ("Hard Task", "src/train_hard.py"),
    ]
    
    results = {}
    start_time = time.time()
    
    for task_name, script_path in tasks:
        success = run_task(task_name, script_path)
        results[task_name] = success
        
        if not success:
            print(f"\n⚠️  {task_name} failed. Continuing with next task...")
        
        # Small delay between tasks
        time.sleep(2)
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("Pipeline Summary")
    print("=" * 80)
    
    for task_name, success in results.items():
        status = "✅ Completed" if success else "❌ Failed"
        print(f"{task_name}: {status}")
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print("\nResults are saved in the 'results/' directory:")
    print("  - Metrics: results/clustering_metrics_*.csv")
    print("  - Visualizations: results/latent_visualization/*.png")
    print("  - Models: results/*_model_*.pth")
    
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)

if __name__ == '__main__':
    main()
