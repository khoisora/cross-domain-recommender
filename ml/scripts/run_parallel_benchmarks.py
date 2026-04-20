"""Parallel benchmark runner for single and cross-domain models.

Spawns each bench_*.py as a subprocess with the correct PYTHONPATH
and virtual-env Python so that ``ml.*`` / ``backend.*`` imports work.
"""

import os
import subprocess
import time
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "ml" / "scripts" / "benchmarks"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# Fall back to current interpreter if venv not found
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
SINGLE_DOMAIN_MODELS = {
    "MF BPR": {
        "script": "bench_mf_bpr.py",
        "type": "single",
    },
    "LightGCN": {
        "script": "bench_lightgcn.py",
        "type": "single",
    },
    "NCF": {
        "script": "bench_ncf.py",
        "type": "single",
    },
}

CROSS_DOMAIN_MODELS = {
    "CMF": {
        "script": "bench_cmf.py",
        "type": "cross",
    },
    "DeepAPF": {
        "script": "bench_deepapf.py",
        "type": "cross",
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(model_name, config):
    """Run a single benchmark script as a subprocess."""
    script_path = BENCHMARKS_DIR / config["script"]

    if not script_path.exists():
        return model_name, {"error": f"Script not found: {script_path}"}

    print(f"🚀 Starting {model_name}...")
    start_time = time.time()

    try:
        # Build env: inherit everything, override PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{BENCHMARKS_DIR}"

        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=5400,  # 90 min timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {model_name} completed in {elapsed:.1f}s")
            metrics = extract_metrics(result.stdout)
            metrics["train_time_s"] = elapsed
            return model_name, metrics
        else:
            print(f"❌ {model_name} failed!")
            # Show last 20 lines of stderr for debugging
            err_lines = result.stderr.strip().split("\n")
            short_err = "\n".join(err_lines[-20:])
            return model_name, {"error": short_err, "train_time_s": elapsed}

    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} timed out!")
        return model_name, {"error": "Timeout", "train_time_s": 1800}
    except Exception as e:
        print(f"💥 {model_name} crashed: {e}")
        return model_name, {"error": str(e), "train_time_s": time.time() - start_time}

def extract_metrics(output):
    """Extract metrics from script output."""
    metrics = {}
    
    for line in output.split('\n'):
        if "Recall@10=" in line:
            try:
                metrics["recall@10"] = float(line.split("Recall@10=")[1].split()[0])
            except:
                pass
        if "NDCG@10=" in line:
            try:
                metrics["ndcg@10"] = float(line.split("NDCG@10=")[1].split()[0])
            except:
                pass

    return metrics

def run_all_benchmarks():
    """Run all benchmarks in parallel."""
    print("🚀 Starting Parallel Benchmark Suite")
    print("="*60)
    
    all_models = {**SINGLE_DOMAIN_MODELS, **CROSS_DOMAIN_MODELS}
    results = {}
    
    # Run in parallel with limited workers to avoid memory issues
    max_workers = min(4, len(all_models))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all benchmarks
        future_to_model = {
            executor.submit(run_benchmark, name, config): name 
            for name, config in all_models.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                name, result = future.result()
                results[name] = result
            except Exception as e:
                print(f"💥 {model_name} failed: {e}")
                results[model_name] = {"error": str(e)}
    
    return results

def create_plots(results):
    """Create comparison plots."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    print("\n📊 Creating plots...")
    
    # Prepare data
    data = []
    for model, metrics in results.items():
        if "error" not in metrics and "recall@10" in metrics:
            data.append({
                "Model": model,
                "Recall@10": metrics["recall@10"],
                "NDCG@10": metrics.get("ndcg@10", 0),
                "Type": "Single-Domain" if model in SINGLE_DOMAIN_MODELS else "Cross-Domain",
                "Time (s)": metrics.get("train_time_s", 0)
            })
    
    if not data:
        print("❌ No valid results to plot!")
        return
    
    df = pd.DataFrame(data)
    df = df.sort_values('Recall@10', ascending=True)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Recall@10 comparison
    ax1 = axes[0, 0]
    colors = ['#FF6B6B' if t == 'Cross-Domain' else '#4ECDC4' for t in df['Type']]
    bars = ax1.barh(df['Model'], df['Recall@10'], color=colors, alpha=0.8)
    ax1.set_title('Recall@10 Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Recall@10')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, df['Recall@10']):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=10)
    
    # Plot 2: Single vs Cross domain
    ax2 = axes[0, 1]
    single_df = df[df['Type'] == 'Single-Domain'].copy()
    cross_df = df[df['Type'] == 'Cross-Domain'].copy()
    
    if not single_df.empty and not cross_df.empty:
        # Use separate x positions for each group
        x_single = np.arange(len(single_df))
        x_cross = np.arange(len(cross_df)) + 0.4  # offset
        width = 0.35
        
        ax2.bar(x_single, single_df['Recall@10'], width, 
                label='Single-Domain', alpha=0.8, color='#4ECDC4')
        ax2.bar(x_cross, cross_df['Recall@10'], width,
                label='Cross-Domain', alpha=0.8, color='#FF6B6B')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Recall@10')
        ax2.set_title('Single vs Cross-Domain')
        ax2.set_xticks(np.concatenate([x_single, x_cross]))
        ax2.set_xticklabels(list(single_df['Model']) + list(cross_df['Model']), rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance vs Time
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['Time (s)'], df['Recall@10'], 
                         s=100, alpha=0.7, c=colors)
    
    # Add model labels
    for i, row in df.iterrows():
        ax3.annotate(row['Model'], 
                    (row['Time (s)'], row['Recall@10']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_ylabel('Recall@10')
    ax3.set_title('Performance vs Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append([
            row['Model'],
            f"{row['Recall@10']:.4f}",
            f"{row['NDCG@10']:.4f}",
            f"{row['Time (s)']:.1f}s",
            row['Type']
        ])
    
    table = ax4.table(cellText=summary_data, 
                     colLabels=['Model', 'Recall@10', 'NDCG@10', 'Time', 'Type'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code by type
    for i in range(len(summary_data)):
        if summary_data[i][-1] == 'Cross-Domain':
            for j in range(4):
                table[(i+1, j)].set_facecolor('#FF6B6B33')
        else:
            for j in range(4):
                table[(i+1, j)].set_facecolor('#4ECDC433')
    
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Parallel Benchmark Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = PROJECT_ROOT / "artifacts" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "parallel_benchmark_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to {output_dir / 'parallel_benchmark_results.png'}")
    
    # Save results
    results_file = output_dir.parent / "results" / "parallel_benchmark_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_file}")

def main():
    """Main function."""
    print("🚀 Parallel Benchmark Runner")
    print("Single-Domain: MF Explicit, MF BPR, LightGCN, NCF, DeepFM")
    print("Cross-Domain: CMF, DeepAPF")
    print("="*60)
    
    # Run benchmarks
    start_time = time.time()
    results = run_all_benchmarks()
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    successful = {k: v for k, v in results.items() if "error" not in v}
    failed = {k: v for k, v in results.items() if "error" in v}
    
    if successful:
        print(f"\n✅ Successful ({len(successful)}):")
        sorted_results = sorted(successful.items(), key=lambda x: x[1].get("recall@10", 0), reverse=True)
        for i, (model, metrics) in enumerate(sorted_results, 1):
            recall = metrics.get("recall@10", 0)
            time_taken = metrics.get("train_time_s", 0)
            print(f"  {i}. {model}: Recall@10={recall:.4f}, Time={time_taken:.1f}s")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for model, error in failed.items():
            print(f"  • {model}: {error.get('error', 'Unknown error')}")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s")
    
    # Create plots
    create_plots(results)
    
    print("\n✅ Parallel benchmarks completed!")

if __name__ == "__main__":
    main()
