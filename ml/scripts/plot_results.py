#!/usr/bin/env python3
"""
Plot results from JSON benchmark files.
Creates comprehensive comparison plots for model performance.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Model categories
SINGLE_DOMAIN_MODELS = ["MF", "MF Explicit", "MF BPR", "NCF", "LightGCN", "DeepFM"]
CROSS_DOMAIN_MODELS = ["CMF", "DeepAPF", "Bi-TGCF", "EMCDR"]

def load_json_results(file_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}

def flatten_results(results: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Flatten nested results into a list of model results."""
    flat_data = []
    
    if file_name == "final_benchmark_results.json":
        # Handle nested structure
        for category, models in results.items():
            if category in ["single_domain", "cross_domain"]:
                for model_name, metrics in models.items():
                    if isinstance(metrics, dict) and "recall@10" in metrics:
                        flat_data.append({
                            "Model": model_name,
                            "Recall@10": metrics["recall@10"],
                            "NDCG@10": metrics.get("ndcg@10", 0),
                            "HitRate@10": metrics.get("hit_rate@10", 0),
                            "Type": category.replace("_", "-").title(),
                            "Time (s)": metrics.get("train_time_s", 0),
                            "Source": file_name
                        })
    else:
        # Handle flat structure - could be individual model files or collection
        if "recall@10" in results:
            # This is an individual model file - extract model name from filename
            model_name = file_name.replace(".json", "").replace("_", " ")
            model_type = "Cross-Domain" if any(name in model_name for name in CROSS_DOMAIN_MODELS) else "Single-Domain"
            flat_data.append({
                "Model": model_name,
                "Recall@10": results["recall@10"],
                "NDCG@10": results.get("ndcg@10", 0),
                "HitRate@10": results.get("hit_rate@10", 0),
                "Type": model_type,
                "Time (s)": results.get("train_time_s", 0),
                "Source": file_name
            })
        else:
            # Handle collection of models
            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and "recall@10" in metrics:
                    model_type = "Cross-Domain" if model_name in CROSS_DOMAIN_MODELS else "Single-Domain"
                    flat_data.append({
                        "Model": model_name,
                        "Recall@10": metrics["recall@10"],
                        "NDCG@10": metrics.get("ndcg@10", 0),
                        "HitRate@10": metrics.get("hit_rate@10", 0),
                        "Type": model_type,
                        "Time (s)": metrics.get("train_time_s", 0),
                        "Source": file_name
                    })
    
    return flat_data

def create_comparison_plots(all_results: List[Dict[str, Any]], output_dir: Path):
    """Create comprehensive comparison plots."""
    if not all_results:
        print("❌ No valid results to plot!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = {
        'Single-Domain': '#4ECDC4',
        'Cross-Domain': '#FF6B6B',
        'Single-Domain': '#4ECDC4',
        'Cross-Domain': '#FF6B6B'
    }
    
    # Plot 1: Recall@10 Comparison (Horizontal Bar)
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('Recall@10', ascending=True)
    bar_colors = [colors.get(t, '#95A5A6') for t in df_sorted['Type']]
    bars = ax1.barh(df_sorted['Model'], df_sorted['Recall@10'], color=bar_colors, alpha=0.8)
    ax1.set_title('Recall@10 Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Recall@10')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, df_sorted['Recall@10']):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=9)
    
    # Plot 2: NDCG@10 Comparison
    ax2 = axes[0, 1]
    df_sorted_ndcg = df.sort_values('NDCG@10', ascending=True)
    bar_colors_ndcg = [colors.get(t, '#95A5A6') for t in df_sorted_ndcg['Type']]
    bars_ndcg = ax2.barh(df_sorted_ndcg['Model'], df_sorted_ndcg['NDCG@10'], color=bar_colors_ndcg, alpha=0.8)
    ax2.set_title('NDCG@10 Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('NDCG@10')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars_ndcg, df_sorted_ndcg['NDCG@10']):
        width = bar.get_width()
        ax2.text(width + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=9)
    
    # Plot 3: Single vs Cross Domain Grouped Bar
    ax3 = axes[0, 2]
    single_df = df[df['Type'].isin(['Single-Domain', 'Single-domain'])].copy()
    cross_df = df[df['Type'].isin(['Cross-Domain', 'Cross-domain'])].copy()
    
    if not single_df.empty and not cross_df.empty:
        x_single = np.arange(len(single_df))
        x_cross = np.arange(len(cross_df)) + 0.4
        width = 0.35
        
        ax3.bar(x_single, single_df['Recall@10'], width, 
                label='Single-Domain', alpha=0.8, color='#4ECDC4')
        ax3.bar(x_cross, cross_df['Recall@10'], width,
                label='Cross-Domain', alpha=0.8, color='#FF6B6B')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Recall@10')
        ax3.set_title('Single vs Cross-Domain')
        ax3.set_xticks(np.concatenate([x_single, x_cross]))
        ax3.set_xticklabels(list(single_df['Model']) + list(cross_df['Model']), rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance vs Efficiency
    ax4 = axes[1, 0]
    scatter_colors = [colors.get(t, '#95A5A6') for t in df['Type']]
    scatter = ax4.scatter(df['Time (s)'], df['Recall@10'], 
                         s=100, alpha=0.7, c=scatter_colors)
    
    # Add model labels
    for i, row in df.iterrows():
        ax4.annotate(row['Model'], 
                    (row['Time (s)'], row['Recall@10']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Recall@10')
    ax4.set_title('Performance vs Efficiency')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Performance Summary Table
    ax5 = axes[1, 1]
    ax5.axis('tight')
    ax5.axis('off')
    
    # Sort by Recall@10 for table
    df_table = df.sort_values('Recall@10', ascending=False)
    table_data = []
    for _, row in df_table.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Recall@10']:.4f}",
            f"{row['NDCG@10']:.4f}",
            f"{row['Time (s)']:.1f}s",
            row['Type']
        ])
    
    table = ax5.table(cellText=table_data, 
                     colLabels=['Model', 'Recall@10', 'NDCG@10', 'Time', 'Type'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Color code by type
    for i in range(len(table_data)):
        if table_data[i][-1] in ['Cross-Domain', 'Cross-domain']:
            for j in range(4):
                table[(i+1, j)].set_facecolor('#FF6B6B33')
        else:
            for j in range(4):
                table[(i+1, j)].set_facecolor('#4ECDC433')
    
    ax5.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Plot 6: Source Distribution
    ax6 = axes[1, 2]
    source_counts = df['Source'].value_counts()
    ax6.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
            colors=['#FF6B6B', '#4ECDC4', '#95A5A6', '#F39C12'])
    ax6.set_title('Results by Source File')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "model_comparison_comprehensive.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive plot saved to {plot_file}")
    
    # Save combined results
    results_file = output_dir / "combined_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Combined results saved to {results_file}")

def main():
    """Main function to plot results from JSON files."""
    results_dir = PROJECT_ROOT / "artifacts" / "results"
    output_dir = PROJECT_ROOT / "artifacts" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Looking for result files in {results_dir}")
    
    # Find all JSON result files
    json_files = list(results_dir.glob("*.json"))
    # Process all JSON files, not just ones with specific keywords
    result_files = json_files
    
    if not result_files:
        print("❌ No benchmark result files found!")
        print(f"   Available files: {[f.name for f in json_files]}")
        return
    
    print(f"📁 Found {len(result_files)} result files:")
    for f in result_files:
        print(f"   - {f.name}")
    
    # Load and combine all results
    all_results = []
    for file_path in result_files:
        print(f"\n📖 Loading {file_path.name}...")
        results = load_json_results(file_path)
        if results:
            flat_results = flatten_results(results, file_path.name)
            all_results.extend(flat_results)
            print(f"   ✅ Loaded {len(flat_results)} model results")
    
    if not all_results:
        print("❌ No valid results found in any file!")
        return
    
    print(f"\n📊 Total results: {len(all_results)} model entries")
    
    # Create plots
    create_comparison_plots(all_results, output_dir)
    
    # Print summary
    df = pd.DataFrame(all_results)
    print("\n📈 Performance Summary:")
    print("=" * 60)
    for _, row in df.sort_values('Recall@10', ascending=False).iterrows():
        print(f"{row['Model']:<15} | Recall@10: {row['Recall@10']:.4f} | "
              f"NDCG@10: {row['NDCG@10']:.4f} | Time: {row['Time (s)']:.1f}s | "
              f"Type: {row['Type']}")

if __name__ == "__main__":
    main()
