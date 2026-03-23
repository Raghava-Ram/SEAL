#!/usr/bin/env python3
"""
Generate publication-ready plots for SEAL research paper.

Focus: M0 (baseline) vs M1 (SEAL) comparison only.
Outputs 3 plots suitable for academic papers.

Usage:
    python generate_paper_plots.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set matplotlib for academic style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Configuration
SEEDS = [42, 123, 999]
METHODS = ["M0", "M1"]  # Only baseline and SEAL
METHOD_FOLDER_MAP = {
    "M0": "baseline",
    "M1": "seal"
}


def triangular_to_square(triangular_matrix: List[List[float]]) -> List[List[float]]:
    """Convert triangular matrix to square matrix format."""
    T = len(triangular_matrix)
    square = [[0.0] * T for _ in range(T)]
    
    for i, row in enumerate(triangular_matrix):
        for j, val in enumerate(row):
            square[i][i + j] = float(val) if val is not None else 0.0
    
    return square


def load_metrics_and_matrix(json_path: str) -> Optional[Tuple[Dict, List[List[float]]]]:
    """Load pre-computed metrics and accuracy matrix from JSON."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract accuracy_matrix
        acc_matrix = data.get('accuracy_matrix', {})
        if not acc_matrix:
            print(f"⚠️  No 'accuracy_matrix' found in {json_path}")
            return None
        
        # Convert dict-based matrix to list format
        if isinstance(acc_matrix, dict):
            matrix_list = []
            for task in acc_matrix.keys():
                row = acc_matrix[task]
                row_clean = [float(v) if v is not None else 0.0 for v in row]
                matrix_list.append(row_clean)
        else:
            matrix_list = acc_matrix
        
        # Check if matrix is triangular (ragged)
        try:
            np.array(matrix_list, dtype=float)
            is_triangular = False
        except (ValueError, TypeError):
            is_triangular = True
        
        # Convert triangular to square if needed
        if is_triangular:
            matrix_for_metrics = triangular_to_square(matrix_list)
        else:
            matrix_for_metrics = matrix_list
        
        # Load pre-computed metrics
        avg_metrics = data.get('average_metrics', {})
        
        if avg_metrics:
            metrics = {
                'average_accuracy': float(avg_metrics.get('average_accuracy', 0.0)),
                'average_forgetting': float(avg_metrics.get('average_forgetting', 0.0)),
                'average_backward_transfer': float(avg_metrics.get('average_backward_transfer', 0.0))
            }
        else:
            print(f"⚠️  No 'average_metrics' found in {json_path}")
            return None
        
        return metrics, matrix_list
    
    except Exception as e:
        print(f"⚠️  Error loading {json_path}: {e}")
        return None


def find_metrics_json(base_path: str, method_folder: str) -> Optional[str]:
    """Find the metrics JSON file in the expected location."""
    candidate = os.path.join(base_path, 'multi_task', method_folder, 'imdb_squad_arc_metrics.json')
    if os.path.exists(candidate):
        return candidate
    return None


def load_summary_data() -> Dict:
    """Load aggregated summary data."""
    root = Path.cwd()
    summary_path = root / 'outputs' / 'multiseed' / 'results_summary.json'
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def load_method_matrices() -> Dict[str, List[np.ndarray]]:
    """Load accuracy matrices for M0 and M1 methods."""
    root = Path.cwd()
    multiseed_dir = root / 'outputs' / 'multiseed'
    
    matrices_by_method = {}
    
    for method in METHODS:
        method_folder = METHOD_FOLDER_MAP[method]
        matrices = []
        
        for seed in SEEDS:
            seed_dir = multiseed_dir / f'seed_{seed}' / method
            
            if not seed_dir.exists():
                print(f"⚠️  Seed {seed} not found for method {method}")
                continue
            
            metrics_path = find_metrics_json(str(seed_dir), method_folder)
            
            if metrics_path is None:
                print(f"⚠️  No metrics JSON found for method {method}, seed {seed}")
                continue
            
            result = load_metrics_and_matrix(metrics_path)
            
            if result is None:
                continue
            
            _, matrix = result
            
            # Convert triangular to square if needed
            try:
                np.array(matrix, dtype=float)
                matrix_square = matrix
            except (ValueError, TypeError):
                matrix_square = triangular_to_square(matrix)
            
            matrices.append(np.array(matrix_square, dtype=float))
        
        matrices_by_method[method] = matrices
    
    return matrices_by_method


def save_comparison_bar_plot(summary_data: Dict, output_path: str):
    """Generate comparison bar plot for M0 vs M1."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    methods = ["M0", "M1"]
    metric_names = ["Final Acc", "Avg Forgetting", "BWT"]
    
    # Prepare data
    means = {
        "M0": [summary_data["M0"]["final_accuracy_mean"], 
               summary_data["M0"]["forgetting_mean"], 
               summary_data["M0"]["bwt_mean"]],
        "M1": [summary_data["M1"]["final_accuracy_mean"], 
               summary_data["M1"]["forgetting_mean"], 
               summary_data["M1"]["bwt_mean"]]
    }
    
    stds = {
        "M0": [summary_data["M0"]["final_accuracy_std"], 
               summary_data["M0"]["forgetting_std"], 
               summary_data["M0"]["bwt_std"]],
        "M1": [summary_data["M1"]["final_accuracy_std"], 
               summary_data["M1"]["forgetting_std"], 
               summary_data["M1"]["bwt_std"]]
    }
    
    x = np.arange(len(methods))
    width = 0.25
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Academic blue, burgundy, and orange
    
    for i, metric in enumerate(metric_names):
        vals = [means[method][i] for method in methods]
        errs = [stds[method][i] for method in methods]
        
        bars = ax.bar(x + i * width, vals, width, yerr=errs, capsize=4, 
                     label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Metric Value')
    ax.set_title('M0 vs M1: Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['M0 (Baseline)', 'M1 (SEAL)'])
    ax.legend(loc='upper right')
    ax.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_combined_forgetting_curves(matrices_by_method: Dict[str, List[np.ndarray]], output_path: str):
    """Generate combined forgetting curves for M0 and M1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    task_names = ['IMDB', 'SQuAD', 'ARC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Standard matplotlib colors
    
    for idx, (method, ax) in enumerate(zip(["M0", "M1"], [ax1, ax2])):
        matrices = matrices_by_method[method]
        
        if not matrices:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{method} - No Data')
            continue
        
        # Compute mean matrix
        mean_matrix = np.stack(matrices, axis=0).mean(axis=0)
        
        # Plot forgetting curves for each task
        for task_idx, task_name in enumerate(task_names):
            accuracies = mean_matrix[task_idx, :]
            ax.plot(range(1, len(accuracies) + 1), accuracies, 
                   marker='o', label=task_name, color=colors[task_idx], 
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('Task Index')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{method} - {"Baseline" if method == "M0" else "SEAL"}')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Forgetting Curves Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_combined_heatmaps(matrices_by_method: Dict[str, List[np.ndarray]], output_path: str):
    """Generate combined accuracy heatmaps for M0 and M1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    task_names = ['IMDB', 'SQuAD', 'ARC']
    
    for idx, (method, ax) in enumerate(zip(["M0", "M1"], [ax1, ax2])):
        matrices = matrices_by_method[method]
        
        if not matrices:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{method} - No Data')
            continue
        
        # Compute mean matrix
        mean_matrix = np.stack(matrices, axis=0).mean(axis=0)
        
        # Create heatmap
        im = ax.imshow(mean_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Add text annotations
        for i in range(len(task_names)):
            for j in range(len(task_names)):
                text = ax.text(j, i, f'{mean_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xticks(range(len(task_names)))
        ax.set_yticks(range(len(task_names)))
        ax.set_xticklabels(['After T1', 'After T2', 'After T3'])
        ax.set_yticklabels(task_names)
        ax.set_title(f'{method} - {"Baseline" if method == "M0" else "SEAL"}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Accuracy', rotation=270, labelpad=15)
    
    plt.suptitle('Accuracy Matrices Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to generate all paper plots."""
    print("=" * 60)
    print("GENERATING PUBLICATION-READY PLOTS")
    print("=" * 60)
    
    # Create output directory
    root = Path.cwd()
    output_dir = root / 'outputs' / 'multiseed' / 'paper_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summary data
    print("\n📊 Loading summary data...")
    try:
        summary_data = load_summary_data()
        print("✅ Summary data loaded")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Load matrices
    print("\n📊 Loading accuracy matrices...")
    matrices_by_method = load_method_matrices()
    
    for method in METHODS:
        count = len(matrices_by_method.get(method, []))
        print(f"  {method}: {count} seed(s) loaded")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    
    # 1. Comparison bar plot
    print("  📊 Comparison bar plot...")
    comparison_path = output_dir / 'comparison_m0_m1.png'
    save_comparison_bar_plot(summary_data, str(comparison_path))
    print(f"    ✅ Saved: {comparison_path}")
    
    # 2. Combined forgetting curves
    print("  📈 Combined forgetting curves...")
    forgetting_path = output_dir / 'forgetting_m0_m1.png'
    save_combined_forgetting_curves(matrices_by_method, str(forgetting_path))
    print(f"    ✅ Saved: {forgetting_path}")
    
    # 3. Combined heatmaps
    print("  🔥 Combined heatmaps...")
    heatmap_path = output_dir / 'heatmap_m0_m1.png'
    save_combined_heatmaps(matrices_by_method, str(heatmap_path))
    print(f"    ✅ Saved: {heatmap_path}")
    
    print("\n" + "=" * 60)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
