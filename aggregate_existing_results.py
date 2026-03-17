"""Aggregate results from completed multi-seed experiments.

This script reads existing results from JSON outputs,
loads PRE-COMPUTED metrics directly (NOT reconstructing from matrices),
aggregates across seeds, and generates publication-ready plots.

NO TRAINING IS PERFORMED - only JSON aggregation and plotting.
NO METRIC RECOMPUTATION - metrics loaded directly from average_metrics.
"""
import os
import json
import statistics
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent hangs

import numpy as np

import plots


SEEDS = [42, 123, 999]
METHODS = ["M0", "M1", "M6"]

# Mapping methods -> folder names
METHOD_FOLDER_MAP = {
    "M0": "baseline",
    "M1": "seal",
    "M6": "hybrid"
}


def triangular_to_square(triangular_matrix: List[List[float]]) -> List[List[float]]:
    """Convert triangular matrix to square matrix format.
    
    Triangular: row i has length T-i
    Example: [[a, b, c], [d, e], [f]]
    Into square: [[a, b, c], [0, d, e], [0, 0, f]]
    """
    T = len(triangular_matrix)
    square = [[0.0] * T for _ in range(T)]
    
    for i, row in enumerate(triangular_matrix):
        for j, val in enumerate(row):
            square[i][i + j] = float(val) if val is not None else 0.0
    
    return square


def load_metrics_and_matrix(json_path: str) -> Optional[Tuple[Dict, List[List[float]]]]:
    """Load pre-computed metrics and accuracy matrix from JSON.
    
    Returns: (metrics_dict, accuracy_matrix_as_list) or None if failed
    
    If average_metrics exists in JSON, use it directly.
    Otherwise, compute metrics from accuracy_matrix as fallback.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract accuracy_matrix (always needed for plotting)
        acc_matrix = data.get('accuracy_matrix', {})
        if not acc_matrix:
            print(f"⚠️  No 'accuracy_matrix' found in {json_path}")
            return None
        
        # Convert dict-based matrix to list format
        if isinstance(acc_matrix, dict):
            # Use insertion order (Python 3.7+), do NOT sort
            matrix_list = []
            for task in acc_matrix.keys():
                row = acc_matrix[task]
                row_clean = [float(v) if v is not None else 0.0 for v in row]
                matrix_list.append(row_clean)
        else:
            # Already a list
            matrix_list = acc_matrix
        
        # Check if matrix is triangular (ragged)
        try:
            np.array(matrix_list, dtype=float)
            is_triangular = False
        except (ValueError, TypeError):
            # Ragged array = triangular format
            is_triangular = True
        
        # Convert triangular to square if needed
        if is_triangular:
            matrix_for_metrics = triangular_to_square(matrix_list)
        else:
            matrix_for_metrics = matrix_list
        
        # Try to load pre-computed metrics from JSON
        avg_metrics = data.get('average_metrics', {})
        
        if avg_metrics:
            # Use pre-computed metrics from JSON
            metrics = {
                'average_accuracy': float(avg_metrics.get('average_accuracy', 0.0)),
                'average_forgetting': float(avg_metrics.get('average_forgetting', 0.0)),
                'average_backward_transfer': float(avg_metrics.get('average_backward_transfer', 0.0))
            }
        else:
            # Fallback: compute metrics from accuracy_matrix (for JSON without average_metrics)
            from seal import metrics as seal_metrics
            
            try:
                metrics = {
                    'average_accuracy': float(seal_metrics.final_average_accuracy(matrix_for_metrics)),
                    'average_forgetting': float(seal_metrics.average_forgetting(matrix_for_metrics)),
                    'average_backward_transfer': float(seal_metrics.backward_transfer(matrix_for_metrics))
                }
            except Exception as e:
                print(f"⚠️  Could not compute metrics from matrix in {json_path}: {e}")
                return None
        
        return metrics, matrix_list
    
    except FileNotFoundError:
        print(f"⚠️  File not found: {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON decode error in {json_path}: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Unexpected error loading {json_path}: {e}")
        return None


def find_metrics_json(base_path: str, method_folder: str) -> Optional[str]:
    """Find the metrics JSON file in the expected location.
    
    Expected: base_path/multi_task/method_folder/imdb_squad_arc_metrics.json
    """
    candidate = os.path.join(base_path, 'multi_task', method_folder, 'imdb_squad_arc_metrics.json')
    if os.path.exists(candidate):
        return candidate
    return None


def aggregate_results():
    """Main aggregation function.
    
    Load PRE-COMPUTED metrics (not recomputed), aggregate across seeds,
    generate plots, and print summary table.
    """
    root = Path.cwd()
    multiseed_dir = root / 'outputs' / 'multiseed'
    
    if not multiseed_dir.exists():
        raise FileNotFoundError(f"multiseed directory not found: {multiseed_dir}")
    
    # Dictionary to store results per method
    # Format: { method: [{ seed: 42, metrics: {...}, matrix: [...] }, ...] }
    results_by_method = {m: [] for m in METHODS}
    
    print("=" * 70)
    print("LOADING PRE-COMPUTED METRICS FROM EXISTING RESULTS")
    print("=" * 70)
    
    for method in METHODS:
        method_folder = METHOD_FOLDER_MAP[method]
        print(f"\n📂 Method: {method} (folder: {method_folder})")
        
        for seed in SEEDS:
            seed_dir = multiseed_dir / f'seed_{seed}' / method
            
            if not seed_dir.exists():
                print(f"  ⚠️  Seed {seed} not found at {seed_dir}")
                continue
            
            # Find metrics JSON
            metrics_path = find_metrics_json(str(seed_dir), method_folder)
            
            if metrics_path is None:
                print(f"  ⚠️  No metrics JSON found for seed {seed}")
                continue
            
            # Load metrics and matrix
            result = load_metrics_and_matrix(metrics_path)
            
            if result is None:
                print(f"  ⚠️  Failed to load from {metrics_path}")
                continue
            
            metrics, matrix = result
            
            results_by_method[method].append({
                'seed': seed,
                'metrics': metrics,
                'matrix': matrix,
                'path': metrics_path
            })
            print(f"  ✅ Seed {seed}: avg_acc={metrics['average_accuracy']:.4f}, " +
                  f"forgetting={metrics['average_forgetting']:.4f}, " +
                  f"bwt={metrics['average_backward_transfer']:.4f}")
    
    # Aggregate metrics across seeds
    print("\n" + "=" * 70)
    print("AGGREGATING METRICS ACROSS SEEDS")
    print("=" * 70)
    
    summary = {}
    plot_data = {}
    
    for method in METHODS:
        runs = results_by_method[method]
        
        if not runs:
            print(f"\n⚠️  No successful runs for method {method}")
            continue
        
        print(f"\n🔢 {method}: {len(runs)} seed(s)")
        
        # Extract metrics per seed
        accuracies = []
        forgettings = []
        bwts = []
        matrices = []
        
        for run in runs:
            seed = run['seed']
            metrics = run['metrics']
            matrix = run['matrix']
            
            # Convert triangular to square if needed (for M6)
            try:
                np.array(matrix, dtype=float)
                matrix_square = matrix  # Already square
            except (ValueError, TypeError):
                # Ragged array = triangular format
                matrix_square = triangular_to_square(matrix)
            
            acc = metrics['average_accuracy']
            forget = metrics['average_forgetting']
            bwt = metrics['average_backward_transfer']
            
            accuracies.append(acc)
            forgettings.append(forget)
            bwts.append(bwt)
            matrices.append(np.array(matrix_square, dtype=float))
            
            print(f"  Seed {seed}: final_acc={acc:.4f}, forgetting={forget:.4f}, bwt={bwt:.4f}")
        
        # Compute mean and std across seeds
        if accuracies:
            summary[method] = {
                'final_accuracy_mean': statistics.mean(accuracies),
                'final_accuracy_std': statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
                'forgetting_mean': statistics.mean(forgettings),
                'forgetting_std': statistics.pstdev(forgettings) if len(forgettings) > 1 else 0.0,
                'bwt_mean': statistics.mean(bwts),
                'bwt_std': statistics.pstdev(bwts) if len(bwts) > 1 else 0.0,
                'num_seeds': len(accuracies)
            }
            
            # Stack matrices for averaging (for heatmaps)
            stacked = np.stack(matrices, axis=0)
            mean_matrix = stacked.mean(axis=0).tolist()
            
            plot_data[method] = {
                'matrices': matrices,
                'mean_matrix': mean_matrix,
                'num_seeds': len(matrices)
            }
    
    # Save summary
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    summary_path = multiseed_dir / 'results_summary.json'
    multiseed_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plots_dir = multiseed_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Heatmaps and forgetting curves per method
    for method in METHODS:
        if method not in plot_data:
            continue
        
        data = plot_data[method]
        mean_matrix = data['mean_matrix']
        num_seeds = data['num_seeds']
        
        # Heatmap
        heatmap_path = str(plots_dir / f'mean_heatmap_{method}.png')
        try:
            plots.save_heatmap(
                mean_matrix,
                heatmap_path,
                title=f'Mean Accuracy Matrix – {method} ({num_seeds} seeds)'
            )
            print(f"✅ Heatmap saved: {heatmap_path}")
        except Exception as e:
            print(f"⚠️  Error generating heatmap for {method}: {e}")
        
        # Forgetting curve
        T = len(mean_matrix)
        task_names = [f'Task {i+1}' for i in range(T)]
        task_acc_hist = {task_names[i]: [mean_matrix[i][j] for j in range(T)] for i in range(T)}
        
        curve_path = str(plots_dir / f'forgetting_curve_{method}.png')
        try:
            plots.save_forgetting_curve(
                task_acc_hist,
                curve_path,
                title=f'Forgetting Curve – {method}'
            )
            print(f"✅ Forgetting curve saved: {curve_path}")
        except Exception as e:
            print(f"⚠️  Error generating forgetting curve for {method}: {e}")
    
    # Comparison bar plot
    if summary:
        metrics_for_plot = {}
        for method in summary:
            s = summary[method]
            metrics_for_plot[method] = {
                'final_accuracy': (s['final_accuracy_mean'], s['final_accuracy_std']),
                'avg_forgetting': (s['forgetting_mean'], s['forgetting_std']),
                'bwt': (s['bwt_mean'], s['bwt_std'])
            }
        
        comparison_path = str(plots_dir / 'comparison_metrics.png')
        try:
            plots.save_comparison_metrics(
                metrics_for_plot,
                comparison_path,
                title='Comparison Metrics (mean ± std)'
            )
            print(f"✅ Comparison plot saved: {comparison_path}")
        except Exception as e:
            print(f"⚠️  Error generating comparison plot: {e}")
    
    # Print publication-ready table
    print("\n" + "=" * 70)
    print("PUBLICATION-READY RESULTS TABLE")
    print("=" * 70 + "\n")
    
    print("Method | Final Acc (±std) | Avg Forgetting (±std) | BWT (±std)")
    print("-" * 75)
    
    for method in METHODS:
        if method not in summary:
            print(f"{method} | [NO DATA]")
            continue
        
        s = summary[method]
        final_str = f"{s['final_accuracy_mean']:.4f} (±{s['final_accuracy_std']:.4f})"
        forget_str = f"{s['forgetting_mean']:.4f} (±{s['forgetting_std']:.4f})"
        bwt_str = f"{s['bwt_mean']:.4f} (±{s['bwt_std']:.4f})"
        
        print(f"{method} | {final_str:20s} | {forget_str:22s} | {bwt_str}")
    
    print("\n" + "=" * 70)
    print("✅ AGGREGATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    aggregate_results()
