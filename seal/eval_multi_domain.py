"""
Evaluation metrics for multi-domain continual learning.

This module provides functions to compute and visualize metrics like
forgetting, backward transfer, and task retention.
"""
from typing import Dict, List, Optional, Any
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_task_accuracy_matrix(results: Dict[str, List[Optional[float]]]) -> Dict[str, List[float]]:
    """
    Compute the accuracy matrix from task results.

    Args:
        results: Dictionary mapping task names to lists of accuracies

    Returns:
        Dictionary with the same structure as input, but with None values replaced with 0.0
    """
    matrix = {}
    for task, accs in results.items():
        matrix[task] = [0.0 if acc is None else acc for acc in accs]
    return matrix

def compute_forgetting(accuracy_matrix: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute forgetting metric for each task.

    Forgetting is defined as the maximum accuracy achieved during training
    minus the final accuracy after all tasks.

    Args:
        accuracy_matrix: Dictionary mapping task names to lists of accuracies

    Returns:
        Dictionary mapping task names to their forgetting scores
    """
    forgetting = {}
    
    for task, accs in accuracy_matrix.items():
        # Filter out None values and ensure we have at least 2 valid accuracies
        valid_accs = [acc for acc in accs if acc is not None]
        if len(valid_accs) <= 1:
            forgetting[task] = 0.0
            continue
            
        max_acc = max(valid_accs[:-1])
        final_acc = valid_accs[-1]
        forgetting[task] = max(0.0, max_acc - final_acc)
    
    return forgetting

def compute_backward_transfer(accuracy_matrix: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute backward transfer for each task.

    Backward transfer measures how learning new tasks affects performance on previous tasks.

    Args:
        accuracy_matrix: Dictionary mapping task names to lists of accuracies

    Returns:
        Dictionary mapping task names to their backward transfer scores
    """
    bwt = {}
    tasks = list(accuracy_matrix.keys())
    num_tasks = len(tasks)
    
    for i, task in enumerate(tasks[:-1]):  # Skip the last task
        accs = [acc for acc in accuracy_matrix[task] if acc is not None]
        if not accs or len(accs) <= i:
            bwt[task] = 0.0
            continue
            
        initial_acc = accs[i]  # Accuracy right after learning the task
        final_acc = accs[-1]   # Final accuracy after all tasks
        
        # Backward transfer: positive is good (improvement), negative is bad (forgetting)
        bwt[task] = final_acc - initial_acc
    
    return bwt

def compute_average_metrics(accuracy_matrix: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute average metrics across all tasks.

    Args:
        accuracy_matrix: Dictionary mapping task names to lists of accuracies

    Returns:
        Dictionary with average metrics
    """
    forgetting = compute_forgetting(accuracy_matrix)
    bwt = compute_backward_transfer(accuracy_matrix)
    
    # Get final accuracies, skipping None values
    final_accs = [accs[-1] for accs in accuracy_matrix.values() 
                 if accs and accs[-1] is not None]
    
    return {
        "average_accuracy": np.mean(final_accs) if final_accs else 0.0,
        "average_forgetting": np.mean(list(forgetting.values())) if forgetting else 0.0,
        "average_backward_transfer": np.mean(list(bwt.values())) if bwt else 0.0
    }

def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """
    Save metrics to a JSON file.

    Args:
        metrics: Dictionary containing metrics to save
        path: Path to save the metrics file
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    try:
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Saved metrics to {path}")
    except Exception as e:
        print(f"⚠️  Error saving metrics: {str(e)}")

def plot_retention_curves(accuracy_matrix: Dict[str, List[float]], out_path: str) -> None:
    """
    Plot retention curves for each task.

    Args:
        accuracy_matrix: Dictionary mapping task names to lists of accuracies
        out_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for task, accs in accuracy_matrix.items():
        x = range(1, len(accs) + 1)
        plt.plot(x, accs, 'o-', label=task)
    
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Task Retention Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    try:
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"⚠️  Error generating retention curves: {str(e)}")

def plot_forgetting_heatmap(accuracy_matrix: Dict[str, List[float]], out_path: str) -> None:
    """
    Plot a heatmap of task accuracies over time.

    Args:
        accuracy_matrix: Dictionary mapping task names to lists of accuracies
        out_path: Path to save the plot
    """
    tasks = list(accuracy_matrix.keys())
    num_tasks = len(tasks)
    
    # Create a matrix for the heatmap
    data = np.zeros((num_tasks, num_tasks))
    
    for i, task in enumerate(tasks):
        accs = accuracy_matrix[task]
        # Pad with zeros if needed
        padded_accs = accs + [0.0] * (num_tasks - len(accs))
        data[i, :] = padded_accs[:num_tasks]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=[f"Task {i+1}" for i in range(num_tasks)],
        yticklabels=tasks
    )
    
    plt.title("Task Accuracy Heatmap")
    plt.xlabel("After Task")
    plt.ylabel("Task")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    try:
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"⚠️  Error generating forgetting heatmap: {str(e)}")

def generate_evaluation_report(accuracy_matrix, output_dir, prefix=""):
    """
    Generate a comprehensive evaluation report with metrics and visualizations.

    Args:
        accuracy_matrix (dict): Dictionary mapping task names to lists of accuracies
        output_dir (str): Directory to save the report and plots
        prefix (str, optional): Prefix for output filenames. Defaults to "".

    Returns:
        dict: Dictionary containing all computed metrics
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize report with basic structure
        report = {
            "accuracy_matrix": accuracy_matrix,
            "forgetting": {},
            "backward_transfer": 0.0,
            "average_metrics": {}
        }
        
        # Only proceed if we have accuracy data
        if not accuracy_matrix:
            print("⚠️  No accuracy data provided for evaluation report")
            return report

        # Compute metrics with error handling
        try:
            report["forgetting"] = compute_forgetting(accuracy_matrix)
        except Exception as e:
            print(f"⚠️  Error computing forgetting metrics: {str(e)}")
            
        try:
            report["backward_transfer"] = compute_backward_transfer(accuracy_matrix)
        except Exception as e:
            print(f"⚠️  Error computing backward transfer: {str(e)}")
            
        try:
            report["average_metrics"] = compute_average_metrics(accuracy_matrix)
        except Exception as e:
            print(f"⚠️  Error computing average metrics: {str(e)}")

        # Save metrics with error handling
        try:
            metrics_path = os.path.join(output_dir, f"{prefix}metrics.json" if prefix else "metrics.json")
            save_metrics(report, metrics_path)
        except Exception as e:
            print(f"⚠️  Error saving metrics: {str(e)}")

        # Generate plots with error handling
        try:
            plot_retention_curves(
                accuracy_matrix,
                os.path.join(output_dir, f"{prefix}retention_curves.png" if prefix else "retention_curves.png")
            )
        except Exception as e:
            print(f"⚠️  Error generating retention curves: {str(e)}")

        try:
            plot_forgetting_heatmap(
                accuracy_matrix,
                os.path.join(output_dir, f"{prefix}forgetting_heatmap.png" if prefix else "forgetting_heatmap.png")
            )
        except Exception as e:
            print(f"⚠️  Error generating forgetting heatmap: {str(e)}")

        return report

    except Exception as e:
        print(f"❌ Error in generate_evaluation_report: {str(e)}")
        # Return a minimal report with the error
        return {
            "error": str(e),
            "accuracy_matrix": accuracy_matrix or {}
        }
