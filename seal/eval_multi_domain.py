""
Evaluation metrics for multi-domain continual learning.

This module provides functions to compute and visualize metrics like
forgetting, backward transfer, and task retention.
"""
from typing import Dict, List, Optional, Tuple
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
    num_tasks = len(accuracy_matrix)
    
    for task, accs in accuracy_matrix.items():
        if len(accs) <= 1:
            forgetting[task] = 0.0
            continue
            
        max_acc = max(accs[:-1])  # Maximum accuracy before the last task
        final_acc = accs[-1]      # Accuracy after all tasks
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
        # Initial accuracy on this task
        initial_acc = accuracy_matrix[task][i]  # Accuracy right after learning the task
        
        # Final accuracy after all tasks
        final_acc = accuracy_matrix[task][-1]
        
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
    
    # Final accuracies (last element of each task's accuracy list)
    final_accs = [accs[-1] for accs in accuracy_matrix.values()]
    
    return {
        "average_accuracy": np.mean(final_accs),
        "average_forgetting": np.mean(list(forgetting.values())),
        "average_backward_transfer": np.mean(list(bwt.values()))
    }

def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing metrics to save
        path: Path to save the metrics file
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

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
    plt.savefig(out_path)
    plt.close()

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
    plt.savefig(out_path)
    plt.close()

def generate_evaluation_report(
    accuracy_matrix: Dict[str, List[float]],
    output_dir: str,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        accuracy_matrix: Dictionary mapping task names to lists of accuracies
        output_dir: Directory to save the report and plots
        prefix: Optional prefix for output filenames
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics
    forgetting = compute_forgetting(accuracy_matrix)
    bwt = compute_backward_transfer(accuracy_matrix)
    avg_metrics = compute_average_metrics(accuracy_matrix)
    
    # Prepare report
    report = {
        "accuracy_matrix": accuracy_matrix,
        "forgetting": forgetting,
        "backward_transfer": bwt,
        "average_metrics": avg_metrics
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{prefix}metrics.json" if prefix else "metrics.json")
    save_metrics(report, metrics_path)
    
    # Generate plots
    plot_retention_curves(
        accuracy_matrix,
        os.path.join(output_dir, f"{prefix}retention_curves.png" if prefix else "retention_curves.png")
    )
    
    plot_forgetting_heatmap(
        accuracy_matrix,
        os.path.join(output_dir, f"{prefix}forgetting_heatmap.png" if prefix else "forgetting_heatmap.png")
    )
    
    return report
