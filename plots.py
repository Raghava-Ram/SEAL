import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def save_heatmap(matrix, out_path, title="Mean Accuracy Matrix", cmap="RdYlGn"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = np.array(matrix, dtype=float)
    T = arr.shape[0]
    plt.figure(figsize=(6, max(3, T * 0.8)))
    sns.heatmap(arr, annot=True, fmt=".2f", cmap=cmap, vmin=0.0, vmax=1.0, cbar_kws={"label": "Accuracy"})
    plt.title(title)
    plt.xlabel("After Task")
    plt.ylabel("Task")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    # also save PDF
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_forgetting_curve(task_acc_hist, out_path, title="Forgetting Curve"):
    """task_acc_hist: dict task_name -> list of accuracies across time (len = T)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    for task, accs in task_acc_hist.items():
        plt.plot(range(1, len(accs) + 1), accs, marker='o', label=task)
    plt.xlabel("Task Index")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_comparison_metrics(metrics_dict, out_path, title="Comparison Metrics"):
    """metrics_dict: {method: {metric_name: (mean, std), ...}, ...}

    Creates grouped bar plot for Final Acc, Avg Forgetting, BWT.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    methods = sorted(metrics_dict.keys())
    metric_names = ["Final Acc", "Avg Forgetting", "BWT"]
    # Prepare arrays
    means = {m: [metrics_dict[m].get('final_accuracy', (0.0, 0.0))[0],
                 metrics_dict[m].get('avg_forgetting', (0.0, 0.0))[0],
                 metrics_dict[m].get('bwt', (0.0, 0.0))[0]] for m in methods}
    stds = {m: [metrics_dict[m].get('final_accuracy', (0.0, 0.0))[1],
                metrics_dict[m].get('avg_forgetting', (0.0, 0.0))[1],
                metrics_dict[m].get('bwt', (0.0, 0.0))[1]] for m in methods}

    x = np.arange(len(methods))
    width = 0.2

    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metric_names):
        vals = [means[m][i] for m in methods]
        errs = [stds[m][i] for m in methods]
        plt.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=5, label=metric)

    plt.xticks(x, methods)
    plt.ylim(-1.0, 1.0 if any(mn > 1.0 for m in methods for mn in [means[m][0]]) else 1.0)
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()
