"""Orchestration script to run multi-seed continual learning experiments.

This script runs `run_sequential_tasks` from `seal.runner` for multiple
seeds and methods, aggregates metrics, computes mean/std and produces
publication-ready plots. It forces CPU execution and calls global
seeding before each run.
"""
import os
import shutil
import yaml
import json
import numpy as np
import copy
import statistics
from pathlib import Path

from seal.utils import set_global_seed
from seal import metrics as seal_metrics
import plots


SEEDS = [42, 123, 999]
METHODS = ["M0", "M1", "M6"]

# Mapping methods -> runner phase and subfolder name
METHOD_MAP = {
    "M0": {"phase": "baseline", "folder": "baseline"},
    "M1": {"phase": "seal", "folder": "seal"},
    "M6": {"phase": "hybrid", "folder": "hybrid", "ewc_enabled": True}
}


def _write_config(base_cfg, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.safe_dump(base_cfg, f)


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def run():
    root = Path.cwd()
    base_config_path = root / 'configs' / 'default.yaml'
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config not found: {base_config_path}")

    with open(base_config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # Force CPU for all runs
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    aggregated = {m: [] for m in METHODS}
    accuracy_matrices_per_method = {m: [] for m in METHODS}

    from seal.runner import run_sequential_tasks

    for seed in SEEDS:
        for method in METHODS:
            mapping = METHOD_MAP[method]
            # Prepare per-run config
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg['phase'] = mapping['phase']
            # EWC only for M6 when requested
            if mapping.get('ewc_enabled'):
                if 'ewc' not in run_cfg:
                    run_cfg['ewc'] = {}
                run_cfg['ewc']['enabled'] = True
            else:
                if 'ewc' in run_cfg:
                    run_cfg['ewc']['enabled'] = False

            save_dir = str(root / 'outputs' / 'multiseed' / f'seed_{seed}' / method)
            run_cfg['save_dir'] = save_dir

            # Ensure output dir is clean for this run
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)

            os.makedirs(save_dir, exist_ok=True)

            # Write config for this run
            run_config_path = Path(save_dir) / 'config.yaml'
            _write_config(run_cfg, str(run_config_path))

            # Set global seed and force deterministic behavior
            set_global_seed(seed)

            print(f"\n=== Running seed={seed} method={method} -> phase={mapping['phase']} ===\n")

            # Run the experiment
            try:
                run_sequential_tasks(config_path=str(run_config_path))
            except Exception as e:
                print(f"⚠️  Run failed for seed={seed} method={method}: {e}")
                continue

            # Locate generated metrics file (imdb_squad_arc_metrics.json)
            # Runner writes to: <save_dir>/multi_task/<folder>/<prefix>metrics.json
            method_folder = mapping['folder']
            metrics_path = Path(save_dir) / 'multi_task' / method_folder / f"imdb_squad_arc_metrics.json"
            if not metrics_path.exists():
                # Try fallback: metrics.json
                alt = Path(save_dir) / 'multi_task' / method_folder / 'metrics.json'
                if alt.exists():
                    metrics_path = alt
                else:
                    print(f"⚠️  Metrics not found for seed={seed} method={method} at expected locations")
                    continue

            metrics = _load_json(str(metrics_path))
            # metrics expected to contain "accuracy_matrix"
            acc_matrix = metrics.get('accuracy_matrix') or metrics.get('accuracy_matrix')
            if not acc_matrix:
                # try task_metrics saved earlier
                alt_metrics = Path(save_dir) / 'multi_task' / method_folder / 'task_results.json'
                if alt_metrics.exists():
                    obj = _load_json(str(alt_metrics))
                    # best-effort: try to extract matrix-like structure
                    acc_matrix = obj.get('accuracy_matrix') or obj
                else:
                    print(f"⚠️  No accuracy_matrix found for seed={seed} method={method}")
                    continue

            # Ensure matrix is in list-of-lists form. Convert from dict if needed.
            if isinstance(acc_matrix, dict):
                # Keep task order consistent
                tasks = list(acc_matrix.keys())
                matrix = [acc_matrix[t] for t in tasks]
            else:
                matrix = acc_matrix

            # Basic validation (square and no NaN)
            try:
                seal_metrics._validate_matrix(matrix)
            except Exception as e:
                print(f"⚠️  Validation failed for seed={seed} method={method}: {e}")
                continue

            accuracy_matrices_per_method[method].append({'matrix': matrix, 'tasks': tasks})

            # Save per-run artifacts
            out_dir = Path(save_dir) / 'artifacts'
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / 'accuracy_matrix.json').write_text(json.dumps({'tasks': tasks, 'matrix': matrix}, indent=2))

    # Aggregate across seeds
    summary = {}
    for method in METHODS:
        runs = accuracy_matrices_per_method[method]
        if not runs:
            print(f"⚠️  No successful runs for method {method}")
            continue

        # Check task order consistency
        task_orders = [r['tasks'] for r in runs]
        first_order = task_orders[0]
        for ord_ in task_orders[1:]:
            if ord_ != first_order:
                raise ValueError(f"Task order mismatch across seeds for method {method}")

        # Stack matrices into numpy arrays (shape: Nseeds x T x T)
        mats = [np.array(r['matrix'], dtype=float) for r in runs]
        stacked = np.stack(mats, axis=0)
        mean_mat = stacked.mean(axis=0).tolist()
        std_mat = stacked.std(axis=0).tolist()

        # Save mean and std matrices
        method_out = Path('outputs') / 'multiseed' / f'mean_matrices'
        method_out.mkdir(parents=True, exist_ok=True)
        mean_path = method_out / f'mean_accuracy_matrix_{method}.json'
        std_path = method_out / f'std_accuracy_matrix_{method}.json'
        mean_path.write_text(json.dumps({'tasks': first_order, 'mean_matrix': mean_mat}, indent=2))
        std_path.write_text(json.dumps({'tasks': first_order, 'std_matrix': std_mat}, indent=2))

        # Compute per-seed scalar metrics
        final_accs = []
        avg_forgettings = []
        bwts = []
        for m in mats:
            # m is T x T, final row is m[-1]
            final_acc = float(np.mean(m[-1]))
            final_accs.append(final_acc)
            # compute forgetting per seed using provided functions
            avg_f = float(seal_metrics.average_forgetting(m.tolist()))
            avg_forgettings.append(avg_f)
            bwt = float(seal_metrics.backward_transfer(m.tolist()))
            bwts.append(bwt)

        summary[method] = {
            'final_accuracy_mean': statistics.mean(final_accs),
            'final_accuracy_std': statistics.pstdev(final_accs) if len(final_accs) > 1 else 0.0,
            'avg_forgetting_mean': statistics.mean(avg_forgettings),
            'avg_forgetting_std': statistics.pstdev(avg_forgettings) if len(avg_forgettings) > 1 else 0.0,
            'bwt_mean': statistics.mean(bwts),
            'bwt_std': statistics.pstdev(bwts) if len(bwts) > 1 else 0.0
        }

    # Save summary
    out_summary = Path('outputs') / 'multiseed' / 'results_summary.json'
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))

    # Print publication-ready table
    print("\nMethod | Final Acc (±std) | Avg Forgetting (±std) | BWT (±std)")
    for method in METHODS:
        if method not in summary:
            continue
        s = summary[method]
        print(f"{method} | {s['final_accuracy_mean']:.4f} (±{s['final_accuracy_std']:.4f}) | {s['avg_forgetting_mean']:.4f} (±{s['avg_forgetting_std']:.4f}) | {s['bwt_mean']:.4f} (±{s['bwt_std']:.4f})")

    # Create plots for each method
    for method in METHODS:
        runs = accuracy_matrices_per_method.get(method, [])
        if not runs:
            continue
        tasks = runs[0]['tasks']
        mats = [np.array(r['matrix'], dtype=float) for r in runs]
        stacked = np.stack(mats, axis=0)
        mean_mat = stacked.mean(axis=0)
        # Heatmap
        plots.save_heatmap(mean_mat.tolist(), f"outputs/multiseed/mean_heatmap_{method}.png", title=f"Mean Accuracy Matrix – {method} ({len(mats)} seeds)")
        # Forgetting curve: task-wise accuracy across time (use mean rows)
        task_acc_hist = {tasks[i]: mean_mat[i, :].tolist() for i in range(mean_mat.shape[0])}
        plots.save_forgetting_curve(task_acc_hist, f"outputs/multiseed/forgetting_curve_{method}.png", title=f"Forgetting Curve – {method}")

    # Comparison bar plot
    metrics_for_plot = {}
    for method in summary:
        metrics_for_plot[method] = {
            'final_accuracy': (summary[method]['final_accuracy_mean'], summary[method]['final_accuracy_std']),
            'avg_forgetting': (summary[method]['avg_forgetting_mean'], summary[method]['avg_forgetting_std']),
            'bwt': (summary[method]['bwt_mean'], summary[method]['bwt_std'])
        }
    plots.save_comparison_metrics(metrics_for_plot, 'outputs/multiseed/comparison_metrics.png', title='Comparison Metrics (mean ± std)')


if __name__ == '__main__':
    run()
