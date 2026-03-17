#!/usr/bin/env python3
"""
Standalone experiment runner for M6 (Hybrid + Multi-Task EWC).

This script runs a single M6 experiment with configurable lambda and seed
for controlled testing of the multi-task EWC fix.

Does NOT modify core trainer.py, runner.py, or default.yaml.
Loads config, overrides parameters programmatically, and runs the experiment.

Usage:
    python run_m6_experiment.py --seed 42 --lam 100 --replay_fraction 0.15
    python run_m6_experiment.py --seed 123 --lam 50
    python run_m6_experiment.py --lam 100  # uses default seed 42
"""

import os
import sys
import yaml
import argparse
import copy
from pathlib import Path

# Import from SEAL
from seal.runner import run_sequential_tasks
from seal.utils import set_global_seed


def main():
    parser = argparse.ArgumentParser(
        description="Run M6 (Hybrid + Multi-Task EWC) experiment with configurable parameters."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic execution (default: 42)"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=100,
        dest='ewc_lambda',
        help="EWC lambda parameter (default: 100)"
    )
    parser.add_argument(
        "--replay_fraction",
        type=float,
        default=0.15,
        help="Replay fraction for hybrid mode (default: 0.15)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base config file (default: configs/default.yaml)"
    )
    
    args = parser.parse_args()
    
    seed = args.seed
    ewc_lambda = args.ewc_lambda
    replay_fraction = args.replay_fraction
    config_path = args.config
    
    # Print experiment header
    print("\n" + "=" * 80)
    print("🔬 M6 EXPERIMENT RUNNER (Hybrid + Multi-Task EWC)")
    print("=" * 80)
    print(f"Seed: {seed}")
    print(f"EWC Lambda: {ewc_lambda}")
    print(f"Replay Fraction: {replay_fraction}")
    print(f"Base Config: {config_path}")
    print("=" * 80 + "\n")
    
    # Verify config exists
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load base config
    print("📖 Loading base configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    print("✅ Config loaded")
    
    # Deep copy to avoid modifying original
    config = copy.deepcopy(config)
    
    # Override parameters for M6
    print("\n🔧 Overriding configuration for M6 (Hybrid mode)...")
    config['phase'] = 'hybrid'
    config['device'] = 'cpu'
    
    # Ensure ewc section exists
    if 'ewc' not in config:
        config['ewc'] = {}
    
    config['ewc']['enabled'] = True
    config['ewc']['lambda'] = ewc_lambda
    
    # Override replay fraction if specified
    if 'replay' not in config:
        config['replay'] = {}
    config['replay']['replay_fraction'] = replay_fraction
    
    print(f"  ✓ phase: {config['phase']}")
    print(f"  ✓ device: {config['device']}")
    print(f"  ✓ ewc.enabled: {config['ewc']['enabled']}")
    print(f"  ✓ ewc.lambda: {config['ewc']['lambda']}")
    print(f"  ✓ replay.replay_fraction: {config['replay']['replay_fraction']}")
    
    # Set output directory
    exp_dir = f"outputs/experiments/m6_lambda_{int(ewc_lambda)}_seed_{seed}"
    config['save_dir'] = exp_dir
    
    print(f"\n📁 Output directory: {exp_dir}")
    
    # Create output directory
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment config for reference
    experiment_config_path = Path(exp_dir) / 'experiment_config.yaml'
    with open(experiment_config_path, 'w') as f:
        yaml.safe_dump(config, f)
    print(f"📄 Experiment config saved: {experiment_config_path}")
    
    # Force CPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set global seed
    print(f"\n🌱 Setting global seed to {seed}...")
    set_global_seed(seed)
    print("✅ Seed set")
    
    # Print effective replay_fraction before training
    effective_replay = config['replay'].get('replay_fraction', 0.15)
    print(f"\n💾 Effective replay_fraction: {effective_replay}")
    
    # Run experiment
    print("\n" + "=" * 80)
    print("🚀 STARTING M6 TRAINING")
    print("=" * 80 + "\n")
    
    try:
        # Run the sequential tasks
        run_sequential_tasks(config_path=str(experiment_config_path))
        
        # Print completion message
        print("\n" + "=" * 80)
        print("✅ M6 EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Find and print results path
        metrics_path = Path(exp_dir) / 'multi_task' / 'hybrid' / 'imdb_squad_arc_metrics.json'
        if metrics_path.exists():
            print(f"\n📊 Results saved to:")
            print(f"   {metrics_path.absolute()}")
            
            # Also print task results
            task_results_path = Path(exp_dir) / 'multi_task' / 'hybrid' / 'task_results.json'
            if task_results_path.exists():
                print(f"\n📈 Task results saved to:")
                print(f"   {task_results_path.absolute()}")
        else:
            print(f"\n⚠️  Metrics file not found at expected location:")
            print(f"   {metrics_path}")
            print(f"\n   Check the experiment output directory:")
            print(f"   {Path(exp_dir).absolute()}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ M6 EXPERIMENT FAILED")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
