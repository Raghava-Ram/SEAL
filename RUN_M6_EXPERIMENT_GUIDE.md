# 🔬 M6 Experiment Runner - Usage Guide

## Overview

`run_m6_experiment.py` is a standalone script for running controlled M6 (Hybrid + Multi-Task EWC) experiments with configurable parameters.

**Key Features:**
- ✅ No modification to core trainer.py or runner.py
- ✅ No permanent changes to default.yaml
- ✅ Configurable lambda and seed via CLI
- ✅ Reproducible results with deterministic seeding
- ✅ Organized output directory structure
- ✅ Clear console feedback

---

## Installation & Setup

No additional installation needed. The script uses existing SEAL modules:
- `seal.runner.run_sequential_tasks`
- `seal.utils.set_global_seed`

**Activate conda environment:**
```bash
conda activate seal_env
cd /path/to/SEAL
```

---

## Usage Examples

### 1. Default Run (seed 42, lambda 100, replay 0.15)
```bash
python run_m6_experiment.py
```

### 2. Custom Lambda
```bash
python run_m6_experiment.py --lam 50
```

### 3. Custom Seed
```bash
python run_m6_experiment.py --seed 123
```

### 4. Custom Replay Fraction
```bash
python run_m6_experiment.py --replay_fraction 0.2
```

### 5. All Custom Parameters
```bash
python run_m6_experiment.py --seed 42 --lam 100 --replay_fraction 0.15
```

### 6. Multiple Runs (Different Lambdas)
```bash
python run_m6_experiment.py --lam 50
python run_m6_experiment.py --lam 100
python run_m6_experiment.py --lam 200
```

### 7. Multiple Runs (Different Seeds)
```bash
python run_m6_experiment.py --seed 42
python run_m6_experiment.py --seed 123
python run_m6_experiment.py --seed 999
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--lam` | float | 100 | EWC lambda parameter |
| `--replay_fraction` | float | 0.15 | Replay fraction for hybrid mode |
| `--config` | str | configs/default.yaml | Path to base config file |

---

## Output Structure

```
outputs/
└── experiments/
    ├── m6_lambda_50_seed_42/
    │   ├── experiment_config.yaml      # Config used for this run
    │   └── multi_task/
    │       └── hybrid/
    │           ├── imdb_squad_arc_metrics.json  # Main results
    │           └── task_results.json
    │
    ├── m6_lambda_100_seed_42/
    │   └── ...
    │
    └── m6_lambda_100_seed_123/
        └── ...
```

Each experiment has its own directory named: `m6_lambda_<lambda>_seed_<seed>`

---

## Console Output

The script provides clear feedback at each stage:

```
================================================================================
🔬 M6 EXPERIMENT RUNNER (Hybrid + Multi-Task EWC)
================================================================================
Seed: 42
EWC Lambda: 100
Replay Fraction: 0.15
Base Config: configs/default.yaml
================================================================================

📖 Loading base configuration...
✅ Config loaded

🔧 Overriding configuration for M6 (Hybrid mode)...
  ✓ phase: hybrid
  ✓ device: cpu
  ✓ ewc.enabled: True
  ✓ ewc.lambda: 100
  ✓ replay.fraction: 0.15

📁 Output directory: outputs/experiments/m6_lambda_100_seed_42
📄 Experiment config saved: outputs/experiments/m6_lambda_100_seed_42/experiment_config.yaml

🌱 Setting global seed to 42...
✅ Seed set

================================================================================
🚀 STARTING M6 TRAINING
================================================================================

🔍 Loading datasets...
[... training progress ...]

================================================================================
✅ M6 EXPERIMENT COMPLETED SUCCESSFULLY
================================================================================

📊 Results saved to:
   C:\Users\ragha\Desktop\SEAL\outputs\experiments\m6_lambda_100_seed_42\multi_task\hybrid\imdb_squad_arc_metrics.json

📈 Task results saved to:
   C:\Users\ragha\Desktop\SEAL\outputs\experiments\m6_lambda_100_seed_42\multi_task\hybrid\task_results.json
```

---

## Lambda Tuning Strategy

The original lambda (1000) caused collapse due to the theta* overwriting bug. With the multi-task EWC fix, test these values:

| Lambda | Expected Effect | Use Case |
|--------|-----------------|----------|
| 1 - 10 | Minimal regularization | Baseline test |
| 50 | Light protection | Trade-off: learning vs stability |
| 100 | Moderate protection | **Recommended starting point** |
| 200 | Strong protection | Heavy catastrophic forgetting scenario |
| 500+ | Very strong (risk of underfitting) | Rare cases |

### Lambda Experiment Protocol

```bash
# Test different lambda values with same seed
for lam in 10 50 100 200; do
    python run_m6_experiment.py --seed 42 --lam $lam
done
```

Then compare results:
```bash
python aggregate_existing_results.py  # Includes experiments/ in scan
```

---

## Multi-Seed Experimentation

To test robustness across seeds with a fixed lambda:

```bash
# Lambda 100, seeds 42, 123, 999
for seed in 42 123 999; do
    python run_m6_experiment.py --seed $seed --lam 100
done
```

Results will be in separate directories:
- `outputs/experiments/m6_lambda_100_seed_42/`
- `outputs/experiments/m6_lambda_100_seed_123/`
- `outputs/experiments/m6_lambda_100_seed_999/`

---

## Configuration Overrides

The script performs these overrides on the loaded config:

1. **Phase:** `hybrid` (enables layer freezing after task 1)
2. **Device:** `cpu` (forced for reproducibility)
3. **EWC Enabled:** `True` (uses multi-task EWC implementation)
4. **EWC Lambda:** Configurable via `--lam` (default: 100)
5. **Replay Fraction:** Configurable via `--replay_fraction` (default: 0.15)

**NOT Modified:**
- Replay policy (remains 'priority' from base config)
- Learning rate
- Task datasets and limits
- Batch size (other than replay config)
- Any other trainer settings

---

## Debugging

If experiment fails, check:

1. **Config file exists:**
   ```bash
   test -f configs/default.yaml && echo "✅ Found"
   ```

2. **Output directory created:**
   ```bash
   ls -la outputs/experiments/
   ```

3. **Experiment config saved:**
   ```bash
   cat outputs/experiments/m6_lambda_100_seed_42/experiment_config.yaml
   ```

4. **Error messages in console output** - The script prints full traceback on failure

---

## Integration with Aggregation Script

Results from `run_m6_experiment.py` can be automatically included in the aggregation:

```bash
# Run multiple experiments
python run_m6_experiment.py --seed 42 --lam 100
python run_m6_experiment.py --seed 123 --lam 100
python run_m6_experiment.py --seed 999 --lam 100

# Run aggregation (automatically discovers experiments/ directory)
python aggregate_existing_results.py
```

The aggregation script will include M6 results from both:
- `outputs/multiseed/seed_<seed>/M6/...`
- `outputs/experiments/m6_lambda_<lam>_seed_<seed>/...`

---

## Expected Results (with Multi-Task EWC Fix)

| Metric | M1 (Baseline) | M6 Before Fix | M6 After Fix | Expected |
|--------|---|---|---|---|
| Final Accuracy | 0.8667 | 0.3333 | ? | ~0.75-0.80 |
| Forgetting | 0.0078 | 0.5889 | ? | ~0.02-0.05 |
| BWT | 0.0817 | -0.4967 | ? | ~-0.10 |

The multi-task EWC fix should bring M6 performance close to M1, demonstrating proper continual learning with principled regularization.

---

## Notes

- Each run is **independent** - Results don't affect other experiments
- **No modification** to default.yaml or trainer.py
- **Fully reproducible** - Same seed + lambda = same results
- **Easy to extend** - Add more CLI args as needed (e.g., `--learning_rate`)
- **Compatible** with existing aggregation pipeline

---

