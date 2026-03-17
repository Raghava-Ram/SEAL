# 🔧 Multi-Task EWC Fix Implementation

## Changes Made

### 1. Trainer Initialization - Multi-Task EWC Storage

**File:** [seal/trainer.py:42-43](seal/trainer.py#L42-L43)

**Before:**
```python
self.ewc_fisher = {}  # accumulated Fisher information for encoder params
self.ewc_theta = {}   # snapshot of encoder parameters (theta*)
```

**After:**
```python
# Multi-task EWC: store per-task fisher and theta snapshots
# Each element: {'task_name': str, 'theta': dict, 'fisher': dict}
self.ewc_tasks = []  # list of {"task_name": str, "theta": {...}, "fisher": {...}}
```

**Why:** Prevents theta* overwriting. Each task's theta and Fisher are stored together in a list, maintaining independence.

---

### 2. Train-on-Batch - Multi-Task Penalty Computation

**File:** [seal/trainer.py:268-293](seal/trainer.py#L268-L293)

**Before:**
```python
if getattr(self, 'ewc_fisher', None) and len(self.ewc_fisher) > 0 and getattr(self, 'ewc_lambda', 0) > 0:
    penalty = None
    for name, param in self.model.named_parameters():
        if name in self.ewc_fisher:
            theta_star = self.ewc_theta.get(name)  # Single snapshot
            F_i = self.ewc_fisher.get(name)         # Accumulated fisher
            # ... compute penalty ...
```

**After:**
```python
if getattr(self, 'ewc_tasks', None) and len(self.ewc_tasks) > 0 and getattr(self, 'ewc_lambda', 0) > 0:
    penalty = None
    # Loop through all stored task snapshots
    for task_snapshot in self.ewc_tasks:
        fisher_dict = task_snapshot.get('fisher', {})
        theta_dict = task_snapshot.get('theta', {})
        task_name = task_snapshot.get('task_name', '?')
        
        for name, param in self.model.named_parameters():
            # Restrict penalty to encoder parameters present in this task's fisher
            if name not in fisher_dict:
                continue
            
            theta_star = theta_dict.get(name)  # Task-specific theta
            F_i = fisher_dict.get(name)        # Task-specific fisher
            # ... compute penalty ...
```

**Key Insights:**
- **Loops through ALL stored tasks** instead of using a single fisher/theta pair
- **Matches fisher with corresponding theta** from the same task
- **Properly penalizes drift** from each task's learned parameters
- **Prevents cross-task confusion** that was causing the collapse

---

### 3. Compute Fisher - Per-Task Storage

**File:** [seal/trainer.py:500-512](seal/trainer.py#L500-L512)

**Before:**
```python
# Merge into existing fisher (accumulate)
if getattr(self, 'ewc_fisher', None) and len(self.ewc_fisher) > 0:
    for name, v in fisher_accum.items():
        if name in self.ewc_fisher:
            self.ewc_fisher[name] = self.ewc_fisher[name] + v
        else:
            self.ewc_fisher[name] = v
else:
    self.ewc_fisher = fisher_accum

# Update theta*: keep the latest snapshot (could be enhanced to store per-task)
self.ewc_theta = theta_snapshot  # 🔴 BUG: OVERWRITES

print(f"EWC: Fisher information computed for task {task_name}")
```

**After:**
```python
# Store per-task fisher and theta snapshot (don't overwrite or accumulate globally)
task_snapshot = {
    'task_name': task_name,
    'fisher': copy.deepcopy(fisher_accum),
    'theta': copy.deepcopy(theta_snapshot)
}
self.ewc_tasks.append(task_snapshot)

print(f"EWC: Fisher information computed for task {task_name}")
print(f"EWC: Stored snapshot for task '{task_name}'. Total tasks stored: {len(self.ewc_tasks)}")
```

**Key Insights:**
- **Appends to list** instead of overwriting
- **Deep copies** ensure no aliasing issues
- **Debug print** shows task-by-task accumulation
- **No cross-task contamination** of snapshots

---

### 4. Config - Lambda Reduction

**File:** [configs/default.yaml:176-178](configs/default.yaml#L176-L178)

**Before:**
```yaml
ewc:
  enabled: true
  lambda: 1000  # Very aggressive
```

**After:**
```yaml
ewc:
  enabled: true
  lambda: 100   # Reduced 10x for better stability
```

**Why:** High lambda with misaligned penalty caused convergence failure. 100 is more reasonable for multi-task learning.

---

## Mathematical Model: Proper Multi-Task EWC

### What We Fixed

For K tasks and parameters θ, the penalty during task k training should be:

$$L_{total} = L_k(θ) + λ \sum_{t=1}^{k-1} \sum_{i} F_{t,i} \cdot (θ_i - θ_t^*)^2$$

Where:
- **t**: Previous task index (1 to k-1)
- **i**: Parameter index
- **F_{t,i}**: Fisher information for parameter i in task t
- **θ_t^***: Parameter snapshot after task t

### What Was Broken (Before)

$$L_{total} = L_k(θ) + λ \sum_{i} F_{accum,i} \cdot (θ_i - θ_{last}^*)^2$$

Where:
- **F_accum**: Fisher accumulated (mixed from all tasks)
- **θ_last^***: Snapshot from ONLY the last task

This mismatch created contradictory gradients.

### What Is Fixed (Now)

Each task's fisher and theta are stored independently:
```
self.ewc_tasks = [
    {"task_name": "imdb", "fisher": F₁, "theta": θ₁*},
    {"task_name": "squad", "fisher": F₂, "theta": θ₂*},
    {"task_name": "arc", "fisher": F₃, "theta": θ₃*}
]
```

During task 3 training, penalty = λ * (F₁·(θ-θ₁*)² + F₂·(θ-θ₂*)² + F₃·(θ-θ₃*)²)

Each term uses the **correct** fisher and theta from the same task.

---

## Expected Improvements

| Metric | Before | Expected After |
|--------|--------|-----------------|
| M6 Final Accuracy | 0.3333 | ~0.75-0.80 (close to M1) |
| M6 Forgetting | 0.5889 | ~0.02-0.05 (much better) |
| M6 BWT | -0.4967 | ~-0.10 (less negative) |
| Training stability | High penalty oscillations | Smooth convergence |
| Gradient flow | Conflicted, contradictory | Consistent per-task |

---

## Debugging Output Location

After each task finishes, check logs for:
```
EWC: Stored snapshot for task 'imdb'. Total tasks stored: 1
EWC: Stored snapshot for task 'squad'. Total tasks stored: 2
EWC: Stored snapshot for task 'arc'. Total tasks stored: 3
```

This confirms all three tasks have independent snapshots.

---

## Testing Next Steps

1. Run with new EWC implementation:
   ```bash
   python seal/runner.py --config configs/default.yaml --phase hybrid
   ```

2. Verify output metrics in `outputs/multi_task/hybrid/imdb_squad_arc_metrics.json`

3. Run aggregation:
   ```bash
   python aggregate_existing_results.py
   ```

4. Compare M6 accuracy to previous run:
   - Before: 0.3333
   - After: Should be significantly higher

---

## Files Modified

1. ✅ [seal/trainer.py](seal/trainer.py) - Multi-task EWC storage and penalty
2. ✅ [configs/default.yaml](configs/default.yaml) - Lambda reduced to 100

## Files NOT Modified

- seal/runner.py (replay and freezing logic untouched)
- aggregate_existing_results.py (metrics unchanged)
- Any plotting or utility code

---

