"""
Replay sampler for SEAL.
"""

from typing import List, Dict, Any
from .memory import EditCache

def mix_batches(new_examples: List[Dict[str, Any]],
               memory: EditCache,
               batch_size: int = 8,
               replay_fraction: float = 0.5,
               policy: str = "priority",
               hybrid: bool = False) -> List[Dict[str, Any]]:
    """
    Mix new examples with replayed examples from memory.
    
    Args:
        new_examples: List of new examples to include in the batch
        memory: EditCache instance containing past edits
        batch_size: Total size of the batch to return
        replay_fraction: Fraction of batch to fill with replayed examples
        policy: Sampling policy ('priority' or 'uniform')
        
    Returns:
        Combined batch of new and replayed examples
    """
    replay_size = int(batch_size * replay_fraction)
    new_size = batch_size - replay_size

    # Take subset of new examples
    new_batch = new_examples[:new_size]

    # Sample from memory if needed
    if replay_size > 0:
        # PHASE 3 EXTENSION: Task-aware replay weighting (only when hybrid=True)
        if hybrid and policy == "priority":
            # Read all edits and compute replay weights = utility * task_age_weight
            edits_all = memory._read_all()
            if len(edits_all) <= replay_size:
                replay_raw = edits_all.copy()
            else:
                # Define task-age weights
                task_weights = {
                    "imdb": 1.5,
                    "squad": 1.2,
                    "arc": 1.0
                }
                # Compute weights
                weights = []
                eps = 1e-6
                for e in edits_all:
                    util = max(float(e.get("utility", 0.0)), 0.0)
                    task = str(e.get("task", "")).lower()
                    taw = task_weights.get(task, 1.0)
                    weights.append(util * taw + eps)

                total = sum(weights)
                if total <= 0:
                    # Fallback to memory.sample when weights are degenerate
                    replay_raw = memory.sample(batch_size=replay_size, policy=policy)
                else:
                    # Weighted sampling without replacement approximation: use random.choices then dedupe
                    import random as _random
                    selected = []
                    indices = list(range(len(edits_all)))
                    w = weights.copy()
                    # Iteratively sample without replacement proportional to weight
                    for _ in range(replay_size):
                        total_w = sum(w)
                        if total_w <= 0:
                            break
                        probs = [x / total_w for x in w]
                        idx = _random.choices(indices, weights=probs, k=1)[0]
                        selected.append(edits_all[idx])
                        # Zero out selected weight to avoid reselection
                        w[idx] = 0.0
                    replay_raw = selected
        else:
            replay_raw = memory.sample(batch_size=replay_size, policy=policy)

        replay_batch = []
        for r in replay_raw:
            label = r.get("label")
            # Ensure label is either 0 or 1, default to 0 if invalid
            if label not in [0, 1]:
                label = 0
            replay_batch.append({
                "text": r.get("edit", r.get("original", "")),
                "label": label
            })
    else:
        replay_batch = []

    return new_batch + replay_batch
