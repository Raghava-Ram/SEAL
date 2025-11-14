"""
Replay sampler for SEAL.
"""

from typing import List, Dict, Any
from .memory import EditCache

def mix_batches(new_examples: List[Dict[str, Any]],
               memory: EditCache,
               batch_size: int = 8,
               replay_fraction: float = 0.5,
               policy: str = "priority") -> List[Dict[str, Any]]:
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
