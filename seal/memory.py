"""
EditCache: JSONL-backed edit memory for SEAL with task support.
"""

import json
import os
import uuid
import random
import time
from random import choices
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

DEFAULT_MAX_SIZE = 20000

class EditCache:
    def __init__(self, path: str = "memory/edits.jsonl", max_size: int = DEFAULT_MAX_SIZE):
        self.path = path
        self.max_size = max_size
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "w", encoding="utf-8").close()

    def _read_all(self) -> List[Dict[str, Any]]:
        edits = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        edits.append(json.loads(line))
                    except Exception:
                        continue
        return edits

    def _write_all(self, edits: List[Dict[str, Any]]):
        with open(self.path, "w", encoding="utf-8") as f:
            for e in edits:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def add_edit(self, edit: Dict[str, Any]) -> str:
        # Ensure consistent structure with required fields
        # Normalize label: coerce to int when possible and ensure non-negative
        raw_label = edit.get("label", 0)
        try:
            label_int = int(float(raw_label))
        except Exception:
            label_int = 0
        if label_int < 0:
            label_int = 0

        e = {
            "id": str(uuid.uuid4()),
            "text": edit.get("text", ""),
            "label": label_int,
            "task": edit.get("task", "unknown"),
            "utility": float(edit.get("utility", 0.0)),
            "original_text": edit.get("original_text", ""),
            "original_label": edit.get("original_label", 0),
            "timestamp": time.time()
        }
        # Add any additional fields
        for k, v in edit.items():
            if k not in e and k != "id":  # Don't override existing fields
                e[k] = v
                
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._enforce_size()
        return e["id"]

    def _enforce_size(self):
        edits = self._read_all()
        if len(edits) <= self.max_size:
            return
        
        # Group edits by task
        task_edits = defaultdict(list)
        for edit in edits:
            task = edit.get('task', 'default')
            task_edits[task].append(edit)
        
        # Calculate how many to keep per task
        num_tasks = max(1, len(task_edits))
        max_per_task = max(1, self.max_size // num_tasks)
        
        # Keep top utility edits per task
        kept_edits = []
        for task, task_edits_list in task_edits.items():
            # Sort by utility (descending)
            task_edits_list.sort(key=lambda x: x.get("utility", 0), reverse=True)
            kept_edits.extend(task_edits_list[:max_per_task])
        
        # If we still have too many, trim the rest by utility
        if len(kept_edits) > self.max_size:
            kept_edits.sort(key=lambda x: x.get("utility", 0), reverse=True)
            kept_edits = kept_edits[:self.max_size]
        
        self._write_all(kept_edits)

    def sample(self, batch_size: int, policy: str = "priority", alpha: float = 1.0, 
              eps: float = 1e-6, task_balance: bool = True) -> List[Dict[str, Any]]:
        """
        Sample edits from memory with optional task balancing.
        
        Args:
            batch_size: Number of samples to return
            policy: Sampling policy ('priority' or 'uniform')
            alpha: Temperature parameter for priority sampling (higher = more uniform)
            eps: Small constant for numerical stability
            task_balance: If True, ensures samples are balanced across tasks
            
        Returns:
            List of sampled edits
        """
        edits = self._read_all()
        n = len(edits)
        if n == 0:
            return []
        if batch_size >= n:
            return edits.copy()
            
        # If task balancing is enabled and we have multiple tasks
        if task_balance and n > 0:
            task_edits = defaultdict(list)
            for edit in edits:
                task = edit.get('task', 'default')
                task_edits[task].append(edit)
                
            if len(task_edits) > 1:  # Only balance if we have multiple tasks
                # Calculate samples per task
                num_tasks = len(task_edits)
                per_task = max(1, batch_size // num_tasks)
                remainder = batch_size % num_tasks
                
                samples = []
                for i, (task, task_edits_list) in enumerate(task_edits.items()):
                    # Distribute remainder
                    count = per_task + (1 if i < remainder else 0)
                    if count > len(task_edits_list):
                        count = len(task_edits_list)
                    
                    if count > 0:
                        if policy == "uniform":
                            samples.extend(random.sample(task_edits_list, count))
                        else:
                            # Apply priority sampling within task
                            utils = [max(e.get("utility", 0.0), 0.0) + eps for e in task_edits_list]
                            weights = [(u ** alpha) for u in utils]
                            total = sum(weights)
                            if total > 0:
                                probs = [w / total for w in weights]
                                samples.extend(choices(task_edits_list, weights=probs, k=count))
                            else:
                                samples.extend(random.sample(task_edits_list, min(count, len(task_edits_list))))
                
                # Shuffle to mix samples from different tasks
                random.shuffle(samples)
                return samples[:batch_size]
        
        # Fall back to non-task-balanced sampling
        if policy == "uniform":
            return random.sample(edits, batch_size)
            
        # Priority sampling
        utils = [max(e.get("utility", 0.0), 0.0) + eps for e in edits]
        weights = [(u ** alpha) for u in utils]
        total = sum(weights)
        if total == 0:
            return random.sample(edits, batch_size)
        probs = [w / total for w in weights]
        return choices(edits, weights=probs, k=batch_size)
        
    def get_task_stats(self) -> Dict[str, int]:
        """
        Get statistics about the number of edits per task.
        
        Returns:
            Dictionary mapping task names to edit counts
        """
        edits = self._read_all()
        task_counts = defaultdict(int)
        
        for edit in edits:
            task = edit.get('task', 'default')
            task_counts[task] += 1
            
        return dict(task_counts)
