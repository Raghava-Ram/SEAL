"""
EditCache: JSONL-backed edit memory for SEAL.
"""

import json
import os
import uuid
from typing import Dict, Any, List, Optional
from random import choices

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
        e = edit.copy()
        e.setdefault("id", str(uuid.uuid4()))
        e.setdefault("utility", float(e.get("utility", 0.0)))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._enforce_size()
        return e["id"]

    def _enforce_size(self):
        edits = self._read_all()
        if len(edits) <= self.max_size:
            return
        edits.sort(key=lambda x: float(x.get("utility", 0.0)), reverse=True)
        edits = edits[: self.max_size]
        self._write_all(edits)

    def sample(self, batch_size: int, policy: str = "priority", alpha: float = 1.0, eps: float = 1e-6):
        edits = self._read_all()
        n = len(edits)
        if n == 0:
            return []
        if batch_size >= n:
            return edits.copy()

        if policy == "uniform":
            from random import sample
            return sample(edits, batch_size)

        utils = [max(e.get("utility", 0.0), 0.0) + eps for e in edits]
        weights = [(u ** alpha) for u in utils]
        total = sum(weights)
        if total == 0:
            from random import sample
            return sample(edits, batch_size)
        probs = [w / total for w in weights]
        return choices(edits, weights=probs, k=batch_size)
