"""
Simple checker to report label distributions for tasks and memory.
Run from the repo root: python tools/check_labels.py
"""
import json
import os
import sys
from collections import Counter

# Ensure repository root is on sys.path so `import seal` works when run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.tasks import get_task_loader

TASKS = ["imdb", "squad", "arc"]


def inspect_task(task_name, sample_size=500):
    loader = get_task_loader(task_name)
    try:
        data = loader(task_size=sample_size)
    except TypeError:
        # Some loaders may not accept task_size; call without it
        data = loader()
    except Exception as e:
        print(f"[ERROR] Loading {task_name}: {e}")
        return None

    labels = [ex.get("label") for ex in data if "label" in ex]
    lbl_counter = Counter(labels)
    unique = sorted([l for l in set(labels) if l is not None])
    print(f"\n--- Task: {task_name} ---")
    print(f"Total examples loaded: {len(labels)}")
    print(f"Unique labels (sample): {unique[:50]}")
    if labels:
        print(f"Label counts: {lbl_counter.most_common(10)}")
        try:
            numeric_labels = [int(l) for l in labels if l is not None]
            print(f"Min label: {min(numeric_labels)}, Max label: {max(numeric_labels)}")
        except Exception:
            pass

    # Task-specific expectations
    if task_name in ("imdb", "squad"):
        bad = [l for l in set(labels) if not (isinstance(l, int) and 0 <= l < 2)]
        if bad:
            print(f"WARNING: Found labels outside [0,1] for binary task {task_name}: {bad}")
    else:
        bad = [l for l in set(labels) if not (isinstance(l, int) and l >= 0)]
        if bad:
            print(f"WARNING: Found non-integer or negative labels for {task_name}: {bad}")

    return labels


def inspect_memory(path="memory/edits.jsonl"):
    print(f"\n--- Inspecting memory file: {path} ---")
    if not os.path.exists(path):
        print("Memory file not found.")
        return

    labels = []
    tasks = []
    bad_entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_entries.append((i+1, "invalid_json"))
                continue
            lbl = obj.get("label")
            t = obj.get("task", "unknown")
            labels.append(lbl)
            tasks.append(t)
            # Check simple issues
            if not (isinstance(lbl, int) and lbl >= 0):
                bad_entries.append((i+1, lbl, t))

    from collections import Counter
    print(f"Memory entries: {len(labels)}")
    if labels:
        ctr = Counter(labels)
        print(f"Top label counts: {ctr.most_common(10)}")
        # Show integer labels separately to avoid mixed-type sorting errors
        int_labels = sorted([l for l in set(labels) if isinstance(l, int)])
        non_int_labels = sorted([l for l in set(labels) if not isinstance(l, int)])
        print(f"Unique integer labels in memory (sample): {int_labels[:50]}")
        if non_int_labels:
            print(f"Unique non-integer/str/None labels (sample): {non_int_labels[:20]}")

    if bad_entries:
        print(f"WARNING: Found {len(bad_entries)} memory entries with non-integer/negative labels or parse errors. Sample:")
        for sample in bad_entries[:10]:
            print(" ", sample)
    else:
        print("No obvious bad labels found in memory (labels are non-negative integers).")


if __name__ == "__main__":
    print("Running label inspection for tasks and memory...\n")
    for t in TASKS:
        inspect_task(t, sample_size=500)

    inspect_memory()
    print("\nInspection complete.")
