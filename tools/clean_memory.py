"""
Backup and clean memory/edits.jsonl by normalizing labels.
Usage: python tools/clean_memory.py
"""
import json
import os
import time
import shutil

SRC = os.path.join(os.path.dirname(__file__), "..", "memory", "edits.jsonl")
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "..", "memory", "backups")

os.makedirs(BACKUP_DIR, exist_ok=True)

if not os.path.exists(SRC):
    print(f"Memory file not found: {SRC}")
    raise SystemExit(1)

# Backup existing file
ts = time.strftime("%Y%m%d_%H%M%S")
bak = os.path.join(BACKUP_DIR, f"edits.jsonl.bak.{ts}")
shutil.copy2(SRC, bak)
print(f"Backup created: {bak}")

# Read and sanitize
cleaned = []
counts_before = {}
counts_after = {}
invalid_entries = []

with open(SRC, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            invalid_entries.append((i, "invalid_json"))
            continue
        lbl = obj.get("label")
        counts_before[lbl] = counts_before.get(lbl, 0) + 1

        # Normalize label: try to coerce to int, clamp to >=0
        try:
            lbl_int = int(float(lbl))
        except Exception:
            lbl_int = 0
        if lbl_int < 0:
            lbl_int = 0

        obj["label"] = lbl_int
        counts_after[lbl_int] = counts_after.get(lbl_int, 0) + 1
        cleaned.append(obj)

# Write cleaned file (overwrite original)
with open(SRC, "w", encoding="utf-8") as f:
    for obj in cleaned:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Processed entries: {len(cleaned)}")
print(f"Invalid/skipped entries: {len(invalid_entries)} (sample: {invalid_entries[:10]})")
print("Sample label counts before (top 10):")
for k, v in sorted(counts_before.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k}: {v}")

print("Sample label counts after (top 10):")
for k, v in sorted(counts_after.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k}: {v}")

print("Cleanup complete. Original backed up.")
