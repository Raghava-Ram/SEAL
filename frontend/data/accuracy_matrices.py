# frontend/data/accuracy_matrices.py

METHOD_RESULTS = {
    "M0: Baseline": {
        "matrix": [
            [0.95, 0.10, 0.22],
            [None, 0.47, 0.00],
            [None, None, 1.00],
        ],
        "source": "baseline.png",
        "description": "No replay or protection. Severe catastrophic forgetting."
    },

    "M1: SEAL (Replay)": {
        "matrix": [
            [0.86, 0.54, 0.02],
            [None, 0.49, 0.01],
            [None, None, 1.00],
        ],
        "source": "seal_replay.png",
        "description": "Replay-based rehearsal only (failure case). Demonstrates insufficiency of replay without parameter-level protection."
    },

    "M2: Hybrid SEAL (LLM + Replay)": {
        "matrix": [
            [0.91, 0.24, 0.85],
            [None, 0.77, 0.45],
            [None, None, 1.00],
        ],
        "source": "hybrid_llm_replay.png",
        "description": "LLM-guided edits with light replay. Partial stability improvement."
    },

    "M3: Hybrid + Freezing": {
        "matrix": [
            [0.94, 1.00, 0.45],
            [None, 0.49, 0.11],
            [None, None, 1.00],
        ],
        "source": "hybrid_freezing.png",
        "description": "Freezing lower encoder layers reduces parameter drift."
    },

    "M4: Hybrid + Task-weighted Replay (v1)": {
        "matrix": [
            [0.81, 0.75, 0.00],
            [None, 0.55, 0.56],
            [None, None, 1.00],
        ],
        "source": "hybrid_task_weighted_replay.png",
        "description": "Initial task-aware replay weighting strategy."
    },

    "M5: Hybrid + Task-weighted Replay (v2 – unstable)": {
        "matrix": [
            [0.87, 1.00, 0.00],
            [None, 0.54, 0.54],
            [None, None, 1.00],
        ],
        "source": "hybrid_task_weighted_replay_v2.png",
        "description": "Task-weighted replay without parameter protection (failure case). Motivates need for EWC."
    },

    "M6: FINAL – Hybrid + EWC": {
        "matrix": [
            [0.95, 0.69, 0.82],
            [None, 0.57, 0.43],
            [None, None, 0.97],
        ],
        "source": "hybrid_ewc_final.png",
        "description": "Elastic Weight Consolidation with task-specific heads. Best retention."
    },
}
