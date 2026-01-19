"""
Utility functions for the SEAL frontend.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import requests


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "outputs" / "multi_task"


def load_json_safe(filepath: Path) -> Optional[Dict]:
    """Safely load a JSON file, returning None on error."""
    try:
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def list_available_approaches() -> List[str]:
    """List available approach directories (baseline, hybrid, etc.)."""
    approaches = []
    if DATA_PATH.exists():
        for item in DATA_PATH.iterdir():
            if item.is_dir() and (item / "imdb_squad_arc_metrics.json").exists():
                approaches.append(item.name)
    return approaches


def get_available_screenshots(base_path: Path) -> List[Path]:
    """Get list of available screenshot files."""
    screenshots = []
    if base_path.exists():
        screenshots = sorted(base_path.glob("*.png"))
    return screenshots


def compute_backward_transfer(accuracy_matrix: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute backward transfer for each task.
    
    Backward transfer measures how learning new tasks affects performance on previous tasks.
    """
    bwt = {}
    tasks = list(accuracy_matrix.keys())
    
    for i, task in enumerate(tasks[:-1]):
        accs = [acc for acc in accuracy_matrix[task] if acc is not None]
        if not accs or len(accs) <= i:
            bwt[task] = 0.0
            continue
        
        initial_acc = accs[i]
        final_acc = accs[-1]
        bwt[task] = final_acc - initial_acc
    
    return bwt


def test_ollama_connection(base_url: str = "http://localhost:11434", timeout: int = 2) -> bool:
    """Test if Ollama is accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False
