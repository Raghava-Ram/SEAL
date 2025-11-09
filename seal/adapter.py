from typing import Optional
import torch

def simulate_edit_locally(text: str, label: Optional[int] = None) -> str:
    """
    Simulate GPT-style edit locally (offline version) with sentiment awareness.
    
    Args:
        text: Input text to be edited
        label: Optional sentiment label (0 for negative, 1 for positive)
        
    Returns:
        Edited text with simple transformations
    """
    # Simple heuristic modifications to simulate editing
    edited = text
    
    # Apply sentiment-aware transformations if label is provided
    if label is not None:
        if label == 1:  # Positive sentiment
            edited = edited.replace("not ", "")
            edited = edited.replace("n't ", " ")  # Remove contractions like "don't"
            edited = edited.replace("no ", "")
            edited = edited.replace("bad", "good")
            edited = edited.replace("terrible", "great")
        else:  # Negative sentiment
            edited = edited.replace("good", "bad")
            edited = edited.replace("great", "terrible")
    
    # General text modifications
    edited = edited.replace("is", "might be").replace("the", "this")
    return edited

def load_config():
    """Load configuration from YAML file."""
    import yaml
    import os
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        return {'editing': {'mode': 'local'}}  # Default fallback

def generate_edit(text: str, label: Optional[int] = None) -> str:
    """
    Generate an edit for the given text using local editing.

    Args:
        text: Input text to be edited
        label: Optional sentiment label (0 for negative, 1 for positive)

    Returns:
        Edited text
    """
    return simulate_edit_locally(text, label)
