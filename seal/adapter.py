from typing import Optional
import torch

def simulate_edit_locally(text: str) -> str:
    """
    Simulate GPT-style edit locally (offline version).
    
    Args:
        text: Input text to be edited
        
    Returns:
        Edited text with simple transformations
    """
    # Simple heuristic modifications to simulate editing
    return text.replace("is", "might be").replace("the", "this")

def generate_edit(text: str, mode: str = "local") -> str:
    """
    Generate an edit for the given text using the specified mode.
    
    Args:
        text: Input text to be edited
        mode: Either "local" for local simulation or "openai" for OpenAI API
        
    Returns:
        Edited text
    """
    if mode == "local":
        return simulate_edit_locally(text)
    elif mode == "openai":
        from .openai_edit import generate_edit_via_openai
        return generate_edit_via_openai(text)
    else:
        raise ValueError(f"Unsupported edit mode: {mode}")
