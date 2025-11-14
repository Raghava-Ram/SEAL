"""SEAL (Self-Edit Adaptive Learning) - A framework for self-improving language models.

This package provides functionality for local text editing, model adaptation 
and fine-tuning, configuration management, and utility functions.
"""

from .adapter import simulate_edit_locally, generate_edit
from .trainer import SEALTrainer
from .runner import run_seal_loop, run_imdb_comparison

# Import demo module if available
try:
    from .demo_seal import run_demo
except ImportError:
    run_demo = None

__all__ = [
    # Core functionality
    'simulate_edit_locally',
    'generate_edit',
    
    # Training and running
    'SEALTrainer',
    'run_seal_loop',
]
