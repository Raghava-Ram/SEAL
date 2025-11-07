"""SEAL (Self-Edit Adaptive Learning) - A framework for self-improving language models.

This package provides functionality for local and OpenAI-based text editing,
model adaptation and fine-tuning, configuration management, and utility functions.
"""

from .adapter import simulate_edit_locally, generate_edit
from .openai_edit import generate_edit_via_openai

__all__ = [
    'simulate_edit_locally',
    'generate_edit',
    'generate_edit_via_openai',
]
