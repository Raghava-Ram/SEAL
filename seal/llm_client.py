from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any, Union

class LLMClient(Protocol):
    """Interface for LLM clients to ensure consistent API across different providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the given prompt.
        
        Args:
            prompt: The input prompt to generate text from
            **kwargs: Additional arguments specific to the implementation
            
        Returns:
            Generated text as a string
        """
        ...

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class OllamaError(LLMError):
    """Exception raised for errors in the Ollama client."""
    pass
