from typing import Dict, Any, Optional, Union
from .llm_client import LLMClient, LLMError
from .ollama_client import OllamaClient

class OllamaAdapter(LLMClient):
    """Adapter for Ollama LLM client."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Ollama adapter.
        
        Args:
            config: Configuration dictionary containing Ollama settings
        """
        self.client = OllamaClient(
            model=config["model"],
            host=config.get("host", "http://localhost:11434"),
            stream=config.get("stream", False),
            timeout=config.get("timeout", 30),
            max_retries=config.get("max_retries", 3)
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the Ollama client.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            Generated text
        """
        return self.client.generate(prompt, **kwargs)
        
    def edit(self, text: str, target_label: int = 1, **kwargs) -> str:
        """Edit the given text to match the target label.
        
        Args:
            text: The text to edit
            target_label: The target sentiment label (0 for negative, 1 for positive)
            **kwargs: Additional parameters
            
        Returns:
            Edited text
        """
        sentiment = "positive" if target_label == 1 else "negative"
        prompt = f"Rewrite the following text to have a {sentiment} sentiment: {text}"
        return self.generate(prompt, **kwargs)

class OpenAIAdapter(LLMClient):
    """Adapter for OpenAI API client (for backward compatibility)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI adapter.
        
        Args:
            config: Configuration dictionary containing OpenAI settings
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install it with 'pip install openai'"
            )
        
        self.client = openai.OpenAI(api_key=config.get("api_key"))
        self.model = config.get("model", "gpt-3.5-turbo")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the OpenAI API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to pass to the OpenAI API
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}")

def get_llm_client(config: Dict[str, Any]) -> LLMClient:
    """Factory function to get an LLM client based on configuration.
    
    Args:
        config: Configuration dictionary containing LLM settings
        
    Returns:
        An instance of an LLMClient implementation
        
    Raises:
        ValueError: If an unsupported backend is specified
    """
    backend = config.get("backend", "ollama").lower()
    
    if backend == "ollama":
        return OllamaAdapter(config.get("ollama", {}))
    elif backend == "openai":
        return OpenAIAdapter(config.get("openai", {}))
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")
