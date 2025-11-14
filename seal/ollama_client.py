import json
import time
from typing import Dict, Any, Optional, Union
import requests
from requests.exceptions import RequestException

from .llm_client import LLMClient, OllamaError

class OllamaClient:
    """Client for interacting with the Ollama API."""
    
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        stream: bool = False,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> None:
        """Initialize the Ollama client.
        
        Args:
            model: The model to use for generation (e.g., "qwen3:4b")
            host: Base URL of the Ollama server
            stream: Whether to use streaming mode
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.base_url = host.rstrip('/')
        self.stream = stream
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text from the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            Generated text
            
        Raises:
            OllamaError: If the request fails after all retries
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self.stream,
            **kwargs
        }
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.stream:
                    return self._handle_streaming_request(url, payload)
                else:
                    return self._handle_standard_request(url, payload)
            except (RequestException, json.JSONDecodeError) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                continue
        
        raise OllamaError(f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}")
    
    def _handle_standard_request(self, url: str, payload: Dict[str, Any]) -> str:
        """Handle a standard (non-streaming) request."""
        response = self.session.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    
    def _handle_streaming_request(self, url: str, payload: Dict[str, Any]) -> str:
        """Handle a streaming request."""
        response = self.session.post(
            url,
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
                
            try:
                data = json.loads(line)
                if "response" in data:
                    full_response += data["response"]
            except json.JSONDecodeError as e:
                continue
                
        return full_response
