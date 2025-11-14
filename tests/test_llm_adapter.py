import pytest
from unittest.mock import patch, MagicMock

from seal.llm_adapter import get_llm_client, OllamaAdapter, OpenAIAdapter, LLMError

def test_get_llm_client_ollama():
    """Test getting an Ollama client with default config."""
    config = {
        "backend": "ollama",
        "ollama": {
            "model": "test-model"
        }
    }
    
    client = get_llm_client(config)
    assert isinstance(client, OllamaAdapter)
    assert client.client.model == "test-model"

def test_get_llm_client_openai():
    """Test getting an OpenAI client with default config."""
    config = {
        "backend": "openai",
        "openai": {
            "api_key": "test-key",
            "model": "gpt-4"
        }
    }
    
    with patch('seal.llm_adapter.OpenAI') as mock_openai:
        client = get_llm_client(config)
        assert isinstance(client, OpenAIAdapter)
        mock_openai.assert_called_once_with(api_key="test-key")

def test_get_llm_client_invalid_backend():
    """Test that an invalid backend raises an error."""
    with pytest.raises(ValueError) as excinfo:
        get_llm_client({"backend": "invalid"})
    assert "Unsupported LLM backend" in str(excinfo.value)

@patch('seal.ollama_client.OllamaClient.generate')
def test_ollama_adapter_generate(mock_generate):
    """Test the Ollama adapter's generate method."""
    # Setup
    mock_generate.return_value = "Generated response"
    adapter = OllamaAdapter({"model": "test-model"})
    
    # Test
    result = adapter.generate("Test prompt", temperature=0.7)
    
    # Assert
    assert result == "Generated response"
    mock_generate.assert_called_once_with("Test prompt", temperature=0.7)

@patch('openai.resources.chat.Completions.create')
def test_openai_adapter_generate(mock_create):
    """Test the OpenAI adapter's generate method."""
    # Setup
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "OpenAI response"
    mock_create.return_value = mock_response
    
    with patch('seal.llm_adapter.openai', create=True):
        adapter = OpenAIAdapter({"api_key": "test-key", "model": "gpt-4"})
        
        # Test
        result = adapter.generate("Test prompt", temperature=0.7)
        
        # Assert
        assert result == "OpenAI response"
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        assert kwargs["model"] == "gpt-4"
        assert kwargs["messages"][0]["content"] == "Test prompt"
        assert kwargs["temperature"] == 0.7

def test_openai_adapter_missing_dependency():
    """Test that OpenAI adapter raises an error when openai package is missing."""
    with patch.dict('sys.modules', {'openai': None}):
        with pytest.raises(ImportError) as excinfo:
            OpenAIAdapter({"api_key": "test-key"})
        assert "OpenAI package not found" in str(excinfo.value)
