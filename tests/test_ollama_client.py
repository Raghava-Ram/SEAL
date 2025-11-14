import json
import pytest
from unittest.mock import patch, MagicMock
import requests

from seal.ollama_client import OllamaClient, OllamaError

def test_ollama_client_initialization():
    """Test that the Ollama client initializes with default values."""
    client = OllamaClient(model="test-model")
    assert client.model == "test-model"
    assert client.base_url == "http://localhost:11434"
    assert client.stream is False
    assert client.timeout == 30
    assert client.max_retries == 3

def test_ollama_client_custom_initialization():
    """Test that the Ollama client initializes with custom values."""
    client = OllamaClient(
        model="custom-model",
        host="http://custom-host:1234",
        stream=True,
        timeout=60,
        max_retries=5
    )
    assert client.model == "custom-model"
    assert client.base_url == "http://custom-host:1234"
    assert client.stream is True
    assert client.timeout == 60
    assert client.max_retries == 5

@patch('requests.Session.post')
def test_generate_success(mock_post):
    """Test successful text generation."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Generated text"}
    mock_post.return_value = mock_response
    
    # Test
    client = OllamaClient(model="test-model")
    result = client.generate("Test prompt")
    
    # Assertions
    assert result == "Generated text"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs['json']['model'] == "test-model"
    assert kwargs['json']['prompt'] == "Test prompt"
    assert kwargs['json']['stream'] is False

@patch('requests.Session.post')
def test_generate_streaming(mock_post):
    """Test streaming text generation."""
    # Setup mock streaming response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        b'{"response": "Gen"}',
        b'{"response": "erated"}',
        b'{"response": " text"}'
    ]
    mock_post.return_value = mock_response
    
    # Test with streaming
    client = OllamaClient(model="test-model", stream=True)
    result = client.generate("Test prompt")
    
    # Assertions
    assert result == "Generated text"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs['stream'] is True

@patch('requests.Session.post')
def test_generate_retry_on_failure(mock_post):
    """Test that the client retries on failure."""
    # Setup mock to fail twice then succeed
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 500
    
    mock_response_success = MagicMock()
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = {"response": "Success after retry"}
    
    mock_post.side_effect = [
        mock_response_fail,
        mock_response_fail,
        mock_response_success
    ]
    
    # Test with 2 retries
    client = OllamaClient(model="test-model", max_retries=3)
    result = client.generate("Test prompt")
    
    # Should succeed after retries
    assert result == "Success after retry"
    assert mock_post.call_count == 3

@patch('requests.Session.post')
def test_generate_max_retries_exceeded(mock_post):
    """Test that the client raises an error when max retries are exceeded."""
    # Setup mock to always fail
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response
    
    # Test with 1 retry
    client = OllamaClient(model="test-model", max_retries=1)
    
    # Should raise OllamaError after max retries
    with pytest.raises(OllamaError):
        client.generate("Test prompt")
    
    # Should have been called twice (initial + 1 retry)
    assert mock_post.call_count == 2

@patch('requests.Session.post')
def test_generate_with_additional_params(mock_post):
    """Test that additional parameters are passed to the API."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Generated text"}
    mock_post.return_value = mock_response
    
    # Test with additional parameters
    client = OllamaClient(model="test-model")
    result = client.generate(
        "Test prompt",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9
    )
    
    # Assertions
    assert result == "Generated text"
    args, kwargs = mock_post.call_args
    assert kwargs['json']['temperature'] == 0.7
    assert kwargs['json']['max_tokens'] == 100
    assert kwargs['json']['top_p'] == 0.9
