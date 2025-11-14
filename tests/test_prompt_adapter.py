import pytest
from seal.prompt_adapter import flatten_messages, create_prompt

def test_flatten_messages_single_user_message():
    """Test flattening a single user message."""
    messages = [{"role": "user", "content": "Hello, world!"}]
    expected = "User: Hello, world!"
    assert flatten_messages(messages) == expected

def test_flatten_messages_system_and_user():
    """Test flattening system and user messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    expected = "System: You are a helpful assistant.\n\nUser: Hello!"
    assert flatten_messages(messages) == expected

def test_flatten_messages_empty_messages():
    """Test flattening an empty messages list."""
    assert flatten_messages([]) == ""

def test_flatten_messages_skips_empty_content():
    """Test that messages with empty content are skipped."""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": ""}
    ]
    assert flatten_messages(messages) == "User: Hello!"

def test_flatten_messages_missing_role_defaults_to_user():
    """Test that messages without a role default to 'user'."""
    messages = [{"content": "No role specified"}]
    assert flatten_messages(messages) == "User: No role specified"

def test_create_prompt_system_only():
    """Test creating a prompt with only a system message."""
    system = "You are a helpful assistant."
    expected = "System: You are a helpful assistant."
    assert create_prompt(system_message=system) == expected

def test_create_prompt_user_only():
    """Test creating a prompt with only a user message."""
    user = "Hello!"
    assert create_prompt(user_message=user) == "User: Hello!"

def test_create_prompt_both_messages():
    """Test creating a prompt with both system and user messages."""
    system = "You are a helpful assistant."
    user = "What's the weather like?"
    expected = "System: You are a helpful assistant.\n\nUser: What's the weather like?"
    assert create_prompt(system, user) == expected

def test_create_prompt_empty():
    """Test creating a prompt with no messages."""
    assert create_prompt() == ""
