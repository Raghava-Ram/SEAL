from typing import List, Dict, Union

def flatten_messages(messages: List[Dict[str, str]]) -> str:
    """Convert a list of message dictionaries into a single string.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        A single string with messages in the format:
        """
    formatted_messages = []
    
    for msg in messages:
        role = msg.get('role', 'user').capitalize()
        content = msg.get('content', '').strip()
        if content:  # Skip empty messages
            formatted_messages.append(f"{role}: {content}")
    
    return "\n\n".join(formatted_messages)

def convert_openai_to_ollama(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style messages to a single prompt string for Ollama.
    
    This is a convenience wrapper around flatten_messages for backward compatibility.
    """
    return flatten_messages(messages)

def create_prompt(system_message: str = "", user_message: str = "") -> str:
    """Create a formatted prompt from system and user messages.
    
    Args:
        system_message: Optional system message (e.g., instructions)
        user_message: User's input message
        
    Returns:
        Formatted prompt string
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return flatten_messages(messages)
