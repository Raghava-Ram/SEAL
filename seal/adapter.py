from typing import Optional, Dict, Any
import os
import yaml

from .llm_adapter import get_llm_client
from .prompt_adapter import create_prompt

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        return {'editing': {'mode': 'local'}}  # Default fallback

def simulate_edit_locally(text: str, label: Optional[int] = None) -> str:
    """
    Simulate GPT-style edit locally (offline version) with sentiment awareness.
    
    Args:
        text: Input text to be edited
        label: Optional sentiment label (0 for negative, 1 for positive)
        
    Returns:
        Edited text with simple transformations
    """
    # Simple heuristic modifications to simulate editing
    edited = text
    
    # Apply sentiment-aware transformations if label is provided
    if label is not None:
        if label == 1:  # Positive sentiment
            edited = edited.replace("not ", "")
            edited = edited.replace("n't ", " ")  # Remove contractions like "don't"
            edited = edited.replace("no ", "")
            edited = edited.replace("bad", "good")
            edited = edited.replace("terrible", "great")
        else:  # Negative sentiment
            edited = edited.replace("good", "bad")
            edited = edited.replace("great", "terrible")
    
    # General text modifications
    edited = edited.replace("is", "might be").replace("the", "this")
    return edited

class TextEditor:
    """Handles text editing using either local or LLM-based methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text editor with configuration."""
        self.config = config or load_config()
        self.llm = None
        self._init_llm_if_needed()
    
    def _init_llm_if_needed(self) -> None:
        """Initialize LLM client if needed."""
        if self.config.get('editing', {}).get('mode') == 'llm' and self.llm is None:
            try:
                self.llm = get_llm_client(self.config)
            except Exception as e:
                print(f"⚠️  Failed to initialize LLM client: {e}")
                print("⚠️  Falling back to local editing mode")
                self.config['editing']['mode'] = 'local'
    
    def edit(self, text: str, label: Optional[int] = None, **kwargs) -> str:
        """
        Generate an edit for the given text.
        
        Args:
            text: Input text to be edited
            label: Optional label for the text (e.g., sentiment)
            **kwargs: Additional arguments for the edit
            
        Returns:
            Edited text
        """
        edit_mode = self.config.get('editing', {}).get('mode', 'local')
        
        if edit_mode == 'local':
            return simulate_edit_locally(text, label)
        
        elif edit_mode == 'llm' and self.llm is not None:
            try:
                # Create a prompt for the LLM
                system_prompt = (
                    "You are a helpful AI assistant that improves text based on the given instructions. "
                    "Make the text more clear, concise, and engaging while preserving its original meaning."
                )
                
                if label is not None:
                    sentiment = "positive" if label == 1 else "negative"
                    system_prompt += f" The text has a {sentiment} sentiment - preserve this in your edits."
                
                prompt = create_prompt(
                    system_message=system_prompt,
                    user_message=f"Improve this text: {text}"
                )
                
                # Get the edited text from the LLM
                edited_text = self.llm.generate(prompt, **kwargs)
                return edited_text.strip()
                
            except Exception as e:
                print(f"⚠️  Error during LLM edit: {e}")
                print("⚠️  Falling back to local editing")
                return simulate_edit_locally(text, label)
        
        else:
            # Fallback to local editing if LLM is not available
            return simulate_edit_locally(text, label)

# Global editor instance for backward compatibility
_editor = TextEditor()

def generate_edit(text: str, label: Optional[int] = None, **kwargs) -> str:
    """
    Generate an edit for the given text using the configured editing mode.
    This is a convenience wrapper around the TextEditor class.
    
    Args:
        text: Input text to be edited
        label: Optional label for the text (e.g., sentiment)
        **kwargs: Additional arguments for the edit
        
    Returns:
        Edited text
    """
    return _editor.edit(text, label, **kwargs)
