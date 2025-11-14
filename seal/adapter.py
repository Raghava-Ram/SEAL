from typing import Optional, Dict, Any
import os
import json
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
    Simulate sentiment classification locally using a rule-based approach.
    
    Args:
        text: Input text to be classified
        label: Optional ground truth label (0 for negative, 1 for positive)
        
    Returns:
        str: Either "positive" or "negative" based on the classification
    """
    # List of positive and negative words for sentiment analysis
    positive_words = {
        "good", "great", "excellent", "amazing", "love", "wonderful",
        "fantastic", "superb", "outstanding", "perfect", "enjoy", "enjoyed",
        "best", "favorite", "loved", "awesome", "brilliant", "fabulous"
    }
    
    negative_words = {
        "bad", "terrible", "awful", "hate", "worst", "boring", "poor",
        "disappointing", "disappointed", "waste", "wasted", "horrible",
        "miserable", "dreadful", "unwatchable", "ridiculous", "stupid"
    }
    
    # Convert text to lowercase and split into words
    words = text.lower().split()
    
    # Count positive and negative word matches
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if negative_words)
    
    # Simple negation handling
    for i in range(1, len(words)):
        if words[i-1] in {"not", "n't", "no", "never", "none"}:
            if words[i] in positive_words:
                neg_count += 1
                pos_count = max(0, pos_count - 1)
            elif words[i] in negative_words:
                pos_count += 1
                neg_count = max(0, neg_count - 1)
    
    # Determine sentiment based on word counts
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    
    # If counts are equal, use the label if provided, otherwise default to positive
    if label is not None:
        return "positive" if label == 1 else "negative"
    return "positive"  # Default to positive if no label and no clear sentiment

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
                print("🔧 Initializing LLM client...")
                
                # Get the configuration
                llm_config = {
                    'backend': self.config.get('llm', {}).get('backend', 'ollama'),
                    'ollama': {
                        'model': self.config.get('ollama', {}).get('model', 'llama2'),
                        'host': self.config.get('ollama', {}).get('host', 'http://localhost:11434'),
                        'stream': self.config.get('ollama', {}).get('stream', True),
                        'timeout': self.config.get('ollama', {}).get('timeout', 30),
                        'max_retries': self.config.get('ollama', {}).get('max_retries', 3)
                    }
                }
                
                print("🔧 LLM Configuration:")
                print(json.dumps(llm_config, indent=2))
                print("🔧 Current working directory:", os.getcwd())
                print("🔧 Python path:", os.environ.get('PYTHONPATH', 'Not set'))
                
                # Initialize the LLM client
                self.llm = get_llm_client(llm_config)
                print(f"✅ LLM client initialized: {self.llm.__class__.__name__}")
                
                # Test the connection
                print("🔧 Testing LLM connection...")
                test_prompt = "Hello, this is a test. Please respond with 'OK' if you can hear me."
                response = self.llm.generate(test_prompt)
                print(f"✅ LLM response: {response[:100]}..." if len(response) > 100 else f"✅ LLM response: {response}")
                
            except Exception as e:
                import traceback
                print(f"⚠️  Failed to initialize LLM client: {e}")
                print(f"⚠️  Error details: {traceback.format_exc()}")
                print("⚠️  Falling back to local editing mode")
                self.config['editing']['mode'] = 'local'
    
    def edit(self, text: str, target_label: int = 1, **kwargs) -> str:
        """
        Edit the given text to match the target sentiment.
        
        Args:
            text: Input text to be edited
            target_label: Target sentiment label (0 for negative, 1 for positive)
            **kwargs: Additional arguments including 'mode' ('local' or 'llm')
            
        Returns:
            str: Edited text with the target sentiment
        """
        mode = kwargs.get('mode', self.config.get('editing', {}).get('mode', 'local'))
        
        if mode == 'llm' and self.llm is not None:
            # Use LLM for editing if available
            try:
                sentiment = "positive" if target_label == 1 else "negative"
                prompt = f"Rewrite the following text to have a {sentiment} sentiment: {text}"
                return self.llm.generate(prompt)
            except Exception as e:
                print(f"⚠️  LLM editing failed: {e}")
                print("Falling back to local editing...")
        
        # Local editing: simple rule-based approach
        return self._edit_text_locally(text, target_label)
    
    def _edit_text_locally(self, text: str, target_label: int) -> str:
        """
        Edit the text to match the target sentiment using local rules.
        
        Args:
            text: Input text to be edited
            target_label: Target sentiment label (0 for negative, 1 for positive)
            
        Returns:
            str: Edited text with the target sentiment
        """
        if target_label == 1:  # Make it positive
            # Replace negative phrases with positive ones
            replacements = {
                "didn't like": "loved",
                "terrible": "amazing",
                "worse": "better",
                "bad": "good",
                "awful": "wonderful",
                "hate": "love",
                "worst": "best",
                "boring": "exciting",
                "poor": "excellent",
                "disappointing": "satisfying",
                "horrible": "fantastic",
                "miserable": "joyful",
                "dreadful": "delightful",
                "unwatchable": "captivating"
            }
        else:  # Make it negative
            # Replace positive phrases with negative ones
            replacements = {
                "love": "hate",
                "good": "bad",
                "great": "terrible",
                "amazing": "awful",
                "wonderful": "dreadful",
                "fantastic": "horrible",
                "excellent": "poor",
                "perfect": "flawed",
                "enjoy": "suffer through",
                "best": "worst",
                "favorite": "least favorite",
                "awesome": "terrible",
                "brilliant": "dull",
                "fabulous": "dismal"
            }
        
        # Apply replacements (case-insensitive)
        edited_text = text
        for old, new in replacements.items():
            # Replace whole words only
            edited_text = ' '.join([new if word.lower() == old.lower() else word for word in edited_text.split()])
        
        # If no replacements were made, add a sentiment indicator
        if edited_text == text:
            if target_label == 1:
                return f"I really loved this! {text}"
            else:
                return f"I really didn't like this. {text}"
                
        return edited_text

    def classify(self, text: str, label: Optional[int] = None, **kwargs) -> str:
        """
        Generate a sentiment classification for the given text.
        
        Args:
            text: Input text to be classified
            label: Optional ground truth label (0 for negative, 1 for positive)
            **kwargs: Additional arguments including 'mode' ('local' or 'llm')
            
        Returns:
            str: Either "positive" or "negative" based on the classification
        """
        # Use mode from kwargs if provided, otherwise fall back to config
        edit_mode = kwargs.pop('mode', None) or self.config.get('editing', {}).get('mode', 'local')
        print(f"🔧 Edit mode: {edit_mode}")
        
        # Update LLM client if needed based on the mode
        if edit_mode == 'llm':
            print("🔧 Checking LLM client...")
            if self.llm is None:
                self._init_llm_if_needed()
            print(f"🔧 LLM client status: {'Initialized' if self.llm is not None else 'Not available'}")
        
        if edit_mode == 'local':
            return simulate_edit_locally(text, label)
        
        elif edit_mode == 'llm' and self.llm is not None:
            try:
                # Create a prompt for the LLM to classify sentiment
                system_prompt = (
                    "You are a sentiment analysis assistant. "
                    "Classify the sentiment of the following text as either 'positive' or 'negative'. "
                    "Respond with a single word: 'positive' or 'negative'."
                )
                
                prompt = create_prompt(
                    system_message=system_prompt,
                    user_message=text
                )
                
                # Get the classification from the LLM
                response = self.llm.generate(prompt, **kwargs).strip().lower()
                
                # Ensure the response is either 'positive' or 'negative'
                if 'positive' in response:
                    return "positive"
                elif 'negative' in response:
                    return "negative"
                else:
                    print(f"⚠️  Unexpected LLM response: {response}")
                    return "positive"  # Default fallback
                
            except Exception as e:
                print(f"⚠️  Error during LLM classification: {e}")
                print("⚠️  Falling back to local classification")
                return simulate_edit_locally(text, label)
        
        else:
            # Fallback to local classification if LLM is not available
            return simulate_edit_locally(text, label)

# Global editor instance for backward compatibility
_editor = None

def get_editor():
    """Get the global editor instance, creating it if necessary."""
    global _editor
    if _editor is None:
        _editor = TextEditor()
    return _editor

def generate_edit(text: str, label: Optional[int] = None, **kwargs) -> str:
    """
    Generate an edit for the given text using the specified editing mode.
    This is a convenience wrapper around the TextEditor class.
    
    Args:
        text: Input text to be edited
        label: Target sentiment label (0 for negative, 1 for positive)
        mode: Either 'local' or 'llm' to specify the editing mode
        **kwargs: Additional arguments for the edit
        
    Returns:
        Edited text
    """
    editor = get_editor()
    # Update the editor's config if mode is specified
    if 'mode' in kwargs:
        if 'editing' not in editor.config:
            editor.config['editing'] = {}
        editor.config['editing']['mode'] = kwargs['mode']
        # Force reinitialization of LLM if needed
        if kwargs['mode'] == 'llm' and editor.llm is None:
            editor._init_llm_if_needed()
    
    # Handle target_label parameter (preferred) or fall back to label
    target_label = kwargs.pop('target_label', label)
    if target_label is None:
        target_label = 1  # Default to positive sentiment if no label provided
    
    return editor.edit(text, target_label=target_label, **kwargs)
