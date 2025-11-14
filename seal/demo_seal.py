"""
SEAL Demo Script

A lightweight demonstration of the SEAL framework's capabilities.
Shows local and LLM-based editing on sample text.
"""

import os
import sys
import torch
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seal.adapter import TextEditor
from seal.llm_adapter import OllamaAdapter, get_llm_client
from seal.utility import score_edit_simple

def load_sample_texts() -> List[Dict[str, Any]]:
    """Load sample texts for demonstration."""
    return [
        {
            "text": "I really didn't like this movie. The plot was terrible and the acting was worse.",
            "label": 0,  # Negative sentiment
            "id": "sample1"
        },
        {
            "text": "This film was absolutely amazing! The cinematography was stunning.",
            "label": 1,  # Positive sentiment
            "id": "sample2"
        },
        {
            "text": "The movie was okay, but I expected more from the director.",
            "label": 0,  # Slightly negative
            "id": "sample3"
        }
    ]

def run_demo():
    """Run the SEAL demonstration."""
    print("\n🎬 Starting SEAL Demo")
    print("-" * 50)
    
    # Initialize editor
    local_editor = TextEditor()
    # Initialize LLM editor with default config
    llm_config = {
        'model': 'llama2',
        'host': 'http://localhost:11434',
        'timeout': 30
    }
    llm_editor = OllamaAdapter(llm_config)
    
    # Load sample texts
    samples = load_sample_texts()
    
    print("\n📝 Sample Texts:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i} (Label: {'Positive' if sample['label'] else 'Negative'}):")
        print(f"  {sample['text']}")
    
    # Demonstrate local editing
    print("\n🔄 Local Editing Examples:")
    for sample in samples:
        original = sample["text"]
        edited = local_editor.edit(original, target_label=1)  # Try to make it positive
        
        print(f"\nOriginal: {original}")
        print(f"Edited:   {edited}")
    
    # Demonstrate LLM editing (if available)
    try:
        print("\n🤖 LLM Editing Examples (if Ollama is running):")
        for sample in samples[:1]:  # Just show one LLM example to be quick
            original = sample["text"]
            edited = llm_editor.edit(original, target_label=1)
            
            print(f"\nOriginal: {original}")
            print(f"LLM Edit: {edited}")
    except Exception as e:
        print(f"\n⚠️ LLM Editing not available: {str(e)}")
        print("Make sure Ollama is running with 'ollama serve'")
    
    # Show utility scoring
    print("\n📊 Utility Scoring Example:")
    sample = samples[0]
    original = sample["text"]
    edited = local_editor.edit(original, target_label=1)
    
    # Mock confidence scores for demo
    score = score_edit_simple({
        "original": original,
        "edit": edited,
        "conf_before": 0.8,  # High confidence in original prediction
        "conf_after": 0.6,   # Slightly less confident after edit
        "pred_before": 0,    # Original prediction
        "pred_after": 1,     # New prediction
        "acc_before": 0.0,   # Accuracy before (unknown in this context)
        "acc_after": 0.0     # Accuracy after (unknown in this context)
    })
    
    print(f"Original: {original}")
    print(f"Edited:   {edited}")
    print(f"Utility Score: {score:.2f}")
    
    print("\n✅ Demo complete!")
    print("\n💡 Try running with '--mode seal' for the full SEAL training loop")
    print("   or '--mode imdb' for a local vs LLM comparison.")

if __name__ == "__main__":
    run_demo()
