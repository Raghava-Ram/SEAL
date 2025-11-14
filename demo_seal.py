#!/usr/bin/env python
"""
SEAL Demonstration Script

This script demonstrates the SEAL (Self-Edit Adaptive Learning) system
in both local and LLM modes on the IMDB dataset.
"""

import os
import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from seal.adapter import generate_edit

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"{text:^80}")
    print("="*80 + "\n")

def load_sample_data(n_samples=3):
    """Load sample data from IMDB dataset."""
    print(f"📊 Loading {n_samples} samples from IMDB test set...")
    dataset = load_dataset("imdb", split='test')
    return dataset.select(range(min(n_samples, len(dataset))))

def run_demo(mode, samples):
    """Run the demo in the specified mode."""
    print_header(f"🛠️  RUNNING IN {mode.upper()} MODE")
    
    results = []
    total_time = 0
    
    for i, sample in enumerate(tqdm(samples, desc=f"Processing ({mode} mode)")):
        text = sample['text']
        true_label = "positive" if sample['label'] == 1 else "negative"
        
        print(f"\n📝 Sample {i+1} - True Sentiment: {true_label.upper()}")
        print("-" * 50)
        print(f"Original text: {text[:200]}..." if len(text) > 200 else f"Original text: {text}")
        
        # Time the edit generation
        start_time = time.time()
        
        if mode == 'llm':
            try:
                print("\n🔍 Generating edit with LLM... (this may take a moment)")
                # For LLM mode, provide more specific instructions
                prompt = f"""You are a sentiment analysis expert. Please edit the following movie review to make it {'more positive' if sample['label'] == 1 else 'more negative'}, 
while keeping the core meaning intact. Return only the edited review, no explanations.

Original review: {text}

Edited review:"""
                
                # Use the LLM directly for more control
                from seal.llm_adapter import get_llm_client
                llm = get_llm_client({
                    'backend': 'ollama',
                    'ollama': {
                        'model': 'llama2',
                        'host': 'http://localhost:11434',
                        'stream': False,
                        'timeout': 120,  # Increased timeout to 2 minutes
                        'max_retries': 2  # Reduced retries to fail faster
                    }
                })
                print("🔄 Sending request to LLM...")
                edited_text = llm.generate(prompt)
                print("✅ Received response from LLM")
                
                # If the response is too short, it might be an error
                if len(edited_text.strip()) < 10:
                    raise ValueError("LLM response too short, falling back to local mode")
                    
            except Exception as e:
                print(f"⚠️  LLM Error: {str(e)}")
                print("🔄 Falling back to local editing...")
                mode = 'local'  # Fall back to local mode
                edited_text = generate_edit(text, label=sample['label'], mode=mode)
        else:
            # For local mode, use the standard generate_edit
            edited_text = generate_edit(text, label=sample['label'], mode=mode)
            
        print(f"🔧 Edit mode: {mode}")
            
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Classify the edited text (simple rule-based for demo)
        words = edited_text.lower().split()
        positive_words = {"good", "great", "excellent", "amazing", "love", "wonderful", "enjoyed", "best", "fantastic", "superb"}
        negative_words = {"bad", "terrible", "awful", "hate", "worst", "boring", "poor", "disappointing", "waste", "horrible"}
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Also check for negations that might flip the sentiment
        negations = {"not", "don't", "doesn't", "isn't", "wasn't", "can't", "couldn't"}
        negation_found = any(word in negations for word in words)
        
        if negation_found:
            predicted_label = "positive" if negative_count > positive_count else "negative"
        else:
            predicted_label = "positive" if positive_count > negative_count else "negative"
        
        print(f"\nEdited text: {edited_text[:200]}..." if len(edited_text) > 200 else f"Edited text: {edited_text}")
        print(f"\n🔍 Analysis:")
        print(f"- Predicted Sentiment: {predicted_label.upper()}")
        print(f"- Processing Time: {inference_time:.2f} seconds")
        print(f"- Edit Length: {len(edited_text)} characters")
        
        results.append({
            'sample': i+1,
            'original_text': text,
            'edited_text': edited_text,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'inference_time': inference_time,
            'is_correct': true_label.lower() == predicted_label.lower()
        })
    
    # Print summary
    accuracy = sum(1 for r in results if r['is_correct']) / len(results)
    avg_time = total_time / len(results)
    
    print_header("📊 DEMO SUMMARY")
    print(f"Mode: {mode.upper()}")
    print(f"Samples Processed: {len(results)}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Average Processing Time: {avg_time:.2f} seconds per sample")
    
    return results

def check_ollama_connection():
    """Check if Ollama server is running and accessible."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main function to run the demo."""
    # Check Ollama connection first
    print("🔍 Checking Ollama connection...")
    if not check_ollama_connection():
        print("⚠️  Ollama server not running or not accessible at http://localhost:11434")
        print("Please make sure Ollama is running by executing 'ollama serve' in a separate terminal")
        print("Falling back to local mode only...\n")
        llm_available = False
    else:
        print("✅ Connected to Ollama server\n")
        llm_available = True
    
    # Load sample data
    samples = load_sample_data(n_samples=3)
    
    # Run in local mode
    local_results = run_demo("local", samples)
    
    # Only offer LLM mode if Ollama is available
    if llm_available:
        run_llm = input("\nWould you like to run the same samples in LLM mode? (y/n): ").lower()
        if run_llm == 'y':
            print("\n" + "="*80)
            print("⚠️  NOTE: LLM mode is much slower than local mode")
            print("    Each sample may take 30-60 seconds to process")
            print("="*80 + "\n")
            llm_results = run_demo("llm", samples)
    else:
        print("\n⚠️  LLM mode not available. To enable it, make sure Ollama is running.")
        print("   Run 'ollama serve' in a separate terminal and restart the demo.")
    
    print("\n🎉 Demo completed! 🎉")
    print("To run the demo again, use: python demo_seal.py")

if __name__ == "__main__":
    main()
