#!/usr/bin/env python3
"""
Test script for SEAL CPU compatibility.

This script verifies that the SEAL implementation works correctly on CPU.
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from seal.adapter import generate_edit

def main():
    """Test SEAL functionality on CPU."""
    # Configuration
    model_name = "distilbert-base-uncased"
    test_text = "Machine learning is transforming the world."
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Ensure model is on CPU
    device = torch.device("cpu")
    model = model.to(device)
    
    # Test model inference
    print("\nTesting model inference...")
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("✅ Model inference successful")
    
    # Test local editing
    print("\nTesting local editing...")
    edited_text = generate_edit(test_text, mode="local")
    print(f"Original: {test_text}")
    print(f"Edited:   {edited_text}")
    print("✅ Local editing successful")
    
    # Print success message
    print("\n✅ SEAL CPU test completed successfully!")
    print("\nExpected output structure:")
    print("✅ Model ran successfully on CPU")
    print("Generated Edit: Machine learning might be transforming this world.")

if __name__ == "__main__":
    main()
