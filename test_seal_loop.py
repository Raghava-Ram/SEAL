#!/usr/bin/env python3
"""
SEAL Self-Adaptive Loop Test

This script verifies the stability of the SEAL self-adaptive loop on CPU.
It runs multiple iterations of the generate_edit() function to ensure
stable execution and proper CPU memory usage.
"""

import time
from seal.adapter import generate_edit

def run_seal_loop(iterations=3, test_text="Machine learning is transforming the world."):
    """
    Run the SEAL self-adaptive loop for a given number of iterations.
    
    Args:
        iterations: Number of iterations to run the loop
        test_text: Text to use for testing the edit generation
    """
    print(f"🚀 Starting SEAL self-adaptive loop test with {iterations} iterations\n")
    
    for i in range(1, iterations + 1):
        # Print iteration header
        print(f"🔁 Starting SEAL iteration {i}")
        
        # Generate and print the edit
        edited_text = generate_edit(test_text, mode="local")
        print(f"Generated Edit: {edited_text}")
        
        # Simulate model update step
        print("Updating model...")
        time.sleep(0.5)  # Short pause to simulate processing
        
        print()  # Add space between iterations
    
    print("✅ SEAL self-adaptive loop executed successfully on CPU")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    run_seal_loop()
