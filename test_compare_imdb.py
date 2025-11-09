#!/usr/bin/env python
"""
SEAL IMDB Comparison Test Script

This script runs the SEAL framework on the IMDB dataset, comparing local and OpenAI edit modes.
"""

import os
import argparse
from dotenv import load_dotenv
from seal.runner import run_imdb_comparison

# Load environment variables from .env file
load_dotenv()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run SEAL IMDB comparison')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Directory to save outputs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Starting SEAL IMDB Comparison Test")
    print(f"📂 Configuration: {args.config}")
    print(f"💾 Output directory: {os.path.abspath(args.output_dir)}")
    
    # Run the comparison
    try:
        run_imdb_comparison(config_path=args.config)
        print("\n✅ SEAL IMDB comparison completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during SEAL IMDB comparison: {str(e)}")
        raise

if __name__ == "__main__":
    main()
