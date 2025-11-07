#!/usr/bin/env python3
"""
SEAL Architecture Test Script

This script tests the SEAL architecture by running a complete self-adaptive loop.
"""

from seal.runner import run_seal_loop

if __name__ == "__main__":
    print("🚀 Testing SEAL Architecture")
    print("=" * 50)
    
    try:
        run_seal_loop("configs/default.yaml")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise
    
    print("\n✅ SEAL architecture test completed successfully!")
