#!/usr/bin/env python3
"""
Quick test script to verify frontend can load properly.
"""

import sys
from pathlib import Path

# Add SEAL root to path
seal_root = Path(__file__).parent.parent
sys.path.insert(0, str(seal_root))

try:
    # Test imports
    print("✓ Testing imports...")
    import streamlit
    import pandas
    import matplotlib.pyplot as plt
    import seaborn
    import requests
    print("  ✓ All Streamlit dependencies imported successfully")
    
    # Test frontend imports
    print("\n✓ Testing frontend modules...")
    from frontend import utils
    print("  ✓ frontend.utils imported successfully")
    
    # Test data loading
    print("\n✓ Testing data loading...")
    metrics = utils.load_json_safe(
        seal_root / "outputs" / "multi_task" / "hybrid" / "imdb_squad_arc_metrics.json"
    )
    if metrics:
        print(f"  ✓ Metrics loaded: {list(metrics.keys())}")
    else:
        print("  ⚠ Metrics file not found (expected if backend not run)")
    
    # Test Ollama connection
    print("\n✓ Testing Ollama connection...")
    is_running = utils.test_ollama_connection()
    if is_running:
        print("  ✓ Ollama is running")
    else:
        print("  ⚠ Ollama not running (optional, chatbot will be disabled)")
    
    # Test utility functions
    print("\n✓ Testing utility functions...")
    test_matrix = {
        "imdb": [0.95, 0.69, 0.82],
        "squad": [0.57, 0.43],
        "arc": [0.97]
    }
    
    # Test forgetting computation
    from frontend.app import compute_forgetting
    forgetting = compute_forgetting(test_matrix)
    print(f"  ✓ Forgetting computed: {forgetting}")
    
    # Test matrix formatting
    from frontend.app import format_matrix_as_table
    df = format_matrix_as_table(test_matrix)
    print(f"  ✓ Matrix formatted to DataFrame with shape {df.shape}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Frontend is ready!")
    print("="*60)
    print("\nTo run the frontend:")
    print("  streamlit run frontend/app.py")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r frontend/requirements_frontend.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    sys.exit(1)
