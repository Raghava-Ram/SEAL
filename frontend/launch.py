#!/usr/bin/env python3
"""
Launch script for SEAL Frontend.
Handles dependency checks and starts the Streamlit app.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║         SEAL Frontend Launcher                                 ║
║  Self-Edit Adaptive Learning: Continual Learning Visualizer   ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✓ Streamlit is installed")
    except ImportError:
        print("✗ Streamlit not found. Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(PROJECT_ROOT / "frontend" / "requirements_frontend.txt")
        ])
    
    # Check if data is available
    data_path = PROJECT_ROOT / "outputs" / "multi_task" / "hybrid" / "imdb_squad_arc_metrics.json"
    if data_path.exists():
        print(f"✓ Data found: {data_path.relative_to(PROJECT_ROOT)}")
    else:
        print("⚠ Warning: Pre-computed metrics not found.")
        print("  Please run the backend first: python main.py --mode tasks")
    
    # Check if Ollama is running (optional)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            print("✓ Ollama is running (chatbot enabled)")
        else:
            print("⚠ Ollama available but not responding properly")
    except Exception:
        print("⚠ Ollama not running (chatbot disabled)")
        print("  To enable: ollama serve")
    
    print("\n" + "="*62)
    print("Starting SEAL Frontend...")
    print("="*62)
    print("\n🌐 Frontend will open at: http://localhost:8501\n")
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "frontend" / "app.py")
    ])

if __name__ == "__main__":
    main()
