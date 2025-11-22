#!/usr/bin/env python3
"""
Main entry point for the SEAL Lightweight Continual Learning Framework.
"""

import argparse
import os

from seal.runner import run_seal_loop, run_imdb_comparison, run_sequential_tasks

# Optional demo import if exists
try:
    from seal.demo_seal import main as run_demo
except ImportError:
    run_demo = None


def main():
    parser = argparse.ArgumentParser(
        description="SEAL: Lightweight Continual Learning Framework (Inspired by SEAL)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="seal",
        choices=["seal", "imdb", "demo", "tasks"],
        help=(
            "Select execution mode:\n"
            "  seal  - Run SEAL continual learning loop\n"
            "  imdb  - Run IMDB local vs LLM comparison\n"
            "  demo  - Run 3-sample demonstration\n"
            "  tasks - Multi-task continual learning (future)\n"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    print("\n🔧 Using configuration:", args.config)
    print("🔧 Selected mode:", args.mode)

    # MAIN EXECUTION LOGIC
    if args.mode == "seal":
        run_seal_loop(args.config)

    elif args.mode == "imdb":
        run_imdb_comparison(args.config)

    elif args.mode == "demo":
        if run_demo is not None:
            run_demo()
        else:
            print("⚠️ demo_seal.py not found — demo mode unavailable.")

    elif args.mode == "tasks":
        print("\n🚀 Starting multi-task continual learning...")
        try:
            run_sequential_tasks(args.config)
        except Exception as e:
            print(f"❌ Error in multi-task learning: {str(e)}")
            import traceback
            traceback.print_exc()

    else:
        print("❌ Unknown mode. Use --help for guidance.")


if __name__ == "__main__":
    main()
