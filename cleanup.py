"""
Cleanup script for SEAL project.
Removes temporary and unnecessary files to keep the repository clean.
"""

import os
import shutil
from pathlib import Path

def remove_files(file_patterns, root_dir='.'):
    """Remove files matching patterns in the given directory."""
    root = Path(root_dir)
    for pattern in file_patterns:
        for file_path in root.rglob(pattern):
            try:
                file_path.unlink()
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def remove_dirs(dir_patterns, root_dir='.'):
    """Remove directories matching patterns in the given directory."""
    root = Path(root_dir)
    for pattern in dir_patterns:
        for dir_path in root.rglob(pattern):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")

def main():
    # Files to remove
    file_patterns = [
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '*.so',
        '*.orig',
        '*.bak',
        '*.swp',
        '*.swo',
        '*~',
        '*.log',
        '.coverage',
        '.pytest_cache',
        'coverage.xml',
        '*.egg-info',
        '.DS_Store',
        'Thumbs.db'
    ]

    # Directories to remove
    dir_patterns = [
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.hypothesis',
        'htmlcov',
        '*.egg-info',
        'dist',
        'build',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '*.so',
        '*.orig',
        '*.bak',
        '*.swp',
        '*.swo',
        '*~',
        '*.log',
        '.coverage',
        '.pytest_cache',
        'coverage.xml',
        '*.egg-info',
        '.DS_Store',
        'Thumbs.db'
    ]

    # Remove files and directories
    remove_files(file_patterns)
    remove_dirs(dir_patterns)

    print("\nCleanup complete!")

if __name__ == "__main__":
    main()
