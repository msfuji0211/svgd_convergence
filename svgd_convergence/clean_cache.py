#!/usr/bin/env python3
"""
Script to clean up __pycache__ directories and .pyc files
"""

import os
import shutil
import glob

def clean_pycache():
    """Remove all __pycache__ directories and .pyc files"""
    print("Cleaning up Python cache files...")
    
    # Find and remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"Removed: {cache_path}")
                except Exception as e:
                    print(f"Error removing {cache_path}: {e}")
    
    # Find and remove .pyc files
    pyc_files = glob.glob('**/*.pyc', recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            print(f"Removed: {pyc_file}")
        except Exception as e:
            print(f"Error removing {pyc_file}: {e}")
    
    # Find and remove .pyo files
    pyo_files = glob.glob('**/*.pyo', recursive=True)
    for pyo_file in pyo_files:
        try:
            os.remove(pyo_file)
            print(f"Removed: {pyo_file}")
        except Exception as e:
            print(f"Error removing {pyo_file}: {e}")
    
    print("Cache cleanup completed!")

if __name__ == '__main__':
    clean_pycache() 