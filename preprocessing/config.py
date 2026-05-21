"""
filename: config.py
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_URL = 'https://github.com/datasciencedojo/datasets/raw/master/titanic.csv'
DATASET_DIR = PROJECT_DIR / 'dataset'