from dotenv import load_dotenv
from pathlib import Path
import os

PROJECT_NAME = 'Titanic System'
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "dataset/clean.csv"
ENV_PATH = PROJECT_DIR / '.env'

if ENV_PATH.exists:
	load_dotenv(ENV_PATH)

DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', '')
DAGSHUB_USER = os.getenv('DAGSHUB_USER', '')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')