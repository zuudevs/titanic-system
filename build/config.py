"""
filename: config.py

Konfigurasi untuk build layer (model training & tuning).
- Lokal: load dari .env file
- CI/CD (GitHub Actions): env vars sudah di-inject lewat secrets, skip .env
"""

import os
from pathlib import Path

PROJECT_NAME = "Titanic System"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "dataset/clean.csv"
ENV_PATH = PROJECT_DIR / ".env"

# -----------------------------------------------------------------------------
# Smart .env Loading
# GitHub Actions set CI=true secara otomatis.
# Jika CI=true, skip .env dan gunakan env vars dari secrets.
# Jika tidak, coba load dari .env file (lokal development).
# -----------------------------------------------------------------------------
_is_ci = os.getenv("CI", "").lower() == "true"

if not _is_ci and ENV_PATH.exists():
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)

DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "").strip()
DAGSHUB_USER = os.getenv("DAGSHUB_USER", "").strip()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()