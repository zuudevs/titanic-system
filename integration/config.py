"""
filename: config.py

Konfigurasi terpusat untuk integration layer.
- Lokal: load dari .env file
- CI/CD (GitHub Actions): env vars sudah di-inject lewat secrets, skip .env
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
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

# -----------------------------------------------------------------------------
# DagsHub / MLflow Configuration
# -----------------------------------------------------------------------------
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "").strip()
DAGSHUB_USER = os.getenv("DAGSHUB_USER", "").strip()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic-system").strip()
MLFLOW_MODEL_ARTIFACT_PATH = os.getenv(
    "MLFLOW_MODEL_ARTIFACT_PATH", "random_forest_tuning_model"
).strip()

# -----------------------------------------------------------------------------
# Base Network Configuration
# -----------------------------------------------------------------------------
HOST = "127.0.0.1"

# -----------------------------------------------------------------------------
# Exporter Service (Simulation)
# -----------------------------------------------------------------------------
EXPORTER_HOST = HOST
EXPORTER_PORT = 8000

# -----------------------------------------------------------------------------
# Inference Service (Production)
# -----------------------------------------------------------------------------
INFERENCE_HOST = HOST
INFERENCE_PORT = 8001

# -----------------------------------------------------------------------------
# Inference API Endpoint (exporter → inference)
# -----------------------------------------------------------------------------
INFERENCE_API_URL = f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/predict"

# -----------------------------------------------------------------------------
# Simulation Configuration
# -----------------------------------------------------------------------------
SIMULATION_INTERVAL_SECONDS = 2  # interval antara simulated requests