"""
filename: inference.py
"""

import logging
import os
import random
import time

import mlflow
import mlflow.sklearn
import pandas as pd
import psutil
from dotenv import load_dotenv
from config import *
from flask import (
    Flask, 
	Response, 
	jsonify, 
	request
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# -----------------------------------------------------------------------------
# Konfigurasi Aplikasi
# -----------------------------------------------------------------------------
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv(
	PROJECT_DIR / '.env'
)

# -----------------------------------------------------------------------------
# Konfigurasi MLflow / DagsHub
# -----------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MLFLOW_MODEL_ARTIFACT_PATH = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "random_forest_model").strip()

MODEL = None
MODEL_URI = None
MODEL_SOURCE = "simulation"

SERVER_START_EPOCH = time.time()

# -----------------------------------------------------------------------------
# Metrik HTTP
# -----------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests yang diterima oleh service",
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Durasi pemrosesan HTTP request dalam detik",
)

THROUGHPUT = Counter(
    "http_requests_throughput",
    "Total jumlah request yang diproses",
)

IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Jumlah request yang sedang diproses",
)

PREDICTION_RESULT = Counter(
    "model_predictions_total",
    "Total hasil prediksi model berdasarkan status survival",
    ["survival_status"],
)

API_ERRORS = Counter(
    "http_requests_errors_total",
    "Total request yang gagal diproses",
)

# -----------------------------------------------------------------------------
# Metrik Sistem
# -----------------------------------------------------------------------------
CPU_USAGE = Gauge(
    "system_cpu_usage",
    "Persentase penggunaan CPU",
)

RAM_USAGE = Gauge(
    "system_ram_usage",
    "Persentase penggunaan RAM",
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Durasi waktu loading model dari MLflow artifact",
)

SERVER_START_TIME = Gauge(
    "model_server_start_timestamp_seconds",
    "Timestamp Unix saat server pertama kali dijalankan",
)

SERVER_START_TIME.set(SERVER_START_EPOCH)

REQUIRED_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]


def load_latest_model():
    """
    Load model dari latest FINISHED run pada MLflow.
    Jika gagal, service tetap berjalan dalam mode simulasi.
    """
    global MODEL, MODEL_URI, MODEL_SOURCE

    if not MLFLOW_TRACKING_URI:
        logger.warning("MLFLOW_TRACKING_URI belum diset. Service berjalan dalam mode simulasi.")
        MODEL = None
        MODEL_URI = None
        MODEL_SOURCE = "simulation"
        MODEL_LOAD_TIME.set(0.0)
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        experiment_ids = [
            exp.experiment_id for exp in mlflow.search_experiments()
        ]

        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            logger.warning("Tidak ada run FINISHED. Service berjalan dalam mode simulasi.")
            MODEL = None
            MODEL_URI = None
            MODEL_SOURCE = "simulation"
            MODEL_LOAD_TIME.set(0.0)
            return

        latest_run_id = str(runs.iloc[0]["run_id"])
        candidate_uri = f"runs:/{latest_run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}"

        start = time.time()
        MODEL = mlflow.sklearn.load_model(candidate_uri)
        elapsed = time.time() - start

        MODEL_URI = candidate_uri
        MODEL_SOURCE = "mlflow-latest"
        MODEL_LOAD_TIME.set(elapsed)

        logger.info("Model berhasil dimuat dari %s", MODEL_URI)
        logger.info("Waktu loading model: %.3f detik", elapsed)

    except Exception as exc:
        logger.exception("Gagal memuat model latest dari MLflow: %s", exc)
        MODEL = None
        MODEL_SOURCE = "simulation"
        MODEL_LOAD_TIME.set(0.0)

def extract_features(payload: dict) -> pd.DataFrame:
    """
    Validasi payload dan ubah menjadi DataFrame satu baris.
    """
    missing = [field for field in REQUIRED_FEATURES if field not in payload]
    if missing:
        raise ValueError(f"Field yang hilang: {', '.join(missing)}")

    row = {field: payload[field] for field in REQUIRED_FEATURES}
    return pd.DataFrame([row], columns=REQUIRED_FEATURES)


# Load model saat startup module
load_latest_model()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "Titanic ML Inference Service",
            "model_source": MODEL_SOURCE,
            "model_uri": MODEL_URI,
            "endpoints": {
                "predict": "/predict",
                "health": "/health",
                "metrics": "/metrics",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": MODEL is not None,
            "model_source": MODEL_SOURCE,
            "model_uri": MODEL_URI,
            "uptime_seconds": round(time.time() - SERVER_START_EPOCH, 2),
        }
    ), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Endpoint untuk Prometheus scraping.
    """
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    RAM_USAGE.set(psutil.virtual_memory().percent)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint prediksi. Jika model latest berhasil diload,
    gunakan model tersebut. Jika tidak, fallback ke simulasi.
    """
    IN_PROGRESS.inc()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    start_time = time.time()

    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            API_ERRORS.inc()
            return jsonify({"error": "Request body harus berformat JSON object"}), 400

        features = extract_features(payload)

        if MODEL is not None:
            prediction = int(MODEL.predict(features)[0])

            if hasattr(MODEL, "predict_proba"):
                confidence = float(max(MODEL.predict_proba(features)[0]))
            else:
                confidence = 1.0

            mode = "mlflow-latest"
        else:
            prediction = random.randint(0, 1)
            confidence = round(random.uniform(0.55, 0.98), 4)
            mode = "simulation"

        label = "Survived" if prediction == 1 else "Not Survived"

        PREDICTION_RESULT.labels(
            survival_status="survived" if prediction == 1 else "not_survived"
        ).inc()

        return jsonify(
            {
                "survived": prediction,
                "confidence": confidence,
                "label": label,
                "mode": mode,
                "model_uri": MODEL_URI,
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            }
        ), 200

    except ValueError as exc:
        API_ERRORS.inc()
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        API_ERRORS.inc()
        logger.exception("Terjadi kesalahan tak terduga: %s", exc)
        return jsonify({"error": "Terjadi kesalahan internal server"}), 500

    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        IN_PROGRESS.dec()


if __name__ == "__main__":
    logger.info("Menjalankan Titanic ML Inference Service di %s:%s", INFERENCE_HOST, INFERENCE_PORT)
    logger.info("Interval konfigurasi: %s ms", METRIC_INTERVAL_MS)
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI or "(not set)")
    logger.info("Model artifact path: %s", MLFLOW_MODEL_ARTIFACT_PATH)

    app.run(
        host=INFERENCE_HOST,
        port=INFERENCE_PORT,
        debug=False,
    )