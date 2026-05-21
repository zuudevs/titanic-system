"""
filename: inference.py

Production Inference Service.
- Load model terbaru (latest FINISHED run) dari DagsHub/MLflow
- Serve prediksi via /predict endpoint
- Expose Prometheus metrics via /metrics endpoint
- Fallback ke mode simulasi jika model gagal di-load
"""

import logging
import random
import time

import mlflow
import mlflow.sklearn
import pandas as pd
import psutil
from config import (
    DAGSHUB_TOKEN,
    INFERENCE_HOST,
    INFERENCE_PORT,
    MLFLOW_MODEL_ARTIFACT_PATH,
    MLFLOW_TRACKING_URI,
)
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model State
# -----------------------------------------------------------------------------
MODEL = None
MODEL_URI = None
MODEL_SOURCE = "simulation"
LATEST_RUN_ID = None

SERVER_START_EPOCH = time.time()

# -----------------------------------------------------------------------------
# Prometheus Metrics — prefix: inference_
# -----------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "inference_http_requests_total",
    "Total HTTP requests yang diterima oleh inference service",
)

REQUEST_LATENCY = Histogram(
    "inference_http_request_duration_seconds",
    "Durasi pemrosesan HTTP request dalam detik",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

THROUGHPUT = Counter(
    "inference_http_requests_throughput",
    "Total jumlah request yang diproses oleh inference",
)

IN_PROGRESS = Gauge(
    "inference_http_requests_in_progress",
    "Jumlah request yang sedang diproses oleh inference",
)

PREDICTION_RESULT = Counter(
    "inference_model_predictions_total",
    "Total hasil prediksi model berdasarkan status survival",
    ["survival_status"],
)

API_ERRORS = Counter(
    "inference_http_requests_errors_total",
    "Total request yang gagal diproses oleh inference",
)

# --- System Metrics ---
CPU_USAGE = Gauge(
    "inference_system_cpu_usage_percent",
    "Persentase penggunaan CPU pada inference service",
)

RAM_USAGE = Gauge(
    "inference_system_ram_usage_percent",
    "Persentase penggunaan RAM pada inference service",
)

MODEL_LOAD_TIME = Gauge(
    "inference_model_load_time_seconds",
    "Durasi waktu loading model dari MLflow artifact",
)

MODEL_LOADED = Gauge(
    "inference_model_loaded",
    "1 jika model production berhasil di-load, 0 jika mode simulasi",
)

SERVER_START_TIME = Gauge(
    "inference_server_start_timestamp_seconds",
    "Timestamp Unix saat server pertama kali dijalankan",
)

SERVER_START_TIME.set(SERVER_START_EPOCH)

# -----------------------------------------------------------------------------
# Required Features
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def load_latest_model():
    """
    Load model dari latest FINISHED run pada MLflow/DagsHub.
    Flow: search experiments → cari latest FINISHED run → load model.
    Jika gagal, service tetap berjalan dalam mode simulasi.
    """
    global MODEL, MODEL_URI, MODEL_SOURCE, LATEST_RUN_ID

    if not MLFLOW_TRACKING_URI:
        logger.warning(
            "MLFLOW_TRACKING_URI belum diset. "
            "Service berjalan dalam mode simulasi."
        )
        MODEL = None
        MODEL_URI = None
        MODEL_SOURCE = "simulation"
        LATEST_RUN_ID = None
        MODEL_LOAD_TIME.set(0.0)
        MODEL_LOADED.set(0)
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Cari semua experiment
        experiment_ids = [
            exp.experiment_id for exp in mlflow.search_experiments()
        ]

        if not experiment_ids:
            logger.warning(
                "Tidak ada experiment ditemukan. "
                "Service berjalan dalam mode simulasi."
            )
            MODEL = None
            MODEL_URI = None
            MODEL_SOURCE = "simulation"
            LATEST_RUN_ID = None
            MODEL_LOAD_TIME.set(0.0)
            MODEL_LOADED.set(0)
            return

        # Cari latest FINISHED run
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            logger.warning(
                "Tidak ada run FINISHED. "
                "Service berjalan dalam mode simulasi."
            )
            MODEL = None
            MODEL_URI = None
            MODEL_SOURCE = "simulation"
            LATEST_RUN_ID = None
            MODEL_LOAD_TIME.set(0.0)
            MODEL_LOADED.set(0)
            return

        # Load model dari latest run
        latest_run_id = str(runs.iloc[0]["run_id"])
        candidate_uri = f"runs:/{latest_run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}"

        logger.info("Loading model dari run_id: %s", latest_run_id)
        logger.info("Model URI: %s", candidate_uri)

        start = time.time()
        MODEL = mlflow.sklearn.load_model(candidate_uri)
        elapsed = time.time() - start

        MODEL_URI = candidate_uri
        MODEL_SOURCE = "mlflow-latest"
        LATEST_RUN_ID = latest_run_id
        MODEL_LOAD_TIME.set(elapsed)
        MODEL_LOADED.set(1)

        logger.info("Model berhasil dimuat dari %s", MODEL_URI)
        logger.info("Waktu loading model: %.3f detik", elapsed)

    except Exception as exc:
        logger.exception("Gagal memuat model latest dari MLflow: %s", exc)
        MODEL = None
        MODEL_SOURCE = "simulation"
        LATEST_RUN_ID = None
        MODEL_LOAD_TIME.set(0.0)
        MODEL_LOADED.set(0)


def extract_features(payload: dict) -> pd.DataFrame:
    """
    Validasi payload dan ubah menjadi DataFrame satu baris.
    """
    missing = [field for field in REQUIRED_FEATURES if field not in payload]
    if missing:
        raise ValueError(f"Field yang hilang: {', '.join(missing)}")

    row = {field: payload[field] for field in REQUIRED_FEATURES}
    return pd.DataFrame([row], columns=REQUIRED_FEATURES)


# Load model saat startup
load_latest_model()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "Titanic ML Inference Service",
            "role": "production",
            "model_source": MODEL_SOURCE,
            "model_uri": MODEL_URI,
            "latest_run_id": LATEST_RUN_ID,
            "endpoints": {
                "predict": "/predict",
                "health": "/health",
                "metrics": "/metrics",
                "reload": "/reload (POST)",
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
            "latest_run_id": LATEST_RUN_ID,
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


@app.route("/reload", methods=["POST"])
def reload_model():
    """
    Hot-reload model dari MLflow tanpa restart server.
    """
    logger.info("Menerima request reload model...")
    load_latest_model()
    return jsonify(
        {
            "status": "reloaded",
            "model_source": MODEL_SOURCE,
            "model_uri": MODEL_URI,
            "latest_run_id": LATEST_RUN_ID,
        }
    ), 200


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
            return jsonify(
                {"error": "Request body harus berformat JSON object"}
            ), 400

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
                "latest_run_id": LATEST_RUN_ID,
                "response_time_ms": round(
                    (time.time() - start_time) * 1000, 2
                ),
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
    logger.info(
        "Menjalankan Titanic ML Inference Service di %s:%s",
        INFERENCE_HOST,
        INFERENCE_PORT,
    )
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI or "(not set)")
    logger.info("Model artifact path: %s", MLFLOW_MODEL_ARTIFACT_PATH)
    logger.info("Model source: %s", MODEL_SOURCE)
    logger.info("Latest run ID: %s", LATEST_RUN_ID or "(none)")

    app.run(
        host=INFERENCE_HOST,
        port=INFERENCE_PORT,
        debug=False,
    )