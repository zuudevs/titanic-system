"""
filename: exporter.py
"""

import logging
import time

import psutil
import requests
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
# App Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# HTTP Metrics
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

API_ERRORS = Counter(
    "http_requests_errors_total",
    "Total request yang gagal diproses",
)

# -----------------------------------------------------------------------------
# System Metrics
# -----------------------------------------------------------------------------
CPU_USAGE = Gauge(
    "system_cpu_usage",
    "Persentase penggunaan CPU",
)

RAM_USAGE = Gauge(
    "system_ram_usage",
    "Persentase penggunaan RAM",
)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "Prometheus Exporter",
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
    Endpoint simulasi request ke MLflow inference server.
    """
    IN_PROGRESS.inc()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    start_time = time.time()

    try:
        payload = request.get_json(silent=True)
        if payload is None:
            API_ERRORS.inc()
            return jsonify({"error": "Request body harus berformat JSON"}), 400

        response = requests.post(MLFLOW_API_URL, json=payload, timeout=30)
        response.raise_for_status()

        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)

        try:
            result = response.json()
        except ValueError:
            API_ERRORS.inc()
            return jsonify({"error": "Respons dari MLflow tidak valid"}), 502

        return jsonify(result), response.status_code

    except requests.RequestException as exc:
        API_ERRORS.inc()
        logger.exception("Gagal menghubungi API MLflow: %s", exc)
        return jsonify({"error": "Gagal menghubungi API MLflow"}), 502

    except Exception as exc:
        API_ERRORS.inc()
        logger.exception("Terjadi kesalahan tak terduga: %s", exc)
        return jsonify({"error": "Terjadi kesalahan internal server"}), 500

    finally:
        IN_PROGRESS.dec()


if __name__ == "__main__":
    logger.info("Menjalankan Prometheus Exporter di %s:%s", EXPORTER_HOST, EXPORTER_PORT)
    logger.info("Interval konfigurasi: %s ms", METRIC_INTERVAL_MS)
    logger.info("MLflow inference endpoint: %s", MLFLOW_API_URL)

    app.run(
        host=EXPORTER_HOST,
        port=EXPORTER_PORT,
        debug=False,
    )