"""
filename: exporter.py

Simulation Exporter Service.
- Background thread otomatis mengirim simulated passenger data ke inference
- Expose Prometheus metrics (prefix: exporter_) via /metrics endpoint
- Control simulasi via /simulate/start dan /simulate/stop
- Digunakan untuk testing & demo monitoring pipeline
"""

import logging
import random
import threading
import time

import psutil
import requests
from config import (
    EXPORTER_HOST,
    EXPORTER_PORT,
    INFERENCE_API_URL,
    SIMULATION_INTERVAL_SECONDS,
)
from flask import Flask, Response, jsonify
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
# Prometheus Metrics — prefix: exporter_
# -----------------------------------------------------------------------------
SIMULATION_REQUESTS_TOTAL = Counter(
    "exporter_simulation_requests_total",
    "Total simulated requests yang dikirim ke inference",
)

SIMULATION_SUCCESS = Counter(
    "exporter_simulation_success_total",
    "Total simulated requests yang berhasil",
)

SIMULATION_ERRORS = Counter(
    "exporter_simulation_errors_total",
    "Total simulated requests yang gagal",
)

SIMULATION_LATENCY = Histogram(
    "exporter_simulation_latency_seconds",
    "Durasi round-trip simulated request ke inference",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

SIMULATION_PREDICTION = Counter(
    "exporter_simulation_predictions_total",
    "Distribusi hasil prediksi dari simulation",
    ["survival_status"],
)

SIMULATION_ACTIVE = Gauge(
    "exporter_simulation_active",
    "1 jika simulation loop sedang berjalan, 0 jika tidak",
)

# --- System Metrics ---
CPU_USAGE = Gauge(
    "exporter_system_cpu_usage_percent",
    "Persentase penggunaan CPU pada exporter service",
)

RAM_USAGE = Gauge(
    "exporter_system_ram_usage_percent",
    "Persentase penggunaan RAM pada exporter service",
)

# --- Inference Response Metrics (from proxy perspective) ---
INFERENCE_RESPONSE_STATUS = Counter(
    "exporter_inference_response_status_total",
    "HTTP status codes dari inference responses",
    ["status_code"],
)

INFERENCE_MODE = Gauge(
    "exporter_inference_mode",
    "Mode inference: 1=mlflow-latest (production), 0=simulation",
)

# -----------------------------------------------------------------------------
# Simulation State
# -----------------------------------------------------------------------------
_simulation_running = False
_simulation_thread = None
_simulation_lock = threading.Lock()


def generate_random_passenger() -> dict:
    """
    Generate random passenger data untuk simulasi.
    """
    return {
        "Pclass": random.choice([1, 2, 3]),
        "Age": round(random.uniform(1, 80), 1),
        "SibSp": random.randint(0, 5),
        "Parch": random.randint(0, 4),
        "Fare": round(random.uniform(5, 500), 2),
        "Sex_male": random.choice([0, 1]),
        "Embarked_Q": random.choice([0, 1]),
        "Embarked_S": random.choice([0, 1]),
    }


def simulation_loop():
    """
    Background loop yang mengirim simulated requests ke inference service.
    """
    global _simulation_running

    logger.info(
        "Simulation loop dimulai (interval: %ss)", SIMULATION_INTERVAL_SECONDS
    )

    while _simulation_running:
        passenger = generate_random_passenger()

        SIMULATION_REQUESTS_TOTAL.inc()
        start = time.time()

        try:
            resp = requests.post(
                INFERENCE_API_URL, json=passenger, timeout=10
            )
            elapsed = time.time() - start
            SIMULATION_LATENCY.observe(elapsed)

            INFERENCE_RESPONSE_STATUS.labels(
                status_code=str(resp.status_code)
            ).inc()

            if resp.status_code == 200:
                SIMULATION_SUCCESS.inc()

                try:
                    result = resp.json()
                    prediction = result.get("survived", -1)
                    mode = result.get("mode", "unknown")

                    SIMULATION_PREDICTION.labels(
                        survival_status=(
                            "survived" if prediction == 1 else "not_survived"
                        )
                    ).inc()

                    INFERENCE_MODE.set(
                        1 if mode == "mlflow-latest" else 0
                    )

                    logger.info(
                        "Simulation → %s | survived=%s | confidence=%.4f | "
                        "latency=%.3fs",
                        mode,
                        prediction,
                        result.get("confidence", 0),
                        elapsed,
                    )
                except ValueError:
                    logger.warning("Response dari inference tidak valid JSON")
            else:
                SIMULATION_ERRORS.inc()
                logger.warning(
                    "Inference returned status %s", resp.status_code
                )

        except requests.RequestException as exc:
            elapsed = time.time() - start
            SIMULATION_LATENCY.observe(elapsed)
            SIMULATION_ERRORS.inc()
            logger.error("Gagal menghubungi inference: %s", exc)

        time.sleep(SIMULATION_INTERVAL_SECONDS)

    logger.info("Simulation loop dihentikan")


def start_simulation():
    """Start simulation background thread."""
    global _simulation_running, _simulation_thread

    with _simulation_lock:
        if _simulation_running:
            return False  # already running

        _simulation_running = True
        SIMULATION_ACTIVE.set(1)
        _simulation_thread = threading.Thread(
            target=simulation_loop, daemon=True
        )
        _simulation_thread.start()
        return True


def stop_simulation():
    """Stop simulation background thread."""
    global _simulation_running, _simulation_thread

    with _simulation_lock:
        if not _simulation_running:
            return False  # already stopped

        _simulation_running = False
        SIMULATION_ACTIVE.set(0)

        if _simulation_thread:
            _simulation_thread.join(timeout=SIMULATION_INTERVAL_SECONDS + 2)
            _simulation_thread = None

        return True


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "Titanic ML Simulation Exporter",
            "role": "simulation",
            "simulation_active": _simulation_running,
            "inference_target": INFERENCE_API_URL,
            "interval_seconds": SIMULATION_INTERVAL_SECONDS,
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "simulate_start": "/simulate/start (POST)",
                "simulate_stop": "/simulate/stop (POST)",
                "simulate_status": "/simulate/status",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "Prometheus Simulation Exporter",
            "simulation_active": _simulation_running,
            "inference_target": INFERENCE_API_URL,
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


@app.route("/simulate/start", methods=["POST"])
def simulate_start():
    """
    Mulai simulation loop.
    """
    started = start_simulation()
    if started:
        logger.info("Simulation loop dimulai via API")
        return jsonify(
            {"status": "started", "interval": SIMULATION_INTERVAL_SECONDS}
        ), 200
    else:
        return jsonify({"status": "already_running"}), 200


@app.route("/simulate/stop", methods=["POST"])
def simulate_stop():
    """
    Hentikan simulation loop.
    """
    stopped = stop_simulation()
    if stopped:
        logger.info("Simulation loop dihentikan via API")
        return jsonify({"status": "stopped"}), 200
    else:
        return jsonify({"status": "already_stopped"}), 200


@app.route("/simulate/status", methods=["GET"])
def simulate_status():
    return jsonify(
        {
            "simulation_active": _simulation_running,
            "interval_seconds": SIMULATION_INTERVAL_SECONDS,
            "inference_target": INFERENCE_API_URL,
        }
    )


if __name__ == "__main__":
    logger.info(
        "Menjalankan Simulation Exporter di %s:%s",
        EXPORTER_HOST,
        EXPORTER_PORT,
    )
    logger.info("Inference target: %s", INFERENCE_API_URL)
    logger.info("Simulation interval: %ss", SIMULATION_INTERVAL_SECONDS)

    # Auto-start simulation saat service dijalankan
    start_simulation()

    app.run(
        host=EXPORTER_HOST,
        port=EXPORTER_PORT,
        debug=False,
    )