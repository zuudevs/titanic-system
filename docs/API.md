# API

Referensi lengkap REST API untuk layanan Inference dan Exporter.

## Inference Service (Port: 8001)

### `POST /predict`
Mengirim data fitur penumpang untuk mendapatkan prediksi *Survived* atau *Not Survived*.
* **Request JSON Payload:**
    ```json
    {
      "Pclass": 3,
      "Age": 22.0,
      "SibSp": 1,
      "Parch": 0,
      "Fare": 7.25,
      "Sex_male": 1,
      "Embarked_Q": 0,
      "Embarked_S": 1
    }
    ```
* **Response (200 OK):**
    ```json
    {
      "survived": 0,
      "confidence": 0.8452,
      "label": "Not Survived",
      "mode": "mlflow-latest",
      "model_uri": "runs:/<RUN_ID>/random_forest_tuning_model",
      "response_time_ms": 15.4
    }
    ```

### `POST /reload`
Melakukan *Hot-Reload*. Memaksa server untuk mencari dan mengunduh model status `FINISHED` terbaru dari MLflow tanpa me-restart server.
* **Response (200 OK):** Mengembalikan status "reloaded" dan informasi model terbaru.

### `GET /health` & `GET /metrics`
* `/health`: Mengecek status model (loaded atau simulation).
* `/metrics`: Endpoint *scraping* metrik untuk Prometheus.

---

## Exporter Service (Port: 8000)

### `POST /simulate/start`
Memulai *background loop* untuk meng-generate data penumpang acak dan melakukan hit ke endpoint `/predict` secara periodik.

### `POST /simulate/stop`
Menghentikan *background loop* simulasi.

### `GET /simulate/status`
Melihat status simulasi saat ini (Aktif/Tidak) dan target URL inference.