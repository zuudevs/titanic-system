# Arsitektur Sistem
Sistem ini memecah kompleksitas menjadi 3 layer fundamental yang saling terisolasi namun terintegrasi:

## 1. Data & Build Layer (`/preprocessing` & `/build`)
* **Data Pipeline (`pipeline.py`)**: Bertugas memuat data mentah (`raw.csv`), melakukan pembersihan data (imputasi *missing values* pada `Age` dan `Embarked`), dan *One-Hot Encoding*. Menghasilkan `clean.csv`.
* **Model Training (`model.py` & `tuning.py`)**: Melatih model klasifikasi *Random Forest*. Modul `tuning.py` menerapkan *GridSearchCV* untuk optimasi *hyperparameter*. Model terbaik, metrik (akurasi, presisi, recall), dan parameter dicatat (*logged*) secara otomatis ke remote server **MLflow (DagsHub)**.

## 2. Integration & Serving Layer (`/integration`)
Layer ini adalah jantung dari aplikasi *production*:
* **Inference Service (`inference.py` - Port 8001)**: Sebuah REST API (Flask) yang secara dinamis mengunduh model dengan status *FINISHED* terbaru dari MLflow. Jika koneksi atau model gagal dimuat, sistem memiliki *fallback* untuk berjalan di **mode simulasi** sehingga layanan tidak *crash*. Menyediakan endpoint `/predict` dan metrik performa `/metrics`.
* **Exporter Service (`exporter.py` - Port 8000)**: Layanan pendamping yang berjalan di *background thread*. Fungsinya adalah mengirimkan *payload* data penumpang acak ke Inference Service. Ini sangat krusial untuk mensimulasikan *traffic real-time* agar metrik di Grafana bisa divisualisasikan tanpa menunggu *traffic* pengguna asli.

## 3. Monitoring & CI/CD Layer
* **Prometheus & Grafana**: Prometheus (Port 9090) melakukan *scraping* metrik (RPS, Latensi, Error Rate, CPU/RAM) dari Inference dan Exporter. Grafana (Port 3000) memvisualisasikan data ini ke dalam *dashboard* terpusat. Dilengkapi juga dengan `alert.rules.yml` untuk memicu peringatan jika sistem *down* atau latensi memburuk.
* **GitHub Actions**: Memastikan kualitas (*Linting* Flake8, Black) dan menguji fungsionalitas kode secara otomatis setiap kali ada *Push* atau *Pull Request* ke branch `main`.

## Flowchart Sistem

```mermaid
graph TD
    subgraph Layer 1: Data & Build Pipeline
        A[dataset/raw.csv] -->|Pembersihan & Encoding| B(preprocessing/pipeline.py)
        B --> C[dataset/clean.csv]
        C --> D(build/model.py & tuning.py)
        D -- Log Metrics & Register Model --> E[(MLflow Tracking Server / DagsHub)]
    end

    subgraph Layer 2: Integration & Serving
        E -.->|Load Latest FINISHED Model| F[Inference Service API:8001]
        F -->|Endpoint /predict| F_Pred(Prediksi Survived/Not)
        G[Exporter Service:8000] -- HTTP POST: Send Simulated Traffic --> F
        F_Pred -.->|Return JSON Response| G
    end

    subgraph Layer 3: Observability & Monitoring
        H((Prometheus:9090)) -- Scrape /metrics --> F
        H -- Scrape /metrics --> G
        I[Grafana Dashboard:3000] -- Query PromQL --> H
        J[AlertManager] -- Evaluate alert.rules.yml --> H
    end

    %% Styling
    classDef service fill:#f9f,stroke:#333,stroke-width:2px;
    classDef db fill:#bbf,stroke:#333,stroke-width:2px;
    class F,G service;
    class E,H,I db;
```

# Flowchart Interaksi API

Alur komunikasi antara User, Exporter, Inference, dan MLflow.

```mermaid
sequenceDiagram
    autonumber
    actor Admin
    participant E as Exporter Service (:8000)
    participant I as Inference Service (:8001)
    participant M as MLflow (DagsHub)

    %% Startup Process
    Note over I,M: Startup & Model Loading
    I->>M: HTTP GET: Search latest FINISHED run
    M-->>I: Return Run ID & Model URI
    I->>M: Download Model Artifacts
    M-->>I: Model Loaded into Memory

    %% Simulation Flow
    Note over Admin,I: Traffic Simulation Loop
    Admin->>E: POST /simulate/start
    E-->>Admin: 200 OK (Started)
    
    activate E
    loop Every 2 Seconds
        E->>E: Generate Random Features
        E->>I: POST /predict (Payload)
        
        activate I
        I->>I: Extract Features -> model.predict()
        I-->>E: 200 OK (Prediction Result)
        deactivate I
        
        E->>E: Record Metrics (Latency, Error, etc)
    end
    
    Admin->>E: POST /simulate/stop
    E-->>Admin: 200 OK (Stopped)
    deactivate E
```