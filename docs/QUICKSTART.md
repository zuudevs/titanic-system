# QUICKSTART

Panduan cepat untuk menjalankan integrasi model (Inference & Simulation) secara lokal.

## 1. Persiapan Lingkungan
Pastikan Anda menggunakan Python 3.12+ dan telah membuat Virtual Environment (opsional namun disarankan).
```bash
# Clone repository
git clone <URL_REPO_ANDA>
cd titanic-system

# Salin template environment
cp .env.example .env
```
Isi variabel di file `.env` dengan kredensial DagsHub / MLflow Anda agar Inference Service bisa mengunduh model.

## 2. Instalasi Dependensi Integrasi
```bash
pip install -r integration/requirements.txt
```

## 3. Menjalankan Layanan
Buka dua terminal terpisah.

**Terminal 1 (Jalankan Inference Service - Port 8001):**
```bash
python integration/inference.py
```
*Catatan: Layanan ini akan mencoba terhubung ke MLflow. Jika gagal, akan otomatis masuk ke "simulation mode".*

**Terminal 2 (Jalankan Exporter/Simulation - Port 8000):**
```bash
python integration/exporter.py
```
*Catatan: Secara otomatis (auto-start), Exporter akan mengirim request prediksi ke port 8001 setiap 2 detik.*