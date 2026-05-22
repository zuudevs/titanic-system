@echo off
chcp 65001

echo ===============================================
echo   Titanic ML System - Local Startup
echo ===============================================
echo.

echo [1/3] Starting Inference Service (port 8001)...
start "Inference Service" cmd /k "cd /d %~dp0 && py -3.12 inference.py"

echo       Menunggu inference service siap...
timeout /t 5 /nobreak > nul

echo [2/3] Starting Exporter / Simulation (port 8000)...
start "Exporter Simulation" cmd /k "cd /d %~dp0 && py -3.12 exporter.py"

echo [3/3] Pastikan Prometheus dan Grafana sudah berjalan!
echo.
echo ===============================================
echo   Endpoints:
echo   Inference  : http://127.0.0.1:8001
echo   Exporter   : http://127.0.0.1:8000
echo   Prometheus : http://127.0.0.1:9090
echo   Grafana    : http://127.0.0.1:3000
echo ===============================================
echo.
echo   Cara menjalankan Prometheus:
echo   prometheus.exe --config.file=prometheus.yml
echo.
echo   Grafana: jalankan grafana-server dari folder instalasi
echo   Salin isi grafana/ ke folder provisioning Grafana
echo ===============================================
pause
