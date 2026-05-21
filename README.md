# Titanic ML System

Titanic ML System adalah sebuah *pipeline* MLOps *end-to-end* yang dirancang untuk melatih, melacak, dan menyajikan (*serve*) model *machine learning* (Random Forest) untuk memprediksi kelangsungan hidup penumpang Titanic. 

Sistem ini dibangun dengan pondasi arsitektur yang solid, mencakup *data preprocessing*, *model tracking* (MLflow/DagsHub), *Inference API* (Flask), generator *traffic* simulasi, serta sistem *monitoring* berkelanjutan menggunakan Prometheus dan Grafana. Seluruh proses pengujian dan *deployment* juga telah diotomatisasi melalui CI/CD GitHub Actions.