"""
filename: model.py
"""

from config import *
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def main():
    # 1. Autentikasi DagsHub (token-based, no OAuth prompt)
    if DAGSHUB_TOKEN:
        os.environ['DAGSHUB_USER_TOKEN'] = DAGSHUB_TOKEN

    # 2. Muat dataset
    print('[INFO] Memuat dataset...')
    if not DATASET_PATH.exists():
        print(f'[ERROR] File tidak ditemukan di {DATASET_PATH}')
        return
    df = pd.read_csv(DATASET_PATH)

    # 3. Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['Survived'])
    y = df['Survived']

	# 4. Split data (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Inisialisasi koneksi ke DagsHub + MLflow
    print('[INFO] Menghubungkan ke DagsHub...')
    dagshub.init(
        repo_owner=DAGSHUB_USER, 
        repo_name=DAGSHUB_REPO, 
        mlflow=True
    )

	# 6. Setup MLflow
    mlflow.set_experiment(PROJECT_NAME)

	# 7. Training & logging
    print('[INFO] Memulai proses training model...')
    with mlflow.start_run(run_name='RandomForest_Basic'):
        n_estimators = 100
        max_depth    = 5

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(
            f'[INFO] Akurasi: {accuracy:.4f} | '
            f'Presisi: {precision:.4f} | '
            f'Recall: {recall:.4f}'
        )

        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.sklearn.log_model(
            model, 
            'random_forest_model'
        )

        print('[INFO] Model dan metrik berhasil dicatat di MLflow')


if __name__ == '__main__':
    main()