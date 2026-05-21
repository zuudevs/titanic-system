"""
filename: tuning.py
"""

from config import *
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
    
# 7. Hyperparameter tuning & logging
    print('[INFO] Memulai proses hyperparameter tuning...')

    with mlflow.start_run(run_name='RandomForest_GridSearch'):
        base_model = RandomForestClassifier(
            random_state=42
        )

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(
            f'[INFO] Akurasi: {accuracy:.4f} | '
            f'Presisi: {precision:.4f} | '
            f'Recall: {recall:.4f}'
        )

        print(f'[INFO] Parameter terbaik: {grid_search.best_params_}')
        print(f'[INFO] Score CV terbaik: {grid_search.best_score_:.4f}')

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('best_cv_score', grid_search.best_score_)

        mlflow.sklearn.log_model(
            best_model,
            'random_forest_tuning_model'
        )

        print('[INFO] Model tuning dan metrik berhasil dicatat di MLflow')


if __name__ == '__main__':
    main()