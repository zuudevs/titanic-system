# Import Library
from config import *
import pandas as pd
import os
import logging


# =============================================================================
# Initialization
# =============================================================================
RAW_FILE = DATASET_DIR / "raw.csv"
PROCESSED_FILE = DATASET_DIR / "clean.csv"


# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================
def load_data(filepath: Path, fallback_url: str) -> pd.DataFrame:
    """
    Load dataset from local file.
    If the file does not exist, download it from the fallback URL and save it locally.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        logger.info("Dataset ditemukan di lokal. Memuat data dari: %s", filepath)
        return pd.read_csv(filepath)

    logger.info("Dataset tidak ditemukan di lokal. Mengunduh dari URL sumber...")
    df = pd.read_csv(fallback_url)
    df.to_csv(filepath, index=False)
    logger.info("Dataset berhasil disimpan ke: %s", filepath)
    return df


# =============================================================================
# Data Preprocessing
# =============================================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform Titanic dataset.
    """
    df_clean = df.copy()

    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=existing_columns_to_drop)

    if "Age" in df_clean.columns:
        df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

    if "Embarked" in df_clean.columns:
        mode_embarked = df_clean["Embarked"].mode(dropna=True)
        if not mode_embarked.empty:
            df_clean["Embarked"] = df_clean["Embarked"].fillna(mode_embarked.iloc[0])

    categorical_columns = [col for col in ["Sex", "Embarked"] if col in df_clean.columns]
    if categorical_columns:
        df_clean = pd.get_dummies(df_clean, columns=categorical_columns, drop_first=True)

    return df_clean


# =============================================================================
# Data Saving
# =============================================================================
def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save processed dataframe to CSV.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Data bersih berhasil disimpan ke: %s", filepath)


# =============================================================================
# Main Pipeline
# =============================================================================
def main() -> None:
    logger.info("Memulai pipeline pemrosesan data Titanic...")

    df_raw = load_data(filepath=RAW_FILE, fallback_url=DATASET_URL)
    logger.info("Data mentah berhasil dimuat. Bentuk data: %s", df_raw.shape)

    df_processed = preprocess_data(df_raw)
    logger.info("Preprocessing selesai. Bentuk data hasil proses: %s", df_processed.shape)

    save_data(df_processed, PROCESSED_FILE)

    logger.info("Pipeline selesai dengan sukses.")


if __name__ == "__main__":
    main()