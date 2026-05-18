import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.data_validation import DataValidation
from src.feature_engineering import FeatureEngineer
from src.logger import logger
from src.model_evaluation import ModelEvaluator
from src.paths import ROOT_DIR
from src.train_model import ModelTrainer


def run_training_pipeline() -> None:
    logger.info("=== 1. Data Ingestion ===")
    df = DataIngestion().load_data()

    logger.info("=== 2. Data Validation ===")
    DataValidation().validate(df)

    config_file = ROOT_DIR / "config/config.yaml"
    with open(config_file, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    target = config["features"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    logger.info("=== 3. Train/Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )
    logger.info("X_train: %s, X_test: %s", X_train.shape, X_test.shape)

    logger.info("=== 4. Preprocessing & Feature Engineering ===")
    data_pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", DataPreprocessing().get_preprocessor()),
        ]
    )
    X_train_processed = data_pipeline.fit_transform(X_train)
    X_test_processed = data_pipeline.transform(X_test)

    logger.info("=== 5. Model Training ===")
    trained_models = ModelTrainer().train_all(X_train_processed, y_train)

    logger.info("=== 6. Model Evaluation ===")
    best_name, best_model, _ = ModelEvaluator().select_best_model(
        trained_models, X_test_processed, y_test
    )
    logger.info("Selected model: %s", best_name)

    logger.info("=== 7. Artifact Saving ===")
    model_path = ROOT_DIR / config["model"]["best_model_path"]
    preprocessor_path = ROOT_DIR / config["model"]["preprocessor_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(data_pipeline, preprocessor_path)
    joblib.dump(best_model, model_path)
    logger.info("Saved preprocessor to %s", preprocessor_path)
    logger.info("Saved model to %s", model_path)

    processed_path = ROOT_DIR / config["data_source"]["processed_data_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_processed).to_csv(processed_path, index=False)

    logger.info("=== Training Pipeline Completed ===")


if __name__ == "__main__":
    run_training_pipeline()
