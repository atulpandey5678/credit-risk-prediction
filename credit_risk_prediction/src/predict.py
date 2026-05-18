import joblib
import pandas as pd
import yaml

from src.logger import logger
from src.paths import ROOT_DIR


class Predictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = ROOT_DIR / config_path
        with open(config_file, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        model_path = ROOT_DIR / self.config["model"]["best_model_path"]
        preprocessor_path = ROOT_DIR / self.config["model"]["preprocessor_path"]

        if not model_path.is_file() or not preprocessor_path.is_file():
            logger.error("Model artifacts not found at %s", ROOT_DIR / "models")
            raise FileNotFoundError(
                "Model artifacts not found. Run pipeline/training_pipeline.py and commit models/*.pkl."
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_df: pd.DataFrame):
        processed = self.preprocessor.transform(input_df)
        preds = self.model.predict(processed)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(processed)[:, 1]
        else:
            probs = preds
        return preds, probs
