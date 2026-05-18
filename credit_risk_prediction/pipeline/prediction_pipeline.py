import pandas as pd

from src.predict import Predictor

LOW_RISK_THRESHOLD = 0.20
MEDIUM_RISK_THRESHOLD = 0.60


def categorize_risk(probability: float) -> str:
    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk"
    if probability < MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


class PredictionPipeline:
    def __init__(self):
        self.predictor = Predictor()

    def predict(self, data_dict: dict) -> dict:
        df = pd.DataFrame([data_dict])
        preds, probs = self.predictor.predict(df)
        prob = float(probs[0])

        return {
            "probability": prob,
            "risk_category": categorize_risk(prob),
            "prediction": int(preds[0]),
        }
