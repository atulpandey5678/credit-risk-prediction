import sys
import os
import pandas as pd

# Append root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import Predictor

class PredictionPipeline:
    def __init__(self):
        self.predictor = Predictor()
        
    def predict(self, data_dict):
        """
        data_dict should be a dictionary containing a single record.
        """
        df = pd.DataFrame([data_dict])
        preds, probs = self.predictor.predict(df)
        
        prob = probs[0]
        
        # Risk Categorization logic
        if prob < 0.20:
            risk = "Low Risk"
        elif prob < 0.60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"
            
        return {
            "probability": float(prob),
            "risk_category": risk,
            "prediction": int(preds[0])
        }

if __name__ == "__main__":
    # Test sample execution
    sample = {
        "loan_amount": 15000,
        "interest_rate": 10.5,
        "employment_length": 5,
        "annual_income": 75000,
        "credit_score": 720,
        "debt_to_income_ratio": 20.0,
        "loan_term": 36,
        "loan_purpose": "debt_consolidation",
        "home_ownership": "MORTGAGE"
    }
    pipeline = PredictionPipeline()
    result = pipeline.predict(sample)
    print("Prediction Result:", result)
