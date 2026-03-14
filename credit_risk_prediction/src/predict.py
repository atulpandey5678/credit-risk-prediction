import joblib
import yaml
import os

class Predictor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        if not os.path.exists(self.config["model"]["best_model_path"]) or not os.path.exists(self.config["model"]["preprocessor_path"]):
            raise FileNotFoundError("Model artifacts not found. Please train the model first.")
            
        self.model = joblib.load(self.config["model"]["best_model_path"])
        self.preprocessor = joblib.load(self.config["model"]["preprocessor_path"])
        
    def predict(self, input_df):
        # Apply preprocessing
        processed_data = self.preprocessor.transform(input_df)
        
        # Make predictions
        preds = self.model.predict(processed_data)
        probs = self.model.predict_proba(processed_data)[:, 1] if hasattr(self.model, "predict_proba") else preds
        return preds, probs
