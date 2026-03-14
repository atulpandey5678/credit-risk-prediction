from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Append root to path so we can import our pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API that predicts the likelihood of loan default.",
    version="1.0.0"
)

# Initialize the pipeline globally on startup
try:
    pipeline = PredictionPipeline()
except Exception as e:
    pipeline = None
    print(f"Warning: Could not initialize pipeline. Did you train the model? Error: {e}")

class BorrowerData(BaseModel):
    loan_amount: float = Field(..., example=15000.0)
    interest_rate: float = Field(..., example=10.5)
    employment_length: int = Field(..., example=5)
    annual_income: float = Field(..., example=75000.0)
    credit_score: int = Field(..., example=720)
    debt_to_income_ratio: float = Field(..., example=20.0)
    loan_term: int = Field(..., example=36)
    loan_purpose: str = Field(..., example="debt_consolidation")
    home_ownership: str = Field(..., example="MORTGAGE")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Credit Risk Prediction API is running"}

@app.post("/predict")
def predict_default(data: BorrowerData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction models are not loaded. Run training pipeline first.")
        
    try:
        # Pydantic model dump maps perfectly to the dictionary expected by pipeline
        input_dict = data.dict()
        result = pipeline.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
