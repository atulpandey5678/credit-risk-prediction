import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts the likelihood of loan default.",
    version="1.0.0",
)

try:
    pipeline = PredictionPipeline()
except Exception as exc:
    pipeline = None
    print(f"Pipeline init failed: {exc}")


class BorrowerData(BaseModel):
    loan_amount: float
    interest_rate: float
    employment_length: int
    annual_income: float
    credit_score: int
    debt_to_income_ratio: float
    loan_term: int
    loan_purpose: str
    home_ownership: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Credit Risk Prediction API is running"}


@app.post("/predict")
def predict_default(data: BorrowerData):
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Models not loaded. Run the training pipeline first.",
        )
    try:
        return pipeline.predict(data.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
