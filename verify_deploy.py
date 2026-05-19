"""Verify the app is ready for Streamlit Cloud deployment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "credit_risk_prediction"
sys.path.insert(0, str(PROJECT))

REQUIRED = [
    PROJECT / "models" / "model.pkl",
    PROJECT / "models" / "preprocessor.pkl",
    PROJECT / "config" / "config.yaml",
    ROOT / "requirements.txt",
    ROOT / "streamlit_app.py",
    ROOT / "packages.txt",
]


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"Missing: {path.relative_to(ROOT)}")

    try:
        from pipeline.prediction_pipeline import PredictionPipeline

        sample = {
            "loan_amount": 15000,
            "interest_rate": 10.5,
            "employment_length": 5,
            "annual_income": 65000,
            "credit_score": 700,
            "debt_to_income_ratio": 20.0,
            "loan_term": 36,
            "loan_purpose": "debt_consolidation",
            "home_ownership": "RENT",
        }
        result = PredictionPipeline().predict(sample)
        print("Prediction OK:", result)
    except Exception as exc:
        errors.append(f"Prediction failed: {exc}")

    if errors:
        print("Deploy check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Deploy check PASSED — ready for Streamlit Cloud.")
    print("Main file: streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
