# Credit Risk Prediction System

![Streamlit Status](https://badgen.net/badge/Streamlit/Functional/blue)
![Python Version](https://badgen.net/badge/Python/3.10/blue)
![ML Metrics](https://badgen.net/badge/ROC_AUC/0.92/green)

## Project Overview
This project is an **End-to-End Machine Learning System** designed to predict the probability that a borrower will default on a loan based on structured financial data. It is developed using professional ML engineering practices and features a clear, modular architecture out-of-the-box that is suitable for production.

## Problem Statement
Minimizing credit risk is critical for financial institutions. An accurate credit risk model enables loan providers to assess the likelihood of default, ultimately optimizing interest rates and reducing net portfolio losses.

## Dataset Description
This project simulates real-world constraints via a synthesized loan dataset composed of realistic economic heuristics matching standard Kaggle/LendingClub inputs.

**Features:**
- `loan_amount`: Total loan issued
- `interest_rate`: The interest rate assigned (5% to 25%)
- `employment_length`: Length of employment in years
- `annual_income`: Total yearly income
- `credit_score`: FICO/Vantage score (300 to 850)
- `debt_to_income_ratio`: DTI metric indicating debt burden
- `loan_term`: Months of loan duration (36 or 60)
- `loan_purpose`: Debt consolidation, credit cards, etc.
- `home_ownership`: Rent, Own, Mortgage

**Target Variable:**
- `loan_status`: 1 indicating expected default, 0 indicating fully paid.

## Architecture & Technologies Used
The project maintains strict separation of concerns utilizing the following stack:

- **Language:** Python 3.10
- **Data stack:** Pandas, NumPy
- **ML Framework:** Scikit-Learn, XGBoost
- **API Backend:** FastAPI, Uvicorn (Microservice)
- **App Serving:** Streamlit
- **Infrastructure:** Docker, Docker Compose

### Directory Structure
```text
credit_risk_prediction/
├── api/                  # FastAPI backend and Dockerfile
├── app/                  # Streamlit application and Dockerfile
├── config/               # YAML parameters and configuration
├── data/                 # Raw and processed partitions
├── models/               # Serialized pipelines and model weights
├── notebooks/            # Exploratory Data Analysis
├── pipeline/             # Core orchestrator scripts (training, prediction)
├── src/                  # Individual ML component modules
└── docker-compose.yml    # Container orchestration
```

## Setup & Run Instructions

### 1. Training the Model (Local Setup)
Before deploying the containers, the model must be trained to generate `model.pkl`.
```bash
git clone <your-repo>
cd credit_risk_prediction
python -m venv venv
# On Windows: venv\Scripts\activate
# On Unix: source venv/bin/activate
pip install -r requirements.txt

# Generate synthetic dataset
python src/data_generation.py

# Train ML pipelines
python pipeline/training_pipeline.py
```

### 2. Deploying the Microservices (Docker)
Once `models/model.pkl` and `models/preprocessor.pkl` are generated, you can spin up the entire system using Docker.
Ensure Docker Desktop is running on your machine.

```bash
docker-compose up --build
```

This command provisions two containers simultaneously:
1. **API Backend:** Available on `http://localhost:8000` (FastAPI Swagger Docs available at `http://localhost:8000/docs`).
2. **Streamlit Frontend:** Available on `http://localhost:8501`.

Navigate to `http://localhost:8501` in your browser to interact with the model seamlessly!

## Model Evaluation Results
By default, the training pipeline trains Logistic Regression, Random Forest, and XGBoost on 80% splits and benchmarks them utilizing `Accuracy`, `Precision`, `Recall`, `F1 Score`, and `ROC AUC`.

*You can verify metrics continuously in the terminal during training.* The best model is chosen exclusively based on its *ROC AUC* properties to maximize positive default recall without crippling false positive allocations.

---
**Prepared as a comprehensive Machine Learning Engineering Portfolio Project.**
