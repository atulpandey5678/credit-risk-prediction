# Credit Risk Prediction System

End-to-end machine learning system that predicts loan default probability from borrower financial and loan details. It includes a training pipeline, saved model artifacts, an optional REST API, and a Streamlit web app for interactive risk scoring.

---

## Project Summary

Lenders need fast, consistent ways to assess whether a borrower is likely to default. This project automates that workflow: raw credit data is validated and transformed, multiple classifiers are trained and compared, and the best model is saved for production inference.

**What it does**

- Ingests and validates a credit risk dataset (~10k borrower records)
- Engineers features (income-to-loan ratio, credit utilization, employment flags, and more)
- Trains and tunes **Logistic Regression**, **Random Forest**, and **XGBoost** with cross-validation
- Selects the best model by **ROC-AUC** and persists artifacts (`model.pkl`, `preprocessor.pkl`)
- Serves predictions via a **Streamlit UI** (bundled model) or an optional **FastAPI** backend
- Classifies risk as **Low**, **Medium**, or **High** from default probability thresholds

**Target variable:** `loan_status` (0 = no default, 1 = default)

---

## Tech Stack

| Layer | Technologies |
|--------|----------------|
| **Language** | Python 3.12 |
| **Data & ML** | Pandas, NumPy, scikit-learn, XGBoost, Joblib |
| **Config** | YAML (`config/config.yaml`) |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Web UI** | Streamlit |
| **HTTP client** | Requests (optional remote API mode) |
| **Experiment tracking** | MLflow (training workflows) |
| **Visualization / EDA** | Matplotlib, Seaborn (notebooks) |
| **Testing** | pytest |
| **Deployment** | Docker, Docker Compose, Streamlit Community Cloud |
| **Logging** | Python `logging` (file + console) |

---

## Architecture

```
credit_risk_dataset.csv
        │
        ▼
┌───────────────────┐
│ Training pipeline │  ingestion → validation → feature engineering
│ (offline job)     │  → preprocessing → train → evaluate → save artifacts
└─────────┬─────────┘
          │
          ▼
   models/model.pkl
   models/preprocessor.pkl
          │
          ├──────────────────────┐
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Streamlit app   │    │ FastAPI (opt.)  │
│ In-process ML   │    │ POST /predict   │
└─────────────────┘    └─────────────────┘
```

- **Streamlit (default):** loads the model directly — suitable for local use and Streamlit Cloud.
- **FastAPI (optional):** set `API_URL` in the Streamlit app to call a separate API (e.g. Docker Compose stack).

---

## Project Structure

```
Loan_wala/
├── streamlit_app.py              # Streamlit Cloud / local entrypoint
├── requirements.txt              # Core runtime dependencies
├── .python-version               # Python 3.12
├── packages.txt                  # System libs for Streamlit Cloud (XGBoost)
└── credit_risk_prediction/
    ├── api/main.py               # FastAPI prediction service
    ├── app/app.py                # Streamlit UI
    ├── config/config.yaml        # Paths, features, training settings
    ├── data/raw/                 # Source dataset
    ├── data/processed/           # Processed training output
    ├── models/                   # Saved model + preprocessor
    ├── pipeline/
    │   ├── training_pipeline.py  # End-to-end training job
    │   └── prediction_pipeline.py
    ├── src/                      # Ingestion, validation, ML modules
    ├── notebooks/eda.ipynb       # Exploratory analysis
    └── docker-compose.yml        # API + Streamlit containers
```

---

## Machine Learning Pipeline

1. **Data ingestion** — load CSV from `data/raw/`
2. **Data validation** — missing values, duplicates, column types
3. **Feature engineering** — ratios and derived numeric features
4. **Preprocessing** — imputation, scaling, one-hot encoding
5. **Model training** — GridSearchCV on three algorithms
6. **Model evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
7. **Artifact export** — best model and fitted preprocessor pipeline

**Risk bands**

| Category | Default probability |
|----------|---------------------|
| Low Risk | &lt; 20% |
| Medium Risk | 20% – 60% |
| High Risk | &gt; 60% |

---

## How to Run

### 1. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

For API development and training extras:

```bash
pip install -r credit_risk_prediction/requirements.txt
```

### 2. Train the model (first time or after data changes)

```bash
cd credit_risk_prediction
python pipeline/training_pipeline.py
```

### 3. Run the Streamlit app

From the repository root:

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

### 4. Optional — FastAPI + Docker

```bash
cd credit_risk_prediction
docker compose up --build
```

- API: http://localhost:8000/docs  
- UI: http://localhost:8501 (uses `API_URL=http://api:8000/predict`)

### 5. Deploy on Streamlit Cloud

| Setting | Value |
|---------|--------|
| Main file | `streamlit_app.py` |
| Python | 3.12 (`.python-version`) |
| Requirements | `requirements.txt` (repo root) |

Ensure `credit_risk_prediction/models/*.pkl` are committed to the repository.

---

## Dataset Features

| Feature | Description |
|---------|-------------|
| `loan_amount` | Requested loan amount |
| `interest_rate` | Annual interest rate (%) |
| `annual_income` | Borrower annual income |
| `employment_length` | Years employed |
| `credit_score` | Credit score (300–850) |
| `debt_to_income_ratio` | DTI (%) |
| `loan_term` | Term in months (36 or 60) |
| `loan_purpose` | Purpose category |
| `home_ownership` | RENT / OWN / MORTGAGE |

---

## Author

**Atul Pandey**
