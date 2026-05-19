

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

In [share.streamlit.io](https://share.streamlit.io) → **App settings**:

| Setting | Value |
|---------|--------|
| **Main file path** | `streamlit_app.py` |
| **Requirements file** | `requirements.txt` |
| **Python version** | `3.12` |

Also commit: `.python-version`, `packages.txt`, `setup.sh`, and `credit_risk_prediction/models/*.pkl`.

If you see `ModuleNotFoundError: joblib`, the requirements file was not installed — usually wrong Python version (use 3.12, not 3.14) or wrong requirements path.

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
