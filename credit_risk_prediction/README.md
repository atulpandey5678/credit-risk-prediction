# Credit Risk Prediction System

An end-to-end machine learning project that predicts the probability of loan default using borrower financial information.
The system helps lenders evaluate borrower risk and make better credit decisions.

---

## Project Overview

Financial institutions must evaluate the risk of lending money to borrowers.
Incorrect risk assessment can lead to loan defaults and financial losses.

This project builds a machine learning pipeline that analyzes borrower attributes and predicts the probability of default.

The system includes:

* Data preprocessing pipeline
* Feature engineering
* Multiple model training
* Model evaluation
* Prediction pipeline
* Interactive Streamlit application

---

## Problem Statement

Banks and lending platforms need automated systems to evaluate loan applications and identify high-risk borrowers.

The goal of this project is to develop a machine learning model that predicts whether a borrower is likely to default on a loan.

---

## Dataset

The dataset contains borrower financial and loan attributes.

Example features include:

* Loan Amount
* Interest Rate
* Annual Income
* Employment Length
* Credit Score
* Debt to Income Ratio
* Loan Purpose
* Home Ownership
* Loan Term

Target variable:

* Loan Default (1 = Default, 0 = No Default)

---

## Project Structure

credit-risk-prediction/

data/
notebooks/
src/
pipeline/
models/
app/

README.md
requirements.txt
.gitignore

---

## Machine Learning Pipeline

The system follows a structured machine learning workflow.

### Data Ingestion

Loads the dataset and prepares raw data.

### Data Validation

Checks dataset quality and structure.

### Data Preprocessing

Handles missing values and encodes categorical features.

### Feature Engineering

Creates additional useful features such as financial ratios.

### Model Training

Trains multiple machine learning models.

Models used:

* Logistic Regression
* Random Forest
* XGBoost

### Model Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The best performing model is selected and saved.

### Prediction Pipeline

Loads the trained model and predicts loan default probability for new borrower inputs.

---

## Streamlit Application

The project includes a Streamlit interface where users can input borrower information and receive a risk prediction.

User inputs:

* Loan Amount
* Annual Income
* Credit Score
* Employment Length
* Debt to Income Ratio
* Loan Term
* Home Ownership

Output:

* Default probability
* Risk classification

Risk levels:

Low Risk
Medium Risk
High Risk

---

## Demo

Screenshots of the application interface:

![App Screenshot](assets/dashboard.png)

![Prediction Result](assets/prediction.png)

---

## Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Joblib
Streamlit

---

## How to Run the Project

Clone the repository

git clone https://github.com/yourusername/credit-risk-prediction.git

Move to project folder

cd credit-risk-prediction

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app/app.py

---

## Future Improvements

Add hyperparameter tuning
Add MLflow experiment tracking
Deploy model using Docker
Create REST API using FastAPI
Use larger financial datasets

---

## Author

Atul Pandey
