import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_eng = X.copy()
        
        # 1. income to loan ratio
        X_eng["income_to_loan_ratio"] = X_eng["annual_income"] / (X_eng["loan_amount"] + 1)
        
        # 2. credit utilization approximation
        X_eng["credit_utilization"] = X_eng["debt_to_income_ratio"] / 100.0
        
        # 3. long term employment indicator
        X_eng["is_long_term_emp"] = (X_eng["employment_length"] >= 10).astype(int)
        
        # 4. loan amount to term ratio
        X_eng["loan_to_term_ratio"] = X_eng["loan_amount"] / X_eng["loan_term"]
        
        return X_eng
