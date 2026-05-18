from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_eng = X.copy()
        X_eng["income_to_loan_ratio"] = X_eng["annual_income"] / (X_eng["loan_amount"] + 1)
        X_eng["credit_utilization"] = X_eng["debt_to_income_ratio"] / 100.0
        X_eng["is_long_term_emp"] = (X_eng["employment_length"] >= 10).astype(int)
        X_eng["loan_to_term_ratio"] = X_eng["loan_amount"] / X_eng["loan_term"]
        return X_eng
