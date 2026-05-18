import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.paths import ROOT_DIR


class DataPreprocessing:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = ROOT_DIR / config_path
        with open(config_file, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        self.numerical_cols = self.config["features"]["numerical"] + [
            "income_to_loan_ratio",
            "credit_utilization",
            "is_long_term_emp",
            "loan_to_term_ratio",
        ]
        self.categorical_cols = self.config["features"]["categorical"]

    def get_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numerical_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )
