import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessing:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # These are original + engineered numerical features
        self.numerical_cols = self.config["features"]["numerical"] + [
            "income_to_loan_ratio", 
            "credit_utilization", 
            "is_long_term_emp", 
            "loan_to_term_ratio"
        ]
        self.categorical_cols = self.config["features"]["categorical"]
        
    def get_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numerical_cols),
                ("cat", categorical_transformer, self.categorical_cols)
            ]
        )
        return preprocessor
