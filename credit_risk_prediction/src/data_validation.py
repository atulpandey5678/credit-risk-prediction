import pandas as pd
import yaml

class DataValidation:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
    def validate(self, df):
        print("Starting data validation...")
        
        # 1. Missing Values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("Warning: Missing values found.")
            print(missing[missing > 0])
        else:
            print("Check passed: No missing values.")
            
        # 2. Duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Warning: {duplicates} duplicate rows found.")
        else:
            print("Check passed: No duplicate rows.")
            
        # 3. Invalid Feature Types
        expected_numerical = self.config["features"]["numerical"]
        expected_categorical = self.config["features"]["categorical"]
        
        for col in expected_numerical:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: {col} is expected to be numerical, but is {df[col].dtype}.")
        
        for col in expected_categorical:
            if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
                print(f"Warning: {col} is expected to be categorical, but is {df[col].dtype}.")
        
        print("Data validation completed.")
        return True

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    df = DataIngestion().load_data()
    validator = DataValidation()
    validator.validate(df)
