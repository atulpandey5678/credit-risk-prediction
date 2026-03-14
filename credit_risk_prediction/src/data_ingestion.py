import pandas as pd
import yaml
import os

class DataIngestion:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
    def load_data(self):
        raw_path = self.config["data_source"]["raw_data_path"]
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
        df = pd.read_csv(raw_path)
        print(f"Data successfully loaded. Shape: {df.shape}")
        return df

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
