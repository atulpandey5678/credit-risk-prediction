import pandas as pd
import yaml

from src.logger import logger
from src.paths import ROOT_DIR


class DataIngestion:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = ROOT_DIR / config_path
        with open(config_file, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

    def load_data(self) -> pd.DataFrame:
        raw_path = ROOT_DIR / self.config["data_source"]["raw_data_path"]
        if not raw_path.is_file():
            logger.error("Raw data file not found at: %s", raw_path)
            raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
        df = pd.read_csv(raw_path)
        logger.info("Data loaded. Shape: %s", df.shape)
        return df
