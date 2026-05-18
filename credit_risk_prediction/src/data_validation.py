import pandas as pd
import yaml

from src.logger import logger
from src.paths import ROOT_DIR


class DataValidation:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_file = ROOT_DIR / config_path
        with open(config_file, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

    def validate(self, df: pd.DataFrame) -> bool:
        logger.info("Starting data validation...")

        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning("Missing values found: %s", missing[missing > 0].to_dict())
        else:
            logger.info("No missing values.")

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning("%s duplicate rows found.", duplicates)
        else:
            logger.info("No duplicate rows.")

        for col in self.config["features"]["numerical"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning("%s expected numerical, got %s.", col, df[col].dtype)

        for col in self.config["features"]["categorical"]:
            if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
                logger.warning("%s expected categorical, got %s.", col, df[col].dtype)

        logger.info("Data validation completed.")
        return True
