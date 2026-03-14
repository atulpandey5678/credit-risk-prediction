import sys
import os
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Append root to path for local execution flexibility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.feature_engineering import FeatureEngineer
from src.data_preprocessing import DataPreprocessing
from src.train_model import ModelTrainer
from src.model_evaluation import ModelEvaluator

def run_training_pipeline():
    print("=== 1. Data Ingestion ===")
    ingestion = DataIngestion()
    df = ingestion.load_data()
    
    print("\n=== 2. Data Validation ===")
    validator = DataValidation()
    validator.validate(df)
    
    # Target variable
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    target = config["features"]["target"]
    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"\n=== 3. Train/Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["training"]["test_size"], 
        random_state=config["training"]["random_state"],
        stratify=y
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    print("\n=== 4. Preprocessing & Feature Engineering ===")
    dp = DataPreprocessing()
    preprocessor = dp.get_preprocessor()
    
    data_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor)
    ])
    
    print("Fitting preprocessing pipeline...")
    X_train_processed = data_pipeline.fit_transform(X_train)
    X_test_processed = data_pipeline.transform(X_test)
    
    print("\n=== 5. Model Training ===")
    trainer = ModelTrainer()
    trained_models = trainer.train_all(X_train_processed, y_train)
    
    print("\n=== 6. Model Evaluation ===")
    evaluator = ModelEvaluator()
    best_name, best_model, eval_results = evaluator.select_best_model(trained_models, X_test_processed, y_test)
    
    print("\n=== 7. Artifact Saving ===")
    os.makedirs(os.path.dirname(config["model"]["best_model_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["model"]["preprocessor_path"]), exist_ok=True)
    
    # Save the pipeline and the best model
    joblib.dump(data_pipeline, config["model"]["preprocessor_path"])
    joblib.dump(best_model, config["model"]["best_model_path"])
    
    print(f"Saved preprocessor to {config['model']['preprocessor_path']}")
    print(f"Saved model to {config['model']['best_model_path']}")
    
    # Save processed data for reference
    os.makedirs(os.path.dirname(config["data_source"]["processed_data_path"]), exist_ok=True)
    # Using float arrays for processed representation
    pd.DataFrame(X_train_processed).to_csv(config["data_source"]["processed_data_path"], index=False)
    
    print("\n=== Training Pipeline Completed Successfully ===")

if __name__ == "__main__":
    run_training_pipeline()
