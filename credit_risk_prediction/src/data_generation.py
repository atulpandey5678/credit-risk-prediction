import pandas as pd
import numpy as np
import os
import yaml

with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

def generate_synthetic_data(num_records=10000):
    np.random.seed(42)
    
    # Generate numerical features
    loan_amount = np.random.uniform(1000, 40000, num_records)
    interest_rate = np.random.uniform(5.0, 25.0, num_records)
    employment_length = np.random.randint(0, 15, num_records)
    annual_income = np.random.lognormal(mean=11.0, sigma=0.6, size=num_records)  # Skewed like real income
    credit_score = np.random.randint(300, 850, num_records)
    debt_to_income_ratio = np.random.uniform(0, 60.0, num_records)
    
    # 36 or 60 months typically
    loan_term = np.random.choice([36, 60], size=num_records, p=[0.7, 0.3])

    # Generate categorical features
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "other"]
    loan_purpose = np.random.choice(purposes, size=num_records, p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05])
    
    home_ownership = np.random.choice(["RENT", "OWN", "MORTGAGE"], size=num_records, p=[0.4, 0.1, 0.5])

    df = pd.DataFrame({
        "loan_amount": np.round(loan_amount, 2),
        "interest_rate": np.round(interest_rate, 2),
        "employment_length": employment_length,
        "annual_income": np.round(annual_income, 2),
        "credit_score": credit_score,
        "debt_to_income_ratio": np.round(debt_to_income_ratio, 2),
        "loan_term": loan_term,
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership
    })

    # Generate Target based on some realistic heuristics
    # Lower income, higher interest rate, lower credit score -> higher default prob
    base_default_risk = 0.05 
    
    risk_score = (
        (df["interest_rate"] / 25.0) * 0.3 + 
        (1 - (df["credit_score"] - 300) / 550) * 0.4 +
        (df["debt_to_income_ratio"] / 60.0) * 0.2 +
        (df["loan_amount"] / df["annual_income"]) * 0.1
    )
    
    # Add noise
    risk_score += np.random.normal(0, 0.1, num_records)
    
    # 1 for default, 0 for paid completely
    # set threshold to yield roughly 20-30% default rate
    threshold = np.percentile(risk_score, 75)
    df["loan_status"] = (risk_score > threshold).astype(int)

    os.makedirs(os.path.dirname(config["data_source"]["raw_data_path"]), exist_ok=True)
    df.to_csv(config["data_source"]["raw_data_path"], index=False)
    print(f"Generated {num_records} records at {config['data_source']['raw_data_path']}")
    print(f"Default class distribution:\n{df['loan_status'].value_counts(normalize=True)}")

if __name__ == "__main__":
    generate_synthetic_data()
