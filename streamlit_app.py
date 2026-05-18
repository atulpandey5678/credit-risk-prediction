import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent / "credit_risk_prediction"
sys.path.insert(0, str(PROJECT_ROOT))

from app.app import main

if __name__ == "__main__":
    main()
