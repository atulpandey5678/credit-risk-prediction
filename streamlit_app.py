import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT / "credit_risk_prediction"
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import joblib  # noqa: F401
except ImportError as exc:
    import streamlit as st

    st.error(
        "Missing Python packages (joblib). "
        "Set **Requirements file** to `requirements.txt` at repo root, "
        "set **Python** to 3.12, then redeploy."
    )
    st.code("pip install -r requirements.txt", language="bash")
    st.stop()

from app.app import main

if __name__ == "__main__":
    main()
