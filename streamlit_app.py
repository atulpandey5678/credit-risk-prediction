import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT / "credit_risk_prediction"
sys.path.insert(0, str(PROJECT_ROOT))

REQUIREMENTS = ROOT / "requirements.txt"
FALLBACK_PACKAGES = [
    "joblib==1.4.2",
    "scikit-learn==1.5.2",
    "xgboost==2.1.3",
    "pandas==2.2.3",
    "numpy==2.1.3",
    "scipy==1.14.1",
    "PyYAML==6.0.2",
    "requests==2.32.3",
]


def ensure_dependencies() -> None:
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
        import xgboost  # noqa: F401
        return
    except ImportError:
        pass

    if REQUIREMENTS.is_file():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)],
        )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *FALLBACK_PACKAGES],
        )


ensure_dependencies()

from app.app import main

if __name__ == "__main__":
    main()
