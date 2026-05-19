import logging
import sys

from src.paths import ROOT_DIR

handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

try:
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers.insert(0, logging.FileHandler(log_dir / "pipeline.log"))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    handlers=handlers,
)

logger = logging.getLogger("CreditRiskLogger")
