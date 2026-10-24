"""
Central configuration module for Customer Lifetime Value and Churn Intelligence System.
Manages file paths, hyperparameters, database URLs, and business assumptions.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_ARTIFACTS_DIR = BASE_DIR / "models_artifacts"
REPORTS_DIR = BASE_DIR / "reports"
SQL_DIR = BASE_DIR / "sql"
DOCS_DIR = BASE_DIR / "docs"

# Ensure all critical runtime directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset path
RAW_DATA_PATH = RAW_DATA_DIR / "Telco_customer_churn.csv"
FALLBACK_DATA_PATH = BASE_DIR / "Telco_customer_churn.csv"

# Database Configuration (PostgreSQL Dedicated)
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# Business Defaults
DEFAULT_CONTRIBUTION_MARGIN = float(os.environ.get("DEFAULT_CONTRIBUTION_MARGIN", "0.65"))
DEFAULT_ANNUAL_DISCOUNT_RATE = float(os.environ.get("DEFAULT_ANNUAL_DISCOUNT_RATE", "0.10"))
DEFAULT_HORIZON_MONTHS = int(os.environ.get("DEFAULT_HORIZON_MONTHS", "24"))

# ML Experimentation Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # Proportion of train split for validation

# Default Decision Threshold
DEFAULT_CHURN_THRESHOLD = 0.40  # Tuned for business retention cost-benefit

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
