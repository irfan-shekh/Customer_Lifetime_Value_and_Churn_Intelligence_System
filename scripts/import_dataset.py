"""
Script to import and upsert customer dataset into the database.
Usage:
    python scripts/import_dataset.py [--path <csv_path>]
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
from src.data.load_data import load_raw_data, standardize_columns
from src.data.validate_data import validate_raw_dataframe
from src.database.repository import DatabaseRepository
from src.database.connection import get_active_storage_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def import_dataset(csv_path: Path = None):
    logger.info("Starting dataset import workflow...")
    df_raw = load_raw_data(csv_path)
    
    # 1. Validation
    val_report = validate_raw_dataframe(df_raw)
    if not val_report["is_valid"]:
        logger.error(f"Dataset validation failed: {val_report['issues']}")
        sys.exit(1)
        
    logger.info(f"Dataset validated successfully ({val_report['row_count']} rows).")
    
    # 2. Standardization
    df_clean = standardize_columns(df_raw)
    
    # 3. Persistence
    repo = DatabaseRepository()
    records_upserted = repo.bulk_upsert_customers(df_clean)
    
    mode, msg = get_active_storage_mode()
    print("\n" + "=" * 60)
    print("DATASET IMPORT SUMMARY")
    print("=" * 60)
    print(f"Total Rows Ingested : {records_upserted}")
    print(f"Active Storage Mode : {mode}")
    print(f"Storage Status      : {msg}")
    print("=" * 60 + "\n")
    return records_upserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import customer churn dataset into database.")
    parser.add_argument("--path", type=str, default=None, help="Optional path to custom CSV dataset.")
    args = parser.parse_args()

    custom_path = Path(args.path) if args.path else None
    import_dataset(custom_path)
