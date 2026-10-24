"""
Script to initialize database schema and verify connectivity.
Usage:
    python scripts/initialize_database.py
    or
    python -m scripts.initialize_database
"""

import sys
from pathlib import Path

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
from src.database.connection import initialize_schema, check_connection_health, get_active_storage_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Customer Intelligence Database...")
    initialize_schema()
    health = check_connection_health()
    mode, msg = get_active_storage_mode()

    print("\n" + "=" * 60)
    print("DATABASE INITIALIZATION STATUS")
    print("=" * 60)
    print(f"Active Storage Mode : {mode}")
    print(f"Health Status       : {health['status']}")
    print(f"Database Target     : {health['database_url']}")
    print(f"Diagnostic Message  : {msg}")
    print(f"Tables Registered   : {len(health['tables_registered'])} tables")
    for tbl in sorted(health["tables_registered"]):
        print(f"  - {tbl}")
    print("=" * 60 + "\n")

    if health["status"] != "HEALTHY":
        sys.exit(1)


if __name__ == "__main__":
    main()
