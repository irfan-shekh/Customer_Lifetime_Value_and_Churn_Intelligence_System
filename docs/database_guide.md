# PostgreSQL Database & Persistence Guide

## 1. Architecture & Persistence Strategy
The Customer Intelligence System employs a normalized PostgreSQL schema as its dedicated relational store.
All customer demographics, historical snapshots, churn probabilities, survival forecasts, CLV projections, and campaign simulations are persisted in PostgreSQL 18.

---

## 2. Configuration & Credentials

All database settings are loaded from the environment or `.env`:
```ini
DATABASE_URL=postgresql://<username>:<password>@<host>:<port>/<database_name>
```

### Example Configurations:
- **Local PostgreSQL Server:**
  ```ini
  DATABASE_URL=postgresql://postgres:postgres@localhost:5432/customer_intelligence
  ```
- **Managed Cloud PostgreSQL (GCP Cloud SQL / AWS RDS):**
  ```ini
  DATABASE_URL=postgresql://app_user:strong_password@10.0.0.5:5432/customer_intelligence
  ```

---

## 3. Database Initialization & Ingestion Commands

### Step 1: Initialize Schema & Tables
Run the schema creation script. This executes SQLAlchemy DDL and creates foreign key constraints, indexes, and tables:
```bash
python scripts/initialize_database.py
```

### Step 2: Ingest Customer Records
Import and standardize raw CSV data into `customers` and `customer_snapshots` tables:
```bash
python scripts/import_dataset.py
```

### Step 3: Populate Model Predictions & Simulations
Score all customer records and persist churn predictions, survival curves, CLV valuations, and retention campaigns:
```bash
python scripts/generate_predictions.py
```

---

## 4. Normalized Table Relationships

```
customers (PK: customer_id)
  ├── 1:N customer_snapshots (FK: customer_id)
  ├── 1:N churn_predictions (FK: customer_id, model_run_id)
  ├── 1:N survival_predictions (FK: customer_id, model_run_id)
  ├── 1:N customer_lifetime_values (FK: customer_id, model_run_id)
  ├── 1:N customer_segments (FK: customer_id, model_run_id)
  └── 1:N campaign_assignments (FK: customer_id, campaign_id)

model_runs (PK: model_run_id)
retention_campaigns (PK: campaign_id)
```

---

## 5. Connection Health & Diagnostic Troubleshooting
To test live database connectivity and latency directly from the CLI:
```python
from src.database.connection import check_connection_health
print(check_connection_health())
```
Or view the live telemetry dashboard at `http://localhost:8501/Data_and_Database_Health`.
