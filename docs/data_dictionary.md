# Data Dictionary & Schema Specification

## 1. Raw Dataset Overview
- **Source:** IBM Telco Customer Churn ([Kaggle BlastChar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- **Total Records:** 7,043 customers
- **Total Columns:** 33 columns (extended IBM version) or 21 columns (standard IBM version)
- **Target Variable:** `Churn Value` (1 = Churned, 0 = Retained)

---

## 2. Feature Catalog & Preprocessing Specifications

| Column Name | Standardized Name | Type | Category | Description & Preprocessing Decisions |
| :--- | :--- | :--- | :--- | :--- |
| `CustomerID` | `customer_id` | String | Identifier | Unique customer identifier. **Excluded from all ML features** to prevent memorization. |
| `Gender` | `gender` | Categorical | Demographic | Male / Female. One-Hot Encoded. |
| `Senior Citizen` | `senior_citizen` | Binary Integer | Demographic | 1 if customer age >= 65, else 0. |
| `Partner` | `partner` | Categorical | Demographic | Yes / No. One-Hot Encoded. |
| `Dependents` | `dependents` | Categorical | Demographic | Yes / No. One-Hot Encoded. |
| `Tenure Months` | `tenure` | Integer | Account | Number of months account has been active (0 to 72). Scaled via StandardScaler. |
| `Phone Service` | `phone_service` | Categorical | Service | Yes / No home phone subscription. |
| `Multiple Lines` | `multiple_lines` | Categorical | Service | Yes, No, or No phone service. |
| `Internet Service` | `internet_service` | Categorical | Service | DSL, Fiber optic, or No. Strong predictive signal. |
| `Online Security` | `online_security` | Categorical | Add-On Service | Yes, No, or No internet service. |
| `Online Backup` | `online_backup` | Categorical | Add-On Service | Yes, No, or No internet service. |
| `Device Protection`| `device_protection`| Categorical | Add-On Service | Yes, No, or No internet service. |
| `Tech Support` | `tech_support` | Categorical | Add-On Service | Yes, No, or No internet service. Reduces hazard rate. |
| `Streaming TV` | `streaming_tv` | Categorical | Add-On Service | Yes, No, or No internet service. |
| `Streaming Movies`| `streaming_movies`| Categorical | Add-On Service | Yes, No, or No internet service. |
| `Contract` | `contract` | Categorical | Account | Month-to-month, One year, Two year. Highest predictive SHAP driver. |
| `Paperless Billing`| `paperless_billing`| Categorical | Billing | Yes / No. |
| `Payment Method` | `payment_method` | Categorical | Billing | Electronic check, Mailed check, Bank transfer, Credit card. |
| `Monthly Charges` | `monthly_charges` | Float | Financial | Monthly recurring bill in USD ($18.25 to $118.75). Scaled. |
| `Total Charges` | `total_charges` | Float | Financial | Cumulative historical billings. **11 blank strings imputed to 0.0 for tenure=0**. |
| `Churn Value` | `churn` | Binary Integer | Target | 1 if churned, 0 if retained. Population churn rate: 26.54%. |

---

## 3. Engineered Features (`src/data/features.py`)

| Feature Name | Type | Derivation Logic | Business Rationale |
| :--- | :--- | :--- | :--- |
| `tenure_group` | Categorical | Binned into `0-12m`, `13-24m`, `25-48m`, `49-60m`, `61-72m` | Captures lifecycle adoption stages. |
| `monthly_charge_group`| Categorical | Binned into `Low`, `Medium`, `High`, `Very High` spend | Isolates premium enterprise subscribers. |
| `avg_monthly_charges` | Float | `total_charges / (tenure + 1)` | Measures average historical billing rate. |
| `charge_ratio` | Float | `avg_monthly_charges / (monthly_charges + 1e-4)` | Detects promotional roll-offs (recent price hikes). |
| `num_services_subscribed`| Integer | Count of active services out of 8 potential add-ons | Quantifies product stickiness and adoption depth. |
| `has_streaming_bundle` | Binary | 1 if customer subscribes to both TV and Movies | Entertainment ecosystem stickiness. |
| `is_month_to_month` | Binary | 1 if `contract == 'Month-to-month'` | Flag for highest-risk churn cohort. |
| `single_senior` | Binary | 1 if senior citizen with no partner and no dependents | Vulnerable demographic requiring proactive support. |

---

## 4. Target Leakage Exclusions (Audited & Dropped)
- **`Churn Reason`:** Only recorded when a customer has already churned (5,174 missing values). Severe leakage.
- **`Churn Score`:** IBM proprietary pre-computed risk score (0-100). Dropped.
- **`CLTV` (IBM Raw):** Pre-calculated commercial index score. Dropped.
- **`Churn Label`:** Redundant duplicate of `Churn Value`. Dropped.
- **`Count`, `Country`, `State`:** Constants across all records (1, United States, California). Dropped.
