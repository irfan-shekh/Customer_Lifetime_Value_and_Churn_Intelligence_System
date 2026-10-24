# Model Card: Churn Prediction & Lifetime Value System

## 1. Model Details
- **Developer:** Internship Data Science & ML Engineering Team
- **Model Architecture:** Gradient Boosted Decision Trees (`LightGBMClassifier`) + Semi-Parametric Cox Proportional Hazards (`lifelines.CoxPHFitter`) + K-Means Clustering (`sklearn.cluster.KMeans`)
- **Model Version:** `v1.0.0`
- **Release Date:** October 2026
- **License:** MIT / Apache-2.0
- **Supported Runtime:** Python 3.13, Scikit-learn 1.9, LightGBM 4.7, Lifelines 0.30

---

## 2. Intended Use & Target Users
- **Primary Use Case:** Proactive identification of telecom subscribers at high risk of contract termination; forward-looking valuation of customer cash flows; simulation of targeted retention discount campaigns.
- **Intended Users:** Retention Marketing Managers, Financial Planning Analysts, Customer Success Directors.
- **Out-of-Scope Uses:** Autonomous credit scoring; automated punitive contractual termination; setting discriminatory consumer prices based on demographic attributes.

---

## 3. Training & Validation Data
- **Dataset:** IBM Telco Customer Churn (7,043 total records).
- **Partitioning Strategy:**
  - Training Set: 4,507 records (64.0%)
  - Validation Set: 1,127 records (16.0%)
  - Held-Out Test Set: 1,409 records (20.0%)
  - All splits stratified on the binary target (`churn`).
- **Leakage Prevention Audit:**
  - Customer identifiers dropped (`customer_id`).
  - Target proxies dropped (`churn_reason`, `churn_score`, `cltv_ibm`).
  - Preprocessors fitted strictly on training data.

---

## 4. Evaluated Performance Metrics (Held-Out Test Set: 1,409 Customers)

| Metric | Primary LightGBM (Tuned T=0.55) | LightGBM (Default T=0.50) | Logistic Regression Baseline |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.8516** | 0.8516 | 0.8560 |
| **PR-AUC** | **0.6634** | 0.6634 | 0.6669 |
| **Accuracy** | **79.84%** | 79.13% | 75.23% |
| **Precision** | **62.71%** | 59.85% | 52.17% |
| **Recall** | **59.36%** | 64.97% | 80.48% |
| **F1-Score** | **0.6099** | 0.6231 | 0.6330 |
| **Brier Score** | **0.1373** (Sharp calibration) | 0.1373 | 0.1599 (Uncalibrated) |

### Survival Model Metric:
- **Cox Proportional Hazards Concordance Index:** **0.8277**

### Segmentation Metric:
- **Optimal K:** 3 Clusters (Silhouette Score: **0.4099**)

---

## 5. Quantitative Subgroup Fairness Analysis

| Demographic Cohort | Test Samples | Empirical Churn Rate | Model Accuracy | Model Precision | Model Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Senior Citizens (Age >= 65)** | 231 | 41.6% | 76.2% | 68.8% | 64.6% |
| **Non-Seniors** | 1,178 | 23.6% | 80.6% | 60.5% | 57.6% |
| **Month-to-Month Contracts** | 775 | 42.7% | 73.0% | 64.1% | 68.3% |
| **Two-Year Contracts** | 339 | 2.7% | 96.8% | 50.0% | 11.1% |

---

## 6. Ethical Considerations & Limitations
1. **Correlation vs Causality:** Model probabilities indicate historical association. High-risk predictions must not be used to prematurely restrict service.
2. **Observational Limitations:** The IBM dataset does not record randomized interventions. True treatment response (uplift) requires running live A/B tests.
