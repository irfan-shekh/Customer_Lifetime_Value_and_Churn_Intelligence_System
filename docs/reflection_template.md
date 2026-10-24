# Internship Assessment Reflection & Engineering Log

## 1. Executive Summary & Personal Contribution
- **Project Name:** Customer Lifetime Value and Churn Intelligence System
- **Candidate Name:** [Your Name]
- **Role:** Data Science & ML Engineering Intern
- **Evaluation Period:** October 2026

---

## 2. Technical Accomplishments & Key Wins

### A. Data Engineering & Integrity
- Designed a normalized schema in **PostgreSQL 18** using SQLAlchemy 2.0 and psycopg, featuring foreign keys, check constraints, and performance indexes.
- Successfully audited and mitigated **4 critical target leakage vectors** in the raw IBM Telco dataset (`Churn Reason`, `Churn Score`, `CLTV`, and duplicate labels).
- Developed a robust preprocessing pipeline handling 11 blank-string `TotalCharges` entries and implementing leak-free featurization fitted strictly on training data.

### B. Machine Learning & Model Performance
- Implemented **LightGBM Classifier** achieving **0.8516 Test ROC-AUC**, **0.6634 PR-AUC**, and **0.1373 Brier Score**.
- Proved why naive 0.50 thresholding was suboptimal and optimized the threshold to **0.550** based on business net profit.
- Integrated **TreeExplainer SHAP** for both global feature importance and individual customer risk attribution.

### C. Advanced Statistical Learning: Survival & Uplift
- Built **Lifelines Survival Engine** (Kaplan-Meier & Cox Proportional Hazards with Concordance Index 0.8277), solving right-censoring for active subscribers.
- Formulated forward-looking **Discounted Cash Flow CLV** based on conditional survival probabilities.
- Built a **T-Learner Meta-Estimator** and evaluated Qini cumulative curves, clearly distinguishing observational correlations from causal counterfactuals.

### D. Full-Stack Product Delivery
- Delivered a multipage **Streamlit Application (10 pages)** with interactive financial waterfall simulators and 2D PCA cluster maps.
- Engineered a production **FastAPI REST inference service** (`/health`, `/predict/customer`, `/clv/estimate`).
- Authored **23 automated Pytest test cases** achieving 100% test pass rate.

---

## 3. Engineering Challenges & Problem Solving

| Challenge Encountered | Root Cause | Engineering Solution Implemented |
| :--- | :--- | :--- |
| **Windows Python 3.13 Build Failure for CausalML** | CausalML tree extensions require Microsoft Visual C++ 14.0+ compiler on Windows. | Built a native, modular Meta-Learner (T-Learner/S-Learner) architecture in scikit-learn/LightGBM with Qini evaluation, while providing dynamic CausalML hooks. |
| **Infinite Median Survival Time** | Cohort churn rate (~26.5%) never crosses the 50% threshold within 72 months. | Handled `np.isinf` in PostgreSQL numeric columns by mapping infinite duration to SQL `NULL` and documenting cohort longevity. |
| **Row-by-Row Pandas CLV Bottleneck** | Repeated `survival_function_at_times` lookups across 7,000 customers took ~40s. | Vectorized CLV calculations by precomputing a 73-point tenure multiplier lookup array, accelerating execution to sub-10ms. |

---

## 4. Candid Assessment of Current Limitations & Next Steps

1. **Observational Treatment Data:**
   The IBM Telco dataset lacks randomized intervention logs. The next step in a corporate deployment is launching an A/B test with 10% holdouts to collect true experimental data for CausalML.
2. **Dynamic Pricing & Contract Upgrades:**
   Current CLV models assume constant monthly spend over the 24-month horizon. Future iterations should incorporate transition Markov matrices modeling plan upgrades or downgrades.
