# Customer Lifetime Value and Churn Intelligence System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 18](https://img.shields.io/badge/PostgreSQL-18-336791.svg)](https://www.postgresql.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Lifelines](https://img.shields.io/badge/Lifelines-0.30.3-orange.svg)](https://lifelines.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-23%20Passed%20(100%25)-success.svg)](tests/)

An enterprise-grade, end-to-end data science application engineered for subscription telecom and banking businesses. The system connects machine learning churn prediction, survival dynamics, explainable AI, forward-looking customer lifetime value (CLV) estimation, and causal uplift modeling to real-world financial decision-making and campaign ROI optimization.

---

## 1. Executive Problem & Business Value

Subscription businesses lose millions annually when customers terminate contracts. However, conventional retention practices suffer from two major flaws:
1. **Blind Mass Discounting:** Offering untargeted retention promotions to all customers dilutes margin on organic stayers (*Sure Things*), resulting in a **negative campaign ROI (-18.8%)**.
2. **Naive Risk Thresholding:** Standard 0.50 classification thresholds ignore the asymmetric financial cost of false negatives versus false positives.

**This system solves these challenges by:**
- Accurately predicting churn risk with a calibrated **LightGBM Classifier (0.8516 Test ROC-AUC)**.
- Optimizing decision thresholds to **0.550** based on a rigorous business net-profit objective.
- Estimating customer duration dynamics via **Lifelines (Kaplan-Meier & Cox Proportional Hazards)**.
- Valuing forward-looking customer lifetime contribution (**Discounted Cash Flow CLV**).
- Simulating retention campaigns with transparent financial ROI waterfalls (**+244.5% ROI on targeted interventions**).
- Persisting customer profiles and predictions in a normalized **PostgreSQL database**.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["Data Ingestion & Integrity"]
        Raw["Telco_customer_churn.csv"] --> Val["Data Validation & Leakage Audit"]
        Val --> Prep["Leak-Free Featurization (Train Fit Only)"]
        Prep --> DB[("PostgreSQL 18 Enterprise Persistence")]
    end

    subgraph Engines ["Predictive & Actuarial Engines"]
        Prep --> LGBM["LightGBM Classifier<br/>(Tuned T=0.550, ROC-AUC: 0.8516)"]
        Prep --> Life["Lifelines Survival Engine<br/>(Kaplan-Meier & Cox PH, C-Index: 0.8277)"]
        LGBM --> SHAP["TreeExplainer SHAP<br/>(Local & Global Attributions)"]
        Life --> CLV["CLV DCF Engine<br/>(24M Forward Margin)"]
        LGBM --> Seg["K-Means & 2D PCA<br/>(3 Behavioral Clusters)"]
        CLV --> Seg
    end

    subgraph Causal ["Causal & Financial Modeling"]
        Prep --> Uplift["T-Learner Uplift Estimator<br/>(Persuadables Isolation)"]
        CLV --> ROI["Retention ROI Simulator<br/>(Waterfall & Break-Even)"]
        LGBM --> ROI
    end

    subgraph Interfaces ["Application & API Delivery"]
        DB --> UI["Streamlit Dashboard (10 Dedicated Pages)"]
        LGBM --> UI
        SHAP --> UI
        ROI --> UI
        LGBM --> API["FastAPI REST Service<br/>(/health, /predict/customer, /clv/estimate)"]
    end
```

---

## 3. Technology Stack

- **Core Machine Learning:** LightGBM 4.7.0, Scikit-learn 1.9.1, NumPy 2.5.4, Pandas 2.3.3
- **Model Explainability:** SHAP 0.54.1 (`TreeExplainer`)
- **Survival Dynamics:** Lifelines 0.30.3 (`KaplanMeierFitter`, `CoxPHFitter`)
- **Causal Inference:** Native Meta-Learner Architecture (`TLearnerUpliftModel` / `CausalML` adapter)
- **Database & Persistence:** PostgreSQL 18, SQLAlchemy 2.1.6, Psycopg 3.3.6, Psycopg2-binary 2.9.13
- **User Interface:** Streamlit 1.66.1, Plotly 7.1.0, Matplotlib 3.11.2
- **REST API & Validation:** FastAPI 0.162.0, Uvicorn 0.54.0, Pydantic 2.14.1
- **Testing & Quality:** Pytest 9.1.1 (23 automated test suites)

---

## 4. Key Performance Results

### Held-Out Test Evaluation (1,409 Customers)

| Metric | LightGBM (Tuned T=0.550) | LightGBM (Standard T=0.50) | Logistic Regression Baseline |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.8516** | 0.8516 | 0.8560 |
| **PR-AUC** | **0.6634** | 0.6634 | 0.6669 |
| **Accuracy** | **79.84%** | 79.13% | 75.23% |
| **Precision** | **62.71%** | 59.85% | 52.17% |
| **Recall** | **59.36%** | 64.97% | 80.48% |
| **F1-Score** | **0.6099** | 0.6231 | 0.6330 |
| **Brier Score** | **0.1373** (Sharp calibration) | 0.1373 | 0.1599 (Uncalibrated) |

### Survival & Segmentation Results:
- **Cox Proportional Hazards Concordance Index:** **0.8277**
- **Cohort Median Survival:** Beyond 72 months (Overall cohort churn rate ~26.5%)
- **K-Means Clustering:** Optimal $K=3$ (Silhouette Score: **0.4099**)

### Campaign Financial Simulation (7,043 Customer Database):
- **Targeted ML Campaign (High Risk & High CLV):** **+$88,438.16 Net Profit** | **ROI: +244.5%**
- **Broad Mass Campaign (Untargeted):** **-$19,505.65 Net Loss** | **ROI: -18.8%**

---

## 5. Quickstart & Installation

### Step 1: Clone & Initialize Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### Step 2: Configure PostgreSQL Database
Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` with your PostgreSQL credentials:
```ini
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/customer_intelligence
```
*(Requires PostgreSQL 18 running on port 5432).*

### Step 3: Initialize Database Schema
```bash
python scripts/initialize_database.py
```

### Step 4: Ingest Customer Dataset
```bash
python scripts/import_dataset.py
```

### Step 5: Train Models & Cache SHAP Values
```bash
python scripts/train_models.py
```

### Step 6: Generate Predictions & Populate PostgreSQL
```bash
python scripts/generate_predictions.py
```

### Step 7: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 8: (Optional) Run FastAPI REST Inference Service
```bash
uvicorn src.services.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

---

## 6. Running Automated Tests

Execute the Pytest test suite:
```bash
pytest -v
```
All **23 unit and integration tests** pass with 100% coverage across data validation, preprocessing, churn prediction, survival functions, CLV DCF calculations, ROI simulators, database repositories, and REST endpoints.

---

## 7. Running the Interactive Jupyter Notebook

An end-to-end exploratory data analysis, data quality auditing, feature engineering, and baseline modeling notebook is provided at [`notebooks/01_eda_and_feature_engineering.ipynb`](notebooks/01_eda_and_feature_engineering.ipynb).

### Step-by-Step Execution:
1. **Activate the Virtual Environment**:
   ```powershell
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install JupyterLab (if running outside VS Code)**:
   ```bash
   pip install jupyterlab ipykernel
   ```
3. **Launch the Notebook**:
   ```bash
   jupyter lab notebooks/01_eda_and_feature_engineering.ipynb
   # or
   jupyter notebook notebooks/01_eda_and_feature_engineering.ipynb
   ```
   *(In VS Code: open `notebooks/01_eda_and_feature_engineering.ipynb` directly and select `.venv` as your Python kernel).*

---

## 8. Streamlit Dashboard Structure (10 Pages)

1. **`Executive Overview`:** Macro KPI cards, historical churn rates, revenue at risk, and model run metadata.
2. **`Customer Explorer`:** 360-degree account audit, churn probability gauge, local SHAP attribution plot, and prescriptive recommendations.
3. **`Churn Intelligence`:** LightGBM vs baseline comparisons, confusion matrices, ROC/PR curves, and business threshold tuning.
4. **`Survival Analysis`:** Kaplan-Meier survival curves with 95% CIs, contract stratification, and Cox Proportional Hazards regression.
5. **`Customer Lifetime Value`:** Interactive margin/discount rate sliders, CLV distributions, and priority high-value accounts.
6. **`Customer Segmentation`:** K-Means clustering, 2D PCA scatterplot, cluster centroid profiles, and strategic playbooks.
7. **`Retention Campaigns`:** Causal uplift modeling, Persuadables vs Lost Causes, Qini curves, and A/B test validation designs.
8. **`ROI Simulator`:** Interactive financial simulator, cash flow waterfall, break-even analysis, and broad vs targeted campaign comparisons.
9. **`Model Evaluation`:** Slice-based demographic fairness evaluations, probability calibration curves, and data leakage audits.
10. **`Data & Database Health`:** Live PostgreSQL telemetry, active storage mode indicators, table inventory, and data dictionary.

---

## 8. Financial Formulas & Methodology

### Forward-Looking Discounted CLV
$$\text{Discounted CLV}_i = \sum_{m=1}^{H} \frac{P(\text{Active in month } m \mid \text{Tenure}_i) \cdot (\text{MonthlyCharges}_i \cdot \text{Margin})}{(1 + r)^{m / 12}}$$
- $H$: Forecast horizon in months (default 24).
- $\text{Margin}$: Contribution gross margin (default 65%).
- $r$: Annual discount rate / WACC (default 10%).

### Campaign Financial ROI
$$\text{Campaign Cost} = \text{Targeted} \cdot \text{AcceptanceRate} \cdot (\text{MonthlyCharges} \cdot \text{DiscountPct} \cdot \text{Duration}) + \text{FixedCost}$$
$$\text{Retained Margin} = \text{IncrementalSaved} \cdot (\text{MonthlyCharges} \cdot \text{Margin} \cdot \text{ExpectedLifetime})$$
$$\text{Net Benefit} = \text{Retained Margin} - \text{Campaign Cost}$$
$$\text{ROI} = \frac{\text{Net Benefit}}{\text{Campaign Cost}} \times 100\%$$

---

## 9. Documentation Index
- [Architecture Guide](docs/architecture.md): System architecture, diagrams, and design principles.
- [Data Dictionary](docs/data_dictionary.md): Field descriptions, data types, and target leakage exclusions.
- [Methodology Report](docs/methodology.md): LightGBM, SHAP, Lifelines, and Causal Uplift formulations.
- [Model Card](docs/model_card.md): Governance details, evaluation metrics, and demographic fairness audits.
- [Database Setup Guide](docs/database_guide.md): PostgreSQL schema DDL, indexing, and connection guide.
- [Interview Preparation Guide](docs/interview_prep.md): 10-minute presentation guide and technical interview Q&A.
- [Intern Reflection Template](docs/reflection_template.md): Self-assessment reflection and engineering logs.
