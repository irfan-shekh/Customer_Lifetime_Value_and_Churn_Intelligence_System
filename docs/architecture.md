# System Architecture: Customer Lifetime Value & Churn Intelligence System

```mermaid
flowchart TD
    subgraph Data_Layer ["Data Ingestion & Persistence"]
        RawCSV[("Raw Telco CSV Data")] --> Loader["load_data.py"]
        Loader --> Validator["validate_data.py"]
        Validator --> Preprocessor["preprocess.py & features.py"]
        Preprocessor --> PostgreSQL[("PostgreSQL 18 Database")]
    end

    subgraph Modeling_Layer ["Predictive & Actuarial Engines"]
        Preprocessor --> LightGBM["LightGBM Classifier<br/>(Class Imbalance & Threshold Tuned)"]
        Preprocessor --> Lifelines["Lifelines Survival Engine<br/>(Kaplan-Meier & Cox PH)"]
        LightGBM --> SHAP["SHAP TreeExplainer<br/>(Risk Attributions)"]
        Lifelines --> CLV["CLV Actuarial Engine<br/>(Discounted Cash Flows)"]
        LightGBM --> Seg["K-Means Clustering & PCA<br/>(Behavioral Personas)"]
        CLV --> Seg
    end

    subgraph Causal_Layer ["Causal & Financial Simulation"]
        Preprocessor --> MetaLearner["T-Learner Uplift Estimator<br/>(Persuadables Isolation)"]
        CLV --> ROISim["Retention ROI Simulator<br/>(Cash Flow Waterfall & Break-Even)"]
        LightGBM --> ROISim
    end

    subgraph Delivery_Layer ["User & API Interfaces"]
        PostgreSQL --> Streamlit["Streamlit Multipage App (10 Pages)"]
        LightGBM --> Streamlit
        SHAP --> Streamlit
        CLV --> Streamlit
        ROISim --> Streamlit
        LightGBM --> FastAPI["FastAPI REST Inference Service<br/>(/predict/customer, /clv/estimate)"]
    end
```

## Architectural Design Principles

1. **Separation of Concerns:** 
   Raw data ingestion, feature transformation, statistical modeling, persistence, and presentation are strictly decoupled into independent modules under `src/`.
2. **Leak-Free Featurization Boundary:** 
   StandardScaler and OneHotEncoder are fitted exclusively on training splits after stratified partitioning. Held-out validation and test sets remain completely unseen.
3. **Enterprise PostgreSQL Persistence:**
   Full PostgreSQL schema support with parameterized SQLAlchemy transactions. Decouples raw operational records from model-generated inferences, with real-time connection telemetry across UI banners.
4. **Distinction Between Prediction and Causality:**
   Predictive churn flags ($P(\text{Churn} \mid X)$) are treated strictly as correlations. Causal treatment response (uplift $\tau(X)$) is modeled separately via Meta-Learners and clearly distinguished from observational data.
