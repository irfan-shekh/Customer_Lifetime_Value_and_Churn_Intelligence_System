# Interview Preparation & Executive Presentation Guide

## 1. 10-Minute Executive Presentation Outline

| Time | Slide / Topic | Key Talking Points |
| :--- | :--- | :--- |
| **0:00 - 1:30** | **Business Problem & Financial Stakes** | Telecom businesses face recurring margin erosion from subscriber attrition. Mass untargeted discounting dilutes margin (-18.8% ROI), while predictive ML targeting yields over +240% ROI. |
| **1:30 - 3:30** | **Data Engineering & PostgreSQL Architecture** | Normalized schema in PostgreSQL 18 with SQLAlchemy 2.0; audited target leakage (`Churn Reason`, `Churn Score`, `CLTV`); leak-free featurization fitted strictly on training data. |
| **3:30 - 5:30** | **Predictive Modeling & Threshold Tuning** | LightGBM achieved 0.8516 Test ROC-AUC and 0.1373 Brier score. Explain why 0.50 threshold was rejected and how 0.550 maximizes net business profit. |
| **5:30 - 7:00** | **Survival Dynamics & Forward CLV** | Kaplan-Meier & Cox PH (C-index: 0.8277); explicit right-censoring treatment; discounted cash flow formulation distinguishing realized spend from future margin. |
| **7:00 - 8:30** | **Causal Uplift & ROI Simulation** | Explain the *Persuadables* concept; Qini curve; demonstrate the ROI simulator waterfall showing why broad discounting fails vs targeted campaigns. |
| **8:30 - 10:00** | **Production Delivery & Q&A Transition** | Multipage Streamlit application; FastAPI REST inference service; 100% passing Pytest suite. |

---

## 2. Likely Technical & Business Interview Questions with Model Answers

### Q1: Why didn't you evaluate the churn model on Accuracy alone?
> **Model Answer:**
> In our dataset, the baseline churn rate is ~26.5%. A naive dummy classifier predicting that every customer stays would achieve 73.5% accuracy while identifying exactly zero churners!
> Furthermore, accuracy treats false positives (giving a discount to someone who would stay) and false negatives (losing an entire customer) as equally costly. Instead, we prioritized **ROC-AUC (0.8516)**, **PR-AUC (0.6634)**, and **Brier Score (0.1373)**, tuning the classification threshold to maximize net business retention profit.

### Q2: How did you select the decision threshold (0.550) instead of the default 0.50?
> **Model Answer:**
> We conducted a business utility simulation on the validation partition across thresholds from 0.10 to 0.85. For each threshold, we computed the expected campaign cost of targeting predicted positives versus the expected margin preserved from true positives. The profit-maximizing curve peaked at $T = 0.550$. On the test set, this threshold achieved a balanced 62.7% Precision and 59.4% Recall, protecting our campaign from excessive dilution.

### Q3: What is right-censoring in survival analysis and why is it critical here?
> **Model Answer:**
> In customer subscriber cohorts, active customers (`Churn = 0`) have not churned *yet*, but will churn at some unknown future date. If you drop them or treat them as having infinite lifetimes, you introduce severe survival bias. The Kaplan-Meier and Cox models in `lifelines` handle right-censoring by keeping active customers in the "risk set" up until their current observed tenure, then correctly removing them from subsequent hazard denominators without penalizing the survival curve.

### Q4: If a customer has a high positive SHAP score for Month-to-Month contract, can we say the contract caused their churn?
> **Model Answer:**
> No! SHAP values quantify predictive attribution inside the decision tree forest—meaning Month-to-Month contracts correlate heavily with churn risk in the observational training data. However, correlation is not causation. Customers who intend to leave soon naturally select Month-to-Month contracts to avoid exit fees. Forcing them onto annual contracts could trigger immediate churn. Proving causality requires an A/B test with randomized holdouts.

### Q5: What is the difference between Churn Prediction and Uplift Modeling?
> **Model Answer:**
> Churn prediction answers: *"Who is most likely to leave?"*
> Uplift modeling answers: *"Who will change their decision and stay if we offer an incentive?"*
> Targeting customers with high churn risk often wastes budget on **Lost Causes** (who leave anyway) or **Sure Things** (who were never leaving). Uplift modeling uses Meta-Learners (like our T-Learner) to isolate the **Persuadables**, ensuring budget is spent only where incremental response is positive.

### Q6: Why did the Broad Campaign produce a negative ROI (-18.8%) while the Targeted Campaign generated +244.5% ROI?
> **Model Answer:**
> In a broad, untargeted campaign across all 7,043 customers, 73.5% of the recipients were organic stayers ("Sure Things"). Giving them a 15% discount for 3 months gave away substantial revenue with zero incremental retention benefit.
> In contrast, the ML-targeted campaign restricted offers to the high-risk, high-CLV cohort (baseline churn ~65%), where incremental uplift yielded substantial retained contribution that easily eclipsed campaign costs.
