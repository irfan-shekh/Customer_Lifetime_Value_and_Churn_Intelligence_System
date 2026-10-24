# Statistical & Machine Learning Methodology Report

## 1. Predictive Churn Modeling (LightGBM)

### 1.1 Objective & Architecture
Customer churn prediction is framed as a supervised binary classification problem:
$$P(Y = 1 \mid X) = \sigma(f(X))$$
Where $Y \in \{0, 1\}$ denotes whether the subscriber churns, and $X$ represents demographic, account, and behavioral attributes.

### 1.2 Class Imbalance Handling
The Telco cohort exhibits moderate class imbalance (~26.5% positive churn). Rather than synthetically inflating data using SMOTE (which risks creating biologically unrealistic feature combinations in mixed categorical spaces), we adjust the loss gradient via LightGBM's `scale_pos_weight = 1.5`, penalizing false negatives proportionally.

### 1.3 Decision Threshold Optimization
Conventional accuracy-maximizing models apply a naive threshold $T = 0.50$. In subscription telecom, the asymmetric business costs are:
- **Cost of False Positive (Targeting stayer):** \$20 incentive cost.
- **Cost of False Negative (Missing churner):** \$120+ lost future customer margin.

We optimize $T$ on the validation set using a business utility objective:
$$\text{Net Business Value}(T) = \sum_{i} \left[ \text{TP}_i(T) \cdot \text{Uplift} \cdot \text{SavedValue} - (\text{TP}_i(T) + \text{FP}_i(T)) \cdot \text{Cost} \right]$$
The validation-optimal threshold is selected at **$T = 0.550$**, yielding a balanced Precision of 62.7% and Recall of 59.4% on held-out test data.

---

## 2. Explainable AI (TreeExplainer SHAP)

To explain LightGBM predictions without computational approximations, we employ `shap.TreeExplainer`:
$$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$
Where $\phi_0 = \mathbb{E}[f(X)]$ is the base expectation, and $\phi_j(x)$ represents the Shapley value (marginal attribution) of feature $j$.

### Critical Methodological Distinction: Attribution vs Causality
- **SHAP values reflect model attribution:** They identify which features the gradient-boosted trees utilized to arrive at a high risk probability.
- **SHAP values do NOT prove causality:** For example, while `is_month_to_month` has a very high positive SHAP score, forcing a customer onto a 2-year contract without consent may cause immediate churn rather than loyalty.

---

## 3. Survival Analysis & Right-Censoring (Lifelines)

### 3.1 Kaplan-Meier Survival Estimator
Observed tenure data is right-censored for active customers ($C_i = 1$ if churned, $C_i = 0$ if censored). The Kaplan-Meier estimator models cumulative survival probability non-parametrically:
$$\hat{S}(t) = \prod_{t_i \le t} \left( 1 - \frac{d_i}{n_i} \right)$$
Where $d_i$ is the number of churn events at time $t_i$, and $n_i$ is the number of customers at risk.

### 3.2 Conditional Retention Horizons
For an existing customer with current tenure $T_0$, the probability of surviving an additional $\Delta t$ months is computed conditionally:
$$P(T > T_0 + \Delta t \mid T > T_0) = \frac{\hat{S}(T_0 + \Delta t)}{\hat{S}(T_0)}$$

### 3.3 Cox Proportional Hazards Model
To incorporate multiple covariates into duration analysis, we fit a semi-parametric Cox model:
$$h(t \mid X) = h_0(t) \exp(\beta_1 X_1 + \dots + \beta_p X_p)$$
The model achieves a **Concordance Index of 0.8277**, indicating strong discriminatory power in ranking customer lifetimes.

---

## 4. Customer Lifetime Value (CLV) Formulation

Traditional models mistakenly equate historical revenue (`TotalCharges`) with forward-looking CLV. We employ an actuarially sound Discounted Cash Flow (DCF) model over a configurable horizon $H$ (default 24 months):
$$\text{Discounted CLV}_i = \sum_{m=1}^{H} \frac{P(\text{Active in month } m \mid \text{Tenure}_i) \cdot (\text{MonthlyCharges}_i \cdot \text{Margin})}{(1 + r)^{m / 12}}$$
- **Monthly Gross Margin ($\text{Margin}$):** Default 65%.
- **Annual WACC / Discount Rate ($r$):** Default 10%.
- **Revenue at Risk:** $\text{CLV}_i \times P(\text{Churn}_i)$.

---

## 5. Causal Uplift & Meta-Learners (T-Learner)

Uplift modeling estimates the Conditional Average Treatment Effect (CATE):
$$\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$$
Using a Two-Model (T-Learner) framework:
1. Model $\mu_0(X) = \mathbb{E}[Y \mid X, W = 0]$ trained on control group.
2. Model $\mu_1(X) = \mathbb{E}[Y \mid X, W = 1]$ trained on treated group.
3. Uplift score: $\hat{\tau}(X) = \hat{\mu}_0(X) - \hat{\mu}_1(X)$.

Customers with high $\hat{\tau}(X)$ are isolated as **Persuadables**, eliminating wasteful marketing spend on **Sure Things** (organic stayers) and **Lost Causes**.
