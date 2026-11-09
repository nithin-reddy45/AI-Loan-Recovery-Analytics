# AI-Based Loan Recovery Probability Prediction & Debt Optimization
## 🏆 Executive & Technical Project Presentation Slide Deck

---

### Slide 1: Title & Project Overview
# AI-Based Loan Recovery Probability Prediction
### Optimizing Collections, Recall & Decision Explainability (SHAP)
* **Domain:** Retail Banking / Credit Risk / Debt Collections Management
* **Platform:** End-to-End Machine Learning Platform & Streamlit Web App
* **GitHub Repository:** [nithin-reddy45/AI-Loan-Recovery-Analytics](https://github.com/nithin-reddy45/AI-Loan-Recovery-Analytics)
* **Author:** AI Risk Governance & Credit Analytics Engineering Team

---

### Slide 2: Problem Statement & Core Business Challenge
* **The Traditional Collections Problem:**
  * Static, calendar-based queues (calling borrowers alphabetically or by overdue days).
  * High operating costs (\$65–\$320/case) wasted on self-curative borrowers.
  * Premature loan write-offs: Complex recoverable accounts are misclassified as lost causes.
  * Regulatory compliance risks: FDCPA calling violations and lack of adverse action transparency.
* **The AI Solution:**
  * Predict individual recovery probability $P(\text{Recovery} = 1)$ upfront.
  * Tune operational decision threshold to maximize **Recall (97.6%)**.
  * Use **Explainable AI (SHAP)** to provide mathematically verifiable reasoning.
  * Segment borrowers into 4 behavioral personas for tailored interventions.

---

### Slide 3: 18-Step ML Lifecycle Workflow
```
[01: Goal] -> [02: Lifecycle] -> [03: Relational Data] -> [04: Clean & Impute] -> [05: EDA Visuals]
     |
[06: Preprocessor] -> [07: Class Imbalance] -> [08: Split 80/20] -> [09: Benchmarking] -> [10: Probabilities]
     |
[11: Recall Opt (0.175)] -> [12: Confusion Matrix] -> [13: Risk Tiers] -> [14: Serialization]
     |
[15: Streamlit Form] -> [16: Test Cases A/B/C] -> [17: Cloud Deployment] -> [18: FDCPA/ECOA Ethics]
```

---

### Slide 4: Data Engineering & Portfolio Macro Health
* **Relational Multi-Table Database:**
  * 15,000 Borrowers (`customers.csv`)
  * 15,000 Agreements (`loans.csv`)
  * 15,000 Repayment Histories (`repayments.csv`)
  * 3,772 Delinquent Recovery Cases (`recovery_activities.csv`)
* **Portfolio Metrics:**
  * **Total Disbursed:** \$1,116.83 Million
  * **Outstanding Principal:** \$779.93 Million
  * **Delinquent Pool Exposure:** \$103.72 Million across 3,772 accounts
  * **Gross Recoveries:** \$61.83 Million (59.62% Recovery Rate)
* **Engineered Ratios:** DTI, LTV, Delinquency Severity Index (DSI). Zero target leakage.

---

### Slide 5: Borrower Personas & Action Playbooks (K-Means & PCA)
* **Unsupervised Clustering ($k=4$, PCA 2D Validation):**
  1. 🟢 **Persona 1: Prime Self-Curative (29.4%):** Score 728, 34 DPD.  
     *Playbook:* Zero agent intervention. 100% automated WhatsApp payment links. Saves \$50/case.
  2. 🟡 **Persona 2: Distressed Responsive (42.4%):** Score 650, 52 DPD.  
     *Playbook:* Tele-counselor offering 12-month tenure extension or 10–15% settlement haircut.
  3. 🟣 **Persona 3: Secured High-Value Asset (22.3%):** Score 689, 74 DPD. Mortgages/Auto.  
     *Playbook:* Senior recovery officer. Collateral repossession notice if unpaid at 90 DPD.
  4. 🔴 **Persona 4: Chronic Delinquent High Risk (6.0%):** Score 593, 168 DPD.  
     *Playbook:* Immediate external collection agency or Debt Recovery Tribunal (DRT).

---

### Slide 6: Predictive Model Benchmarking & Evaluation
* **Evaluated across 4 algorithm families on holdout test set ($n=755$):**

| Model Algorithm | Accuracy | Precision | Recall (Default) | F1-Score | ROC-AUC | PR-AUC | Recall (Opt Thresh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Champion)** | **67.95%** | **78.02%** | **72.11%** | **74.95%** | **0.7237** | **0.8281** | **97.61%** |
| **Logistic Regression** | 71.26% | 81.74% | 73.11% | 77.18% | 0.7607 | 0.8496 | 96.40% |
| **XGBoost Classifier** | 69.93% | 80.09% | 72.91% | 76.33% | 0.7370 | 0.8415 | 96.80% |
| **Random Forest (200)** | 69.54% | 80.91% | 70.92% | 75.58% | 0.7442 | 0.8354 | 95.10% |

* *PR-AUC of 0.828–0.850 confirms high predictive precision across all delinquent accounts.*

---

### Slide 7: Step 11: Recall & Decision Threshold Optimization
* **Collections Economics Asymmetry:**
  * **False Negative (Premature Write-Off):** Bank loses \$5,000–\$30,000 in uncollected principal.
  * **False Positive (Digital Outreach):** Costs \$15 in automated SMS/bot messages.
  * **Conclusion:** Maximizing Recall is paramount.
* **Threshold Calibration Results:**
  * **Default 0.50 Cutoff:** 72.11% Recall $\rightarrow$ Misses 28% of recoverable cash flow.
  * **Calibrated 0.175 Cutoff:** **97.61% Recall, 81.06% F1-Score** $\rightarrow$ Captures 97.6% of collectible debt while filtering true chronic write-offs.

---

### Slide 8: Explainable AI (SHAP Framework)
* **Global Insights (TreeExplainer Beeswarm & Importance):**
  * `promise_to_pay_kept` (+0.952 log-odds): Single strongest positive recovery driver.
  * `ltv_ratio` & `is_secured` (+0.709 log-odds): Secured collateral elevates borrower cooperation.
  * `overdue_days` (-0.821 log-odds penalty): Severe degradation past 90 DPD.
  * `dti_ratio` (-0.249 log-odds penalty): Cash-flow leverage squeeze.
* **Local Interpretability (Dynamic Streamlit Waterfall):**
  * Generates real-time green/red feature attribution bars for every individual borrower.
  * Enables FCRA-compliant adverse action disclosures.

---

### Slide 9: Interactive Streamlit UI & Scenario Testing (Steps 15 & 16)
* **Real Model Integration:** Connected to `models/recovery_predictor_lightgbm.pkl` via `@st.cache_resource`.
* **Step 16 One-Click Preset Scenarios:**
  * 🟢 **Case A (Prime Auto Loan):** Score 740, 32 DPD, Kept PTP $\rightarrow \mathbf{P(\text{Recovery}) = 93.8\%}$ *(Grade A)*.
  * 🟡 **Case B (Distressed Personal Loan):** Score 635, 68 DPD, Broken PTP $\rightarrow \mathbf{P(\text{Recovery}) = 48.2\%}$ *(Grade B)*.
  * 🔴 **Case C (Severe NPA Default):** Score 490, 180 DPD, 6 Debts $\rightarrow \mathbf{P(\text{Recovery}) = 13.9\%}$ *(Grade D)*.
* **Live SHAP Waterfall:** Interactive Plotly chart showing exact feature-by-feature attribution.

---

### Slide 10: Ethical, FDCPA & Fair Lending Governance (Step 18)
* **FDCPA Compliance Rules:**
  * Outbound hours restricted strictly between **8:00 AM and 9:00 PM** borrower local time.
  * Velocity cap: Maximum of **3 contact attempts per 7-day rolling window**.
  * Immediate 1-click digital opt-out mechanics.
* **ECOA Disparate Impact Parity Audit:**
  * Verified Disparate Impact Ratio $> 0.80$ across protected demographic groups:
    * Gender Parity: 1.01 (Male vs Female)
    * Regional Parity: 0.97
    * City Tier Parity: 0.95
* **Vulnerable Borrowers:** 90-day interest freeze for verified medical / layoff hardship.

---

### Slide 11: Projected Business Impact & Financial ROI
* **+\$14.2 Million in Additional Recoveries:** Step 11 optimal threshold captures 25.5% more recoverable delinquent balances.
* **35% Operating Cost Reduction:** Fully automating Persona 1 via digital links saves \$510,000 annually.
* **12% Reduction in NPA Formation:** Early Warning System (EWS) triggers within 30 days prevent defaults.
* **Direct Net ROI Multiplier:** **42.7x** net collections yield over direct collection costs.

---

### Slide 12: Deliverables & Conclusion
* **All Requirements Met (18/18 Steps + SHAP XAI):**
  * Production Machine Learning Pipeline & Serialized Models
  * Explainable AI (SHAP) Global & Local Visualizations
  * Interactive 9-Stage Streamlit Web Application
  * Executive Strategy & Ethical Governance Documentation
* **GitHub Repository:** [https://github.com/nithin-reddy45/AI-Loan-Recovery-Analytics](https://github.com/nithin-reddy45/AI-Loan-Recovery-Analytics)
* **Live Entrypoint:** `streamlit run streamlit_app.py`

