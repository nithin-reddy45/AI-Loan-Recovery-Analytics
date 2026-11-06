# 🏦 AI-Driven Loan Recovery Probability Prediction & Risk Analytics
### 🏆 Full ML Lifecycle Platform: Classification, Recall Optimization & Explainable AI (SHAP)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-yellow.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-purple.svg)](https://shap.readthedocs.io/)
[![Deploy](https://img.shields.io/badge/Streamlit_Cloud-Deployment_Ready-brightgreen.svg)](https://streamlit.io/cloud)

---

## 📌 Project Overview
**AI-Based Loan Recovery Probability Prediction** is an end-to-end, production-grade machine learning platform designed for retail banking, credit risk underwriting, and collections optimization. It enables financial institutions to predict debt recovery probability, optimize recall to prevent premature charge-offs, and inspect individual borrower decisions with **Explainable AI (SHAP)**.

### 🎯 Key Project Highlights:
1. 🔄 **Classification Target**: `recovery_target` (1 = Successfully Recovered / Resolved, 0 = Written Off / Non-Performing Asset Loss).
2. 🔬 **Explainable AI (SHAP TreeExplainer)**: Unpacks black-box model decisions into exact feature log-odds contributions (Beeswarm, Mean |SHAP|, and Live Borrower Waterfall plots).
3. 🎯 **Decision Threshold & Recall Optimization**: Calibrated decision threshold (**0.175** vs default 0.50) achieving **97.6% Recall** for maximized cash-flow recovery.
4. 📱 **Full Streamlit Application**: Interactive 9-week ML lifecycle platform with one-click test case presets, global filters, and live SHAP explanations.
5. ⚖️ **Ethical & Fair Lending Governance**: Full compliance documentation with FDCPA, ECOA disparate impact audits, and FCRA adverse action transparency.

---

## 📋 18-Step Step-by-Step Workflow Checklist

| Step | Requirement | Implementation & File Reference | Status |
| :---: | :--- | :--- | :---: |
| **01** | Define loan recovery prediction goal | [README.md](README.md), [executive_business_report.md](reports/executive_business_report.md) | ✅ Complete |
| **02** | Understand loan lifecycle and recovery process | SMA-0, SMA-1, SMA-2, NPA delinquency stages & channel dynamics | ✅ Complete |
| **03** | Collect borrower and loan history data | Multi-table relational datasets in [`data/raw/`](data/raw/) (15,000 customers) | ✅ Complete |
| **04** | Clean missing and inconsistent entries | Median/mode imputation, IQR outlier capping in [`src/data_cleaning.py`](src/data_cleaning.py) | ✅ Complete |
| **05** | Exploratory analysis on repayment behavior | Delinquency distribution & vintage analysis in [`src/eda_visualizations.py`](src/eda_visualizations.py) | ✅ Complete |
| **06** | Encode categorical and scale numerical features | `ColumnTransformer` with `StandardScaler` & `OneHotEncoder` | ✅ Complete |
| **07** | Handle class imbalance if present | `class_weight='balanced'` & `scale_pos_weight` in [`src/predictive_model.py`](src/predictive_model.py) | ✅ Complete |
| **08** | Split dataset into training and testing sets | 80/20 Stratified split maintaining class distribution | ✅ Complete |
| **09** | Select suitable classification algorithms | Logistic Regression, Random Forest, XGBoost, and LightGBM | ✅ Complete |
| **10** | Train recovery probability model | Multi-model benchmarking with continuous calibration $P(\text{Recovery})$ | ✅ Complete |
| **11** | Optimize recall and ROC-AUC metrics | Threshold scanning finding optimal threshold **0.175** (97.6% Recall, 81.1% F1) | ✅ Complete |
| **12** | Analyze confusion matrix results | Diagnostics plotted in [`reports/figures/11_confusion_matrix_and_feature_importance.png`](reports/figures/11_confusion_matrix_and_feature_importance.png) | ✅ Complete |
| **13** | Interpret recovery probability scores | Stratified into 4 Risk Tiers (Grade A–D) with tailored action playbooks | ✅ Complete |
| **14** | Save trained classification model | Serialized in [`models/recovery_predictor_lightgbm.pkl`](models/recovery_predictor_lightgbm.pkl) & metadata | ✅ Complete |
| **15** | Build Streamlit UI for borrower inputs | Form with financial & credit attributes in [`dashboard/modules/week9_dashboard.py`](dashboard/modules/week9_dashboard.py) | ✅ Complete |
| **16** | Test model using recovery/no-recovery cases | One-click test case presets (Prime Auto, Distressed Personal, Severe NPA) | ✅ Complete |
| **17** | Deploy app to Streamlit Cloud | Ready via [`streamlit_app.py`](streamlit_app.py), [`.streamlit/config.toml`](.streamlit/config.toml), and dependencies | ✅ Complete |
| **18** | Document business and ethical considerations | Dedicated report in [`reports/ethical_and_regulatory_considerations.md`](reports/ethical_and_regulatory_considerations.md) | ✅ Complete |
| **XAI**| **Explainable AI (SHAP Requirement)** | SHAP TreeExplainer, Beeswarm, Mean |SHAP| bar, and live Streamlit waterfall | ✅ Complete |

---

## 🏗️ Repository Architecture

```
LoanRecovery/
├── data/
│   ├── raw/                              # Raw multi-table banking datasets
│   │   ├── customers.csv                 # 15,000 borrower profiles (Demographics, Income, Credit Score)
│   │   ├── loans.csv                     # Loan agreements (Type, Amount, Term, Collateral, Channel)
│   │   ├── repayments.csv                # Repayment schedules, DPD, delinquency stages, defaults
│   │   └── recovery_activities.csv       # Recovery activities (Channel, Officer, Cost)
│   └── processed/                        # Cleaned analytical datasets
│       ├── loan_recovery_master.csv      # Unified 57-column analytical master dataset
│       └── delinquent_recovery_master.csv# Filtered delinquent accounts (>30 DPD) for ML
├── notebooks/                            # 4 Jupyter Notebooks
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_customer_segmentation_kmeans.ipynb
│   └── 04_predictive_modeling_loan_recovery.ipynb # Updated with SHAP & threshold tuning
├── src/                                  # Production analytics scripts
│   ├── generate_synthetic_data.py        # Relational financial data generator
│   ├── data_cleaning.py                  # Imputation, outlier handling, feature engineering
│   ├── eda_visualizations.py             # Portfolio visual charts generator
│   ├── segmentation.py                   # K-Means clustering, PCA, and persona profiling
│   ├── predictive_model.py               # ML training (LightGBM, XGBoost, RF, LR) + SHAP + Optimization
│   ├── generate_notebooks.py             # Automated notebook generator
│   └── run_pipeline.py                   # Master end-to-end pipeline runner
├── dashboard/                            # Streamlit BI & AI UI
│   ├── app.py                            # Streamlit 9-stage entrypoint
│   └── modules/                          # Modular weekly ML stages
│       ├── week7_evaluation.py           # Model diagnostics, ROC, SHAP tabs, threshold curve
│       └── week9_dashboard.py            # Live AI simulator + one-click test cases + SHAP waterfall
├── reports/                              # Strategic documentation & visual deliverables
│   ├── executive_business_report.md      # C-suite recommendations & EWS framework
│   ├── ethical_and_regulatory_considerations.md # Step 18: FDCPA, ECOA, FCRA, Ethics & AI Governance
│   └── figures/                          # Exported high-res PNG visual charts (14 plots)
│       ├── 10_model_roc_pr_curves.png
│       ├── 11_confusion_matrix_and_feature_importance.png
│       ├── 12_shap_summary_beeswarm.png  # SHAP Beeswarm plot
│       ├── 13_shap_feature_importance.png# SHAP Global Importance bar
│       └── 14_threshold_optimization_curve.png # Step 11 Recall curve
├── models/                               # Serialized ML models & metadata
│   ├── recovery_predictor_lightgbm.pkl   # Champion LightGBM pipeline
│   ├── recovery_predictor_xgboost.pkl    # Challenger XGBoost pipeline
│   ├── shap_explainer_data.pkl           # SHAP cache for fast UI inference
│   └── model_metadata.json               # Benchmark metrics, hyperparameters, threshold rules
├── requirements.txt                      # Project dependencies (includes shap)
├── streamlit_app.py                      # Streamlit Community Cloud entrypoint
└── README.md                             # Documentation
```

---

## 🔬 Explainable AI (SHAP) Insights

Traditional black-box machine learning models cannot be audited in regulated banking. Our architecture applies **SHAP (SHapley Additive exPlanations)** to interpret both global behavior and individual predictions:

### 1. Global Feature Impact (SHAP Beeswarm Plot)
- **`promise_to_pay_kept`**: Strongest positive predictor. Delinquent borrowers who previously honored commitments exhibit dramatically higher recovery probability (+0.95 log-odds).
- **`ltv_ratio` & `is_secured`**: Asset-backed loans (Auto/Home) provide legal and physical leverage that elevates recovery propensity.
- **`overdue_days` (DPD) & `dti_ratio`**: Prolonged default (>90 DPD) and excessive debt-to-income severely lower recovery likelihood.

![SHAP Beeswarm](reports/figures/12_shap_summary_beeswarm.png)
![SHAP Feature Importance](reports/figures/13_shap_feature_importance.png)

### 2. Local Borrower Explanations (Interactive Streamlit Waterfall)
For any borrower entered in the UI, the system generates real-time green/red feature attribution bars explaining exactly why the account was assigned a specific risk grade.

---

## 🎯 Step 11: Recall & Decision Threshold Optimization

In debt collections, a **False Negative** (classifying a recoverable borrower as unrecoverable) leads to unnecessary write-offs, while a **False Positive** merely results in an automated digital outreach. Therefore, **Recall** is the primary economic metric.

| Threshold | Operational Meaning | Recall | Precision | F1-Score | Strategy Implication |
|:---:|:---|:---:|:---:|:---:|:---|
| **0.50** | Standard Default | 72.11% | 78.02% | 74.95% | Misses 28% of collectible debt ($17M+ lost). |
| **0.175** | **Optimal Calibrated** | **97.61%** | **69.31%** | **81.06%** | **Captures 97.6% of collectible debt, maximizing net cash flow.** |

![Threshold Optimization](reports/figures/14_threshold_optimization_curve.png)

---

## 📊 Machine Learning Model Benchmark

Benchmarking 4 classification models on delinquent accounts (>30 DPD) without target leakage:

| Model | Accuracy | Precision | Recall (Default) | F1-Score | ROC-AUC | PR-AUC | Recall (Optimal Thresh) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM (Champion)** | **67.95%** | **78.02%** | **72.11%** | **74.95%** | **0.7237** | **0.8281** | **97.61%** |
| **Logistic Regression** | 71.26% | 81.74% | 73.11% | 77.18% | 0.7607 | 0.8496 | 96.40% |
| **XGBoost** | 69.93% | 80.09% | 72.91% | 76.33% | 0.7370 | 0.8415 | 96.80% |
| **Random Forest** | 69.54% | 80.91% | 70.92% | 75.58% | 0.7442 | 0.8354 | 95.10% |

![ROC Curves](reports/figures/10_model_roc_pr_curves.png)
![Confusion Matrix](reports/figures/11_confusion_matrix_and_feature_importance.png)

---

## 🚀 Step 17: Streamlit Cloud Deployment Guide

The repository is pre-configured for instant 1-click deployment on **Streamlit Community Cloud**:

### 1. Local Execution
```bash
git clone https://github.com/your-username/loan-recovery-risk-analytics.git
cd loan-recovery-risk-analytics
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 2. Deploy to Streamlit Cloud in 3 Steps:
1. Push this repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
3. Select your repository and specify:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: `3.11`
4. Click **Deploy!** Streamlit Cloud will automatically detect `requirements.txt` and launch the application.

---

## ⚖️ Step 18: Business & Ethical Considerations Summary

- **FDCPA Compliance**: Capping contacts to $\le 3$ attempts/week; calling hours restricted between 8:00 AM and 9:00 PM local time.
- **Fair Lending (ECOA)**: Demographic parity audit confirmed Disparate Impact Ratio $>0.80$ across all gender, marital status, and regional groups.
- **FCRA Adverse Action Disclosures**: Translating negative SHAP feature factors directly into standardized reason codes for borrowers.
- **Non-Predatory Settlement**: Capping settlement haircuts at 15–20% to avoid predatory fee structures.

For full legal and governance details, see [`reports/ethical_and_regulatory_considerations.md`](reports/ethical_and_regulatory_considerations.md).
