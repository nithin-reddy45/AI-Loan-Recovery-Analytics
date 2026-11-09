# Screenshots and Visual Analytics Index
## Project: AI-Based Loan Recovery Probability Prediction

This directory contains the visual artifacts, model diagnostic plots, Explainable AI (SHAP) charts, and dashboard views generated across the 18-step ML lifecycle:

| Screenshot File | Description & Workflow Step |
| :--- | :--- |
| **`01_portfolio_overview_kpis.png`** | Executive macro portfolio metrics: Disbursed volume ($1.12B), outstanding balance ($779M), delinquent pool ($103M), and net recovery ($60M). |
| **`02_delinquency_by_loan_type_and_region.png`** | Delinquency distribution across loan categories (Auto, Home, Personal, Education, Business) and geographic regions. |
| **`03_recovery_rate_by_channel.png`** | Channel resolution performance & ROI multipliers: AI Voice Bot (52.4x ROI) vs Tele-Calling vs Field Visits vs Legal Agency. |
| **`04_settlement_discount_vs_recovery.png`** | Settlement haircut elasticity curve showing optimal net recovery sweet spot between 10% and 20% discount. |
| **`05_correlation_heatmap.png`** | Pairwise correlation matrix across credit score, income, overdue days, DTI, LTV, and default indicators. |
| **`06_vintage_curve_analysis.png`** | Cumulative default vintage curves tracking default velocity by origination vintage quarter. |
| **`07_collector_performance_matrix.png`** | Recovery officer performance leaderboard, total collections, and efficiency scores. |
| **`08_customer_segmentation_pca.png`** | 2D Principal Component Analysis (PCA) projection of borrower clusters showing clear separation of 4 personas. |
| **`09_segment_profile_radar_boxplots.png`** | Persona profile boxplots comparing credit score, income, DTI, and delinquency severity across clusters. |
| **`10_model_roc_pr_curves.png`** | Multi-algorithm benchmark comparison curves: ROC Curves (left) and Precision-Recall Curves (right) for LR, RF, XGBoost, and LightGBM. |
| **`11_confusion_matrix_and_feature_importance.png`** | Champion model confusion matrix diagnostics (Step 12) alongside tree split-gain feature importance. |
| **`12_shap_summary_beeswarm.png`** | **Explainable AI (SHAP)** Summary Beeswarm plot showing global distribution and directional impact of features on recovery odds. |
| **`13_shap_feature_importance.png`** | **Explainable AI (SHAP)** Global feature importance bar chart ranked by mean absolute SHAP value (log-odds impact). |
| **`14_threshold_optimization_curve.png`** | **Step 11 Recall Optimization** tradeoff curve across 33 decision thresholds, highlighting the calibrated 0.175 optimal cutoff. |
