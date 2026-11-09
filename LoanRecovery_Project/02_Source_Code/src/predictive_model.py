"""
Machine Learning Pipeline: Loan Recovery Probability Prediction & Explainable AI (SHAP)
Author: AI Collections & Risk Analytics Team
Tasks:
- Multi-algorithm classification benchmarking (LR, RF, XGBoost, LightGBM)
- Stratified train/test splitting & class imbalance mitigation
- Recall and ROC-AUC threshold optimization
- Model diagnostics (ROC, PR, Confusion Matrix)
- Model serialization (Pipeline + Metadata)
- Explainable AI: SHAP Beeswarm, Feature Importance, and Local Explanations
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def train_and_evaluate_models(
    input_file="data/processed/loan_recovery_master.csv",
    model_dir="models",
    fig_dir="reports/figures"
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    print("======================================================================")
    print("🚀 EXECUTING ML PIPELINE: AI LOAN RECOVERY PREDICTION & EXPLAINABLE AI")
    print("======================================================================")
    
    df = pd.read_csv(input_file)
    
    # Filter to delinquent accounts with recovery activities (>30 DPD)
    delinquent_df = df[df["overdue_days"] > 30].copy()
    print(f"Delinquent sample size for ML modeling: {delinquent_df.shape[0]} accounts")
    
    # Target variable: 1 = Recovered, 0 = Unrecovered (Written Off / Legal / In Progress)
    delinquent_df["recovery_target"] = delinquent_df["recovery_status"].apply(
        lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0
    )
    
    target_dist = delinquent_df["recovery_target"].value_counts(normalize=True)
    print(f"Target distribution (Recovery = 1, Default/Loss = 0):\n{target_dist}")
    
    # EXCLUDE target leakage features (e.g. settlement_discount_pct, recovered_amount, recovery_cost)
    numeric_features = [
        "credit_score", "annual_income", "loan_amount", "interest_rate",
        "loan_term_months", "overdue_days", "dti_ratio", "ltv_ratio",
        "is_secured", "contact_attempts", "promise_to_pay_kept",
        "delinquency_severity_score", "existing_debts_count", "age"
    ]
    
    categorical_features = [
        "loan_type", "region", "city_tier", "employment_status",
        "housing_status", "collateral_type", "origination_channel",
        "primary_channel"
    ]
    
    X = delinquent_df[numeric_features + categorical_features]
    y = delinquent_df["recovery_target"]
    
    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    # Class imbalance weight for XGBoost
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos = round(neg_count / max(1, pos_count), 2)
    print(f"Class imbalance ratio (0/1): {scale_pos:.2f}. Configuring class-weight adjustments.")
    
    # Candidate classification models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos, random_state=42, eval_metric="logloss"
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1
        )
    }
    
    results = {}
    fitted_pipelines = {}
    
    plt.figure(figsize=(14, 6))
    
    for name, clf in models.items():
        print(f"\n--- Training & Benchmarking: {name} ---")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4)
        }
        fitted_pipelines[name] = pipeline
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)
        
        # Plot PR curve
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_proba)
        plt.subplot(1, 2, 2)
        plt.plot(r_curve, p_curve, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)
        
    # Finalize Curves Plot (ROC & PR)
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Baseline")
    plt.title("ROC Curves Comparison (Step 10-11)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.suptitle("Model Evaluation: ROC-AUC & Precision-Recall Benchmarks", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_roc_path = os.path.join(fig_dir, "10_model_roc_pr_curves.png")
    plt.savefig(fig_roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_roc_path}")
    
    # ------------------------------------------------------------------
    # Step 11: Recall & Decision Threshold Optimization (LightGBM)
    # ------------------------------------------------------------------
    champion_name = "LightGBM"
    champion_pipeline = fitted_pipelines[champion_name]
    y_test_proba = champion_pipeline.predict_proba(X_test)[:, 1]
    
    threshold_range = np.linspace(0.10, 0.90, 33)
    threshold_metrics = []
    
    for t in threshold_range:
        pred_t = (y_test_proba >= t).astype(int)
        r = recall_score(y_test, pred_t, zero_division=0)
        p = precision_score(y_test, pred_t, zero_division=0)
        f = f1_score(y_test, pred_t, zero_division=0)
        threshold_metrics.append({"threshold": round(t, 3), "recall": r, "precision": p, "f1": f})
        
    thresh_df = pd.DataFrame(threshold_metrics)
    
    # Optimal threshold selected by maximizing F1 while maintaining Recall >= 0.80
    eligible_df = thresh_df[thresh_df["recall"] >= 0.80]
    if not eligible_df.empty:
        best_thresh_row = eligible_df.sort_values(by="f1", ascending=False).iloc[0]
        optimal_threshold = float(best_thresh_row["threshold"])
    else:
        optimal_threshold = float(thresh_df.sort_values(by="f1", ascending=False).iloc[0]["threshold"])
        
    print(f"\n[Step 11 Optimization] Optimal Recovery Threshold: {optimal_threshold:.2f}")
    print(f" -> Recall at Optimal Threshold: {thresh_df.loc[thresh_df['threshold'] == optimal_threshold, 'recall'].values[0]:.4f}")
    print(f" -> Precision at Optimal Threshold: {thresh_df.loc[thresh_df['threshold'] == optimal_threshold, 'precision'].values[0]:.4f}")
    print(f" -> F1-Score at Optimal Threshold: {thresh_df.loc[thresh_df['threshold'] == optimal_threshold, 'f1'].values[0]:.4f}")
    
    # Plot Threshold Optimization Curves
    plt.figure(figsize=(10, 5))
    plt.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall (Recovery Coverage)", color="#10b981", lw=2.5)
    plt.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision (Efficiency)", color="#3b82f6", lw=2)
    plt.plot(thresh_df["threshold"], thresh_df["f1"], label="F1-Score (Harmonic Mean)", color="#8b5cf6", lw=2.5)
    plt.axvline(optimal_threshold, color="#ef4444", linestyle="--", lw=2, label=f"Optimal Threshold ({optimal_threshold:.2f})")
    plt.axvline(0.50, color="gray", linestyle=":", lw=1.5, label="Default Threshold (0.50)")
    plt.title("Step 11: Recall & Decision Threshold Optimization Curve (LightGBM)", fontsize=13, fontweight="bold")
    plt.xlabel("Probability Decision Threshold", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower center")
    plt.tight_layout()
    fig_thresh_path = os.path.join(fig_dir, "14_threshold_optimization_curve.png")
    plt.savefig(fig_thresh_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_thresh_path}")
    
    # ------------------------------------------------------------------
    # Step 12: Confusion Matrix & Diagnostic Feature Importance
    # ------------------------------------------------------------------
    best_y_pred_opt = (y_test_proba >= optimal_threshold).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix (at optimal threshold)
    cm = confusion_matrix(y_test, best_y_pred_opt)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Predicted Default (0)", "Predicted Recovered (1)"],
                yticklabels=["Actual Default (0)", "Actual Recovered (1)"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title(f"{champion_name} Confusion Matrix (Threshold = {optimal_threshold:.2f})", fontweight="bold", pad=10)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # Extract feature names after OneHotEncoding
    ohe_cols = list(champion_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features))
    all_features = numeric_features + ohe_cols
    importances = champion_pipeline.named_steps["classifier"].feature_importances_
    
    feat_df = pd.DataFrame({"feature": all_features, "importance": importances})
    top_feat = feat_df.sort_values("importance", ascending=False).head(15)
    
    sns.barplot(data=top_feat, x="importance", y="feature", ax=axes[1], palette="viridis")
    axes[1].set_title(f"Top 15 Predictive Features (Tree Split Gain)", fontweight="bold", pad=10)
    axes[1].set_xlabel("Split Gain Importance")
    axes[1].set_ylabel("Feature Name")
    
    plt.suptitle("Champion Model Diagnostics & Feature Drivers", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_cm_path = os.path.join(fig_dir, "11_confusion_matrix_and_feature_importance.png")
    plt.savefig(fig_cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_cm_path}")
    
    # ------------------------------------------------------------------
    # EXPLAINABLE AI (SHAP): Global Beeswarm & Bar Plots
    # ------------------------------------------------------------------
    print("\n--- Generating Explainable AI (SHAP) Insights ---")
    fitted_preprocessor = champion_pipeline.named_steps["preprocessor"]
    fitted_classifier = champion_pipeline.named_steps["classifier"]
    
    X_test_transformed = fitted_preprocessor.transform(X_test)
    X_test_trans_df = pd.DataFrame(X_test_transformed, columns=all_features)
    
    # TreeExplainer for LightGBM
    explainer = shap.TreeExplainer(fitted_classifier)
    shap_values = explainer.shap_values(X_test_trans_df)
    
    # In binary LightGBM, shap_values can be a matrix or a list of matrices
    if isinstance(shap_values, list):
        shap_vals_matrix = shap_values[1]  # positive class (Recovery = 1)
    else:
        shap_vals_matrix = shap_values
        
    # 1. SHAP Summary Beeswarm Plot
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_vals_matrix, X_test_trans_df, max_display=15, show=False)
    plt.title("Explainable AI: SHAP Value Distribution (Impact on Recovery Likelihood)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig_shap_bee = os.path.join(fig_dir, "12_shap_summary_beeswarm.png")
    plt.savefig(fig_shap_bee, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_shap_bee}")
    
    # 2. SHAP Global Feature Importance Bar Plot
    mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
    shap_imp_df = pd.DataFrame({"feature": all_features, "mean_abs_shap": mean_abs_shap})
    shap_top15 = shap_imp_df.sort_values(by="mean_abs_shap", ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=shap_top15, x="mean_abs_shap", y="feature", palette="magma")
    plt.title("Explainable AI: Top 15 Drivers of Recovery (Mean |SHAP Value|)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)")
    plt.ylabel("Feature")
    plt.tight_layout()
    fig_shap_imp = os.path.join(fig_dir, "13_shap_feature_importance.png")
    plt.savefig(fig_shap_imp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_shap_imp}")
    
    # ------------------------------------------------------------------
    # Step 14: Save Trained Models & SHAP Artifacts
    # ------------------------------------------------------------------
    lgb_path = os.path.join(model_dir, "recovery_predictor_lightgbm.pkl")
    xgb_path = os.path.join(model_dir, "recovery_predictor_xgboost.pkl")
    shap_data_path = os.path.join(model_dir, "shap_explainer_data.pkl")
    
    joblib.dump(fitted_pipelines["LightGBM"], lgb_path)
    joblib.dump(fitted_pipelines["XGBoost"], xgb_path)
    
    # Save SHAP explainer cache for instant Streamlit UI explanations
    # Sample background for fast computation
    background_sample = X_test.head(100)
    shap_cache = {
        "all_features": all_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "expected_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]),
        "optimal_threshold": optimal_threshold,
        "top_shap_features": shap_top15["feature"].tolist()
    }
    joblib.dump(shap_cache, shap_data_path)
    
    metadata = {
        "model_name": "LightGBM Classifier",
        "task": "Loan Recovery Probability Estimation",
        "training_sample_size": len(X_train),
        "test_sample_size": len(X_test),
        "metrics_default_threshold": results["LightGBM"],
        "metrics_optimal_threshold": {
            "threshold": optimal_threshold,
            "recall": round(float(thresh_df.loc[thresh_df["threshold"] == optimal_threshold, "recall"].values[0]), 4),
            "precision": round(float(thresh_df.loc[thresh_df["threshold"] == optimal_threshold, "precision"].values[0]), 4),
            "f1_score": round(float(thresh_df.loc[thresh_df["threshold"] == optimal_threshold, "f1"].values[0]), 4),
            "roc_auc": results["LightGBM"]["ROC-AUC"]
        },
        "benchmark_comparison": results,
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "total_transformed_features": len(all_features)
        },
        "optimal_recovery_threshold": optimal_threshold,
        "explainable_ai": {
            "framework": "SHAP (SHapley Additive exPlanations)",
            "explainer_type": "TreeExplainer",
            "top_drivers": shap_top15["feature"].tolist()[:5]
        }
    }
    
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    perf_df = pd.DataFrame(results).T
    print("\n=======================================================")
    print("          MACHINE LEARNING BENCHMARK SUMMARY           ")
    print("=======================================================")
    print(perf_df.to_string())
    print("\nModel Artifacts and SHAP Insights Successfully Saved:")
    print(f" - {lgb_path}")
    print(f" - {xgb_path}")
    print(f" - {shap_data_path}")
    print(f" - {meta_path}")
    
    return perf_df

if __name__ == "__main__":
    train_and_evaluate_models()
