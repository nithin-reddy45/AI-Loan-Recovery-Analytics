
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

def train_and_evaluate_models(input_file="data/processed/loan_recovery_master.csv", model_dir="models", fig_dir="reports/figures"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    print("Executing Machine Learning Pipeline for Loan Recovery Prediction...")
    
    df = pd.read_csv(input_file)
    
    # Filter to delinquent accounts with recovery activities
    delinquent_df = df[df["overdue_days"] > 30].copy()
    print(f"Delinquent sample size for ML modeling: {delinquent_df.shape[0]} accounts")
    
    # Target variable: 1 = Recovered, 0 = Unrecovered
    delinquent_df["recovery_target"] = delinquent_df["recovery_status"].apply(
        lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0
    )
    
    print(f"Target distribution:\n{delinquent_df['recovery_target'].value_counts(normalize=True)}")
    
    numeric_features = [
        "credit_score", "annual_income", "loan_amount", "interest_rate",
        "loan_term_months", "overdue_days", "dti_ratio", "ltv_ratio",
        "is_secured", "contact_attempts", "promise_to_pay_kept",
        "settlement_discount_pct", "delinquency_severity_score"
    ]
    
    categorical_features = [
        "loan_type", "region", "city_tier", "employment_status",
        "housing_status", "collateral_type", "origination_channel",
        "primary_channel"
    ]
    
    X = delinquent_df[numeric_features + categorical_features]
    y = delinquent_df["recovery_target"]
    
    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    # Model Dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    }
    
    results = {}
    fitted_pipelines = {}
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: ROC Curves
    plt.subplot(1, 2, 1)
    
    # Subplot 2: Precision-Recall Curves
    # (will plot below)
    
    for name, clf in models.items():
        print(f"\n--- Training & Evaluating: {name} ---")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
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
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)
        
        # Plot PR
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_proba)
        plt.subplot(1, 2, 2)
        plt.plot(r_curve, p_curve, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)
        
    # Finalize Curves Plot
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    plt.title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
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
    
    # -------------------------------------------------------------
    # Best Model Deep-Dive: LightGBM
    # -------------------------------------------------------------
    best_model_name = "LightGBM"
    best_pipeline = fitted_pipelines[best_model_name]
    best_y_pred = best_pipeline.predict(X_test)
    best_y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Failed Recovery (0)", "Recovered (1)"],
                yticklabels=["Failed Recovery (0)", "Recovered (1)"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title(f"{best_model_name} Confusion Matrix", fontweight="bold", pad=10)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # 2. Feature Importances
    # Extract feature names after OneHotEncoding
    ohe_cols = list(best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features))
    all_features = numeric_features + ohe_cols
    importances = best_pipeline.named_steps["classifier"].feature_importances_
    
    feat_df = pd.DataFrame({"feature": all_features, "importance": importances})
    top_feat = feat_df.sort_values("importance", ascending=False).head(15)
    
    sns.barplot(data=top_feat, x="importance", y="feature", ax=axes[1], palette="viridis")
    axes[1].set_title(f"Top 15 Predictive Features ({best_model_name})", fontweight="bold", pad=10)
    axes[1].set_xlabel("Feature Importance (Split Gain)")
    axes[1].set_ylabel("Feature")
    
    plt.suptitle("Champion Model Diagnostics & Feature Drivers", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_cm_path = os.path.join(fig_dir, "11_confusion_matrix_and_feature_importance.png")
    plt.savefig(fig_cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_cm_path}")
    
    # Performance Summary Table
    perf_df = pd.DataFrame(results).T
    print("\n=======================================================")
    print("          MACHINE LEARNING BENCHMARK SUMMARY           ")
    print("=======================================================")
    print(perf_df.to_string())
    
    # Save Model Artifacts
    lgb_path = os.path.join(model_dir, "recovery_predictor_lightgbm.pkl")
    xgb_path = os.path.join(model_dir, "recovery_predictor_xgboost.pkl")
    
    joblib.dump(fitted_pipelines["LightGBM"], lgb_path)
    joblib.dump(fitted_pipelines["XGBoost"], xgb_path)
    
    metadata = {
        "model_name": "LightGBM Classifier",
        "task": "Loan Recovery Probability Estimation",
        "training_sample_size": len(X_train),
        "test_sample_size": len(X_test),
        "metrics": results["LightGBM"],
        "benchmark_comparison": results,
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features
        },
        "optimal_recovery_threshold": 0.50
    }
    
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nTrained models and metadata successfully saved:")
    print(f" - {lgb_path}")
    print(f" - {xgb_path}")
    print(f" - {meta_path}")
    
    return perf_df

if __name__ == "__main__":
    train_and_evaluate_models()
