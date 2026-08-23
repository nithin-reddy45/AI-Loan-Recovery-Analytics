import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week6():
    st.header("⚙️ Week 6: Hyperparameter Tuning & Grid Search Optimization")
    st.markdown("Fine-tune model hyperparameters, analyze parameter sensitivity, and promote the champion model.")
    
    df_features = st.session_state.get("df_features", None)
    base_models = st.session_state.get("model_baseline", {})
    if df_features is None:
        show_prerequisite_warning("df_features (Feature Matrix)", "Week 4: 🧩 Feature Engineering")
        return
        
    df_filtered = apply_global_filters(df_features)
    render_global_kpi_cards(df_filtered)
    
    # Target Mode
    target_mode = st.selectbox(
        "🎯 Select Model Objective for Tuning",
        [
            "🔄 Classification: Loan Recovery Status (recovery_status_binary)",
            "💰 Regression: Recovered Dollar Amount (recovered_amount)"
        ]
    )
    is_classification = "Classification" in target_mode
    key_mode = "classification" if is_classification else "regression"
    
    # -------------------------------------------------------------
    # 1. Hyperparameter Search Space Configuration
    # -------------------------------------------------------------
    with st.expander("🛠️ Configure Hyperparameter Grid Space", expanded=True):
        c_g1, c_g2, c_g3 = st.columns(3)
        with c_g1:
            n_estimators_list = st.multiselect("Number of Trees (n_estimators)", [50, 100, 150, 200], default=[50, 100, 150])
        with c_g2:
            max_depth_list = st.multiselect("Max Depth (max_depth)", [4, 6, 8, 12, None], default=[4, 8, 12])
        with c_g3:
            min_samples_split_list = st.multiselect("Min Samples Split", [2, 5, 10], default=[2, 5])
            
        cv_folds = st.slider("Cross-Validation Folds (k-fold)", 2, 5, 3)
        run_tuning_btn = st.button("⚡ Run Grid Search & Optimize", use_container_width=True)

    # -------------------------------------------------------------
    # 2. Execution & Grid Search
    # -------------------------------------------------------------
    ignore_cols = ["customer_id", "loan_id", "recovery_id", "repayment_id", "recovery_status", "recovery_date", 
                   "last_payment_date", "default_date", "disbursement_date", "delinquency_bucket", "officer_name", 
                   "recovery_officer_id", "officer_designation", "loan_purpose"]
    feat_cols = [c for c in df_features.select_dtypes(include=['number']).columns 
                 if c not in ignore_cols and c not in ["recovery_status_binary", "recovered_amount", "net_recovery_amount", "default_flag"]]

    if run_tuning_btn:
        if not n_estimators_list or not max_depth_list or not min_samples_split_list:
            st.error("Please select at least one value for each hyperparameter.")
            return
            
        param_grid = {
            "n_estimators": n_estimators_list,
            "max_depth": max_depth_list,
            "min_samples_split": min_samples_split_list
        }
        
        progress_bar = st.progress(0, text="Initializing Grid Search optimization...")
        start_time = time.time()
        
        if is_classification:
            if "recovery_status_binary" not in df_features.columns:
                if "recovery_target_binary" in df_features.columns:
                    df_features["recovery_status_binary"] = df_features["recovery_target_binary"].fillna(0).astype(int)
                elif "recovery_status" in df_features.columns:
                    df_features["recovery_status_binary"] = df_features["recovery_status"].apply(
                        lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0
                    )
                elif "default_flag" in df_features.columns:
                    df_features["recovery_status_binary"] = (1 - df_features["default_flag"]).astype(int)
                else:
                    df_features["recovery_status_binary"] = 0
                    
            valid_mask = df_features["recovery_status_binary"].notnull()
            X = df_features.loc[valid_mask, feat_cols].fillna(0)
            y = df_features.loc[valid_mask, "recovery_status_binary"].astype(int)
            
            clf = RandomForestClassifier(random_state=42)
            grid = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=cv_folds), scoring="roc_auc", n_jobs=-1)
            grid.fit(X, y)
            
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            best_score = grid.best_score_
            
            # Record results
            results_df = pd.DataFrame(grid.cv_results_)
            results_df["Run #"] = range(1, len(results_df) + 1)
            results_df["Mean ROC-AUC Score"] = results_df["mean_test_score"].round(4)
            results_df["Fit Time (s)"] = results_df["mean_fit_time"].round(3)
            results_df["Parameters"] = results_df["params"].astype(str)
            
            st.session_state["model_tuned"]["classification"] = best_model
            st.session_state["best_params"]["classification"] = best_params
            st.session_state["metrics_tuned"]["classification"] = {
                "Best ROC-AUC": best_score,
                "Params": best_params,
                "Grid Results": results_df[["Run #", "Parameters", "Mean ROC-AUC Score", "Fit Time (s)"]]
            }
            
        else:
            valid_mask = (df_features["overdue_days"] > 30) if "overdue_days" in df_features.columns else df_features["recovered_amount"].notnull()
            X = df_features.loc[valid_mask, feat_cols].fillna(0)
            y = df_features.loc[valid_mask, "recovered_amount"].fillna(0)
            
            reg = RandomForestRegressor(random_state=42)
            grid = GridSearchCV(reg, param_grid, cv=KFold(n_splits=cv_folds), scoring="r2", n_jobs=-1)
            grid.fit(X, y)
            
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            best_score = grid.best_score_
            
            results_df = pd.DataFrame(grid.cv_results_)
            results_df["Run #"] = range(1, len(results_df) + 1)
            results_df["Mean R² Score"] = results_df["mean_test_score"].round(4)
            results_df["Fit Time (s)"] = results_df["mean_fit_time"].round(3)
            results_df["Parameters"] = results_df["params"].astype(str)
            
            st.session_state["model_tuned"]["regression"] = best_model
            st.session_state["best_params"]["regression"] = best_params
            st.session_state["metrics_tuned"]["regression"] = {
                "Best R²": best_score,
                "Params": best_params,
                "Grid Results": results_df[["Run #", "Parameters", "Mean R² Score", "Fit Time (s)"]]
            }
            
        progress_bar.progress(100, text=f"Tuning completed in {time.time()-start_time:.2f}s!")
        st.success("🎉 Champion tuned model successfully saved into session state as `model_tuned`!")

    # -------------------------------------------------------------
    # 3. Best Model Results & Comparison Display
    # -------------------------------------------------------------
    tuned_info = st.session_state.get("metrics_tuned", {}).get(key_mode)
    if tuned_info:
        st.markdown("### 🏆 Best Hyperparameter Solution")
        
        c_b1, c_b2 = st.columns([3, 2])
        with c_b1:
            st.json(tuned_info["Params"])
        with c_b2:
            metric_label = "CV ROC-AUC Score" if is_classification else "CV R² Score"
            metric_val = tuned_info.get("Best ROC-AUC" if is_classification else "Best R²", 0.0)
            st.metric(f"🌟 Champion {metric_label}", f"{metric_val:.4f}")
            st.button("✅ Promoted as Production Champion Model", disabled=True, use_container_width=True)
            
        st.markdown("#### 📋 Grid Search Iteration Log")
        res_table = tuned_info["Grid Results"]
        st.dataframe(res_table.sort_values(res_table.columns[2], ascending=False), use_container_width=True)
        
        # Plotly chart: Score across parameter sets
        fig_grid = px.bar(
            res_table, x="Run #", y=res_table.columns[2], color=res_table.columns[2],
            hover_data=["Parameters", "Fit Time (s)"],
            title=f"<b>Cross-Validation Performance Across Tuning Iterations ({metric_label})</b>",
            color_continuous_scale="Viridis", template="plotly_white"
        )
        st.plotly_chart(fig_grid, use_container_width=True)
    else:
        st.info("Click '⚡ Run Grid Search & Optimize' above to begin hyperparameter tuning.")
