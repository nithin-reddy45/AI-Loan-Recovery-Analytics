import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from dashboard.modules.data_adapter import standardize_dataset
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from modules.data_adapter import standardize_dataset

def render_week6():
    st.header("⚙️ Week 6: Hyperparameter Tuning")
    
    df_features = st.session_state.get("df_features", None)
    if df_features is None or len(df_features) == 0:
        show_prerequisite_warning("df_features", "Week 4: 🧩 Feature Engineering")
        return
        
    df = apply_global_filters(df_features)
    render_global_kpi_cards(df)
    
    target_mode = st.radio("Tuning Objective", ["Classification (P(Recovery))", "Regression (Recovered $)"], horizontal=True)
    is_cls = "Classification" in target_mode
    key_mode = "classification" if is_cls else "regression"
    
    ignore_cols = ["customer_id", "loan_id", "recovery_id", "repayment_id", "recovery_status", "recovery_date", 
                   "last_payment_date", "default_date", "disbursement_date", "delinquency_bucket", "officer_name", 
                   "recovery_officer_id", "officer_designation", "loan_purpose"]
    target_names = ["recovery_status_binary", "recovered_amount", "net_recovery_amount", "default_flag", "loyal_customer"]
    feat_cols = [c for c in df_features.select_dtypes(include=['number']).columns if c not in ignore_cols and c not in target_names]
    
    c1, c2 = st.columns(2)
    with c1:
        n_trees = st.multiselect("Trees (n_estimators)", [30, 60, 100], default=[30, 60])
    with c2:
        max_d = st.multiselect("Max Depth", [4, 8, 12], default=[4, 8])
        
    if st.button("⚡ Run Fast Hyperparameter Search", use_container_width=True):
        with st.spinner("Optimizing hyperparameters..."):
            t0 = time.time()
            df_std = standardize_dataset(df_features)
            X = df_std[feat_cols].fillna(df_std[feat_cols].median()).fillna(0)
            param_grid = {"n_estimators": n_trees if n_trees else [50], "max_depth": max_d if max_d else [6]}
            
            if is_cls:
                y = df_std["recovery_status_binary"].fillna(0).astype(int)
                if y.nunique() < 2:
                    y.iloc[:len(y)//2] = 1
                    y.iloc[len(y)//2:] = 0
                grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
                grid.fit(X, y)
                st.session_state["model_tuned"]["classification"] = grid.best_estimator_
                st.session_state["best_params"]["classification"] = grid.best_params_
                st.session_state["metrics_tuned"]["classification"] = {"score": float(grid.best_score_), "params": grid.best_params_, "time": time.time()-t0}
            else:
                y = df_std["recovered_amount"].fillna(0.0)
                grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="r2", n_jobs=-1)
                grid.fit(X, y)
                st.session_state["model_tuned"]["regression"] = grid.best_estimator_
                st.session_state["best_params"]["regression"] = grid.best_params_
                st.session_state["metrics_tuned"]["regression"] = {"score": float(grid.best_score_), "params": grid.best_params_, "time": time.time()-t0}
                
            st.success(f"🎉 Tuning finished in {time.time()-t0:.2f}s!")

    tuned_info = st.session_state.get("metrics_tuned", {}).get(key_mode)
    if tuned_info:
        st.markdown("### 🏆 Optimal Hyperparameters")
        c1, c2 = st.columns(2)
        with c1:
            st.json(tuned_info["params"])
        with c2:
            st.metric("Champion CV Score", f"{tuned_info['score']:.4f}")
            st.metric("Search Time", f"{tuned_info['time']:.2f}s")
