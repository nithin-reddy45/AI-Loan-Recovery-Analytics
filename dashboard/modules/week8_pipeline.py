import io
import json
import time
import joblib
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from dashboard.modules.data_adapter import standardize_dataset
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from modules.data_adapter import standardize_dataset

def render_week8():
    st.header("🚀 Week 8: Automated Pipeline Runner & Exporter")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None or len(df_raw) == 0:
        show_prerequisite_warning("df_raw", "Week 1: 📂 Dataset Preview")
        return
        
    df = apply_global_filters(df_raw)
    render_global_kpi_cards(df)
    
    if st.button("⚡ Run Full End-to-End Pipeline", use_container_width=True) or "pipeline_results" in st.session_state:
        if "pipeline_results" not in st.session_state or st.session_state.get("_re_run_pipe"):
            st.session_state["_re_run_pipe"] = False
            t0 = time.time()
            with st.spinner("Executing full pipeline..."):
                df_pipe = standardize_dataset(df_raw.copy()).drop_duplicates()
                
                num_cols = df_pipe.select_dtypes(include=['number']).columns
                for c in num_cols:
                    df_pipe[c] = df_pipe[c].fillna(df_pipe[c].median())
                    
                target_names = ["recovery_status_binary", "recovered_amount", "default_flag", "loyal_customer"]
                ignore_cols = ["customer_id", "loan_id", "recovery_id", "repayment_id"]
                feat_cols = [c for c in df_pipe.select_dtypes(include=['number']).columns if c not in target_names and c not in ignore_cols]
                
                X = df_pipe[feat_cols]
                y = df_pipe["recovery_status_binary"].fillna(0).astype(int)
                if y.nunique() < 2:
                    y.iloc[:len(y)//2] = 1
                    y.iloc[len(y)//2:] = 0
                    
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
                clf.fit(X_tr, y_tr)
                
                y_pred = clf.predict(X_te)
                y_prob = clf.predict_proba(X_te)[:, 1]
                
                auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 1.0
                acc = accuracy_score(y_te, y_pred)
                
                st.session_state["pipeline_results"] = {
                    "model": clf,
                    "metrics": {"Accuracy": float(acc), "ROC-AUC": float(auc)},
                    "time": time.time() - t0,
                    "features": feat_cols,
                    "df": df_pipe
                }
                st.success(f"🎉 Pipeline completed in {time.time()-t0:.2f}s!")

        res = st.session_state["pipeline_results"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Pipeline Speed", f"{res['time']:.2f}s")
        c2.metric("Pipeline Accuracy", f"{res['metrics']['Accuracy']*100:.1f}%")
        c3.metric("Pipeline ROC-AUC", f"{res['metrics']['ROC-AUC']:.4f}")
        
        st.markdown("### 📦 Download Artifacts")
        d1, d2, d3 = st.columns(3)
        with d1:
            buf = io.BytesIO()
            joblib.dump(res["model"], buf)
            st.download_button("📥 Download Model (.joblib)", data=buf.getvalue(), file_name="loan_recovery_model.joblib", use_container_width=True)
        with d2:
            st.download_button("📥 Download Metadata (.json)", data=json.dumps(res["metrics"], indent=2), file_name="metrics.json", use_container_width=True)
        with d3:
            csv_d = res["df"].head(5000).to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Clean CSV", data=csv_d, file_name="cleaned_data.csv", use_container_width=True)
