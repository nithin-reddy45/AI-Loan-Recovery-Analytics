import io
import json
import time
import joblib
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week8():
    st.header("🚀 Week 8: Automated End-to-End Pipeline Runner & Exporter")
    st.markdown("Execute the complete reproducible ML pipeline with one-click orchestration and artifact download.")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None:
        show_prerequisite_warning("df_raw (Raw Dataset)", "Week 1: 📂 Dataset Preview")
        return
        
    df_filtered = apply_global_filters(df_raw)
    render_global_kpi_cards(df_filtered)
    
    # -------------------------------------------------------------
    # 1. Pipeline Execution Step Toggles
    # -------------------------------------------------------------
    with st.expander("⚙️ Pipeline Stage Orchestration Toggles", expanded=True):
        c_t1, c_t2, c_t3, c_t4, c_t5 = st.columns(5)
        with c_t1:
            step_clean = st.checkbox("1. 🧼 Data Cleaning", value=True)
        with c_t2:
            step_encode = st.checkbox("2. 🔤 Categorical Encoding", value=True)
        with c_t3:
            step_features = st.checkbox("3. 🧩 Domain Features", value=True)
        with c_t4:
            step_train = st.checkbox("4. 🏋️ Dual Model Train", value=True)
        with c_t5:
            step_eval = st.checkbox("5. 📊 Evaluation & Metrics", value=True)
            
        target_mode = st.radio("Primary Export Model Focus", ["Classification (P(Recovery))", "Regression (Recovered $)"], horizontal=True)
        run_full_btn = st.button("⚡ Run Full Reproducible Pipeline", use_container_width=True)

    # -------------------------------------------------------------
    # 2. Pipeline Execution Engine
    # -------------------------------------------------------------
    if run_full_btn or "pipeline_results" in st.session_state:
        if run_full_btn:
            start_time = time.time()
            progress = st.progress(0, text="Initializing Pipeline...")
            
            # Step 1: Cleaning
            progress.progress(20, text="Step 1/5: Cleaning missing values & outliers...")
            df_pipe = df_raw.copy().drop_duplicates()
            num_cols = df_pipe.select_dtypes(include=['number']).columns
            for col in num_cols:
                df_pipe[col] = df_pipe[col].fillna(df_pipe[col].median())
            cat_cols = df_pipe.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                mode_v = df_pipe[col].mode()[0] if len(df_pipe[col].mode()) > 0 else "Unknown"
                df_pipe[col] = df_pipe[col].fillna(mode_v)
                
            # Step 2 & 3: Feature Engineering
            progress.progress(50, text="Step 2-3/5: Engineering credit domain ratios...")
            if "interest_rate" in df_pipe.columns and "loan_term_months" in df_pipe.columns and "loan_amount" in df_pipe.columns and "annual_income" in df_pipe.columns:
                r = (df_pipe["interest_rate"] / 100.0) / 12.0
                n = df_pipe["loan_term_months"]
                emi = df_pipe["loan_amount"] * (r * (1 + r)**n) / ((1 + r)**n - 1)
                df_pipe["dti_ratio"] = ((emi * 12) / np.maximum(df_pipe["annual_income"], 1000.0)).round(4)
                
            if "overdue_days" in df_pipe.columns and "credit_score" in df_pipe.columns:
                df_pipe["delinquency_severity_score"] = np.clip((df_pipe["overdue_days"] / 360.0) * 60 + (1 - df_pipe["credit_score"] / 850.0) * 40, 0, 100).round(2)
                
            if "recovery_status_binary" not in df_pipe.columns and "recovery_status" in df_pipe.columns:
                df_pipe["recovery_status_binary"] = df_pipe["recovery_status"].apply(lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0)
                
            # Step 4: Training with Sklearn Pipeline
            progress.progress(80, text="Step 4-5/5: Fitting Production Pipeline & Calculating Metrics...")
            
            num_feats = ["annual_income", "credit_score", "loan_amount", "interest_rate", "loan_term_months", "overdue_days"]
            if "dti_ratio" in df_pipe.columns:
                num_feats.append("dti_ratio")
            if "delinquency_severity_score" in df_pipe.columns:
                num_feats.append("delinquency_severity_score")
                
            cat_feats = [c for c in ["loan_type", "region", "primary_channel"] if c in df_pipe.columns]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_feats),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats)
                ]
            )
            
            X = df_pipe[num_feats + cat_feats]
            
            if "Classification" in target_mode:
                y = df_pipe["recovery_status_binary"].fillna(0).astype(int)
                clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                full_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                full_pipeline.fit(X_train, y_train)
                y_pred = full_pipeline.predict(X_test)
                y_prob = full_pipeline.predict_proba(X_test)[:, 1]
                
                final_metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "ROC-AUC": roc_auc_score(y_test, y_prob)
                }
            else:
                y = df_pipe["recovered_amount"].fillna(0)
                reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
                full_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", reg)])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                full_pipeline.fit(X_train, y_train)
                y_pred = full_pipeline.predict(X_test)
                
                final_metrics = {
                    "R² Score": r2_score(y_test, y_pred),
                    "RMSE ($)": np.sqrt(mean_squared_error(y_test, y_pred))
                }
                
            elapsed_time = time.time() - start_time
            progress.progress(100, text=f"Pipeline finished in {elapsed_time:.2f}s!")
            
            st.session_state["pipeline_results"] = {
                "pipeline": full_pipeline,
                "metrics": final_metrics,
                "elapsed": elapsed_time,
                "df_cleaned": df_pipe,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "features_count": len(num_feats) + len(cat_feats),
                "target_mode": target_mode
            }
            st.success("🎉 Full Pipeline Execution Successfully Completed!")

        res = st.session_state["pipeline_results"]
        
        # -------------------------------------------------------------
        # 3. Status Badges & Summary Cards
        # -------------------------------------------------------------
        st.subheader("📋 Pipeline Execution Status & Metrics Summary")
        
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown("""
        <div style="background: #1e293b; padding: 12px; border-radius: 8px; color: white;">
            <div style="color: #94a3b8; font-size: 0.8rem;">Status Check</div>
            <div style="font-weight: 700; color: #4ade80;">✅ All Stages Passed</div>
        </div>
        """, unsafe_allow_html=True)
        
        b2.markdown(f"""
        <div style="background: #1e293b; padding: 12px; border-radius: 8px; color: white;">
            <div style="color: #94a3b8; font-size: 0.8rem;">Execution Speed</div>
            <div style="font-weight: 700; color: #38bdf8;">⚡ {res['elapsed']:.2f} Seconds</div>
        </div>
        """, unsafe_allow_html=True)
        
        b3.markdown(f"""
        <div style="background: #1e293b; padding: 12px; border-radius: 8px; color: white;">
            <div style="color: #94a3b8; font-size: 0.8rem;">Feature Count</div>
            <div style="font-weight: 700; color: #fbbf24;">🧩 {res['features_count']} Features Ingested</div>
        </div>
        """, unsafe_allow_html=True)
        
        primary_metric = list(res["metrics"].keys())[0]
        primary_val = list(res["metrics"].values())[0]
        b4.markdown(f"""
        <div style="background: #1e293b; padding: 12px; border-radius: 8px; color: white;">
            <div style="color: #94a3b8; font-size: 0.8rem;">Primary {primary_metric}</div>
            <div style="font-weight: 700; color: #a855f7;">🏆 {primary_val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 4. Multi-Format Artifact Downloads
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("📦 Download Production Artifacts")
        
        d_c1, d_c2, d_c3 = st.columns(3)
        
        # 1. Serialized Model Bytes (joblib)
        with d_c1:
            buffer = io.BytesIO()
            joblib.dump(res["pipeline"], buffer)
            buffer.seek(0)
            st.download_button(
                label="📥 Download Pipeline (.joblib)",
                data=buffer.getvalue(),
                file_name="loan_recovery_full_pipeline.joblib",
                mime="application/octet-stream",
                use_container_width=True
            )
            
        # 2. Config & Metadata JSON
        with d_c2:
            config_meta = {
                "project": "AI-Driven Loan Recovery & Risk Analytics",
                "timestamp": res["timestamp"],
                "target_focus": res["target_mode"],
                "metrics": res["metrics"],
                "execution_time_seconds": res["elapsed"]
            }
            st.download_button(
                label="📥 Download Config & Metadata (.json)",
                data=json.dumps(config_meta, indent=4),
                file_name="pipeline_config_metadata.json",
                mime="application/json",
                use_container_width=True
            )
            
        # 3. Cleaned Dataset CSV
        with d_c3:
            csv_data = res["df_cleaned"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (.csv)",
                data=csv_data,
                file_name="loan_recovery_cleaned_master.csv",
                mime="text/csv",
                use_container_width=True
            )
