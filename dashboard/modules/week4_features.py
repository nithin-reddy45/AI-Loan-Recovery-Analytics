import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from dashboard.modules.data_adapter import standardize_dataset
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from modules.data_adapter import standardize_dataset

def render_week4():
    st.header("🧩 Week 4: Feature Engineering & Ratios")
    
    df_clean = st.session_state.get("df_clean", None)
    if df_clean is None or len(df_clean) == 0:
        show_prerequisite_warning("df_clean", "Week 3: 🧼 Data Cleaning")
        return
        
    df = apply_global_filters(df_clean)
    render_global_kpi_cards(df)
    
    c1, c2 = st.columns(2)
    with c1:
        add_dti = st.checkbox("Generate Debt-to-Income (DTI)", value=True)
        add_sev = st.checkbox("Generate Delinquency Severity Score (0-100)", value=True)
    with c2:
        scale_num = st.checkbox("StandardScale Numeric Features", value=True)
        encode_cat = st.checkbox("One-Hot Encode Categorical Features", value=True)
        
    if st.button("⚡ Transform Features", use_container_width=True) or st.session_state.get("df_features") is None:
        with st.spinner("Transforming features..."):
            df_feat = df_clean.copy()
            
            if add_dti:
                income = np.maximum(df_feat["annual_income"], 1000.0) if "annual_income" in df_feat.columns else 50000.0
                loan_amt = df_feat["loan_amount"] if "loan_amount" in df_feat.columns else 25000.0
                df_feat["dti_ratio"] = ((loan_amt * 0.35) / income).round(4)
                
            if add_sev:
                dpd = df_feat["overdue_days"] if "overdue_days" in df_feat.columns else 0.0
                score = df_feat["credit_score"] if "credit_score" in df_feat.columns else 650.0
                df_feat["delinquency_severity_score"] = np.clip((dpd / 360.0) * 60.0 + (1.0 - score / 850.0) * 40.0, 0.0, 100.0).round(2)
                
            cat_cols = list(df_feat.select_dtypes(include=['object', 'category']).columns)
            target_names = ["recovery_status_binary", "recovered_amount", "default_flag"]
            cat_to_encode = [c for c in cat_cols if c not in target_names and df_feat[c].nunique() < 30]
            
            if encode_cat and cat_to_encode:
                df_feat = pd.get_dummies(df_feat, columns=cat_to_encode, drop_first=True)
                
            num_cols = [c for c in df_feat.select_dtypes(include=['number']).columns if c not in target_names and not c.endswith("_scaled")]
            if scale_num and num_cols:
                scaler = StandardScaler()
                for c in num_cols[:8]:
                    df_feat[f"{c}_scaled"] = scaler.fit_transform(df_feat[[c]]).round(3)
                    
            df_feat = standardize_dataset(df_feat)
            st.session_state["df_features"] = df_feat
            st.success("✅ Features engineered and saved to memory!")

    df_feat = st.session_state.get("df_features", df_clean)
    
    st.markdown(f"**Engineered Feature Matrix ({df_feat.shape[1]} Columns):**")
    st.dataframe(df_feat.head(20), use_container_width=True)
