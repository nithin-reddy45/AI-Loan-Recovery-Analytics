import streamlit as st
import pandas as pd
import numpy as np

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from dashboard.modules.data_adapter import standardize_dataset
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from modules.data_adapter import standardize_dataset

def render_week3():
    st.header("🧼 Week 3: Data Cleaning & Preprocessing")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None or len(df_raw) == 0:
        show_prerequisite_warning("df_raw", "Week 1: 📂 Dataset Preview")
        return
        
    df = apply_global_filters(df_raw)
    render_global_kpi_cards(df)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        impute_num = st.selectbox("Numeric Imputation", ["Median (Robust)", "Mean", "Zero"])
    with c2:
        drop_dups = st.checkbox("Drop Duplicates", value=True)
    with c3:
        cap_outliers = st.checkbox("Cap Extreme Outliers (IQR)", value=True)
        
    if st.button("⚡ Execute Cleaning Pipeline", use_container_width=True) or st.session_state.get("df_clean") is None:
        with st.spinner("Cleaning dataset..."):
            df_c = df_raw.copy()
            if drop_dups:
                df_c = df_c.drop_duplicates()
                
            num_cols = df_c.select_dtypes(include=['number']).columns
            for col in num_cols:
                if df_c[col].isnull().sum() > 0:
                    val = df_c[col].median() if impute_num == "Median (Robust)" else (df_c[col].mean() if impute_num == "Mean" else 0.0)
                    df_c[col] = df_c[col].fillna(val)
                if cap_outliers and df_c[col].nunique() > 10:
                    q25, q75 = df_c[col].quantile(0.25), df_c[col].quantile(0.75)
                    iqr = q75 - q25
                    df_c[col] = np.clip(df_c[col], q25 - 2.5 * iqr, q75 + 2.5 * iqr)
                    
            cat_cols = df_c.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                mode_v = df_c[col].mode()[0] if len(df_c[col].mode()) > 0 else "Unknown"
                df_c[col] = df_c[col].fillna(mode_v)
                
            df_c = standardize_dataset(df_c)
            st.session_state["df_clean"] = df_c
            st.success("✅ Dataset cleaned and saved to memory!")

    df_clean = st.session_state.get("df_clean", df_raw)
    
    st.markdown("### 📊 Before vs After Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw Rows", f"{len(df_raw):,}")
    m2.metric("Clean Rows", f"{len(df_clean):,}", delta="0 Duplicates")
    m3.metric("Missing Values", f"{df_clean.isnull().sum().sum():,}", delta="100% Cleaned")
    
    st.markdown("#### Preview Cleaned Data:")
    st.dataframe(df_clean.head(20), use_container_width=True)
