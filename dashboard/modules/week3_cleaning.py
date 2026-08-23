import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week3():
    st.header("🧼 Week 3: Data Cleaning & Preprocessing Comparison")
    st.markdown("Handle missing values, rectify data anomalies, cap statistical outliers, and compare **Before vs After**.")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None:
        show_prerequisite_warning("df_raw (Raw Dataset)", "Week 1: 📂 Dataset Preview")
        return
        
    # Global filter toggle
    use_filters = st.checkbox("🔍 Apply Global Filters to Cleaning View", value=True)
    df_current_raw = apply_global_filters(df_raw) if use_filters else df_raw
    
    render_global_kpi_cards(df_current_raw)
    
    # -------------------------------------------------------------
    # 1. Interactive Cleaning Configuration Panel
    # -------------------------------------------------------------
    with st.expander("⚙️ Configure Data Cleansing Pipeline Rules", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.markdown("**1. Missing Value Strategy**")
            num_impute = st.selectbox("Numerical Imputation", ["Median (Robust)", "Mean", "Drop Missing Rows", "Constant Zero"], index=0)
            cat_impute = st.selectbox("Categorical Imputation", ["Mode (Most Frequent)", "Category 'Unknown'", "Drop Rows"], index=0)
            
        with col_c2:
            st.markdown("**2. Duplicates & Type Integrity**")
            drop_dups = st.checkbox("Drop Exact Duplicate Records", value=True)
            fix_dates = st.checkbox("Standardize Date Formats (YYYY-MM-DD)", value=True)
            
        with col_c3:
            st.markdown("**3. Outlier Handling (Income/Loans)**")
            outlier_action = st.selectbox("Outlier Treatment", ["Winsorize / Cap (IQR k=2.5)", "Remove Outliers", "Keep Untouched"], index=0)
            target_outlier_col = st.selectbox("Target Column for Outlier Capping", ["annual_income", "loan_amount", "All Numeric"], index=0)

        run_clean_btn = st.button("⚡ Execute Cleaning Pipeline & Save to Memory", use_container_width=True)

    # Execute Cleaning
    if run_clean_btn or st.session_state.get("df_clean") is None:
        df_clean = df_raw.copy()
        
        # 1. Duplicates
        if drop_dups:
            df_clean = df_clean.drop_duplicates()
            
        # 2. Missing Numerical Imputation
        num_cols = df_clean.select_dtypes(include=['number']).columns
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                if num_impute == "Median (Robust)":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif num_impute == "Mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif num_impute == "Constant Zero":
                    df_clean[col] = df_clean[col].fillna(0.0)
                elif num_impute == "Drop Missing Rows":
                    df_clean = df_clean.dropna(subset=[col])
                    
        # 3. Missing Categorical Imputation
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                if cat_impute == "Mode (Most Frequent)":
                    mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown"
                    df_clean[col] = df_clean[col].fillna(mode_val)
                elif cat_impute == "Category 'Unknown'":
                    df_clean[col] = df_clean[col].fillna("Unknown")
                elif cat_impute == "Drop Rows":
                    df_clean = df_clean.dropna(subset=[col])
                    
        # 4. Outlier Treatment
        if "Winsorize" in outlier_action:
            cols_to_cap = ["annual_income"] if target_outlier_col == "annual_income" else (["loan_amount"] if target_outlier_col == "loan_amount" else list(num_cols))
            for c in cols_to_cap:
                if c in df_clean.columns:
                    q25 = df_clean[c].quantile(0.25)
                    q75 = df_clean[c].quantile(0.75)
                    iqr = q75 - q25
                    lower = q25 - 2.5 * iqr
                    upper = q75 + 2.5 * iqr
        elif outlier_action == "Remove Outliers":
            if "annual_income" in df_clean.columns:
                q25 = df_clean["annual_income"].quantile(0.25)
                q75 = df_clean["annual_income"].quantile(0.75)
                iqr = q75 - q25
                df_clean = df_clean[(df_clean["annual_income"] >= (q25 - 2.5 * iqr)) & (df_clean["annual_income"] <= (q75 + 2.5 * iqr))]
                
        # Ensure target columns are preserved
        if "recovery_status_binary" not in df_clean.columns:
            if "recovery_target_binary" in df_clean.columns:
                df_clean["recovery_status_binary"] = df_clean["recovery_target_binary"].fillna(0).astype(int)
            elif "recovery_status" in df_clean.columns:
                df_clean["recovery_status_binary"] = df_clean["recovery_status"].apply(
                    lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0
                )
            elif "default_flag" in df_clean.columns:
                df_clean["recovery_status_binary"] = (1 - df_clean["default_flag"]).astype(int)
            else:
                df_clean["recovery_status_binary"] = 0
                
        if "recovered_amount" not in df_clean.columns:
            df_clean["recovered_amount"] = 0.0
            
        st.session_state["df_clean"] = df_clean
        if run_clean_btn:
            st.success("✅ Cleaned dataset successfully stored in session memory as `df_clean`!")

    df_clean_all = st.session_state["df_clean"]
    df_clean_view = apply_global_filters(df_clean_all) if use_filters else df_clean_all

    # -------------------------------------------------------------
    # 2. Before vs After Metrics Ribbon
    # -------------------------------------------------------------
    raw_missing_pct = (df_current_raw.isnull().sum().sum() / (df_current_raw.shape[0] * df_current_raw.shape[1])) * 100
    clean_missing_pct = (df_clean_view.isnull().sum().sum() / (df_clean_view.shape[0] * df_clean_view.shape[1])) * 100
    
    st.markdown("### 📊 Cleansing Impact Comparison")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw Rows vs Clean Rows", f"{len(df_current_raw):,} ➔ {len(df_clean_view):,}", delta=f"{len(df_clean_view)-len(df_current_raw)} rows")
    m2.metric("Missing Value Reduction", f"{raw_missing_pct:.2f}% ➔ {clean_missing_pct:.2f}%", delta=f"-{raw_missing_pct - clean_missing_pct:.2f}%", delta_color="inverse")
    m3.metric("Duplicate Count", f"{df_current_raw.duplicated().sum():,} ➔ {df_clean_view.duplicated().sum():,}", delta="0 Duplicates")
    m4.metric("Clean Data Columns", f"{df_clean_view.shape[1]} Columns", "Ready for Feature Eng.")

    # -------------------------------------------------------------
    # 3. Before vs After Tabs & Visual Comparison
    # -------------------------------------------------------------
    tab_before, tab_after, tab_diff, tab_vis = st.tabs([
        "📄 Raw Dataset (Before)", "✨ Cleaned Dataset (After)", "🔍 Data Differences", "📈 Visual Distribution Overlay"
    ])
    
    with tab_before:
        st.dataframe(df_current_raw.head(15), use_container_width=True)
        
    with tab_after:
        st.dataframe(df_clean_view.head(15), use_container_width=True)
        csv_clean_bytes = df_clean_view.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned Dataset (CSV)", data=csv_clean_bytes, file_name="cleaned_loan_recovery_data.csv")
        
    with tab_diff:
        st.markdown("#### 📋 Column-by-Column Missing Value Reduction")
        diff_df = pd.DataFrame({
            "Column": df_current_raw.columns,
            "Missing in Raw": df_current_raw.isnull().sum(),
            "Missing in Clean": df_clean_view.reindex(columns=df_current_raw.columns).isnull().sum()
        })
        diff_df["Status"] = np.where(diff_df["Missing in Clean"] == 0, "✅ Cleaned", "⚠️ Has Nulls")
        st.dataframe(diff_df.reset_index(drop=True), use_container_width=True)
        
    with tab_vis:
        st.markdown("#### 📈 Distribution Overlay: Raw vs Cleaned")
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            if "annual_income" in df_current_raw.columns and "annual_income" in df_clean_view.columns:
                fig_over = go.Figure()
                fig_over.add_trace(go.Histogram(x=df_current_raw["annual_income"], name="Raw (Before)", marker_color="#ef4444", opacity=0.6))
                fig_over.add_trace(go.Histogram(x=df_clean_view["annual_income"], name="Cleaned (After)", marker_color="#10b981", opacity=0.6))
                fig_over.update_layout(barmode="overlay", title="<b>Annual Income Distribution Before vs After Outlier Treatment</b>", template="plotly_white")
                st.plotly_chart(fig_over, use_container_width=True)
        with c_v2:
            # Missing reduction bar
            fig_bar_red = px.bar(
                diff_df[diff_df["Missing in Raw"] > 0],
                x="Column", y=["Missing in Raw", "Missing in Clean"],
                barmode="group", title="<b>Missing Value Reduction Count by Column</b>",
                color_discrete_sequence=["#ef4444", "#10b981"], template="plotly_white"
            )
            st.plotly_chart(fig_bar_red, use_container_width=True)
