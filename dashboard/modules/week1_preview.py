import os
import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters

def render_week1():
    st.header("📂 Week 1: Dataset Ingestion & Exploratory Preview")
    st.markdown("Upload raw loan portfolios, inspect column metadata, and initialize enterprise global filters.")
    
    # -------------------------------------------------------------
    # 1. Dataset Upload & Ingestion Controls
    # -------------------------------------------------------------
    c_up1, c_up2 = st.columns([3, 2])
    
    with c_up1:
        uploaded_file = st.file_uploader("📥 Upload Loan & Recovery CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            st.session_state["df_raw"] = df_uploaded
            st.success(f"✅ Successfully loaded custom dataset ({len(df_uploaded):,} records, {df_uploaded.shape[1]} columns) into session memory!")
            
    with c_up2:
        st.markdown("**Or Load Built-in Portfolio:**")
        if st.button("🚀 Load 15,000 Banking Loan Dataset", use_container_width=True):
            def_path = "data/processed/loan_recovery_master.csv"
            if not os.path.exists(def_path):
                def_path = "data/raw/loans.csv"
            df_def = pd.read_csv(def_path)
            st.session_state["df_raw"] = df_def
            st.success("✅ Loaded 15,000-loan master dataset into app memory!")
            st.rerun()

    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None:
        st.warning("⚠️ No dataset uploaded yet. Please upload a CSV or click the button above to load the sample banking dataset.")
        return

    # Apply Global Filters
    df = apply_global_filters(df_raw)
    
    # Display 5 Global KPI Cards
    render_global_kpi_cards(df)

    # -------------------------------------------------------------
    # 2. Multi-Tab Data Inspection
    # -------------------------------------------------------------
    tab_prev, tab_stats, tab_info, tab_charts = st.tabs([
        "📋 Data Preview", "📊 Summary Statistics", "ℹ️ Schema & Types", "📈 Quick Distribution Charts"
    ])
    
    with tab_prev:
        st.markdown(f"**Showing Top Rows (Filtered: {len(df):,} of {len(df_raw):,} total rows):**")
        st.dataframe(df.head(25), use_container_width=True)
        
        # Download Filtered Data
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Current Filtered CSV",
            data=csv_bytes,
            file_name="loan_recovery_filtered_data.csv",
            mime="text/csv"
        )
        
    with tab_stats:
        c_st1, c_st2 = st.columns(2)
        with c_st1:
            st.markdown("#### 🔢 Numerical Columns Summary")
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                st.dataframe(df[num_cols].describe().T.style.format("{:,.2f}"), use_container_width=True)
        with c_st2:
            st.markdown("#### 🔤 Categorical Columns Summary")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                st.dataframe(df[cat_cols].describe().T, use_container_width=True)
                
    with tab_info:
        st.markdown("#### 🔍 Schema Diagnostics & Memory Footprint")
        info_data = []
        for col in df.columns:
            info_data.append({
                "Column Name": col,
                "Data Type": str(df[col].dtype),
                "Non-Null Count": f"{df[col].notnull().sum():,} / {len(df):,}",
                "Null %": f"{(df[col].isnull().mean() * 100):.2f}%",
                "Unique Values": df[col].nunique(),
                "Sample Value": str(df[col].dropna().iloc[0]) if df[col].notnull().sum() > 0 else "None"
            })
        st.dataframe(pd.DataFrame(info_data), use_container_width=True)
        
    with tab_charts:
        st.markdown("#### 📊 Initial Plotly Express Macro Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            if "loan_amount" in df.columns:
                fig_amt = px.histogram(
                    df, x="loan_amount", nbins=40, color="loan_type" if "loan_type" in df.columns else None,
                    title="<b>Loan Amount Disbursal Distribution</b>", template="plotly_white"
                )
                st.plotly_chart(fig_amt, use_container_width=True)
                
            if "recovered_amount" in df.columns and "loan_type" in df.columns:
                fig_box = px.box(
                    df[df["recovered_amount"] > 0], x="loan_type", y="recovered_amount",
                    color="loan_type", title="<b>Recovered Amount Distribution by Product Line</b>",
                    template="plotly_white"
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
        with c2:
            if "credit_score" in df.columns:
                fig_score = px.histogram(
                    df, x="credit_score", nbins=35,
                    title="<b>Borrower Credit Score Distribution (300 - 850)</b>",
                    color_discrete_sequence=["#0ea5e9"], template="plotly_white"
                )
                st.plotly_chart(fig_score, use_container_width=True)
                
            if "region" in df.columns:
                reg_counts = df["region"].value_counts().reset_index()
                reg_counts.columns = ["region", "count"]
                fig_reg = px.bar(
                    reg_counts, x="region", y="count", color="region",
                    title="<b>Account Portfolio Distribution by Geographic Region</b>",
                    template="plotly_white"
                )
                st.plotly_chart(fig_reg, use_container_width=True)
