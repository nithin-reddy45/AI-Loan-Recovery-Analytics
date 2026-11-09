import os
import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, set_active_dataset
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, set_active_dataset

def render_week1():
    st.header("📂 Week 1: Dataset Ingestion & Exploratory Preview")
    
    c_up1, c_up2 = st.columns([3, 2])
    with c_up1:
        uploaded_file = st.file_uploader("📥 Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            last_name = st.session_state.get("_last_file_loaded", "")
            if uploaded_file.name != last_name:
                df_uploaded = pd.read_csv(uploaded_file)
                st.session_state["_last_file_loaded"] = uploaded_file.name
                set_active_dataset(df_uploaded)
                st.success(f"✅ Ingested `{uploaded_file.name}` ({len(df_uploaded):,} rows, {df_uploaded.shape[1]} cols)")
                st.rerun()
            
    with c_up2:
        st.write("")
        st.write("")
        if st.button("🚀 Load Sample 15,000 Banking Dataset", use_container_width=True):
            def_path = "data/processed/loan_recovery_master.csv"
            if not os.path.exists(def_path):
                def_path = "data/raw/loans.csv"
            if os.path.exists(def_path):
                df_def = pd.read_csv(def_path)
                st.session_state["_last_file_loaded"] = "sample_15k"
                set_active_dataset(df_def)
                st.success("✅ Sample banking dataset loaded!")
                st.rerun()

    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None or len(df_raw) == 0:
        st.info("Please upload a CSV dataset or load the sample banking dataset above to proceed.")
        return

    df = apply_global_filters(df_raw)
    render_global_kpi_cards(df)

    tab1, tab2, tab3 = st.tabs(["📋 Data Table Preview", "📊 Summary Statistics", "📈 Quick Visualizations"])
    
    with tab1:
        st.markdown(f"**Showing Records ({len(df):,} rows):**")
        st.dataframe(df.head(30), use_container_width=True)
        csv_bytes = df.head(5000).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv_bytes, file_name="portfolio_data.csv", mime="text/csv")
        
    with tab2:
        c1, c2 = st.columns(2)
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        with c1:
            st.markdown("#### 🔢 Numerical Columns")
            if len(num_cols) > 0:
                st.dataframe(df[num_cols].describe().T.style.format("{:,.2f}"), use_container_width=True)
            else:
                st.write("No numeric columns found.")
        with c2:
            st.markdown("#### 🔤 Categorical Columns")
            if len(cat_cols) > 0:
                st.dataframe(df[cat_cols].describe().T, use_container_width=True)
            else:
                st.write("No categorical columns found.")
                
    with tab3:
        c1, c2 = st.columns(2)
        num_list = list(df.select_dtypes(include=['number']).columns)
        cat_list = list(df.select_dtypes(include=['object', 'category']).columns)
        
        sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
        
        with c1:
            plot_col = "loan_amount" if "loan_amount" in df.columns else (num_list[0] if num_list else None)
            if plot_col:
                fig = px.histogram(sample_df, x=plot_col, nbins=30, title=f"Distribution: {plot_col.replace('_', ' ').title()}")
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            cat_col = "loan_type" if "loan_type" in df.columns else (cat_list[0] if cat_list else None)
            if cat_col:
                vc = df[cat_col].value_counts().reset_index()
                vc.columns = [cat_col, "count"]
                fig2 = px.bar(vc, x=cat_col, y="count", title=f"Counts by {cat_col.replace('_', ' ').title()}")
                fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
                st.plotly_chart(fig2, use_container_width=True)
