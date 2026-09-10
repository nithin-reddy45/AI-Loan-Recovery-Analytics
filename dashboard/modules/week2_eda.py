import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week2():
    st.header("🔎 Week 2: Exploratory Data Analysis & Outliers")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None or len(df_raw) == 0:
        show_prerequisite_warning("df_raw", "Week 1: 📂 Dataset Preview")
        return
        
    df = apply_global_filters(df_raw)
    render_global_kpi_cards(df)
    
    tab_miss, tab_out, tab_corr = st.tabs(["⚠️ Missing Values", "🎯 Outlier Detection", "📊 Correlation & Scatter"])
    
    with tab_miss:
        null_counts = df.isnull().sum()
        miss_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": null_counts,
            "Missing %": (df.isnull().mean() * 100).round(2)
        }).sort_values("Missing %", ascending=False)
        
        c1, c2 = st.columns([5, 7])
        with c1:
            st.dataframe(miss_df.reset_index(drop=True), use_container_width=True)
        with c2:
            m_plot = miss_df[miss_df["Missing %"] > 0]
            if len(m_plot) > 0:
                fig = px.bar(m_plot, x="Missing %", y="Column", orientation="h", title="Missing Values (%)")
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No missing values detected!")

    with tab_out:
        num_cols = list(df.select_dtypes(include=['number']).columns)
        if num_cols:
            c_s1, c_s2 = st.columns([2, 3])
            with c_s1:
                sel_col = st.selectbox("Select Feature to Inspect", num_cols)
                series = df[sel_col].dropna()
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                outliers = series[(series < q25 - 1.5 * iqr) | (series > q75 + 1.5 * iqr)]
                st.metric("Detected Outliers", f"{len(outliers):,} records", f"{(len(outliers)/max(1, len(series)))*100:.1f}% of data")
            with c_s2:
                sample_n = min(1000, len(df))
                fig_box = px.box(df.sample(sample_n, random_state=42) if len(df) > sample_n else df, y=sel_col, title=f"Boxplot: {sel_col}")
                fig_box.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab_corr:
        num_cols = list(df.select_dtypes(include=['number']).columns)
        if len(num_cols) > 1:
            corr_df = df[num_cols[:10]].corr()
            fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Matrix")
            fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 2 numeric features for correlation.")
