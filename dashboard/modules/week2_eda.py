import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week2():
    st.header("🔎 Week 2: Exploratory Data Analysis (EDA) & Data Quality Viewer")
    st.markdown("Analyze missing value distributions, detect statistical outliers, and uncover multi-factor risk correlations.")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None:
        show_prerequisite_warning("df_raw (Raw Dataset)", "Week 1: 📂 Dataset Preview")
        return
        
    df = apply_global_filters(df_raw)
    render_global_kpi_cards(df)
    
    tab_missing, tab_outliers, tab_charts = st.tabs([
        "⚠️ Missing Value Audit", "🎯 Statistical Outlier Detection", "📊 Risk & Recovery Charts"
    ])
    
    # -------------------------------------------------------------
    # 1. Missing Value Audit Tab
    # -------------------------------------------------------------
    with tab_missing:
        st.subheader("📋 Missing Value Diagnostics")
        null_counts = df.isnull().sum()
        null_pcts = (df.isnull().mean() * 100).round(2)
        
        miss_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": null_counts,
            "Missing Percentage (%)": null_pcts
        }).sort_values("Missing Percentage (%)", ascending=False)
        
        c_m1, c_m2 = st.columns([5, 7])
        with c_m1:
            st.dataframe(miss_df.reset_index(drop=True), use_container_width=True)
            csv_miss = miss_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Missing Value Report (CSV)", data=csv_miss, file_name="missing_value_report.csv")
            
        with c_m2:
            fig_miss = px.bar(
                miss_df[miss_df["Missing Percentage (%)"] > 0],
                x="Missing Percentage (%)", y="Column", orientation="h",
                title="<b>Missing Value Share by Column (%)</b>",
                color="Missing Percentage (%)", color_continuous_scale="Reds",
                template="plotly_white"
            )
            fig_miss.update_layout(height=350)
            st.plotly_chart(fig_miss, use_container_width=True)
            
    # -------------------------------------------------------------
    # 2. Outlier Detection Tab
    # -------------------------------------------------------------
    with tab_outliers:
        st.subheader("🎯 Outlier Diagnostics (IQR vs Z-Score)")
        
        c_ctrl1, c_ctrl2 = st.columns(2)
        with c_ctrl1:
            det_method = st.radio("Detection Methodology", ["IQR (Interquartile Range)", "Z-Score (|z| > 3)"], horizontal=True)
        with c_ctrl2:
            iqr_k = st.selectbox("IQR Multiplier (k)", [1.5, 3.0], index=0) if "IQR" in det_method else None
            
        num_cols = list(df.select_dtypes(include=['number']).columns)
        outlier_summary = []
        
        for col in num_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            if "IQR" in det_method:
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - (iqr_k * iqr)
                upper_bound = q75 + (iqr_k * iqr)
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            else:
                mean = series.mean()
                std = series.std()
                if std > 0:
                    z_scores = (series - mean) / std
                    outliers = series[z_scores.abs() > 3.0]
                    lower_bound = mean - 3.0 * std
                    upper_bound = mean + 3.0 * std
                else:
                    outliers = pd.Series(dtype=float)
                    lower_bound, upper_bound = mean, mean
                    
            outlier_summary.append({
                "Column": col,
                "Total Count": len(series),
                "Outlier Count": len(outliers),
                "Outlier %": round((len(outliers) / len(series)) * 100, 2),
                "Lower Threshold": round(lower_bound, 2),
                "Upper Threshold": round(upper_bound, 2)
            })
            
        outlier_df = pd.DataFrame(outlier_summary).sort_values("Outlier %", ascending=False)
        st.dataframe(outlier_df.reset_index(drop=True), use_container_width=True)
        
        sel_col = st.selectbox("Select Column for Box & Distribution Plot", num_cols)
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            fig_box_col = px.box(
                df, y=sel_col, color="loan_type" if "loan_type" in df.columns else None,
                title=f"<b>Boxplot: {sel_col} Outlier Spread</b>", template="plotly_white"
            )
            st.plotly_chart(fig_box_col, use_container_width=True)
        with c_p2:
            fig_viol = px.violin(
                df, y=sel_col, box=True, points="all",
                title=f"<b>Violin Distribution: {sel_col}</b>", template="plotly_white"
            )
            st.plotly_chart(fig_viol, use_container_width=True)

    # -------------------------------------------------------------
    # 3. Interactive Risk Charts & Correlation Tab
    # -------------------------------------------------------------
    with tab_charts:
        st.subheader("📊 Interactive Risk Matrix & Correlation Heatmap")
        
        # Correlation Heatmap
        corr_cols = [c for c in ["annual_income", "credit_score", "loan_amount", "interest_rate", "overdue_days", "recovered_amount", "recovery_cost", "default_flag"] if c in df.columns]
        if len(corr_cols) > 1:
            corr_matrix = df[corr_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                title="<b>Correlation Heatmap of Financial & Recovery Metrics</b>",
                zmin=-1.0, zmax=1.0
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            if "credit_score" in df.columns and "overdue_days" in df.columns:
                fig_sc1 = px.scatter(
                    df.sample(min(1500, len(df))),
                    x="credit_score", y="overdue_days", color="loan_type" if "loan_type" in df.columns else None,
                    size="loan_amount" if "loan_amount" in df.columns else None,
                    title="<b>Credit Score vs Overdue Duration (DPD)</b>", template="plotly_white"
                )
                st.plotly_chart(fig_sc1, use_container_width=True)
        with c_sc2:
            if "annual_income" in df.columns and "recovered_amount" in df.columns:
                fig_sc2 = px.scatter(
                    df[df["recovered_amount"] > 0].sample(min(1500, (df["recovered_amount"] > 0).sum())),
                    x="annual_income", y="recovered_amount", color="primary_channel" if "primary_channel" in df.columns else None,
                    title="<b>Annual Income vs Amount Recovered ($)</b>", template="plotly_white"
                )
                st.plotly_chart(fig_sc2, use_container_width=True)
