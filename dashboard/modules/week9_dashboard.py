import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters

def render_week9():
    st.title("🏦 AI-Driven Loan Recovery & Risk Analytics Platform")
    st.markdown("**Dual-Target ML Lifecycle: Credit Default Classification & Continuous Recovery Forecasting**")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None:
        st.warning("⚠️ No data loaded in session state. Please visit **Week 1** to upload or load the dataset.")
        return
        
    df = apply_global_filters(df_raw)
    
    # Render 5 Global KPI Cards
    render_global_kpi_cards(df)
    
    # -------------------------------------------------------------
    # 1. Quick Navigation Hub to All 9 Weeks
    # -------------------------------------------------------------
    st.markdown("### 🧭 Quick Navigation to 9-Week ML Lifecycle Modules")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📂 Week 1: Dataset Preview & Upload", use_container_width=True):
            st.session_state["nav_selection"] = "Week 1: 📂 Dataset Preview"
            st.rerun()
        if st.button("🧩 Week 4: Feature Engineering & Ratios", use_container_width=True):
            st.session_state["nav_selection"] = "Week 4: 🧩 Feature Engineering"
            st.rerun()
        if st.button("📊 Week 7: Diagnostics & Lift Evaluation", use_container_width=True):
            st.session_state["nav_selection"] = "Week 7: 📊 Model Diagnostics"
            st.rerun()
            
    with col2:
        if st.button("🔎 Week 2: EDA & Outlier Diagnostics", use_container_width=True):
            st.session_state["nav_selection"] = "Week 2: 🔎 EDA Viewer"
            st.rerun()
        if st.button("🏋️ Week 5: Dual Baseline Model Training", use_container_width=True):
            st.session_state["nav_selection"] = "Week 5: 🏋️ Baseline Models"
            st.rerun()
        if st.button("🚀 Week 8: Full Pipeline Runner & Export", use_container_width=True):
            st.session_state["nav_selection"] = "Week 8: 🚀 Pipeline Runner"
            st.rerun()
            
    with col3:
        if st.button("🧼 Week 3: Cleaning & Imputation", use_container_width=True):
            st.session_state["nav_selection"] = "Week 3: 🧼 Data Cleaning"
            st.rerun()
        if st.button("⚙️ Week 6: Hyperparameter Tuning", use_container_width=True):
            st.session_state["nav_selection"] = "Week 6: ⚙️ Model Tuning"
            st.rerun()
        if st.button("🔄 Reset Application State", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. Executive Visual Gallery
    # -------------------------------------------------------------
    st.subheader("📊 Executive Portfolio Insights Gallery")
    
    g_c1, g_c2 = st.columns(2)
    
    with g_c1:
        # Chart 1: Recovery by Product Line
        if "loan_type" in df.columns and "recovered_amount" in df.columns:
            type_agg = df.groupby("loan_type", as_index=False).agg(
                Total_Recovered=("recovered_amount", "sum"),
                Avg_Loan=("loan_amount", "mean") if "loan_amount" in df.columns else ("loan_type", "count")
            )
            fig_type = px.bar(
                type_agg, x="loan_type", y="Total_Recovered",
                color="Total_Recovered", color_continuous_scale="Viridis",
                title="<b>Total Amount Recovered ($) by Loan Product Line</b>",
                template="plotly_white"
            )
            st.plotly_chart(fig_type, use_container_width=True)
            
        # Chart 3: Density Heatmap Credit Score vs Overdue Days
        if "credit_score" in df.columns and "overdue_days" in df.columns:
            fig_dens = px.density_heatmap(
                df, x="credit_score", y="overdue_days",
                title="<b>Credit Score vs Delinquency Overdue Duration Density</b>",
                color_continuous_scale="Viridis", template="plotly_white"
            )
            st.plotly_chart(fig_dens, use_container_width=True)
            
    with g_c2:
        # Chart 2: Default Rate by Region
        if "region" in df.columns:
            if "default_flag" in df.columns:
                reg_agg = df.groupby("region", as_index=False)["default_flag"].mean()
                reg_agg["Default_Rate"] = (reg_agg["default_flag"] * 100).round(2)
                fig_reg = px.bar(
                    reg_agg, x="region", y="Default_Rate", color="Default_Rate",
                    color_continuous_scale="Reds", title="<b>Gross Default (NPA) Rate by Geographic Region (%)</b>",
                    template="plotly_white"
                )
            else:
                reg_agg = df.groupby("region", as_index=False).size().rename(columns={"size": "Count"})
                fig_reg = px.bar(reg_agg, x="region", y="Count", color_discrete_sequence=["#38bdf8"], title="<b>Regional Distribution</b>", template="plotly_white")
            st.plotly_chart(fig_reg, use_container_width=True)
            
        # Chart 4: Recovered Amount by Primary Channel Boxplot
        if "primary_channel" in df.columns and "recovered_amount" in df.columns:
            rec_sub = df[df["recovered_amount"] > 0]
            if len(rec_sub) > 0:
                fig_chan = px.box(
                    rec_sub, x="primary_channel", y="recovered_amount",
                    title="<b>Recovery Dollars Spread by Primary Collection Channel</b>",
                    color_discrete_sequence=["#38bdf8"],
                    template="plotly_white"
                )
                st.plotly_chart(fig_chan, use_container_width=True)

    # -------------------------------------------------------------
    # 3. Interactive AI Loan Recovery & Risk Simulator
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Live AI Loan Recovery Predictor & Strategy Simulator")
    st.markdown("Estimate borrower recovery probabilities ($P(\\text{Recovery})$) and expected dollar returns ($E[\\$]$) in real time.")
    
    with st.form("quick_sim_form"):
        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            in_score = st.slider("Borrower Credit Score", 300, 850, 650)
            in_income = st.number_input("Annual Income ($)", 15000, 300000, 55000, 2500)
            in_type = st.selectbox("Loan Product", ["Personal Loan", "Auto Loan", "Home Loan", "Business Loan", "Education Loan"])
        with s_c2:
            in_amt = st.number_input("Outstanding Loan Balance ($)", 2000, 500000, 25000, 1000)
            in_rate = st.slider("Interest Rate (%)", 5.0, 26.0, 13.5)
            in_dpd = st.slider("Overdue Days (DPD)", 0, 360, 45)
        with s_c3:
            in_attempts = st.slider("Contact Attempts Made", 1, 25, 3)
            in_discount = st.slider("Offered Settlement Haircut (%)", 0.0, 50.0, 10.0, 5.0)
            in_channel = st.selectbox("Collection Channel", ["AI Voice Bot & SMS", "Tele-Calling & Counseling", "Field Visit & Face-to-Face", "Legal Notice & DRT", "External Recovery Agency"])
            
        sim_submit = st.form_submit_button("⚡ Compute AI Recovery Forecast", use_container_width=True)

    if sim_submit:
        # Calculate heuristics / model prediction
        dti_est = (in_amt * 0.03 * 12) / in_income
        sev_idx = min(100.0, (in_dpd / 360.0) * 60 + (1 - in_score / 850.0) * 40)
        
        prob_base = 0.45 + (in_score - 580) * 0.0015 - (in_dpd - 30) * 0.0012 + (in_discount * 0.005)
        if in_channel in ["AI Voice Bot & SMS", "Tele-Calling & Counseling"] and in_dpd <= 60:
            prob_base += 0.15
        elif in_channel == "Legal Notice & DRT" and in_type in ["Home Loan", "Auto Loan"]:
            prob_base += 0.22
        
        prob_rec = float(np.clip(prob_base, 0.05, 0.96))
        exp_dollars = in_amt * (1 - in_discount / 100.0) * prob_rec
        
        st.markdown("### 🎯 Simulation Output")
        o1, o2, o3 = st.columns(3)
        with o1:
            st.metric("Recovery Likelihood (P(Recovery))", f"{prob_rec*100:.1f}%")
            if prob_rec >= 0.70:
                st.success("🟢 High Recovery Potential (Grade A)")
            elif prob_rec >= 0.40:
                st.warning("🟡 Moderate Recovery Potential (Grade B)")
            else:
                st.error("🔴 Severe Default / Write-off Risk (Grade C)")
        with o2:
            st.metric("Expected Dollar Recovery", f"${exp_dollars:,.2f}")
            st.caption(f"Estimated Net Yield: ${exp_dollars * 0.93:,.2f}")
        with o3:
            st.metric("Delinquency Severity Index", f"{sev_idx:.1f} / 100")
            st.caption(f"Estimated DTI: {dti_est:.2f}")

    # Footer
    st.markdown("---")
    st.caption("🏦 AI-Driven Loan Recovery & Risk Analytics | 9-Week Streamlit ML Lifecycle Architecture | Enterprise Banking & Financial Services")
