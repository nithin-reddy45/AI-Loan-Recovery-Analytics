import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters

def render_week9():
    st.title("🏦 AI Loan Recovery & Risk Analytics Dashboard")
    
    df_raw = st.session_state.get("df_raw", None)
    if df_raw is None or len(df_raw) == 0:
        st.info("⚠️ No data loaded yet. Please visit **Week 1** to upload or load data.")
        return
        
    df = apply_global_filters(df_raw)
    render_global_kpi_cards(df)
    
    # -------------------------------------------------------------
    # 1. Quick Navigation Hub
    # -------------------------------------------------------------
    st.markdown("### 🧭 Fast Navigation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📂 1. Upload Data", use_container_width=True):
            st.session_state["nav_selection"] = "Week 1: 📂 Dataset Preview"
            st.rerun()
    with c2:
        if st.button("🧼 3. Clean Data", use_container_width=True):
            st.session_state["nav_selection"] = "Week 3: 🧼 Data Cleaning"
            st.rerun()
    with c3:
        if st.button("🏋️ 5. Train Models", use_container_width=True):
            st.session_state["nav_selection"] = "Week 5: 🏋️ Baseline Models"
            st.rerun()
    with c4:
        if st.button("🚀 8. Run Pipeline", use_container_width=True):
            st.session_state["nav_selection"] = "Week 8: 🚀 Pipeline Runner"
            st.rerun()

    # -------------------------------------------------------------
    # 2. Portfolio Visual Summary
    # -------------------------------------------------------------
    st.markdown("### 📊 Portfolio Visual Highlights")
    c_p1, c_p2 = st.columns(2)
    sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
    
    with c_p1:
        cat_col = "loan_type" if "loan_type" in df.columns else (list(df.select_dtypes(include=['object']).columns)[0] if len(df.select_dtypes(include=['object']).columns) > 0 else None)
        num_col = "recovered_amount" if "recovered_amount" in df.columns else "loan_amount"
        if cat_col and num_col in df.columns:
            agg = df.groupby(cat_col, as_index=False)[num_col].sum()
            fig = px.bar(agg, x=cat_col, y=num_col, title=f"Total {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}")
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
            
    with c_p2:
        if "credit_score" in df.columns and "overdue_days" in df.columns:
            fig2 = px.scatter(sample_df, x="credit_score", y="overdue_days", title="Credit Score vs Overdue Days (Sampled)")
            fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------
    # 3. Live AI Recovery Simulator with Risk Level Prediction
    # -------------------------------------------------------------
    st.markdown("### 🤖 Live AI Recovery Simulator & Risk Predictor")
    st.caption("Input borrower parameters to compute recovery likelihood, continuous dollar yield, and the specific **Level of Risk**.")
    
    with st.form("sim_form"):
        s1, s2, s3 = st.columns(3)
        with s1:
            score = st.slider("Borrower Credit Score", 300, 850, 660)
            income = st.number_input("Annual Income ($)", 15000, 300000, 60000, 5000)
        with s2:
            balance = st.number_input("Outstanding Balance ($)", 1000, 200000, 20000, 1000)
            dpd = st.slider("Overdue Days (DPD)", 0, 360, 45)
        with s3:
            discount = st.slider("Offered Settlement Discount (%)", 0, 50, 10)
            channel = st.selectbox("Channel", ["AI Digital Bot / SMS", "Tele-Calling & Counseling", "Field Visit", "Legal Notice"])
            
        sim_btn = st.form_submit_button("⚡ Predict Recovery & Risk Level", use_container_width=True)
        
    if sim_btn:
        # Base probability calculation
        prob = np.clip(0.50 + (score - 600) * 0.0015 - (dpd - 30) * 0.0012 + (discount * 0.004), 0.05, 0.98)
        risk_score_pct = float(np.clip((1.0 - prob) * 100.0, 2.0, 98.0))
        exp_yield = balance * (1.0 - discount / 100.0) * prob
        
        # Determine Level of Risk & Actionable Strategy
        if risk_score_pct < 30.0:
            risk_level = "🟢 Low Risk (Grade A)"
            risk_color = "#10b981"
            strategy = "Self-Curative / Automated AI Reminders"
            recommendation = "Low default probability. Route to automated WhatsApp/SMS digital payment links with zero agent intervention needed."
        elif risk_score_pct < 55.0:
            risk_level = "🟡 Moderate / Medium Risk (Grade B)"
            risk_color = "#f59e0b"
            strategy = "Tele-Calling & Restructuring Counselor"
            recommendation = "Moderate slip risk. Assign to inside tele-counselor to offer a 6-month tenure extension or structured repayment plan."
        elif risk_score_pct < 75.0:
            risk_level = "🟠 High Risk (Grade C)"
            risk_color = "#f97316"
            strategy = "Field Officer Visit & Settlement Haircut"
            recommendation = "High default severity. Dispatch field recovery officer for physical contact and offer a 10%-15% standardized settlement haircut."
        else:
            risk_level = "🔴 Critical / Severe Risk (Grade D - NPA)"
            risk_color = "#ef4444"
            strategy = "Immediate Legal Escalation & Recovery Agency"
            recommendation = "Imminent write-off risk. Issue formal legal notice under debt recovery tribunal and escalate to specialized collection agency."

        st.markdown("---")
        st.markdown("#### 🎯 Prediction & Risk Diagnostic Results")
        
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Level of Risk", risk_level)
        with r2:
            st.metric("Default Risk Score", f"{risk_score_pct:.1f}%", delta=f"{100-risk_score_pct:.1f}% Recovery Prob", delta_color="inverse")
        with r3:
            st.metric("Expected Dollar Yield", f"${exp_yield:,.2f}")
        with r4:
            st.metric("Recovery Likelihood (P)", f"{prob*100:.1f}%")
            
        st.markdown(f"**Risk Severity Gauge:**")
        st.progress(risk_score_pct / 100.0)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 10px; padding: 16px; border-left: 5px solid {risk_color}; margin-top: 10px; color: white;">
            <div style="font-size: 0.85rem; font-weight: 700; color: {risk_color}; text-transform: uppercase;">
                📋 Recommended Strategy: {strategy}
            </div>
            <div style="font-size: 0.95rem; margin-top: 6px; color: #e2e8f0;">
                {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)
