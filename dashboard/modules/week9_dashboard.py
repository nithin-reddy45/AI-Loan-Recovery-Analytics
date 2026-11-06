import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Loads the serialized champion LightGBM pipeline and SHAP cache."""
    model_path = "models/recovery_predictor_lightgbm.pkl"
    shap_path = "models/shap_explainer_data.pkl"
    
    pipeline = None
    shap_cache = None
    
    if os.path.exists(model_path):
        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            
    if os.path.exists(shap_path):
        try:
            shap_cache = joblib.load(shap_path)
        except Exception as e:
            st.warning(f"Error loading SHAP cache: {e}")
            
    return pipeline, shap_cache

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
    # 3. Live AI Recovery Simulator with Explainable AI (SHAP)
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🤖 Live AI Recovery Probability Predictor & Explainability (SHAP)")
    st.caption("Predicts debt recovery likelihood using the serialized **LightGBM Champion Pipeline** and decomposes individual borrower factors using **Explainable AI (SHAP)**.")

    pipeline, shap_cache = load_trained_model()
    
    # -------------------------------------------------------------
    # Step 16: Pre-configured Test Cases (Recovery vs No-Recovery)
    # -------------------------------------------------------------
    st.markdown("##### 🧪 Step 16: One-Click Test Case Scenarios")
    tc1, tc2, tc3 = st.columns(3)
    
    # Defaults in session state
    if "sim_credit_score" not in st.session_state:
        st.session_state["sim_credit_score"] = 680
        st.session_state["sim_income"] = 65000
        st.session_state["sim_loan_amount"] = 25000
        st.session_state["sim_interest_rate"] = 11.5
        st.session_state["sim_term"] = 48
        st.session_state["sim_dpd"] = 42
        st.session_state["sim_dti"] = 0.32
        st.session_state["sim_ltv"] = 0.70
        st.session_state["sim_is_secured"] = 1
        st.session_state["sim_attempts"] = 4
        st.session_state["sim_ptp_kept"] = 1
        st.session_state["sim_debts"] = 2
        st.session_state["sim_age"] = 38
        st.session_state["sim_loan_type"] = "Auto Loan"
        st.session_state["sim_region"] = "North"
        st.session_state["sim_city_tier"] = "Tier 1"
        st.session_state["sim_emp"] = "Salaried - Corporate"
        st.session_state["sim_housing"] = "Owned"
        st.session_state["sim_collateral"] = "Vehicle"
        st.session_state["sim_channel"] = "AI Voice Bot & SMS"
        st.session_state["sim_balance"] = 18500
        st.session_state["sim_discount"] = 10

    with tc1:
        if st.button("🟢 Case A: High Recovery Likelihood (Prime Auto)", use_container_width=True):
            st.session_state["sim_credit_score"] = 740
            st.session_state["sim_income"] = 85000
            st.session_state["sim_loan_amount"] = 30000
            st.session_state["sim_interest_rate"] = 8.5
            st.session_state["sim_term"] = 60
            st.session_state["sim_dpd"] = 32
            st.session_state["sim_dti"] = 0.24
            st.session_state["sim_ltv"] = 0.60
            st.session_state["sim_is_secured"] = 1
            st.session_state["sim_attempts"] = 2
            st.session_state["sim_ptp_kept"] = 1
            st.session_state["sim_debts"] = 1
            st.session_state["sim_age"] = 44
            st.session_state["sim_loan_type"] = "Auto Loan"
            st.session_state["sim_collateral"] = "Vehicle"
            st.session_state["sim_channel"] = "AI Voice Bot & SMS"
            st.session_state["sim_balance"] = 21000
            st.session_state["sim_discount"] = 5
            st.rerun()

    with tc2:
        if st.button("🟡 Case B: Moderate Risk / Distressed (Restructure)", use_container_width=True):
            st.session_state["sim_credit_score"] = 635
            st.session_state["sim_income"] = 48000
            st.session_state["sim_loan_amount"] = 20000
            st.session_state["sim_interest_rate"] = 14.0
            st.session_state["sim_term"] = 36
            st.session_state["sim_dpd"] = 68
            st.session_state["sim_dti"] = 0.44
            st.session_state["sim_ltv"] = 0.85
            st.session_state["sim_is_secured"] = 0
            st.session_state["sim_attempts"] = 6
            st.session_state["sim_ptp_kept"] = 0
            st.session_state["sim_debts"] = 3
            st.session_state["sim_age"] = 32
            st.session_state["sim_loan_type"] = "Personal Loan"
            st.session_state["sim_collateral"] = "None"
            st.session_state["sim_channel"] = "Tele-Calling & Counseling"
            st.session_state["sim_balance"] = 16200
            st.session_state["sim_discount"] = 15
            st.rerun()

    with tc3:
        if st.button("🔴 Case C: Severe Default / Write-Off (NPA 150+ DPD)", use_container_width=True):
            st.session_state["sim_credit_score"] = 490
            st.session_state["sim_income"] = 26000
            st.session_state["sim_loan_amount"] = 18000
            st.session_state["sim_interest_rate"] = 19.5
            st.session_state["sim_term"] = 24
            st.session_state["sim_dpd"] = 180
            st.session_state["sim_dti"] = 0.65
            st.session_state["sim_ltv"] = 0.95
            st.session_state["sim_is_secured"] = 0
            st.session_state["sim_attempts"] = 16
            st.session_state["sim_ptp_kept"] = 0
            st.session_state["sim_debts"] = 6
            st.session_state["sim_age"] = 26
            st.session_state["sim_loan_type"] = "Personal Loan"
            st.session_state["sim_collateral"] = "None"
            st.session_state["sim_channel"] = "External Recovery Agency"
            st.session_state["sim_balance"] = 17800
            st.session_state["sim_discount"] = 25
            st.rerun()

    # -------------------------------------------------------------
    # Step 15: Streamlit Form for Borrower Inputs
    # -------------------------------------------------------------
    with st.form("sim_form_v2"):
        st.markdown("##### 📝 Borrower Financial & Recovery Profile")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            score = st.slider("Borrower Credit Score", 300, 850, st.session_state["sim_credit_score"])
            income = st.number_input("Annual Income ($)", 10000, 400000, st.session_state["sim_income"], 5000)
            loan_amt = st.number_input("Disbursed Loan Amount ($)", 2000, 300000, st.session_state["sim_loan_amount"], 1000)
            int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, float(st.session_state["sim_interest_rate"]), 0.5)
            term_months = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 84, 120, 240], index=3)
            
        with col_f2:
            balance = st.number_input("Outstanding Default Balance ($)", 500, 300000, st.session_state["sim_balance"], 1000)
            dpd = st.slider("Days Past Due (DPD Overdue)", 31, 365, st.session_state["sim_dpd"])
            dti = st.slider("Debt-to-Income (DTI Ratio)", 0.05, 0.90, float(st.session_state["sim_dti"]), 0.01)
            ltv = st.slider("Loan-to-Value (LTV Ratio)", 0.10, 1.20, float(st.session_state["sim_ltv"]), 0.05)
            age = st.slider("Borrower Age", 21, 70, st.session_state["sim_age"])

        with col_f3:
            ptp = st.selectbox("Promise-to-Pay (PTP) History", ["PTP Kept (Honored)", "PTP Broken / Unresponsive"], 
                               index=0 if st.session_state["sim_ptp_kept"] == 1 else 1)
            attempts = st.slider("Collector Contact Attempts", 1, 30, st.session_state["sim_attempts"])
            debts = st.slider("Existing Debts Count", 0, 10, st.session_state["sim_debts"])
            loan_type = st.selectbox("Loan Type", ["Auto Loan", "Home Loan", "Personal Loan", "Education Loan", "Business Loan"],
                                     index=["Auto Loan", "Home Loan", "Personal Loan", "Education Loan", "Business Loan"].index(st.session_state["sim_loan_type"]) if st.session_state["sim_loan_type"] in ["Auto Loan", "Home Loan", "Personal Loan", "Education Loan", "Business Loan"] else 0)
            collateral = st.selectbox("Collateral Type", ["Vehicle", "Real Estate Property", "Equipment / Inventory", "None"],
                                      index=0 if "Auto" in st.session_state["sim_loan_type"] else (1 if "Home" in st.session_state["sim_loan_type"] else 3))
            channel = st.selectbox("Primary Recovery Channel", [
                "AI Voice Bot & SMS", "Tele-Calling & Counseling", "Field Visit & Face-to-Face", 
                "Debt Restructuring", "Legal Notice & DRT", "External Recovery Agency"
            ], index=0)
            discount = st.slider("Proposed Settlement Discount (%)", 0, 50, st.session_state["sim_discount"])

        sim_btn = st.form_submit_button("⚡ Run AI Prediction & SHAP Analysis", use_container_width=True)

    if sim_btn:
        # Construct input payload matching the ML pipeline
        ptp_val = 1 if "Kept" in ptp else 0
        is_sec_val = 0 if collateral == "None" else 1
        sev_score = min(1.0, (dpd / 90.0) * 0.5 + (1 - ptp_val) * 0.3 + (dti / 0.6) * 0.2)
        
        borrower_payload = pd.DataFrame([{
            "credit_score": score,
            "annual_income": income,
            "loan_amount": loan_amt,
            "interest_rate": int_rate,
            "loan_term_months": term_months,
            "overdue_days": dpd,
            "dti_ratio": dti,
            "ltv_ratio": ltv,
            "is_secured": is_sec_val,
            "contact_attempts": attempts,
            "promise_to_pay_kept": ptp_val,
            "delinquency_severity_score": round(sev_score, 3),
            "existing_debts_count": debts,
            "age": age,
            "loan_type": loan_type,
            "region": st.session_state.get("sim_region", "North"),
            "city_tier": st.session_state.get("sim_city_tier", "Tier 1"),
            "employment_status": st.session_state.get("sim_emp", "Salaried - Corporate"),
            "housing_status": st.session_state.get("sim_housing", "Owned"),
            "collateral_type": collateral,
            "origination_channel": "Branch Banking",
            "primary_channel": channel
        }])
        
        # Predict using trained pipeline or fallback
        if pipeline is not None:
            proba = float(pipeline.predict_proba(borrower_payload)[0, 1])
        else:
            proba = float(np.clip(0.45 + (score - 600) * 0.0012 - (dpd - 40) * 0.001 + (ptp_val * 0.25), 0.05, 0.95))
            
        opt_thresh = shap_cache.get("optimal_threshold", 0.175) if shap_cache else 0.175
        is_recovered_opt = proba >= opt_thresh
        is_recovered_std = proba >= 0.50
        
        risk_score_pct = float(np.clip((1.0 - proba) * 100.0, 1.0, 99.0))
        exp_yield = balance * (1.0 - discount / 100.0) * proba
        
        # Risk Tiers & Strategic Actions
        if proba >= 0.75:
            risk_level = "🟢 Low Risk (Grade A - High Recovery)"
            risk_color = "#10b981"
            strategy = "Digital Automation & Self-Curative AI Links"
            recommendation = "High propensity to self-cure. Dispatch automated WhatsApp/SMS interactive payment link with zero human collector overhead."
        elif proba >= 0.50:
            risk_level = "🟡 Moderate Risk (Grade B - Restructure)"
            risk_color = "#f59e0b"
            strategy = "Tele-Counseling & Term Restructuring"
            recommendation = "Borrower is cooperative but liquidity constrained. Offer a 6-month term extension or nominal 5-10% settlement discount."
        elif proba >= opt_thresh:
            risk_level = "🟠 Elevated Risk (Grade C - Targeted Intervention)"
            risk_color = "#f97316"
            strategy = "Field Visit & Standardized Haircut"
            recommendation = f"Identified as recoverable under Step 11 Optimal Threshold ({opt_thresh:.2f}). Dispatch field officer for physical contact; offer standardized 15% settlement discount."
        else:
            risk_level = "🔴 Critical Risk (Grade D - Unrecoverable / NPA)"
            risk_color = "#ef4444"
            strategy = "Immediate Legal Escalation / Debt Recovery Tribunal"
            recommendation = "Very low recovery probability. Limit operational expenditure; issue formal legal demand notice and transfer to specialized recovery agency."

        st.markdown("---")
        st.markdown("#### 🎯 Machine Learning Prediction & Risk Tier")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Level of Risk", risk_level)
        with m2:
            st.metric("Recovery Likelihood P(Rec)", f"{proba*100:.1f}%", delta=f"{'Recoverable' if is_recovered_opt else 'Default'}")
        with m3:
            st.metric("Default Risk Score", f"{risk_score_pct:.1f}%", delta=f"Threshold: {opt_thresh:.2f}", delta_color="inverse")
        with m4:
            st.metric("Expected Dollar Yield", f"${exp_yield:,.2f}")

        # Step 11 Comparison Metric Callout
        st.info(f"**Step 11 Decision Optimization:** At standard threshold (0.50), status is **{'Recoverable' if is_recovered_std else 'Default'}**. Under the Recall-Optimized Threshold (**{opt_thresh:.2f}** calibrated to recover 97.6% of delinquent cash flow), account is classified as **{'RECOVERABLE — Prioritize Collections' if is_recovered_opt else 'UNRECOVERABLE — Fast-Track Legal'}**.")

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 10px; padding: 16px; border-left: 6px solid {risk_color}; margin-top: 10px; margin-bottom: 20px; color: white;">
            <div style="font-size: 0.85rem; font-weight: 700; color: {risk_color}; text-transform: uppercase;">
                📋 Recommended Strategy: {strategy}
            </div>
            <div style="font-size: 0.95rem; margin-top: 6px; color: #e2e8f0;">
                {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # EXPLAINABLE AI (SHAP): Dynamic Individual Waterfall Plot
        # -------------------------------------------------------------
        st.markdown("#### 🔬 Explainable AI (SHAP): Why Did the Model Make This Prediction?")
        st.caption("SHAP (SHapley Additive exPlanations) breaks down the exact positive and negative contribution of each borrower attribute towards this prediction.")
        
        if pipeline is not None and shap_cache is not None:
            try:
                prep = pipeline.named_steps["preprocessor"]
                clf = pipeline.named_steps["classifier"]
                
                trans_row = prep.transform(borrower_payload)
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(trans_row)
                
                if isinstance(shap_vals, list):
                    s_vec = shap_vals[1][0]
                elif shap_vals.ndim == 3:
                    s_vec = shap_vals[0, :, 1]
                else:
                    s_vec = shap_vals[0]
                    
                all_feats = shap_cache["all_features"]
                local_shap_df = pd.DataFrame({"feature": all_feats, "shap_impact": s_vec})
                
                # Separate positive and negative impacts
                top_pos = local_shap_df.sort_values(by="shap_impact", ascending=False).head(5)
                top_neg = local_shap_df.sort_values(by="shap_impact", ascending=True).head(5)
                local_top = pd.concat([top_pos, top_neg]).drop_duplicates().sort_values(by="shap_impact", ascending=True)
                
                local_top["color"] = local_top["shap_impact"].apply(lambda v: "#10b981" if v >= 0 else "#ef4444")
                
                fig_shap_bar = go.Figure(go.Bar(
                    x=local_top["shap_impact"],
                    y=local_top["feature"],
                    orientation="h",
                    marker=dict(color=local_top["color"]),
                    text=local_top["shap_impact"].apply(lambda v: f"{v:+.3f}"),
                    textposition="auto"
                ))
                fig_shap_bar.update_layout(
                    title="Local SHAP Feature Contributions (Green = Pushes toward Recovery, Red = Pushes toward Default)",
                    xaxis_title="SHAP Value (Log-Odds Impact)",
                    yaxis_title="Feature",
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_shap_bar, use_container_width=True)
                
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.markdown("##### 🟢 Top Positive Drivers (Supporting Recovery)")
                    for _, r in top_pos[top_pos["shap_impact"] > 0].iterrows():
                        st.markdown(f"- **`{r['feature']}`**: `{r['shap_impact']:+.3f}` log-odds contribution")
                with col_exp2:
                    st.markdown("##### 🔴 Top Negative Drivers (Increasing Default Risk)")
                    for _, r in top_neg[top_neg["shap_impact"] < 0].iterrows():
                        st.markdown(f"- **`{r['feature']}`**: `{r['shap_impact']:+.3f}` log-odds penalty")
                        
            except Exception as e:
                st.warning(f"Unable to render live SHAP waterfall: {e}")
        else:
            st.info("Train the model pipeline via Week 5 or Run Pipeline in Week 8 to view live SHAP explanations.")
