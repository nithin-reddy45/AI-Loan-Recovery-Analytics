import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week4():
    st.header("🧩 Week 4: Feature Engineering, Encoding & Scaling")
    st.markdown("Transform raw variables, build credit risk ratios, and encode/scale feature matrices.")
    
    df_clean = st.session_state.get("df_clean", None)
    if df_clean is None:
        show_prerequisite_warning("df_clean (Cleaned Dataset)", "Week 3: 🧼 Cleaning Comparison")
        return
        
    df_filtered = apply_global_filters(df_clean)
    render_global_kpi_cards(df_filtered)
    
    # -------------------------------------------------------------
    # 1. Feature Engineering Configuration
    # -------------------------------------------------------------
    with st.expander("⚙️ Configure Transformations & Feature Construction", expanded=True):
        c_fe1, c_fe2, c_fe3 = st.columns(3)
        
        with c_fe1:
            st.markdown("**1. Categorical Encoding Method**")
            enc_method = st.selectbox("Encoding Strategy", ["One-Hot Encoding", "Ordinal / Label Encoding", "Frequency Encoding"], index=0)
            target_cat_cols = st.multiselect("Select Categorical Columns", ["loan_type", "region", "city_tier", "employment_status", "collateral_type", "primary_channel"], default=["loan_type", "region", "primary_channel"])
            
        with c_fe2:
            st.markdown("**2. Numerical Scaling Strategy**")
            scaler_choice = st.selectbox("Scaling Algorithm", ["StandardScaler (z-score)", "MinMaxScaler (0 to 1)", "RobustScaler (IQR)"], index=0)
            target_scale_cols = st.multiselect("Select Numeric Columns to Scale", ["annual_income", "credit_score", "loan_amount", "interest_rate", "overdue_days"], default=["annual_income", "credit_score", "loan_amount", "overdue_days"])
            
        with c_fe3:
            st.markdown("**3. Domain Engineered Features (≥3)**")
            feat_dti = st.checkbox("🟩 Debt-to-Income (DTI Ratio)", value=True)
            feat_ltv = st.checkbox("🟩 Loan-to-Value (LTV Ratio)", value=True)
            feat_sev = st.checkbox("🟩 Delinquency Severity Index (0-100)", value=True)
            feat_eff = st.checkbox("🟩 Recovery Efficiency Multiplier", value=True)

        apply_fe_btn = st.button("⚡ Transform Features & Save `df_features` to Memory", use_container_width=True)

    # Execute Feature Engineering
    if apply_fe_btn or st.session_state.get("df_features") is None:
        df_feat = df_clean.copy()
        
        # 1. Engineered Features
        if feat_dti and "interest_rate" in df_feat.columns and "loan_term_months" in df_feat.columns and "loan_amount" in df_feat.columns and "annual_income" in df_feat.columns:
            r = (df_feat["interest_rate"] / 100.0) / 12.0
            n = df_feat["loan_term_months"]
            emi = df_feat["loan_amount"] * (r * (1 + r)**n) / ((1 + r)**n - 1)
            df_feat["dti_ratio"] = ((emi * 12) / np.maximum(df_feat["annual_income"], 1000.0)).round(4)
            
        if feat_ltv and "loan_amount" in df_feat.columns and "collateral_value" in df_feat.columns:
            df_feat["ltv_ratio"] = np.where(
                df_feat["collateral_value"] > 0,
                np.clip(df_feat["loan_amount"] / df_feat["collateral_value"], 0.1, 1.5),
                1.0
            ).round(4)
            
        if feat_sev and "overdue_days" in df_feat.columns and "credit_score" in df_feat.columns:
            df_feat["delinquency_severity_score"] = np.clip(
                (df_feat["overdue_days"] / 360.0) * 60 + (1 - df_feat["credit_score"] / 850.0) * 40,
                0, 100
            ).round(2)
            
        if feat_eff and "recovered_amount" in df_feat.columns and "recovery_cost" in df_feat.columns:
            df_feat["recovery_efficiency"] = (df_feat["recovered_amount"] / np.maximum(df_feat["recovery_cost"], 1.0)).round(2)
            
        # Ensure target columns are preserved
        if "recovery_status_binary" not in df_feat.columns:
            if "recovery_target_binary" in df_feat.columns:
                df_feat["recovery_status_binary"] = df_feat["recovery_target_binary"].fillna(0).astype(int)
            elif "recovery_status" in df_feat.columns:
                df_feat["recovery_status_binary"] = df_feat["recovery_status"].apply(
                    lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0
                )
            elif "default_flag" in df_feat.columns:
                df_feat["recovery_status_binary"] = (1 - df_feat["default_flag"]).astype(int)
            else:
                df_feat["recovery_status_binary"] = 0
                
        if "recovered_amount" not in df_feat.columns:
            df_feat["recovered_amount"] = 0.0
            
        # 2. Scaling
        if scaler_choice == "StandardScaler (z-score)":
            scaler = StandardScaler()
        elif scaler_choice == "MinMaxScaler (0 to 1)":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        valid_scale_cols = [c for c in target_scale_cols if c in df_feat.columns]
        if valid_scale_cols:
            scaled_vals = scaler.fit_transform(df_feat[valid_scale_cols])
            for idx, c in enumerate(valid_scale_cols):
                df_feat[f"{c}_scaled"] = scaled_vals[:, idx].round(4)
                
        # 3. Categorical Encoding
        valid_cat_cols = [c for c in target_cat_cols if c in df_feat.columns]
        if enc_method == "One-Hot Encoding" and valid_cat_cols:
            df_feat = pd.get_dummies(df_feat, columns=valid_cat_cols, drop_first=True)
        elif enc_method == "Ordinal / Label Encoding" and valid_cat_cols:
            for c in valid_cat_cols:
                le = LabelEncoder()
                df_feat[f"{c}_encoded"] = le.fit_transform(df_feat[c].astype(str))
                
        st.session_state["df_features"] = df_feat
        if apply_fe_btn:
            st.success("✅ Features transformed and saved as `df_features` in session state!")

    df_features_view = st.session_state["df_features"]

    # -------------------------------------------------------------
    # 2. Feature Tags & Metadata
    # -------------------------------------------------------------
    st.markdown("### 🏷️ Active Feature Inventory by Role")
    
    encoded_cols = [c for c in df_features_view.columns if any(c.startswith(f"{x}_") for x in ["loan_type", "region", "city_tier", "primary_channel"]) or c.endswith("_encoded")]
    scaled_cols = [c for c in df_features_view.columns if c.endswith("_scaled")]
    engineered_cols = [c for c in ["dti_ratio", "ltv_ratio", "delinquency_severity_score", "recovery_efficiency"] if c in df_features_view.columns]
    
    c_t1, c_t2, c_t3 = st.columns(3)
    with c_t1:
        st.markdown(f"**🟦 Encoded Features ({len(encoded_cols)})**")
        st.caption(", ".join(encoded_cols[:6]) + ("..." if len(encoded_cols) > 6 else ""))
    with c_t2:
        st.markdown(f"**🟧 Scaled Features ({len(scaled_cols)})**")
        st.caption(", ".join(scaled_cols))
    with c_t3:
        st.markdown(f"**🟩 Domain Engineered ({len(engineered_cols)})**")
        st.caption(", ".join(engineered_cols))

    # -------------------------------------------------------------
    # 3. Transformed Data Table & Visualizations
    # -------------------------------------------------------------
    tab_data, tab_vis = st.tabs(["📋 Transformed Feature Matrix (`df_features`)", "📈 Visual Impact Comparison"])
    
    with tab_data:
        st.dataframe(df_features_view.head(20), use_container_width=True)
        csv_feat_bytes = df_features_view.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Feature Engineered Dataset (CSV)", data=csv_feat_bytes, file_name="feature_engineered_dataset.csv")
        
    with tab_vis:
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            # Raw vs Scaled comparison
            if "credit_score" in df_clean.columns and "credit_score_scaled" in df_features_view.columns:
                fig_sc = px.histogram(
                    df_features_view, x="credit_score_scaled", nbins=30,
                    title="<b>Credit Score after Scaling Distribution</b>",
                    color_discrete_sequence=["#f97316"], template="plotly_white"
                )
                st.plotly_chart(fig_sc, use_container_width=True)
        with c_p2:
            # Engineered feature vs recovery target
            if "delinquency_severity_score" in df_features_view.columns and "recovered_amount" in df_features_view.columns:
                fig_eng = px.scatter(
                    df_features_view[df_features_view["recovered_amount"] > 0].sample(min(1200, (df_features_view["recovered_amount"] > 0).sum())),
                    x="delinquency_severity_score", y="recovered_amount",
                    color="recovery_status_binary" if "recovery_status_binary" in df_features_view.columns else None,
                    title="<b>Delinquency Severity Score vs Amount Recovered ($)</b>", template="plotly_white"
                )
                st.plotly_chart(fig_eng, use_container_width=True)
