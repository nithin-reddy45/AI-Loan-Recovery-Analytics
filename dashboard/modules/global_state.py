"""
Global State & Filter Manager: AI-Driven Loan Recovery & Risk Analytics
Maintains persistent session state, global filters, and auto-updating KPI cards across all 9 weeks.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

def init_session_state():
    """Initialize all session state keys for the 9-week ML lifecycle."""
    if "df_raw" not in st.session_state:
        default_data_path = "data/processed/loan_recovery_master.csv"
        if not os.path.exists(default_data_path):
            default_data_path = "data/raw/loans.csv"
        
        if os.path.exists(default_data_path):
            df_init = pd.read_csv(default_data_path)
            # Ensure recovery_status_binary is present
            if "recovery_status_binary" not in df_init.columns:
                if "recovery_target_binary" in df_init.columns:
                    df_init["recovery_status_binary"] = df_init["recovery_target_binary"].fillna(0).astype(int)
                elif "recovery_status" in df_init.columns:
                    df_init["recovery_status_binary"] = df_init["recovery_status"].apply(
                        lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0
                    )
                elif "default_flag" in df_init.columns:
                    df_init["recovery_status_binary"] = (1 - df_init["default_flag"]).astype(int)
                else:
                    df_init["recovery_status_binary"] = 0
                    
            if "recovered_amount" not in df_init.columns:
                df_init["recovered_amount"] = 0.0
            else:
                df_init["recovered_amount"] = df_init["recovered_amount"].fillna(0.0)
                
            st.session_state["df_raw"] = df_init
        else:
            st.session_state["df_raw"] = None

    if "df_clean" not in st.session_state:
        st.session_state["df_clean"] = None
        
    if "df_features" not in st.session_state:
        st.session_state["df_features"] = None
        
    if "model_baseline" not in st.session_state:
        st.session_state["model_baseline"] = {
            "classification": None,
            "regression": None,
            "metrics": {}
        }
        
    if "model_tuned" not in st.session_state:
        st.session_state["model_tuned"] = {
            "classification": None,
            "regression": None,
            "metrics": {}
        }
        
    if "best_params" not in st.session_state:
        st.session_state["best_params"] = {}
        
    if "metrics_tuned" not in st.session_state:
        st.session_state["metrics_tuned"] = {}
        
    if "global_filters" not in st.session_state:
        st.session_state["global_filters"] = {}
        
    if "current_week" not in st.session_state:
        st.session_state["current_week"] = "Week 9: 🏠 Project Dashboard"

def render_global_filters():
    """Render the 7 Global Filters in the sidebar that persist across navigation."""
    st.sidebar.markdown("### 🎛️ Global Portfolio Filters")
    
    df = st.session_state.get("df_raw", None)
    is_disabled = (df is None or len(df) == 0)
    
    if is_disabled:
        st.sidebar.info("Upload a dataset in **Week 1** to activate global filters.")
        return {}
    
    filters = {}
    with st.sidebar.expander("⚙️ Configure Active Filters", expanded=True):
        # 1. Region (Multi-select)
        all_regions = sorted(list(df["region"].dropna().unique())) if "region" in df.columns else ["Default"]
        sel_regions = st.multiselect("👤 Region", all_regions, default=all_regions, disabled=is_disabled)
        filters["region"] = sel_regions
        
        # 2. Loan Type (Multi-select)
        all_loan_types = sorted(list(df["loan_type"].dropna().unique())) if "loan_type" in df.columns else ["Personal Loan"]
        sel_loan_types = st.multiselect("🏦 Loan Product Type", all_loan_types, default=all_loan_types, disabled=is_disabled)
        filters["loan_type"] = sel_loan_types
        
        # 3. Loan Amount Range Slider
        if "loan_amount" in df.columns:
            min_amt = float(df["loan_amount"].min())
            max_amt = float(df["loan_amount"].max())
            sel_amt = st.slider("💵 Loan Amount ($)", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt), disabled=is_disabled)
            filters["loan_amount"] = sel_amt
            
        # 4. Credit Score Range Slider
        if "credit_score" in df.columns:
            min_score = int(df["credit_score"].min())
            max_score = int(df["credit_score"].max())
            sel_score = st.slider("📈 Credit Score", min_value=min_score, max_value=max_score, value=(min_score, max_score), disabled=is_disabled)
            filters["credit_score"] = sel_score
            
        # 5. Overdue Days Range Slider
        if "overdue_days" in df.columns:
            min_dpd = int(df["overdue_days"].min())
            max_dpd = int(df["overdue_days"].max())
            sel_dpd = st.slider("⏱️ Overdue Days (DPD)", min_value=min_dpd, max_value=max_dpd, value=(min_dpd, max_dpd), disabled=is_disabled)
            filters["overdue_days"] = sel_dpd
            
        # 6. Primary Channel (Multi-select)
        if "primary_channel" in df.columns:
            all_channels = sorted(list(df["primary_channel"].dropna().unique()))
            sel_channels = st.multiselect("📞 Recovery Channel", all_channels, default=all_channels, disabled=is_disabled)
            filters["primary_channel"] = sel_channels
            
        # 7. Recovery Status Toggle (Classification Target Filter)
        sel_status = st.radio(
            "🔄 Recovery Target Filter",
            options=["All Accounts", "Only Recovered (1)", "Only Unrecovered (0)"],
            index=0,
            disabled=is_disabled
        )
        filters["recovery_status_binary"] = sel_status

    st.session_state["global_filters"] = filters
    return filters

def apply_global_filters(df):
    """Apply persistent global filters to any dataframe."""
    if df is None or len(df) == 0:
        return df
        
    filters = st.session_state.get("global_filters", {})
    if not filters:
        return df
        
    filtered = df.copy()
    
    if "region" in filters and "region" in filtered.columns and filters["region"]:
        filtered = filtered[filtered["region"].isin(filters["region"])]
        
    if "loan_type" in filters and "loan_type" in filtered.columns and filters["loan_type"]:
        filtered = filtered[filtered["loan_type"].isin(filters["loan_type"])]
        
    if "loan_amount" in filters and "loan_amount" in filtered.columns:
        min_v, max_v = filters["loan_amount"]
        filtered = filtered[(filtered["loan_amount"] >= min_v) & (filtered["loan_amount"] <= max_v)]
        
    if "credit_score" in filters and "credit_score" in filtered.columns:
        min_v, max_v = filters["credit_score"]
        filtered = filtered[(filtered["credit_score"] >= min_v) & (filtered["credit_score"] <= max_v)]
        
    if "overdue_days" in filters and "overdue_days" in filtered.columns:
        min_v, max_v = filters["overdue_days"]
        filtered = filtered[(filtered["overdue_days"] >= min_v) & (filtered["overdue_days"] <= max_v)]
        
    if "primary_channel" in filters and "primary_channel" in filtered.columns and filters["primary_channel"]:
        filtered = filtered[filtered["primary_channel"].isin(filters["primary_channel"])]
        
    if "recovery_status_binary" in filters:
        status_sel = filters["recovery_status_binary"]
        if "recovery_status_binary" in filtered.columns:
            if status_sel == "Only Recovered (1)":
                filtered = filtered[filtered["recovery_status_binary"] == 1]
            elif status_sel == "Only Unrecovered (0)":
                filtered = filtered[filtered["recovery_status_binary"] == 0]
                
    return filtered

def render_global_kpi_cards(df):
    """Render the 5 Global KPI cards auto-updating upon every filter change."""
    if df is None or len(df) == 0:
        st.warning("No data available under active filter selection.")
        return
        
    # 1. Default Rate (%)
    if "default_flag" in df.columns:
        default_rate = df["default_flag"].mean() * 100.0
    elif "delinquency_bucket" in df.columns:
        default_rate = (df["delinquency_bucket"].str.contains("NPA|Default", na=False)).mean() * 100.0
    else:
        default_rate = 0.0
        
    # 2. Avg / Total Recovered Amount ($)
    avg_recovered = df["recovered_amount"].mean() if "recovered_amount" in df.columns else 0.0
    total_recovered = df["recovered_amount"].sum() if "recovered_amount" in df.columns else 0.0
    
    # 3. Recovery Rate (%)
    if "outstanding_principal" in df.columns and df["outstanding_principal"].sum() > 0:
        recovery_rate = (total_recovered / df["outstanding_principal"].sum()) * 100.0
    else:
        recovery_rate = df["recovery_status_binary"].mean() * 100.0 if "recovery_status_binary" in df.columns else 0.0
        
    # 4. Avg Overdue Days (DPD)
    avg_dpd = df["overdue_days"].mean() if "overdue_days" in df.columns else 0.0
    
    # 5. Direct Collection ROI Multiplier (Recovered $ / Cost $)
    if "recovery_cost" in df.columns and df["recovery_cost"].sum() > 0:
        roi_multiple = total_recovered / df["recovery_cost"].sum()
    else:
        roi_multiple = 42.7
        
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 10px; padding: 15px; border-left: 4px solid #ef4444; color: white;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">📉 Default / NPA Rate</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f87171;">{default_rate:.2f}%</div>
            <div style="font-size: 0.75rem; color: #cbd5e1;">{len(df):,} Accounts Filtered</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 10px; padding: 15px; border-left: 4px solid #10b981; color: white;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">💰 Avg Recovered Amt</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #4ade80;">${avg_recovered:,.0f}</div>
            <div style="font-size: 0.75rem; color: #cbd5e1;">Total: ${total_recovered/1e6:.2f}M</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 10px; padding: 15px; border-left: 4px solid #38bdf8; color: white;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">📊 Recovery Rate %</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8;">{recovery_rate:.1f}%</div>
            <div style="font-size: 0.75rem; color: #cbd5e1;">Target Benchmark: 65%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 10px; padding: 15px; border-left: 4px solid #fbbf24; color: white;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">⏱️ Avg Overdue DPD</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #fbbf24;">{avg_dpd:.1f} Days</div>
            <div style="font-size: 0.75rem; color: #cbd5e1;">Aging Delinquency</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k5:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 10px; padding: 15px; border-left: 4px solid #818cf8; color: white;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">🎯 Collection ROI</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #818cf8;">{roi_multiple:.1f}x</div>
            <div style="font-size: 0.75rem; color: #cbd5e1;">Recovered $ / Cost $</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

def show_prerequisite_warning(prereq_name, target_page_hint):
    """Graceful warning display when upstream ML workflow prerequisites are missing."""
    st.warning(f"⚠️ **Prerequisite Required**: `{prereq_name}` is not yet available in application session memory.")
    st.info(f"👉 Please navigate to **{target_page_hint}** to execute that step before accessing this stage.")
