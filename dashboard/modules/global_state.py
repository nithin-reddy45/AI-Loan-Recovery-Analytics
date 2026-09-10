"""
Global State & Filter Manager: AI-Driven Loan Recovery & Risk Analytics
Ultra-fast, responsive session state and lightweight global filter synchronization.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np

try:
    from dashboard.modules.data_adapter import standardize_dataset
except ModuleNotFoundError:
    from modules.data_adapter import standardize_dataset

@st.cache_data(show_spinner=False)
def load_default_data():
    """Fast cached loader for default banking dataset."""
    path = "data/processed/loan_recovery_master.csv"
    if not os.path.exists(path):
        path = "data/raw/loans.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def init_session_state():
    """Initialize all session state keys."""
    if "dataset_id" not in st.session_state:
        st.session_state["dataset_id"] = 1

    if "df_raw" not in st.session_state or st.session_state["df_raw"] is None:
        raw_df = load_default_data()
        if raw_df is not None:
            st.session_state["df_raw"] = standardize_dataset(raw_df)
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

def set_active_dataset(df_new: pd.DataFrame):
    """Updates active dataset and resets downstream states."""
    if df_new is None or len(df_new) == 0:
        return
        
    std_df = standardize_dataset(df_new)
    st.session_state["df_raw"] = std_df
    st.session_state["df_clean"] = None
    st.session_state["df_features"] = None
    st.session_state["model_baseline"] = {"classification": None, "regression": None, "metrics": {}}
    st.session_state["model_tuned"] = {"classification": None, "regression": None, "metrics": {}}
    st.session_state["best_params"] = {}
    st.session_state["metrics_tuned"] = {}
    if "pipeline_results" in st.session_state:
        del st.session_state["pipeline_results"]
        
    st.session_state["global_filters"] = {}
    st.session_state["dataset_id"] = int(time.time())

def render_global_filters():
    """Simple, fast sidebar filters that do not block UI performance."""
    st.sidebar.markdown("### 🎛️ Quick Filters")
    
    df = st.session_state.get("df_raw", None)
    if df is None or len(df) == 0:
        st.sidebar.info("Upload data in **Week 1** to activate filters.")
        return {}
        
    ds_id = st.session_state.get("dataset_id", 1)
    filters = {}
    
    # Simple form to avoid rerunning on every single slider tick
    with st.sidebar.form(f"filter_form_{ds_id}"):
        # 1. Loan Product Type
        if "loan_type" in df.columns and df["loan_type"].nunique() > 1:
            all_types = sorted([str(x) for x in df["loan_type"].dropna().unique()])
            sel_types = st.multiselect("Product Type", all_types, default=all_types)
            filters["loan_type"] = sel_types
            
        # 2. Overdue DPD Threshold
        if "overdue_days" in df.columns:
            max_dpd = int(df["overdue_days"].max())
            if max_dpd > 0:
                sel_dpd = st.slider("Max Overdue Days (DPD)", 0, max_dpd, max_dpd)
                filters["max_dpd"] = sel_dpd
                
        # 3. Target Filter
        if "recovery_status_binary" in df.columns:
            sel_target = st.selectbox("Target Filter", ["All Records", "Recovered Only (1)", "Unrecovered Only (0)"])
            filters["target"] = sel_target
            
        apply_btn = st.form_submit_button("🔍 Apply Filters", use_container_width=True)
        if apply_btn:
            st.session_state["global_filters"] = filters
            st.rerun()

    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state["global_filters"] = {}
        st.rerun()

    return st.session_state.get("global_filters", {})

def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Fast filter application."""
    if df is None or len(df) == 0:
        return df
        
    filters = st.session_state.get("global_filters", {})
    if not filters:
        return df
        
    filtered = df
    if "loan_type" in filters and "loan_type" in filtered.columns and filters["loan_type"]:
        filtered = filtered[filtered["loan_type"].astype(str).isin([str(x) for x in filters["loan_type"]])]
        
    if "max_dpd" in filters and "overdue_days" in filtered.columns:
        filtered = filtered[filtered["overdue_days"] <= filters["max_dpd"]]
        
    if "target" in filters and "recovery_status_binary" in filtered.columns:
        if filters["target"] == "Recovered Only (1)":
            filtered = filtered[filtered["recovery_status_binary"] == 1]
        elif filters["target"] == "Unrecovered Only (0)":
            filtered = filtered[filtered["recovery_status_binary"] == 0]
            
    if len(filtered) == 0:
        return df
        
    return filtered

def render_global_kpi_cards(df: pd.DataFrame):
    """Fast, clean 5-card metric row."""
    if df is None or len(df) == 0:
        return
        
    default_rate = float(df["default_flag"].mean() * 100.0) if "default_flag" in df.columns else 0.0
    avg_rec = float(df["recovered_amount"].mean()) if "recovered_amount" in df.columns else 0.0
    tot_rec = float(df["recovered_amount"].sum()) if "recovered_amount" in df.columns else 0.0
    avg_dpd = float(df["overdue_days"].mean()) if "overdue_days" in df.columns else 0.0
    rec_rate = float(df["recovery_status_binary"].mean() * 100.0) if "recovery_status_binary" in df.columns else 65.0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📉 Default / Risk Rate", f"{default_rate:.1f}%", f"{len(df):,} Records")
    c2.metric("💰 Avg Recovered", f"${avg_rec:,.0f}", f"Total: ${tot_rec/1e6:.2f}M")
    c3.metric("📊 Recovery Rate", f"{rec_rate:.1f}%", "Target: 65%")
    c4.metric("⏱️ Avg Overdue DPD", f"{avg_dpd:.0f} Days", "Aging")
    c5.metric("🎯 Portfolio Yield", f"${tot_rec/max(1, len(df)):,.0f}/acct", "Performance")
    st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)

def show_prerequisite_warning(prereq_name: str, target_page_hint: str):
    """Simple warning for missing stage."""
    st.warning(f"⚠️ **Prerequisite Required**: `{prereq_name}` is not yet computed.")
    st.info(f"👉 Please navigate to **{target_page_hint}** and click run.")
