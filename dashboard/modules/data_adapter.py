"""
Data Adapter & Schema Standardizer: AI-Driven Loan Recovery & Risk Analytics
Intelligently ingests, validates, standardizes, and enriches custom uploaded datasets
so that all 9 weekly ML modules work seamlessly with any tabular data format.
"""

import numpy as np
import pandas as pd

# Canonical column alias mappings
COLUMN_ALIASES = {
    "loan_amount": [
        "loan_amount", "loan_amt", "principal_amount", "disbursed_amount",
        "outstanding_balance", "loan_balance", "balance", "amount", "principal"
    ],
    "annual_income": [
        "annual_income", "income", "yearly_income", "salary",
        "gross_income", "applicant_income", "monthly_income"
    ],
    "credit_score": [
        "credit_score", "cibil_score", "fico_score", "bureau_score",
        "credit_rating", "score", "risk_score"
    ],
    "interest_rate": [
        "interest_rate", "int_rate", "rate", "interest", "roi", "interest_rate_pct"
    ],
    "loan_term_months": [
        "loan_term_months", "loan_term", "term_months", "term",
        "tenure_months", "tenure", "loan_duration_months"
    ],
    "overdue_days": [
        "overdue_days", "dpd", "days_past_due", "delinquency_days",
        "delay_days", "days_overdue", "overdue_duration"
    ],
    "loan_type": [
        "loan_type", "product_type", "loan_category", "product", "loan_name", "type"
    ],
    "region": [
        "region", "zone", "state", "location", "city", "territory", "geography"
    ],
    "primary_channel": [
        "primary_channel", "recovery_channel", "channel", "collection_channel",
        "collection_method", "recovery_method", "contact_channel"
    ],
    "recovery_cost": [
        "recovery_cost", "collection_cost", "cost", "expense",
        "direct_cost", "operating_cost"
    ],
    "recovered_amount": [
        "recovered_amount", "amount_recovered", "recovery_amount",
        "total_recovered", "recovered", "collection_amount", "net_recovered"
    ],
    "contact_attempts": [
        "contact_attempts", "attempts", "calls_made", "followups",
        "contact_count", "recovery_agent_experience"
    ],
    "settlement_discount_pct": [
        "settlement_discount_pct", "settlement_offered", "settlement_discount",
        "discount_pct", "haircut_pct", "discount", "settlement_pct"
    ],
    "default_flag": [
        "default_flag", "is_default", "defaulted", "npa_flag", "default", "bad_loan"
    ]
}

def standardize_dataset(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and derives essential banking metrics
    so that downstream EDA, cleaning, feature engineering, and ML models
    execute smoothly without column mismatch errors.
    """
    if df_input is None or len(df_input) == 0:
        return df_input

    df = df_input.copy()
    col_map = {}
    lower_cols = {col.strip().lower(): col for col in df.columns}

    # Match aliases to canonical names
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower_cols:
                original_name = lower_cols[alias.lower()]
                if canonical not in col_map.values():
                    col_map[original_name] = canonical
                    break

    if col_map:
        df = df.rename(columns=col_map)

    # 1. Standardize / Derive Numerical Attributes
    if "loan_amount" in df.columns:
        df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce").fillna(25000.0)
    else:
        df["loan_amount"] = 25000.0

    if "annual_income" in df.columns:
        df["annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce")
        # If annual income looks like monthly income (< 10000), scale up
        if df["annual_income"].median() < 12000 and df["annual_income"].median() > 500:
            df["annual_income"] = df["annual_income"] * 12.0
        df["annual_income"] = df["annual_income"].fillna(60000.0)
    else:
        df["annual_income"] = 60000.0

    if "credit_score" in df.columns:
        df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce").fillna(675.0)
    else:
        # Synthesize realistic credit score based on overdue_days / defaults
        if "overdue_days" in df.columns:
            df["credit_score"] = np.clip(720 - pd.to_numeric(df["overdue_days"], errors="coerce").fillna(0) * 1.2, 350, 850).round(0)
        else:
            df["credit_score"] = 680.0

    if "interest_rate" in df.columns:
        df["interest_rate"] = pd.to_numeric(df["interest_rate"], errors="coerce").fillna(12.5)
    else:
        df["interest_rate"] = 12.5

    if "loan_term_months" in df.columns:
        df["loan_term_months"] = pd.to_numeric(df["loan_term_months"], errors="coerce").fillna(36).astype(int)
    else:
        df["loan_term_months"] = 36

    if "overdue_days" in df.columns:
        df["overdue_days"] = pd.to_numeric(df["overdue_days"], errors="coerce").fillna(0).astype(int)
    else:
        df["overdue_days"] = 0

    if "recovery_cost" in df.columns:
        df["recovery_cost"] = pd.to_numeric(df["recovery_cost"], errors="coerce").fillna(0.0)
    else:
        df["recovery_cost"] = 0.0

    # 2. Derive Targets: recovery_status_binary and recovered_amount
    if "recovery_status_binary" not in df.columns:
        if "recovery_target_binary" in df.columns:
            df["recovery_status_binary"] = pd.to_numeric(df["recovery_target_binary"], errors="coerce").fillna(0).astype(int)
        elif "recovery_target" in df.columns:
            df["recovery_status_binary"] = pd.to_numeric(df["recovery_target"], errors="coerce").fillna(0).astype(int)
        elif "recovery_status" in df.columns:
            df["recovery_status_binary"] = df["recovery_status"].astype(str).str.lower().apply(
                lambda s: 1 if any(x in s for x in ["fully", "partially", "recovered", "resolved", "paid", "closed", "settled"]) else 0
            )
        elif "repayment_history" in df.columns:
            df["recovery_status_binary"] = df["repayment_history"].astype(str).str.lower().apply(
                lambda s: 1 if any(x in s for x in ["good", "paid", "on-time", "standard", "resolved"]) else 0
            )
        elif "customer_response" in df.columns:
            df["recovery_status_binary"] = df["customer_response"].astype(str).str.lower().apply(
                lambda s: 1 if any(x in s for x in ["paid", "settled", "promised", "cooperative"]) else 0
            )
        elif "default_flag" in df.columns:
            df["recovery_status_binary"] = (1 - pd.to_numeric(df["default_flag"], errors="coerce").fillna(0)).astype(int)
        elif "loyal_customer" in df.columns:
            df["recovery_status_binary"] = pd.to_numeric(df["loyal_customer"], errors="coerce").fillna(1).astype(int)
        else:
            # Derive based on overdue days and credit score
            p_rec = 1.0 / (1.0 + np.exp(-((df["credit_score"] - 620) / 60.0 - (df["overdue_days"] - 30) / 45.0)))
            df["recovery_status_binary"] = (p_rec >= 0.50).astype(int)

    # 3. Recovered Amount
    if "recovered_amount" in df.columns:
        df["recovered_amount"] = pd.to_numeric(df["recovered_amount"], errors="coerce").fillna(0.0)
    else:
        # If user dataset has monthly_spending, use that as continuous target
        if "monthly_spending" in df.columns:
            df["recovered_amount"] = pd.to_numeric(df["monthly_spending"], errors="coerce").fillna(0.0)
        else:
            # Estimate recovered amount from loan amount and recovery status
            rec_pct = np.where(df["recovery_status_binary"] == 1, np.random.uniform(0.60, 0.95, size=len(df)), 0.0)
            df["recovered_amount"] = (df["loan_amount"] * rec_pct).round(2)

    # 4. Default Flag
    if "default_flag" not in df.columns:
        if "delinquency_bucket" in df.columns:
            df["default_flag"] = df["delinquency_bucket"].astype(str).str.contains("NPA|Default|90\\+", case=False, na=False).astype(int)
        else:
            df["default_flag"] = (df["overdue_days"] > 90).astype(int)
    else:
        df["default_flag"] = pd.to_numeric(df["default_flag"], errors="coerce").fillna(0).astype(int)

    # 5. Outstanding principal
    if "outstanding_principal" not in df.columns:
        df["outstanding_principal"] = df["loan_amount"]

    # 6. Ensure Categoricals Have Sensible Defaults if Present
    if "loan_type" not in df.columns:
        df["loan_type"] = "Personal Loan"
    else:
        df["loan_type"] = df["loan_type"].fillna("Personal Loan").astype(str)

    if "region" not in df.columns:
        df["region"] = "General"
    else:
        df["region"] = df["region"].fillna("General").astype(str)

    if "primary_channel" not in df.columns:
        df["primary_channel"] = "Tele-Calling & Counseling"
    else:
        df["primary_channel"] = df["primary_channel"].fillna("Tele-Calling & Counseling").astype(str)

    return df
