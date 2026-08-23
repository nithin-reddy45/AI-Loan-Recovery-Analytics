"""
Data Cleaning & Feature Engineering Pipeline: AI-Driven Loan Recovery & Risk Analytics
Cleans raw relational banking tables, handles missing values & outliers,
performs feature engineering, and compiles the analytical master dataset.
"""

import os
import numpy as np
import pandas as pd

def clean_and_preprocess(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    print("Loading raw banking datasets for data cleansing and preprocessing...")
    
    customers_df = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
    loans_df = pd.read_csv(os.path.join(raw_dir, "loans.csv"))
    repayments_df = pd.read_csv(os.path.join(raw_dir, "repayments.csv"))
    recovery_df = pd.read_csv(os.path.join(raw_dir, "recovery_activities.csv"))
    
    print(f"Loaded: Customers={customers_df.shape}, Loans={loans_df.shape}, Repayments={repayments_df.shape}, Recovery={recovery_df.shape}")
    
    # -------------------------------------------------------------
    # 1. CLEANING CUSTOMERS TABLE
    # -------------------------------------------------------------
    # Impute missing employment_length_years with median by employment_status
    emp_median = customers_df.groupby("employment_status")["employment_length_years"].transform("median")
    customers_df["employment_length_years"] = customers_df["employment_length_years"].fillna(emp_median).fillna(3).astype(int)
    
    # Impute missing housing_status with mode
    mode_housing = customers_df["housing_status"].mode()[0]
    customers_df["housing_status"] = customers_df["housing_status"].fillna(mode_housing)
    
    # Outlier treatment for annual_income using IQR capping (by city_tier)
    def cap_outliers(group):
        q25 = group["annual_income"].quantile(0.25)
        q75 = group["annual_income"].quantile(0.75)
        iqr = q75 - q25
        upper_bound = q75 + 3.0 * iqr # conservative threshold for income
        lower_bound = max(12000, q25 - 1.5 * iqr)
        group["annual_income"] = np.clip(group["annual_income"], lower_bound, upper_bound)
        return group

    customers_df = customers_df.groupby("city_tier", group_keys=False).apply(cap_outliers)
    
    # -------------------------------------------------------------
    # 2. CLEANING LOANS TABLE
    # -------------------------------------------------------------
    # Fill missing collateral_value with 0.0 for unsecured loans
    loans_df["collateral_value"] = loans_df["collateral_value"].fillna(0.0)
    loans_df["disbursement_date"] = pd.to_datetime(loans_df["disbursement_date"])
    loans_df["vintage_year"] = loans_df["disbursement_date"].dt.year
    loans_df["vintage_quarter"] = loans_df["disbursement_date"].dt.to_period("Q").astype(str)
    
    # -------------------------------------------------------------
    # 3. CLEANING REPAYMENTS TABLE
    # -------------------------------------------------------------
    repayments_df["last_payment_date"] = pd.to_datetime(repayments_df["last_payment_date"])
    repayments_df["default_date"] = pd.to_datetime(repayments_df["default_date"])
    
    # Calculate Outstanding Principal and Total Outstanding Balance
    loans_rep_df = pd.merge(loans_df, repayments_df, on=["loan_id", "customer_id"], how="inner")
    loans_rep_df["outstanding_principal"] = np.maximum(0.0, loans_rep_df["loan_amount"] - loans_rep_df["principal_paid"])
    
    # -------------------------------------------------------------
    # 4. MERGING WITH RECOVERY ACTIVITIES
    # -------------------------------------------------------------
    master_df = pd.merge(
        customers_df,
        loans_rep_df,
        on="customer_id",
        how="inner"
    )
    
    # Left join recovery activities
    master_df = pd.merge(
        master_df,
        recovery_df.drop(columns=["customer_id"]),
        on="loan_id",
        how="left"
    )
    
    # Fill nulls for loans that had no recovery activities (healthy/current accounts)
    master_df["recovery_status"] = master_df["recovery_status"].fillna("No Delinquency / Not Applicable")
    master_df["primary_channel"] = master_df["primary_channel"].fillna("None")
    master_df["recovery_officer_id"] = master_df["recovery_officer_id"].fillna("None")
    master_df["officer_name"] = master_df["officer_name"].fillna("None")
    master_df["officer_designation"] = master_df["officer_designation"].fillna("None")
    master_df["contact_attempts"] = master_df["contact_attempts"].fillna(0).astype(int)
    master_df["promise_to_pay_kept"] = master_df["promise_to_pay_kept"].fillna(0).astype(int)
    master_df["settlement_discount_pct"] = master_df["settlement_discount_pct"].fillna(0.0)
    master_df["recovered_amount"] = master_df["recovered_amount"].fillna(0.0)
    master_df["recovery_cost"] = master_df["recovery_cost"].fillna(0.0)
    
    # -------------------------------------------------------------
    # 5. ADVANCED FEATURE ENGINEERING
    # -------------------------------------------------------------
    # Monthly EMI
    r = (master_df["interest_rate"] / 100.0) / 12.0
    n = master_df["loan_term_months"]
    master_df["monthly_emi"] = master_df["loan_amount"] * (r * (1 + r)**n) / ((1 + r)**n - 1)
    master_df["monthly_emi"] = master_df["monthly_emi"].round(2)
    
    # Debt-to-Income (DTI) Ratio
    master_df["dti_ratio"] = ((master_df["monthly_emi"] * 12) / master_df["annual_income"]).round(4)
    
    # Loan-to-Value (LTV) Ratio (where collateral exists)
    master_df["is_secured"] = np.where(master_df["collateral_type"] != "None", 1, 0)
    master_df["ltv_ratio"] = np.where(
        master_df["is_secured"] == 1,
        np.clip(master_df["loan_amount"] / np.maximum(master_df["collateral_value"], 1.0), 0.1, 1.5),
        1.0
    ).round(4)
    
    # Credit Risk Tiers
    bins = [0, 549, 649, 699, 749, 900]
    labels = ["Deep Subprime (<550)", "Subprime (550-649)", "Near-Prime (650-699)", "Prime (700-749)", "Super-Prime (750+)"]
    master_df["credit_risk_tier"] = pd.cut(master_df["credit_score"], bins=bins, labels=labels)
    
    # Delinquency Severity Score (0-100)
    master_df["delinquency_severity_score"] = np.clip(
        (master_df["overdue_days"] / 360.0) * 60 + (1 - master_df["credit_score"] / 850.0) * 40,
        0, 100
    ).round(2)
    
    # Net Recovery Amount (Recovered - Direct Recovery Cost)
    master_df["net_recovery_amount"] = (master_df["recovered_amount"] - master_df["recovery_cost"]).round(2)
    
    # Recovery Rate Percentage on Outstanding Principal
    master_df["recovery_rate_pct"] = np.where(
        master_df["outstanding_principal"] > 0,
        np.clip((master_df["recovered_amount"] / master_df["outstanding_principal"]) * 100.0, 0.0, 100.0),
        0.0
    ).round(2)
    
    # Recovery Target Flag for Machine Learning
    # 1 = Successfully Recovered (Fully or Partially Recovered), 0 = Failed Recovery / Written Off / Severe Legal / Unrecovered
    master_df["recovery_target_binary"] = master_df["recovery_status"].apply(
        lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else (0 if s in ["Written Off", "Legal Escalation", "In Progress"] else np.nan)
    )
    
    # Export Clean Processed Datasets
    customers_df.to_csv(os.path.join(processed_dir, "clean_customers.csv"), index=False)
    loans_df.to_csv(os.path.join(processed_dir, "clean_loans.csv"), index=False)
    repayments_df.to_csv(os.path.join(processed_dir, "clean_repayments.csv"), index=False)
    recovery_df.to_csv(os.path.join(processed_dir, "clean_recovery.csv"), index=False)
    
    master_df.to_csv(os.path.join(processed_dir, "loan_recovery_master.csv"), index=False)
    
    # Export Delinquent Master for Collections & ML modeling
    delinquent_df = master_df[master_df["overdue_days"] > 30].copy()
    delinquent_df.to_csv(os.path.join(processed_dir, "delinquent_recovery_master.csv"), index=False)
    
    print(f"Data cleaning & preprocessing complete!")
    print(f"Master Analytical Dataset: {master_df.shape} saved to {os.path.join(processed_dir, 'loan_recovery_master.csv')}")
    print(f"Delinquent & Recovery Master: {delinquent_df.shape} saved to {os.path.join(processed_dir, 'delinquent_recovery_master.csv')}")
    return master_df

if __name__ == "__main__":
    clean_and_preprocess()
