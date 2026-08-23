"""
Data Generation Script: AI-Driven Loan Recovery & Risk Analytics
Generates realistic multi-table relational financial datasets with realistic banking distributions,
correlations, delinquency patterns, and controlled real-world data quality anomalies.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def generate_datasets(n_customers=15000, output_dir="data/raw"):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic enterprise banking dataset with {n_customers} customer records...")

    # -------------------------------------------------------------
    # 1. CUSTOMERS DATASET
    # -------------------------------------------------------------
    customer_ids = [f"CUST-{100000 + i}" for i in range(n_customers)]
    
    # Age: 21 to 68, skewed towards 28-45
    ages = np.clip(np.random.normal(loc=37, scale=10, size=n_customers).astype(int), 21, 68)
    
    genders = np.random.choice(["Male", "Female", "Other"], size=n_customers, p=[0.58, 0.40, 0.02])
    
    regions = np.random.choice(
        ["North", "South", "East", "West", "Central", "North-East"], 
        size=n_customers, 
        p=[0.28, 0.26, 0.16, 0.20, 0.07, 0.03]
    )
    
    city_tiers = np.random.choice(["Tier 1", "Tier 2", "Tier 3", "Rural"], size=n_customers, p=[0.42, 0.35, 0.18, 0.05])
    
    employment_statuses = np.random.choice(
        ["Salaried - Corporate", "Salaried - Public Sector", "Self-Employed Professional", "Self-Employed Business", "Contractual", "Unemployed"],
        size=n_customers,
        p=[0.45, 0.18, 0.14, 0.15, 0.06, 0.02]
    )
    
    employment_lengths = []
    for emp, age in zip(employment_statuses, ages):
        if emp == "Unemployed":
            employment_lengths.append(0)
        else:
            max_exp = max(1, age - 21)
            exp = min(max_exp, int(np.random.exponential(scale=6)))
            employment_lengths.append(exp)
            
    # Annual Income (in USD): Log-normal distribution depending on employment & city tier
    base_incomes = []
    for emp, tier in zip(employment_statuses, city_tiers):
        multiplier = 1.3 if tier == "Tier 1" else (1.0 if tier == "Tier 2" else 0.75)
        if "Corporate" in emp:
            inc = np.random.lognormal(mean=10.9, sigma=0.5) * multiplier
        elif "Public" in emp:
            inc = np.random.lognormal(mean=10.8, sigma=0.4) * multiplier
        elif "Professional" in emp:
            inc = np.random.lognormal(mean=11.1, sigma=0.6) * multiplier
        elif "Business" in emp:
            inc = np.random.lognormal(mean=11.2, sigma=0.7) * multiplier
        elif "Contractual" in emp:
            inc = np.random.lognormal(mean=10.2, sigma=0.45) * multiplier
        else:
            inc = np.random.lognormal(mean=9.5, sigma=0.4) * multiplier
        base_incomes.append(round(inc, 2))
    
    # Credit Score (300 - 850)
    credit_scores = []
    for inc, emp, exp in zip(base_incomes, employment_statuses, employment_lengths):
        score_base = 650 + (np.log1p(inc) - 10.5) * 45 + (exp * 3)
        if emp == "Unemployed":
            score_base -= 90
        elif emp == "Contractual":
            score_base -= 40
        noise = np.random.normal(0, 45)
        final_score = int(np.clip(score_base + noise, 300, 850))
        credit_scores.append(final_score)
        
    housing_statuses = np.random.choice(["Owned", "Mortgaged", "Rented", "Family Owned"], size=n_customers, p=[0.32, 0.28, 0.30, 0.10])
    marital_statuses = np.random.choice(["Married", "Single", "Divorced"], size=n_customers, p=[0.64, 0.29, 0.07])
    existing_debts = np.random.poisson(lam=1.8, size=n_customers)
    
    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "region": regions,
        "city_tier": city_tiers,
        "employment_status": employment_statuses,
        "employment_length_years": employment_lengths,
        "annual_income": base_incomes,
        "credit_score": credit_scores,
        "housing_status": housing_statuses,
        "marital_status": marital_statuses,
        "existing_debts_count": existing_debts
    })
    
    # Introduce deliberate real-world dirty data points into customers_df
    # 1. Missing values in employment_length_years and housing_status
    mask_nan_emp = np.random.rand(n_customers) < 0.035
    customers_df.loc[mask_nan_emp, "employment_length_years"] = np.nan
    mask_nan_house = np.random.rand(n_customers) < 0.02
    customers_df.loc[mask_nan_house, "housing_status"] = np.nan
    
    # 2. Outliers in annual income
    outlier_idx = np.random.choice(n_customers, size=15, replace=False)
    customers_df.loc[outlier_idx, "annual_income"] = customers_df.loc[outlier_idx, "annual_income"] * 10
    
    # -------------------------------------------------------------
    # 2. LOANS DATASET
    # -------------------------------------------------------------
    loan_ids = [f"LN-{200000 + i}" for i in range(n_customers)]
    loan_types = np.random.choice(
        ["Personal Loan", "Auto Loan", "Home Loan", "Business Loan", "Education Loan"],
        size=n_customers,
        p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )
    
    loan_amounts = []
    interest_rates = []
    loan_terms = []
    collateral_types = []
    collateral_values = []
    purposes = []
    
    start_date = datetime(2021, 1, 1)
    disbursal_dates = [start_date + timedelta(days=int(np.random.uniform(0, 1095))) for _ in range(n_customers)]
    
    for l_type, score, inc in zip(loan_types, credit_scores, base_incomes):
        if l_type == "Home Loan":
            amt = min(inc * random.uniform(3.0, 5.5), random.uniform(150000, 600000))
            base_rate = random.uniform(6.5, 9.2) - (score - 650) * 0.008
            term = random.choice([180, 240, 300, 360])
            col_type = "Residential Property"
            col_val = amt * random.uniform(1.2, 1.6)
            purpose = random.choice(["Home Purchase", "Home Construction", "Refinancing"])
        elif l_type == "Auto Loan":
            amt = min(inc * random.uniform(0.4, 0.9), random.uniform(15000, 75000))
            base_rate = random.uniform(8.0, 12.5) - (score - 650) * 0.010
            term = random.choice([36, 48, 60, 72, 84])
            col_type = "Vehicle"
            col_val = amt * random.uniform(1.05, 1.3)
            purpose = random.choice(["New Car", "Used Car", "Commercial Vehicle"])
        elif l_type == "Business Loan":
            amt = min(inc * random.uniform(1.2, 2.8), random.uniform(30000, 250000))
            base_rate = random.uniform(11.0, 16.5) - (score - 650) * 0.012
            term = random.choice([24, 36, 48, 60])
            col_type = random.choice(["Commercial Property", "Equipment", "Inventory", "None"])
            col_val = amt * random.uniform(1.1, 1.5) if col_type != "None" else 0.0
            purpose = random.choice(["Working Capital", "Business Expansion", "Equipment Purchase"])
        elif l_type == "Education Loan":
            amt = min(inc * random.uniform(0.5, 1.5), random.uniform(10000, 80000))
            base_rate = random.uniform(8.5, 11.5) - (score - 650) * 0.007
            term = random.choice([48, 60, 84, 120])
            col_type = random.choice(["Third Party Guarantee", "None"])
            col_val = amt * 1.0 if col_type != "None" else 0.0
            purpose = random.choice(["Higher Studies Abroad", "Domestic Degree", "Professional Certification"])
        else: # Personal Loan (Unsecured)
            amt = min(inc * random.uniform(0.15, 0.6), random.uniform(3000, 40000))
            base_rate = random.uniform(13.0, 22.0) - (score - 650) * 0.015
            term = random.choice([12, 24, 36, 48, 60])
            col_type = "None"
            col_val = 0.0
            purpose = random.choice(["Medical Emergency", "Debt Consolidation", "Wedding/Travel", "Home Improvement"])
            
        rate = round(float(np.clip(base_rate, 5.5, 26.0)), 2)
        loan_amounts.append(round(amt, 2))
        interest_rates.append(rate)
        loan_terms.append(term)
        collateral_types.append(col_type)
        collateral_values.append(round(col_val, 2))
        purposes.append(purpose)
        
    origination_channels = np.random.choice(["Digital Portal", "Branch Network", "DSA / Direct Selling Agent", "Mobile App", "Tele-Sales"], size=n_customers, p=[0.38, 0.30, 0.16, 0.12, 0.04])
    
    loans_df = pd.DataFrame({
        "loan_id": loan_ids,
        "customer_id": customer_ids,
        "loan_type": loan_types,
        "loan_amount": loan_amounts,
        "interest_rate": interest_rates,
        "loan_term_months": loan_terms,
        "disbursement_date": [d.strftime("%Y-%m-%d") for d in disbursal_dates],
        "collateral_type": collateral_types,
        "collateral_value": collateral_values,
        "loan_purpose": purposes,
        "origination_channel": origination_channels
    })
    
    # Introduce deliberate missing values
    mask_nan_colval = (loans_df["collateral_type"] == "None") & (np.random.rand(n_customers) < 0.15)
    loans_df.loc[mask_nan_colval, "collateral_value"] = np.nan
    
    # -------------------------------------------------------------
    # 3. REPAYMENTS & DELINQUENCY DATASET
    # -------------------------------------------------------------
    repayment_ids = [f"REP-{300000 + i}" for i in range(n_customers)]
    
    total_inst_due = []
    inst_paid = []
    amt_paid = []
    principal_paid_list = []
    interest_paid_list = []
    overdue_days_list = []
    delinquency_buckets = []
    default_flags = []
    default_dates = []
    last_pay_dates = []
    
    study_end_date = datetime(2024, 12, 31)
    
    for i in range(n_customers):
        disb_dt = disbursal_dates[i]
        months_active = max(1, min(loans_df.loc[i, "loan_term_months"], (study_end_date.year - disb_dt.year) * 12 + (study_end_date.month - disb_dt.month)))
        
        amt = loans_df.loc[i, "loan_amount"]
        rate = loans_df.loc[i, "interest_rate"]
        term = loans_df.loc[i, "loan_term_months"]
        score = customers_df.loc[i, "credit_score"]
        inc = customers_df.loc[i, "annual_income"] if pd.notnull(customers_df.loc[i, "annual_income"]) else 50000
        l_type = loans_df.loc[i, "loan_type"]
        
        # Monthly EMI calculation: P * r * (1+r)^n / ((1+r)^n - 1)
        r = (rate / 100.0) / 12.0
        n = term
        emi = amt * (r * (1 + r)**n) / ((1 + r)**n - 1)
        
        # Calculate Debt-to-Income (DTI)
        dti = (emi * 12) / max(inc, 10000)
        
        # Realistic credit risk default / delinquency logic
        risk_score_norm = (650 - score) / 100.0
        dti_impact = (dti - 0.35) * 2.0
        rate_impact = (rate - 11.0) * 0.08
        
        type_adj = 0.3 if l_type == "Personal Loan" else (-0.4 if l_type == "Home Loan" else 0.1)
        risk_index = risk_score_norm + dti_impact + rate_impact + type_adj + np.random.normal(0, 0.6)
        
        total_due_count = months_active
        total_inst_due.append(total_due_count)
        
        if risk_index < -0.3:
            # Fully current / healthy
            overdue_dpd = 0
            paid_count = int(total_due_count)
            is_def = 0
            def_dt = np.nan
        elif risk_index < 0.4:
            # SMA-0 (1-30 DPD)
            overdue_dpd = int(np.random.randint(1, 31))
            paid_count = max(0, int(total_due_count - 1))
            is_def = 0
            def_dt = np.nan
        elif risk_index < 0.9:
            # SMA-1 (31-60 DPD)
            overdue_dpd = int(np.random.randint(31, 61))
            paid_count = max(0, int(total_due_count - 2))
            is_def = 0
            def_dt = np.nan
        elif risk_index < 1.4:
            # SMA-2 (61-90 DPD)
            overdue_dpd = int(np.random.randint(61, 91))
            paid_count = max(0, int(total_due_count - 3))
            is_def = 0
            def_dt = np.nan
        else:
            # NPA / Default (91+ DPD)
            overdue_dpd = int(np.random.randint(91, 540))
            paid_count = max(0, int(total_due_count * random.uniform(0.1, 0.65)))
            is_def = 1
            def_dt = (disb_dt + timedelta(days=int(paid_count * 30 + 90))).strftime("%Y-%m-%d")
            
        inst_paid.append(paid_count)
        paid_val = round(float(paid_count * emi), 2)
        amt_paid.append(paid_val)
        
        p_paid = round(float(paid_val * random.uniform(0.68, 0.82)), 2)
        i_paid = round(float(paid_val - p_paid), 2)
        principal_paid_list.append(p_paid)
        interest_paid_list.append(i_paid)
        
        overdue_days_list.append(overdue_dpd)
        default_flags.append(is_def)
        default_dates.append(def_dt)
        
        if overdue_dpd == 0:
            bucket = "Current"
        elif overdue_dpd <= 30:
            bucket = "1-30 DPD (SMA-0)"
        elif overdue_dpd <= 60:
            bucket = "31-60 DPD (SMA-1)"
        elif overdue_dpd <= 90:
            bucket = "61-90 DPD (SMA-2)"
        else:
            bucket = "90+ DPD (NPA / Default)"
        delinquency_buckets.append(bucket)
        
        last_dt = disb_dt + timedelta(days=int(paid_count * 30))
        last_pay_dates.append(last_dt.strftime("%Y-%m-%d"))
        
    repayments_df = pd.DataFrame({
        "repayment_id": repayment_ids,
        "loan_id": loan_ids,
        "customer_id": customer_ids,
        "total_installments_due": total_inst_due,
        "installments_paid": inst_paid,
        "total_amount_paid": amt_paid,
        "principal_paid": principal_paid_list,
        "interest_paid": interest_paid_list,
        "last_payment_date": last_pay_dates,
        "overdue_days": overdue_days_list,
        "delinquency_bucket": delinquency_buckets,
        "default_flag": default_flags,
        "default_date": default_dates
    })
    
    # -------------------------------------------------------------
    # 4. RECOVERY ACTIVITIES DATASET (Delinquent & Default Accounts)
    # -------------------------------------------------------------
    recovery_mask = (repayments_df["overdue_days"] > 30)
    recovery_indices = repayments_df[recovery_mask].index.tolist()
    
    officers = [
        ("OFF-101", "Marcus Vance", "Senior Recovery Manager"),
        ("OFF-102", "Elena Rostova", "Special Asset Resolution Lead"),
        ("OFF-103", "Aarav Sharma", "Field Collections Specialist"),
        ("OFF-104", "Sarah Jenkins", "Legal & Settlements Officer"),
        ("OFF-105", "Carlos Mendez", "Digital & Tele-Recovery Lead"),
        ("OFF-106", "Priya Patel", "Early Stage Collections Specialist"),
        ("OFF-107", "David Kim", "Corporate Debt Recovery Executive"),
        ("OFF-108", "Zainab Al-Mansoor", "Restructuring & Workout Lead")
    ]
    
    rec_records = []
    rec_counter = 1
    
    for idx in recovery_indices:
        l_id = loans_df.loc[idx, "loan_id"]
        c_id = customers_df.loc[idx, "customer_id"]
        l_amt = loans_df.loc[idx, "loan_amount"]
        amt_p = repayments_df.loc[idx, "total_amount_paid"]
        dpd = repayments_df.loc[idx, "overdue_days"]
        c_score = customers_df.loc[idx, "credit_score"]
        col_type = loans_df.loc[idx, "collateral_type"]
        l_type = loans_df.loc[idx, "loan_type"]
        
        outstanding_principal = max(0.0, l_amt - repayments_df.loc[idx, "principal_paid"])
        
        if dpd <= 60:
            assigned_channel = np.random.choice(["AI Voice Bot & SMS", "Tele-Calling & Counseling"], p=[0.55, 0.45])
            officer = random.choice([officers[4], officers[5]])
        elif dpd <= 120:
            assigned_channel = np.random.choice(["Tele-Calling & Counseling", "Field Visit & Face-to-Face", "Debt Restructuring"], p=[0.40, 0.40, 0.20])
            officer = random.choice([officers[0], officers[2], officers[7]])
        else:
            assigned_channel = np.random.choice(["Field Visit & Face-to-Face", "Legal Notice & DRT", "External Recovery Agency"], p=[0.30, 0.40, 0.30])
            officer = random.choice([officers[1], officers[3], officers[6]])
            
        attempts = int(np.clip(dpd / 20 + np.random.poisson(3), 1, 28))
        ptp_kept = np.random.choice([1, 0], p=[0.38, 0.62])
        
        recovery_propensity = 0.45 + (c_score - 550) * 0.001 - (dpd - 60) * 0.0008
        if col_type != "None":
            recovery_propensity += 0.25
        if assigned_channel == "Legal Notice & DRT" and col_type != "None":
            recovery_propensity += 0.15
        if ptp_kept == 1:
            recovery_propensity += 0.30
            
        recovery_propensity = np.clip(recovery_propensity, 0.05, 0.95)
        is_recovered = np.random.rand() < recovery_propensity
        
        if is_recovered:
            if random.random() < 0.65:
                status = "Fully Recovered"
                settlement_discount = round(random.uniform(0.0, 15.0), 2)
                recovered_amt = round(outstanding_principal * (1 - settlement_discount / 100.0), 2)
            else:
                status = "Partially Recovered"
                settlement_discount = round(random.uniform(15.0, 45.0), 2)
                recovered_amt = round(outstanding_principal * random.uniform(0.35, 0.75), 2)
            rec_date = (study_end_date - timedelta(days=int(random.uniform(5, 180)))).strftime("%Y-%m-%d")
        else:
            if dpd > 360:
                status = "Written Off"
            elif assigned_channel == "Legal Notice & DRT":
                status = "Legal Escalation"
            else:
                status = "In Progress"
            settlement_discount = 0.0
            recovered_amt = 0.0
            rec_date = np.nan
            
        cost_base = {
            "AI Voice Bot & SMS": 15,
            "Tele-Calling & Counseling": 65,
            "Debt Restructuring": 180,
            "Field Visit & Face-to-Face": 320,
            "Legal Notice & DRT": 750,
            "External Recovery Agency": round(recovered_amt * 0.12 + 100, 2)
        }
        rec_cost = round(cost_base[assigned_channel] + attempts * 8.5, 2)
        
        rec_records.append({
            "recovery_id": f"REC-{400000 + rec_counter}",
            "loan_id": l_id,
            "customer_id": c_id,
            "recovery_status": status,
            "primary_channel": assigned_channel,
            "recovery_officer_id": officer[0],
            "officer_name": officer[1],
            "officer_designation": officer[2],
            "contact_attempts": attempts,
            "promise_to_pay_kept": ptp_kept,
            "settlement_discount_pct": settlement_discount,
            "recovered_amount": recovered_amt,
            "recovery_cost": rec_cost,
            "recovery_date": rec_date
        })
        rec_counter += 1
        
    recovery_df = pd.DataFrame(rec_records)
    
    # Save raw datasets to CSV
    customers_df.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    loans_df.to_csv(os.path.join(output_dir, "loans.csv"), index=False)
    repayments_df.to_csv(os.path.join(output_dir, "repayments.csv"), index=False)
    recovery_df.to_csv(os.path.join(output_dir, "recovery_activities.csv"), index=False)
    
    print(f"Raw datasets successfully saved to {output_dir}:")
    print(f" - customers.csv: {customers_df.shape}")
    print(f" - loans.csv: {loans_df.shape}")
    print(f" - repayments.csv: {repayments_df.shape}")
    print(f" - recovery_activities.csv: {recovery_df.shape}")

if __name__ == "__main__":
    generate_datasets()
