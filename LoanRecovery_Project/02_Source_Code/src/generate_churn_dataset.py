"""
Dataset Generator: Customer Churn Prediction & Monthly Spending Analysis
Generates a realistic 10,000-row customer dataset matching the exact 12-column minimum schema.
"""

import os
import random
import numpy as np
import pandas as pd

def generate_churn_dataset(n_samples=10000, output_path="data/raw/customer_churn_spending_dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating realistic dataset with {n_samples} customer profiles...")
    
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    
    # Age: 18 - 75, normal distribution centered around 38
    ages = np.clip(np.random.normal(loc=38, scale=12, size=n_samples).astype(int), 18, 75)
    
    # Gender
    genders = np.random.choice(["Male", "Female", "Other"], size=n_samples, p=[0.49, 0.49, 0.02])
    
    # City (Top 10 Metro Hubs)
    cities = np.random.choice(
        ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "San Francisco", "Miami", "Seattle", "Boston", "Austin"],
        size=n_samples,
        p=[0.20, 0.16, 0.12, 0.10, 0.08, 0.10, 0.07, 0.06, 0.06, 0.05]
    )
    
    # Annual Income ($)
    city_income_mult = {
        "San Francisco": 1.4, "New York": 1.35, "Seattle": 1.25, "Boston": 1.2,
        "Los Angeles": 1.15, "Austin": 1.1, "Chicago": 1.05, "Miami": 0.95, "Houston": 0.95, "Phoenix": 0.9
    }
    
    annual_incomes = []
    for c in cities:
        mult = city_income_mult.get(c, 1.0)
        inc = np.random.lognormal(mean=10.9, sigma=0.5) * mult
        annual_incomes.append(round(float(np.clip(inc, 18000, 250000)), 2))
        
    annual_incomes = np.array(annual_incomes)
    
    # Membership Years (0.5 to 15.0)
    membership_years = []
    for age in ages:
        max_mem = max(0.5, min(15.0, (age - 18) * 0.4))
        mem = round(float(np.random.uniform(0.5, max_mem)), 1)
        membership_years.append(mem)
    membership_years = np.array(membership_years)
    
    # Visit Frequency (Monthly store/app visits: 1 to 30)
    visit_frequency = np.clip(
        (np.random.poisson(lam=6.5, size=n_samples) + (membership_years * 0.4)).astype(int),
        1, 30
    )
    
    # Avg Purchase Value ($10 to $400)
    avg_purchase_values = []
    for inc in annual_incomes:
        base_val = (inc / 1000.0) * random.uniform(1.2, 2.8) + np.random.normal(0, 15)
        avg_purchase_values.append(round(float(np.clip(base_val, 10.0, 450.0)), 2))
    avg_purchase_values = np.array(avg_purchase_values)
    
    # Online Activity Score (1.0 to 100.0)
    online_activity_scores = []
    for age, mem in zip(ages, membership_years):
        # Younger customers and older members tend to have higher online engagement
        score_base = 75 - (age * 0.4) + (mem * 2.5) + np.random.normal(0, 12)
        online_activity_scores.append(round(float(np.clip(score_base, 1.0, 100.0)), 1))
    online_activity_scores = np.array(online_activity_scores)
    
    # Discount Usage (0.00 to 0.85)
    discount_usages = []
    for inc, online in zip(annual_incomes, online_activity_scores):
        disc_base = 0.45 - (inc / 300000.0) + (online / 250.0) + np.random.normal(0, 0.08)
        discount_usages.append(round(float(np.clip(disc_base, 0.0, 0.85)), 2))
    discount_usages = np.array(discount_usages)
    
    # -------------------------------------------------------------------------
    # DUAL TARGETS:
    # 1. Monthly Spending (Regression Target)
    # 2. Loyal Customer (Classification Target: 1 = Loyal, 0 = Churned)
    # -------------------------------------------------------------------------
    monthly_spendings = []
    for vf, apv, disc, inc in zip(visit_frequency, avg_purchase_values, discount_usages, annual_incomes):
        # Base spend = visits * avg purchase * discount efficiency
        spend = (vf * apv) * (1.0 - (disc * 0.25)) + (inc * 0.003) + np.random.normal(0, 35)
        monthly_spendings.append(round(float(np.clip(spend, 45.0, 4800.0)), 2))
    monthly_spendings = np.array(monthly_spendings)
    
    # Churn / Loyalty Calculation:
    # Probability of being loyal depends on visit frequency, online score, membership, and satisfaction
    loyal_customers = []
    for mem, online, vf, disc, spend in zip(membership_years, online_activity_scores, visit_frequency, discount_usages, monthly_spendings):
        online_val = 50.0 if np.isnan(online) else online
        disc_val = 0.35 if np.isnan(disc) else disc
        # Centered logit for ~72% loyal / 28% churned
        logit = -2.8 + (mem * 0.25) + (online_val / 100.0 * 2.2) + (vf * 0.15) - (disc_val * 1.8) + (spend / 1500.0 * 0.3) + np.random.normal(0, 0.6)
        p_loyal = 1.0 / (1.0 + np.exp(-logit))
        is_loyal = 1 if p_loyal >= 0.50 else 0
        loyal_customers.append(is_loyal)
    loyal_customers = np.array(loyal_customers)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "city": cities,
        "annual_income": annual_incomes,
        "visit_frequency": visit_frequency,
        "avg_purchase_value": avg_purchase_values,
        "online_activity_score": online_activity_scores,
        "membership_years": membership_years,
        "discount_usage": discount_usages,
        "monthly_spending": monthly_spendings,
        "loyal_customer": loyal_customers
    })
    
    # Introduce controlled missing values and outliers for Week 2/3 Data Cleansing & EDA
    # 2.5% missing values in discount_usage and online_activity_score
    mask_nan_disc = np.random.rand(n_samples) < 0.025
    df.loc[mask_nan_disc, "discount_usage"] = np.nan
    mask_nan_online = np.random.rand(n_samples) < 0.020
    df.loc[mask_nan_online, "online_activity_score"] = np.nan
    
    # A few mild outliers in annual_income and monthly_spending
    outlier_idx = np.random.choice(n_samples, size=12, replace=False)
    df.loc[outlier_idx, "annual_income"] = df.loc[outlier_idx, "annual_income"] * 2.2
    
    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully generated and saved to {output_path} with shape {df.shape}")
    print(f"Loyal Customers Share: {df['loyal_customer'].mean()*100:.1f}%")
    print(f"Average Monthly Spending: ${df['monthly_spending'].mean():.2f}")
    return df

if __name__ == "__main__":
    generate_churn_dataset()
