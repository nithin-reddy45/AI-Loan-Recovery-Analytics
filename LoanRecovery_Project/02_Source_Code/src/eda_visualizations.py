"""
Exploratory Data Analysis & Visualization Module: AI-Driven Loan Recovery & Risk Analytics
Generates publication-quality figures and plots for risk assessment and recovery intelligence.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#ebebeb",
    "figure.autolayout": True,
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def run_eda(input_file="data/processed/loan_recovery_master.csv", output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("Executing Exploratory Data Analysis & generating visual reports...")
    
    df = pd.read_csv(input_file)
    delinquent_df = df[df["overdue_days"] > 30].copy()
    
    # ------------------------------------------------------------------
    # 1. Portfolio Financial & Risk Distributions
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loan Amount Distribution by Type
    sns.histplot(data=df, x="loan_amount", hue="loan_type", kde=True, ax=axes[0, 0], bins=35, multiple="stack", palette="tab10")
    axes[0, 0].set_title("Loan Amount Distribution by Product Line", fontweight="bold", pad=10)
    axes[0, 0].set_xlabel("Loan Amount ($)")
    axes[0, 0].set_ylabel("Account Count")
    
    # Credit Score Distribution by Delinquency Bucket
    sns.kdeplot(data=df, x="credit_score", hue="delinquency_bucket", common_norm=False, ax=axes[0, 1], palette="turbo", fill=True, alpha=0.25)
    axes[0, 1].set_title("Credit Score Density across Delinquency Stages", fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("Credit Score (300 - 850)")
    axes[0, 1].set_ylabel("Density")
    
    # Interest Rate vs Delinquency Days (Scatter)
    sns.scatterplot(data=df.sample(min(2500, len(df))), x="interest_rate", y="overdue_days", hue="loan_type", alpha=0.6, ax=axes[1, 0], palette="Set1")
    axes[1, 0].set_title("Interest Rate vs Overdue Duration (DPD)", fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("Interest Rate (%)")
    axes[1, 0].set_ylabel("Overdue Days (DPD)")
    
    # Debt-to-Income (DTI) by Risk Tier
    sns.boxplot(data=df, x="credit_risk_tier", y="dti_ratio", ax=axes[1, 1], palette="Blues_r", showfliers=False)
    axes[1, 1].set_title("Debt-to-Income (DTI) Distribution across Risk Tiers", fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("Credit Risk Tier")
    axes[1, 1].set_ylabel("DTI Ratio")
    axes[1, 1].tick_params(axis="x", rotation=20)
    
    plt.suptitle("Portfolio Macro Overview & Risk Profile", fontsize=16, fontweight="bold", y=1.02)
    fig_path1 = os.path.join(output_dir, "01_portfolio_overview_kpis.png")
    plt.savefig(fig_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path1}")
    
    # ------------------------------------------------------------------
    # 2. Delinquency & Default Rates by Segment & Region
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    delinq_rate_type = df.groupby("loan_type")["default_flag"].mean().reset_index()
    delinq_rate_type["default_rate_pct"] = delinq_rate_type["default_flag"] * 100
    delinq_rate_type = delinq_rate_type.sort_values("default_rate_pct", ascending=False)
    
    sns.barplot(data=delinq_rate_type, x="loan_type", y="default_rate_pct", ax=axes[0], palette="Reds_r")
    axes[0].set_title("Default Rate (%) by Loan Product", fontweight="bold", pad=10)
    axes[0].set_ylabel("Default Rate (%)")
    axes[0].set_xlabel("Loan Type")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="baseline", fontsize=11, color="black", xytext=(0, 4), textcoords="offset points")
        
    delinq_region = df.groupby(["region", "delinquency_bucket"]).size().unstack(fill_value=0)
    delinq_region_pct = delinq_region.div(delinq_region.sum(axis=1), axis=0) * 100
    delinq_region_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Spectral", edgecolor="none")
    axes[1].set_title("Delinquency Stage Breakdown by Geographic Region", fontweight="bold", pad=10)
    axes[1].set_ylabel("Share of Accounts (%)")
    axes[1].set_xlabel("Region")
    axes[1].legend(title="Delinquency Bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=0)
    
    plt.suptitle("Portfolio Delinquency & Default Exposure", fontsize=16, fontweight="bold", y=1.02)
    fig_path2 = os.path.join(output_dir, "02_delinquency_by_loan_type_and_region.png")
    plt.savefig(fig_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path2}")
    
    # ------------------------------------------------------------------
    # 3. Recovery Rate & Efficiency by Channel
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    rec_channel = delinquent_df.groupby("primary_channel").agg(
        total_outstanding=("outstanding_principal", "sum"),
        total_recovered=("recovered_amount", "sum"),
        total_cost=("recovery_cost", "sum"),
        avg_attempts=("contact_attempts", "mean")
    ).reset_index()
    rec_channel["recovery_rate_pct"] = (rec_channel["total_recovered"] / rec_channel["total_outstanding"]) * 100
    rec_channel["roi_multiple"] = (rec_channel["total_recovered"] / np.maximum(rec_channel["total_cost"], 1.0))
    rec_channel = rec_channel.sort_values("recovery_rate_pct", ascending=False)
    
    sns.barplot(data=rec_channel, x="recovery_rate_pct", y="primary_channel", ax=axes[0], palette="Greens_r")
    axes[0].set_title("Recovery Rate (%) by Collection Channel", fontweight="bold", pad=10)
    axes[0].set_xlabel("Recovery Rate (%)")
    axes[0].set_ylabel("Primary Channel")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_width():.1f}%", (p.get_width(), p.get_y() + p.get_height() / 2.),
                         ha="left", va="center", fontsize=11, color="black", xytext=(4, 0), textcoords="offset points")
        
    sns.barplot(data=rec_channel.sort_values("roi_multiple", ascending=False), x="roi_multiple", y="primary_channel", ax=axes[1], palette="Blues_r")
    axes[1].set_title("Channel Efficiency Multiplier (Recovered $ / Cost $)", fontweight="bold", pad=10)
    axes[1].set_xlabel("ROI Multiplier (x)")
    axes[1].set_ylabel("")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_width():.1f}x", (p.get_width(), p.get_y() + p.get_height() / 2.),
                         ha="left", va="center", fontsize=11, color="black", xytext=(4, 0), textcoords="offset points")
        
    plt.suptitle("Debt Recovery Performance & Channel Cost-Effectiveness", fontsize=16, fontweight="bold", y=1.02)
    fig_path3 = os.path.join(output_dir, "03_recovery_rate_by_channel.png")
    plt.savefig(fig_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path3}")
    
    # ------------------------------------------------------------------
    # 4. Settlement Discount Elasticity & Collateral Impact
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    settled_df = delinquent_df[delinquent_df["settlement_discount_pct"] > 0]
    sns.scatterplot(data=settled_df, x="settlement_discount_pct", y="recovered_amount", hue="loan_type", size="outstanding_principal", sizes=(20, 200), alpha=0.6, ax=axes[0])
    axes[0].set_title("Settlement Discount % vs Amount Recovered", fontweight="bold", pad=10)
    axes[0].set_xlabel("Settlement Discount (%)")
    axes[0].set_ylabel("Recovered Amount ($)")
    
    sns.boxplot(data=delinquent_df, x="collateral_type", y="recovery_rate_pct", ax=axes[1], palette="viridis", showfliers=False)
    axes[1].set_title("Recovery Rate (%) by Collateral Structure", fontweight="bold", pad=10)
    axes[1].set_xlabel("Collateral Type")
    axes[1].set_ylabel("Recovery Rate (%)")
    axes[1].tick_params(axis="x", rotation=20)
    
    plt.suptitle("Settlement Optimization & Collateral Protection", fontsize=16, fontweight="bold", y=1.02)
    fig_path4 = os.path.join(output_dir, "04_settlement_discount_vs_recovery.png")
    plt.savefig(fig_path4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path4}")
    
    # ------------------------------------------------------------------
    # 5. Correlation Heatmap of Financial & Risk Drivers
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 10))
    numeric_cols = [
        "age", "annual_income", "credit_score", "loan_amount", "interest_rate", 
        "loan_term_months", "overdue_days", "dti_ratio", "ltv_ratio", "contact_attempts",
        "recovered_amount", "recovery_cost", "default_flag"
    ]
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f", annot_kws={"size": 9})
    plt.title("Correlation Heatmap of Key Risk, Financial & Recovery Variables", fontsize=15, fontweight="bold", pad=15)
    fig_path5 = os.path.join(output_dir, "05_correlation_heatmap.png")
    plt.savefig(fig_path5, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path5}")
    
    # ------------------------------------------------------------------
    # 6. Vintage Cohort Recovery Curve
    # ------------------------------------------------------------------
    plt.figure(figsize=(13, 6))
    vintage_grp = df.groupby(["vintage_quarter", "loan_type"]).agg(
        total_originated=("loan_amount", "sum"),
        total_recovered=("recovered_amount", "sum"),
        total_principal_paid=("principal_paid", "sum")
    ).reset_index()
    vintage_grp["cumulative_repayment_rate"] = ((vintage_grp["total_principal_paid"] + vintage_grp["total_recovered"]) / vintage_grp["total_originated"]) * 100
    
    sns.lineplot(data=vintage_grp, x="vintage_quarter", y="cumulative_repayment_rate", hue="loan_type", marker="o", linewidth=2.5)
    plt.title("Cumulative Principal & Recovery Vintage Curve by Origination Quarter", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Origination Cohort (Quarter)")
    plt.ylabel("Cumulative Repayment & Recovery (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Loan Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig_path6 = os.path.join(output_dir, "06_vintage_curve_analysis.png")
    plt.savefig(fig_path6, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path6}")
    
    # ------------------------------------------------------------------
    # 7. Recovery Officer Leaderboard & Productivity Matrix
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    officer_perf = delinquent_df[delinquent_df["recovery_officer_id"] != "None"].groupby(["recovery_officer_id", "officer_name"]).agg(
        total_accounts=("loan_id", "count"),
        total_recovered=("recovered_amount", "sum"),
        avg_recovery_rate=("recovery_rate_pct", "mean"),
        total_cost=("recovery_cost", "sum")
    ).reset_index()
    officer_perf["net_recovered"] = officer_perf["total_recovered"] - officer_perf["total_cost"]
    officer_perf = officer_perf.sort_values("total_recovered", ascending=False)
    
    sns.barplot(data=officer_perf, x="total_recovered", y="officer_name", ax=axes[0], palette="mako")
    axes[0].set_title("Total Recovered Portfolio ($) by Recovery Officer", fontweight="bold", pad=10)
    axes[0].set_xlabel("Recovered Amount ($)")
    axes[0].set_ylabel("Recovery Officer")
    
    sns.scatterplot(data=officer_perf, x="avg_recovery_rate", y="total_recovered", size="total_accounts", hue="officer_name", sizes=(100, 500), ax=axes[1], legend=False)
    for _, row in officer_perf.iterrows():
        axes[1].annotate(row["officer_name"].split()[0], (row["avg_recovery_rate"] + 0.5, row["total_recovered"]), fontsize=10)
    axes[1].set_title("Officer Performance Matrix: Recovery Rate % vs Total Recovered", fontweight="bold", pad=10)
    axes[1].set_xlabel("Average Recovery Rate (%)")
    axes[1].set_ylabel("Total Recovered ($)")
    
    plt.suptitle("Recovery Agent Benchmarking & Productivity", fontsize=16, fontweight="bold", y=1.02)
    fig_path7 = os.path.join(output_dir, "07_collector_performance_matrix.png")
    plt.savefig(fig_path7, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path7}")
    
    print("EDA visual generation successfully completed!")

if __name__ == "__main__":
    run_eda()
