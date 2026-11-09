"""
Customer & Loan Segmentation Module: AI-Driven Loan Recovery & Risk Analytics
Implements K-Means Clustering, PCA Dimensionality Reduction, and Behavioral Persona Profiling.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def run_segmentation(input_file="data/processed/loan_recovery_master.csv", output_dir="data/processed", fig_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    print("Performing Customer & Loan Behavioral Segmentation...")
    
    df = pd.read_csv(input_file)
    
    # Feature selection for clustering
    features = [
        "credit_score", "annual_income", "loan_amount", "interest_rate",
        "overdue_days", "dti_ratio", "installments_paid", "delinquency_severity_score"
    ]
    
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Evaluate Elbow & Silhouette
    k_range = range(2, 8)
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
        
    print(f"Silhouette scores for k=2..7: {[round(s, 3) for s in silhouettes]}")
    
    # Optimal k = 4 for clear banking personas
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Assign cluster labels
    df["cluster_id"] = cluster_labels
    
    # Profile clusters to assign business persona names
    cluster_profiles = df.groupby("cluster_id")[features + ["default_flag", "recovered_amount"]].mean()
    print("Cluster Profiling (Mean values):")
    print(cluster_profiles)
    
    # Persona mapping logic based on profile traits
    persona_map = {}
    for cid in range(optimal_k):
        c_score = cluster_profiles.loc[cid, "credit_score"]
        c_dpd = cluster_profiles.loc[cid, "overdue_days"]
        c_amt = cluster_profiles.loc[cid, "loan_amount"]
        c_def = cluster_profiles.loc[cid, "default_flag"]
        
        if c_score >= 690 and c_def < 0.05:
            persona_map[cid] = "Prime / Low-Risk Self-Curative"
        elif c_amt > 100000 and c_score >= 630:
            persona_map[cid] = "Secured High-Value / Asset-Backed"
        elif c_dpd > 90 or c_score < 600 or c_def > 0.30:
            persona_map[cid] = "Chronic Delinquent / High Risk (Agency/Legal)"
        else:
            persona_map[cid] = "Distressed but Responsive (Restructure)"
            
    df["segment_name"] = df["cluster_id"].map(persona_map)
    print("\nAssigned Business Personas:")
    for cid, name in persona_map.items():
        count = (df["cluster_id"] == cid).sum()
        print(f" - Cluster {cid}: {name} (n={count}, {count/len(df)*100:.1f}%)")
        
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_dim1"] = pca_coords[:, 0]
    df["pca_dim2"] = pca_coords[:, 1]
    
    # -------------------------------------------------------------
    # Plot 1: PCA 2D Cluster Space
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, 
        x="pca_dim1", 
        y="pca_dim2", 
        hue="segment_name", 
        palette="tab10", 
        alpha=0.6, 
        s=30
    )
    plt.title(f"Borrower & Loan Segmentation Clusters (PCA 2D Projection, Var Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(title="Customer Persona", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig_pca_path = os.path.join(fig_dir, "08_customer_segmentation_pca.png")
    plt.savefig(fig_pca_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_pca_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Persona Key Driver Comparison Boxplots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    sns.boxplot(data=df, x="segment_name", y="credit_score", ax=axes[0, 0], palette="Set2", showfliers=False)
    axes[0, 0].set_title("Credit Score Distribution by Segment", fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=25)
    
    sns.boxplot(data=df, x="segment_name", y="loan_amount", ax=axes[0, 1], palette="Set2", showfliers=False)
    axes[0, 1].set_title("Loan Amount Distribution by Segment", fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=25)
    
    sns.barplot(data=df, x="segment_name", y="default_flag", ax=axes[1, 0], palette="Reds_r")
    axes[1, 0].set_title("Default Rate (%) by Segment", fontweight="bold")
    axes[1, 0].set_ylabel("Default Rate")
    axes[1, 0].tick_params(axis="x", rotation=25)
    for p in axes[1, 0].patches:
        axes[1, 0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="baseline", fontsize=10, xytext=(0, 3), textcoords="offset points")
        
    sns.boxplot(data=df, x="segment_name", y="delinquency_severity_score", ax=axes[1, 1], palette="Purples_r", showfliers=False)
    axes[1, 1].set_title("Delinquency Severity Score by Segment", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=25)
    
    plt.suptitle("Customer & Loan Segmentation Persona Deep-Dive", fontsize=16, fontweight="bold", y=1.02)
    fig_profile_path = os.path.join(fig_dir, "09_segment_profile_radar_boxplots.png")
    plt.savefig(fig_profile_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_profile_path}")
    
    # Save segmented master dataset
    output_file = os.path.join(output_dir, "segmented_customers.csv")
    df.to_csv(output_file, index=False)
    print(f"Segmented customer dataset saved to {output_file}")
    
    return df

if __name__ == "__main__":
    run_segmentation()
