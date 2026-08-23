"""
Master End-to-End Analytics Pipeline: AI-Driven Loan Recovery & Risk Analytics
Executes the full pipeline: Data Generation -> Data Cleaning -> EDA Visuals -> SQL Analytics -> Segmentation -> ML Modeling.
"""

import sys
import os
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("      AI-DRIVEN LOAN RECOVERY & RISK ANALYTICS: MASTER PIPELINE")
    print("=" * 80)
    
    # 1. Synthetic Data Generation
    print("\n>>> STEP 1: Generating Realistic Relational Banking Datasets...")
    from src.generate_synthetic_data import generate_datasets
    generate_datasets(n_customers=15000, output_dir="data/raw")
    
    # 2. Data Cleaning & Feature Engineering
    print("\n>>> STEP 2: Data Cleaning & Advanced Feature Engineering...")
    from src.data_cleaning import clean_and_preprocess
    clean_and_preprocess(raw_dir="data/raw", processed_dir="data/processed")
    
    # 3. Exploratory Data Analysis & Visuals
    print("\n>>> STEP 3: Exploratory Data Analysis & Chart Generation...")
    from src.eda_visualizations import run_eda
    run_eda(input_file="data/processed/loan_recovery_master.csv", output_dir="reports/figures")
    
    # 4. SQL Analytics & Database Population
    print("\n>>> STEP 4: Relational Database Build & Advanced SQL Analysis (15 Queries)...")
    from sql.run_sql_analysis import build_database, execute_and_export_queries
    conn = build_database(db_path="loan_recovery.db", processed_dir="data/processed")
    execute_and_export_queries(conn, sql_file="sql/advanced_sql_analytics.sql", report_path="reports/sql_query_results.md")
    conn.close()
    
    # 5. Customer & Loan Segmentation (K-Means + PCA)
    print("\n>>> STEP 5: Customer & Loan Segmentation (K-Means & Personas)...")
    from src.segmentation import run_segmentation
    run_segmentation(input_file="data/processed/loan_recovery_master.csv", output_dir="data/processed", fig_dir="reports/figures")
    
    # 6. Predictive Machine Learning Modeling
    print("\n>>> STEP 6: Predictive Machine Learning Modeling (LightGBM/XGBoost/RF/LR)...")
    from src.predictive_model import train_and_evaluate_models
    train_and_evaluate_models(input_file="data/processed/loan_recovery_master.csv", model_dir="models", fig_dir="reports/figures")
    
    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"      PIPELINE EXECUTION SUCCESSFULLY COMPLETED IN {elapsed} SECONDS")
    print("=" * 80)

if __name__ == "__main__":
    run_full_pipeline()
