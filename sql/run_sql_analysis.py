"""
SQL Execution Engine & Report Generator: AI-Driven Loan Recovery & Risk Analytics
Initializes SQLite database, loads cleaned banking tables, executes all 15 advanced SQL queries,
and formats results into a comprehensive Markdown business report.
"""

import os
import sqlite3
import pandas as pd

def build_database(db_path="loan_recovery.db", processed_dir="data/processed"):
    print(f"Building SQLite database at: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Execute DDL
    with open("sql/schema_and_load.sql", "r") as f:
        ddl_script = f.read()
    cursor.executescript(ddl_script)
    conn.commit()
    
    # Load clean data
    customers_df = pd.read_csv(os.path.join(processed_dir, "clean_customers.csv"))
    loans_df = pd.read_csv(os.path.join(processed_dir, "clean_loans.csv"))
    repayments_df = pd.read_csv(os.path.join(processed_dir, "clean_repayments.csv"))
    recovery_df = pd.read_csv(os.path.join(processed_dir, "clean_recovery.csv"))
    
    # Filter columns to match DDL exactly
    cust_cols = ["customer_id", "age", "gender", "region", "city_tier", "employment_status", 
                 "employment_length_years", "annual_income", "credit_score", "housing_status", 
                 "marital_status", "existing_debts_count"]
    customers_df[cust_cols].to_sql("customers", conn, if_exists="append", index=False)
    
    loan_cols = ["loan_id", "customer_id", "loan_type", "loan_amount", "interest_rate", 
                 "loan_term_months", "disbursement_date", "collateral_type", "collateral_value", 
                 "loan_purpose", "origination_channel"]
    loans_df[loan_cols].to_sql("loans", conn, if_exists="append", index=False)
    
    rep_cols = ["repayment_id", "loan_id", "customer_id", "total_installments_due", "installments_paid", 
                "total_amount_paid", "principal_paid", "interest_paid", "last_payment_date", 
                "overdue_days", "delinquency_bucket", "default_flag", "default_date"]
    repayments_df[rep_cols].to_sql("repayments", conn, if_exists="append", index=False)
    
    rec_cols = ["recovery_id", "loan_id", "recovery_status", "primary_channel", "recovery_officer_id", 
                "officer_name", "officer_designation", "contact_attempts", "promise_to_pay_kept", 
                "settlement_discount_pct", "recovered_amount", "recovery_cost", "recovery_date"]
    recovery_df[rec_cols].to_sql("recovery_activities", conn, if_exists="append", index=False)
    
    conn.commit()
    print("Database tables successfully populated!")
    return conn

def execute_and_export_queries(conn, sql_file="sql/advanced_sql_analytics.sql", report_path="reports/sql_query_results.md"):
    print("Executing advanced SQL queries and compiling report...")
    with open(sql_file, "r") as f:
        full_sql = f.read()
        
    # Split by '-- QUERY'
    raw_sections = full_sql.split("-- QUERY ")
    
    report_content = [
        "# Advanced SQL Analytics Report: Loan Recovery & Risk Intelligence",
        "",
        "This report provides executive intelligence extracted through production-grade SQL analytical queries, covering portfolio rollups, delinquency migration, window functions (RANK, DENSE_RANK, LAG/LEAD, NTILE), CTEs, and cohort analysis.",
        "",
        "---",
        ""
    ]
    
    for section in raw_sections[1:]:
        header_part, sql_part = section.split(";", 1)[0], ""
        lines = section.splitlines()
        title_line = f"QUERY {lines[0].strip()}"
        
        goal_text = ""
        sql_lines = []
        for line in lines[1:]:
            if "Business Goal:" in line:
                goal_text = line.replace("-- Business Goal:", "").replace("--", "").strip()
            elif not line.strip().startswith("--"):
                sql_lines.append(line)
                
        sql_statement = "\n".join(sql_lines).strip()
        # Remove trailing semicolon if any
        if sql_statement.endswith(";"):
            sql_statement = sql_statement[:-1].strip()
            
        if not sql_statement:
            continue
            
        print(f"Running {title_line}...")
        
        try:
            res_df = pd.read_sql_query(sql_statement, conn)
            
            report_content.append(f"## {title_line}")
            if goal_text:
                report_content.append(f"**Business Purpose**: *{goal_text}*")
            report_content.append("")
            report_content.append("```sql")
            report_content.append(sql_statement.strip() + ";")
            report_content.append("```")
            report_content.append("")
            report_content.append("**Execution Output:**")
            report_content.append("")
            report_content.append(res_df.to_markdown(index=False))
            report_content.append("")
            report_content.append("---")
            report_content.append("")
        except Exception as e:
            print(f"Error executing {title_line}: {e}")
            
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
        
    print(f"SQL Analytics Report successfully generated and saved to: {report_path}")

if __name__ == "__main__":
    conn = build_database()
    execute_and_export_queries(conn)
    conn.close()
