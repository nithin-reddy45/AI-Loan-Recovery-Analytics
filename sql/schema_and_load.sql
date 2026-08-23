-- =================================================================================
-- AI-Driven Loan Recovery & Risk Analytics: Schema & Table DDL
-- Compatible with SQLite, PostgreSQL, MySQL
-- =================================================================================

DROP TABLE IF EXISTS recovery_activities;
DROP TABLE IF EXISTS repayments;
DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS customers;

-- 1. Dim_Customers Table
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    region VARCHAR(20),
    city_tier VARCHAR(15),
    employment_status VARCHAR(40),
    employment_length_years INT,
    annual_income REAL,
    credit_score INT,
    housing_status VARCHAR(20),
    marital_status VARCHAR(20),
    existing_debts_count INT
);

-- 2. Dim_Loans Table
CREATE TABLE loans (
    loan_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    loan_type VARCHAR(30),
    loan_amount REAL,
    interest_rate REAL,
    loan_term_months INT,
    disbursement_date DATE,
    collateral_type VARCHAR(30),
    collateral_value REAL,
    loan_purpose VARCHAR(50),
    origination_channel VARCHAR(30),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 3. Fact_Repayments Table
CREATE TABLE repayments (
    repayment_id VARCHAR(20) PRIMARY KEY,
    loan_id VARCHAR(20),
    customer_id VARCHAR(20),
    total_installments_due INT,
    installments_paid INT,
    total_amount_paid REAL,
    principal_paid REAL,
    interest_paid REAL,
    last_payment_date DATE,
    overdue_days INT,
    delinquency_bucket VARCHAR(30),
    default_flag INT,
    default_date DATE,
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 4. Fact_Recovery_Activities Table
CREATE TABLE recovery_activities (
    recovery_id VARCHAR(20) PRIMARY KEY,
    loan_id VARCHAR(20),
    recovery_status VARCHAR(30),
    primary_channel VARCHAR(40),
    recovery_officer_id VARCHAR(20),
    officer_name VARCHAR(50),
    officer_designation VARCHAR(50),
    contact_attempts INT,
    promise_to_pay_kept INT,
    settlement_discount_pct REAL,
    recovered_amount REAL,
    recovery_cost REAL,
    recovery_date DATE,
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
);

-- Create Indexes for Analytical Performance
CREATE INDEX idx_loans_customer ON loans(customer_id);
CREATE INDEX idx_repay_loan ON repayments(loan_id);
CREATE INDEX idx_repay_bucket ON repayments(delinquency_bucket);
CREATE INDEX idx_rec_loan ON recovery_activities(loan_id);
CREATE INDEX idx_rec_channel ON recovery_activities(primary_channel);
