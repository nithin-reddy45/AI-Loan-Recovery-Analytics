-- =================================================================================
-- AI-Driven Loan Recovery & Risk Analytics: 15 Advanced Business SQL Queries
-- Demonstrating CTEs, Window Functions (RANK, DENSE_RANK, LAG, LEAD, NTILE),
-- Multi-Table Joins, Case Statements, Rollups, and Date Aggregations.
-- =================================================================================

-- ---------------------------------------------------------------------------------
-- QUERY 1: Portfolio Executive Summary & Global Loss Metrics
-- Business Goal: Calculate high-level KPIs across the entire lending portfolio.
-- ---------------------------------------------------------------------------------
SELECT 
    COUNT(DISTINCT l.loan_id) AS total_loans_originated,
    ROUND(SUM(l.loan_amount), 2) AS total_loan_disbursed,
    ROUND(SUM(r.principal_paid), 2) AS total_principal_repaid,
    ROUND(SUM(l.loan_amount - r.principal_paid), 2) AS total_outstanding_principal,
    ROUND(COALESCE(SUM(rec.recovered_amount), 0), 2) AS total_recovered_amount,
    ROUND(COALESCE(SUM(rec.recovery_cost), 0), 2) AS total_recovery_costs,
    ROUND(COALESCE(SUM(rec.recovered_amount - rec.recovery_cost), 0), 2) AS net_recovery_yield,
    ROUND(AVG(l.interest_rate), 2) AS avg_interest_rate,
    ROUND(100.0 * SUM(CASE WHEN r.default_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS portfolio_default_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN r.overdue_days > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS portfolio_overdue_rate_pct,
    ROUND(100.0 * COALESCE(SUM(rec.recovered_amount), 0) / NULLIF(SUM(CASE WHEN r.overdue_days > 30 THEN (l.loan_amount - r.principal_paid) ELSE 0 END), 0), 2) AS overall_recovery_rate_pct
FROM loans l
INNER JOIN repayments r ON l.loan_id = r.loan_id
LEFT JOIN recovery_activities rec ON l.loan_id = rec.loan_id;


-- ---------------------------------------------------------------------------------
-- QUERY 2: Delinquency Roll-Forward & Migration Analysis
-- Business Goal: Classify delinquency stages (SMA-0 to NPA) and quantify portfolio exposure.
-- ---------------------------------------------------------------------------------
SELECT 
    r.delinquency_bucket,
    COUNT(l.loan_id) AS account_count,
    ROUND(100.0 * COUNT(l.loan_id) / (SELECT COUNT(*) FROM loans), 2) AS pct_of_total_accounts,
    ROUND(SUM(l.loan_amount - r.principal_paid), 2) AS outstanding_exposure,
    ROUND(AVG(r.overdue_days), 1) AS avg_overdue_days,
    ROUND(AVG(c.credit_score), 0) AS avg_credit_score,
    ROUND(AVG(l.interest_rate), 2) AS avg_interest_rate
FROM loans l
INNER JOIN repayments r ON l.loan_id = r.loan_id
INNER JOIN customers c ON l.customer_id = c.customer_id
GROUP BY r.delinquency_bucket
ORDER BY 
    CASE r.delinquency_bucket
        WHEN 'Current' THEN 1
        WHEN '1-30 DPD (SMA-0)' THEN 2
        WHEN '31-60 DPD (SMA-1)' THEN 3
        WHEN '61-90 DPD (SMA-2)' THEN 4
        ELSE 5
    END;


-- ---------------------------------------------------------------------------------
-- QUERY 3: Month-over-Month (MoM) Recovery Trend using Window Functions (LAG)
-- Business Goal: Track monthly recovery trajectory, MoM change, and running total.
-- ---------------------------------------------------------------------------------
WITH MonthlyRecovery AS (
    SELECT 
        STRFTIME('%Y-%m', rec.recovery_date) AS recovery_month,
        COUNT(rec.recovery_id) AS recovered_accounts,
        ROUND(SUM(rec.recovered_amount), 2) AS total_monthly_recovered,
        ROUND(SUM(rec.recovery_cost), 2) AS total_monthly_cost,
        ROUND(SUM(rec.recovered_amount - rec.recovery_cost), 2) AS net_monthly_recovered
    FROM recovery_activities rec
    WHERE rec.recovery_date IS NOT NULL
    GROUP BY STRFTIME('%Y-%m', rec.recovery_date)
)
SELECT 
    recovery_month,
    recovered_accounts,
    total_monthly_recovered,
    LAG(total_monthly_recovered, 1) OVER (ORDER BY recovery_month) AS prev_month_recovered,
    ROUND(
        100.0 * (total_monthly_recovered - LAG(total_monthly_recovered, 1) OVER (ORDER BY recovery_month)) 
        / NULLIF(LAG(total_monthly_recovered, 1) OVER (ORDER BY recovery_month), 0), 2
    ) AS mom_recovery_growth_pct,
    ROUND(SUM(total_monthly_recovered) OVER (ORDER BY recovery_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_recovered_to_date
FROM MonthlyRecovery
ORDER BY recovery_month;


-- ---------------------------------------------------------------------------------
-- QUERY 4: Recovery Officer Performance & Dense Ranking
-- Business Goal: Rank collection officers based on net recovery yield and productivity.
-- ---------------------------------------------------------------------------------
WITH OfficerMetrics AS (
    SELECT 
        rec.recovery_officer_id,
        rec.officer_name,
        rec.officer_designation,
        COUNT(rec.recovery_id) AS assigned_cases,
        SUM(CASE WHEN rec.recovery_status IN ('Fully Recovered', 'Partially Recovered') THEN 1 ELSE 0 END) AS resolved_cases,
        ROUND(SUM(rec.recovered_amount), 2) AS gross_recovered_amount,
        ROUND(SUM(rec.recovery_cost), 2) AS total_incurred_cost,
        ROUND(SUM(rec.recovered_amount - rec.recovery_cost), 2) AS net_recovered_amount,
        ROUND(AVG(rec.contact_attempts), 1) AS avg_attempts_per_case
    FROM recovery_activities rec
    WHERE rec.recovery_officer_id != 'None'
    GROUP BY rec.recovery_officer_id, rec.officer_name, rec.officer_designation
)
SELECT 
    recovery_officer_id,
    officer_name,
    officer_designation,
    assigned_cases,
    resolved_cases,
    ROUND(100.0 * resolved_cases / assigned_cases, 2) AS resolution_rate_pct,
    gross_recovered_amount,
    net_recovered_amount,
    avg_attempts_per_case,
    DENSE_RANK() OVER (ORDER BY net_recovered_amount DESC) AS net_recovery_rank,
    RANK() OVER (ORDER BY (100.0 * resolved_cases / assigned_cases) DESC) AS efficiency_rank
FROM OfficerMetrics
ORDER BY net_recovery_rank;


-- ---------------------------------------------------------------------------------
-- QUERY 5: Credit Score Quintile (NTILE) Risk & Default Concentration
-- Business Goal: Segment the portfolio into 5 equal credit risk tiers to inspect default distribution.
-- ---------------------------------------------------------------------------------
WITH RankedCustomers AS (
    SELECT 
        c.customer_id,
        c.credit_score,
        l.loan_amount,
        r.principal_paid,
        (l.loan_amount - r.principal_paid) AS outstanding_principal,
        r.default_flag,
        NTILE(5) OVER (ORDER BY c.credit_score ASC) AS credit_quintile
    FROM customers c
    INNER JOIN loans l ON c.customer_id = l.customer_id
    INNER JOIN repayments r ON l.loan_id = r.loan_id
)
SELECT 
    credit_quintile,
    MIN(credit_score) AS min_score_in_tier,
    MAX(credit_score) AS max_score_in_tier,
    COUNT(*) AS total_customers,
    SUM(default_flag) AS defaulted_customers,
    ROUND(100.0 * SUM(default_flag) / COUNT(*), 2) AS default_rate_pct,
    ROUND(SUM(loan_amount), 2) AS total_origination_amt,
    ROUND(SUM(outstanding_principal), 2) AS total_outstanding_amt
FROM RankedCustomers
GROUP BY credit_quintile
ORDER BY credit_quintile;


-- ---------------------------------------------------------------------------------
-- QUERY 6: Recovery Channel Cost-Effectiveness & ROI Multiplier
-- Business Goal: Identify the highest yield channels per dollar spent on debt recovery.
-- ---------------------------------------------------------------------------------
SELECT 
    rec.primary_channel,
    COUNT(rec.recovery_id) AS total_cases,
    SUM(CASE WHEN rec.recovery_status = 'Fully Recovered' THEN 1 ELSE 0 END) AS fully_recovered_count,
    SUM(CASE WHEN rec.recovery_status = 'Partially Recovered' THEN 1 ELSE 0 END) AS partially_recovered_count,
    SUM(CASE WHEN rec.recovery_status = 'Written Off' THEN 1 ELSE 0 END) AS written_off_count,
    ROUND(SUM(rec.recovered_amount), 2) AS total_recovered_dollars,
    ROUND(SUM(rec.recovery_cost), 2) AS total_cost_dollars,
    ROUND(SUM(rec.recovered_amount - rec.recovery_cost), 2) AS net_recovered_dollars,
    ROUND(SUM(rec.recovered_amount) / NULLIF(SUM(rec.recovery_cost), 0), 2) AS roi_multiple_x,
    ROUND(AVG(rec.contact_attempts), 1) AS avg_contact_attempts
FROM recovery_activities rec
GROUP BY rec.primary_channel
ORDER BY total_recovered_dollars DESC;


-- ---------------------------------------------------------------------------------
-- QUERY 7: Settlement Discount Tier Elasticity Analysis
-- Business Goal: Measure whether offering higher settlement discounts improves net dollars recovered.
-- ---------------------------------------------------------------------------------
WITH DiscountTiers AS (
    SELECT 
        recovery_id,
        loan_id,
        settlement_discount_pct,
        recovered_amount,
        recovery_cost,
        CASE 
            WHEN settlement_discount_pct = 0 THEN '0% (No Haircut)'
            WHEN settlement_discount_pct <= 15 THEN '1% - 15% (Low Discount)'
            WHEN settlement_discount_pct <= 30 THEN '16% - 30% (Moderate Discount)'
            ELSE '> 30% (Aggressive Haircut)'
        END AS haircut_tier
    FROM recovery_activities
    WHERE recovery_status IN ('Fully Recovered', 'Partially Recovered')
)
SELECT 
    haircut_tier,
    COUNT(*) AS settlements_count,
    ROUND(AVG(settlement_discount_pct), 2) AS avg_discount_pct,
    ROUND(SUM(recovered_amount), 2) AS total_recovered,
    ROUND(AVG(recovered_amount), 2) AS avg_recovered_per_account,
    ROUND(SUM(recovered_amount - recovery_cost), 2) AS net_recovered_after_cost
FROM DiscountTiers
GROUP BY haircut_tier
ORDER BY total_recovered DESC;


-- ---------------------------------------------------------------------------------
-- QUERY 8: Loss Given Default (LGD) and Collateral Protection
-- Business Goal: Quantify Loss Given Default (LGD) for secured vs unsecured loan portfolios.
-- ---------------------------------------------------------------------------------
WITH DefaultedLoans AS (
    SELECT 
        l.loan_id,
        l.loan_type,
        l.collateral_type,
        CASE WHEN l.collateral_type != 'None' THEN 'Secured' ELSE 'Unsecured' END AS collateral_status,
        (l.loan_amount - r.principal_paid) AS exposure_at_default_ead,
        COALESCE(rec.recovered_amount, 0) AS actual_recovered,
        COALESCE(rec.recovery_cost, 0) AS direct_cost
    FROM loans l
    INNER JOIN repayments r ON l.loan_id = r.loan_id
    LEFT JOIN recovery_activities rec ON l.loan_id = rec.loan_id
    WHERE r.default_flag = 1
)
SELECT 
    collateral_status,
    loan_type,
    COUNT(*) AS defaulted_loan_count,
    ROUND(SUM(exposure_at_default_ead), 2) AS total_ead_exposure,
    ROUND(SUM(actual_recovered), 2) AS total_recovered,
    ROUND(SUM(actual_recovered - direct_cost), 2) AS net_recovered,
    ROUND(100.0 * (1.0 - (SUM(actual_recovered - direct_cost) / NULLIF(SUM(exposure_at_default_ead), 0))), 2) AS loss_given_default_pct
FROM DefaultedLoans
GROUP BY collateral_status, loan_type
ORDER BY collateral_status, loss_given_default_pct DESC;


-- ---------------------------------------------------------------------------------
-- QUERY 9: Early Delinquency / Underwriting Vintage Flaw Identifier
-- Business Goal: Detect loans that defaulted within first 6 installments (< 180 days from origination).
-- ---------------------------------------------------------------------------------
SELECT 
    l.origination_channel,
    l.loan_type,
    COUNT(l.loan_id) AS total_originated,
    SUM(CASE WHEN r.installments_paid <= 6 AND r.default_flag = 1 THEN 1 ELSE 0 END) AS early_default_count,
    ROUND(100.0 * SUM(CASE WHEN r.installments_paid <= 6 AND r.default_flag = 1 THEN 1 ELSE 0 END) / COUNT(l.loan_id), 2) AS early_default_rate_pct,
    ROUND(SUM(CASE WHEN r.installments_paid <= 6 AND r.default_flag = 1 THEN (l.loan_amount - r.principal_paid) ELSE 0 END), 2) AS early_default_loss_exposure
FROM loans l
INNER JOIN repayments r ON l.loan_id = r.loan_id
GROUP BY l.origination_channel, l.loan_type
HAVING early_default_count > 0
ORDER BY early_default_rate_pct DESC;


-- ---------------------------------------------------------------------------------
-- QUERY 10: Multi-CTE Customer 360 Risk & Over-Leveraged Profile
-- Business Goal: Aggregate customer total exposure across debts, DTI, and delinquency.
-- ---------------------------------------------------------------------------------
WITH CustomerDebtAgg AS (
    SELECT 
        c.customer_id,
        c.annual_income,
        c.credit_score,
        c.region,
        c.existing_debts_count,
        COUNT(l.loan_id) AS active_loans_with_bank,
        SUM(l.loan_amount) AS total_borrowed,
        SUM(l.loan_amount - r.principal_paid) AS total_bank_exposure,
        MAX(r.overdue_days) AS max_overdue_days,
        MAX(r.default_flag) AS has_defaulted
    FROM customers c
    INNER JOIN loans l ON c.customer_id = l.customer_id
    INNER JOIN repayments r ON l.loan_id = r.loan_id
    GROUP BY c.customer_id, c.annual_income, c.credit_score, c.region, c.existing_debts_count
)
SELECT 
    CASE 
        WHEN credit_score >= 750 THEN 'Super Prime'
        WHEN credit_score >= 700 THEN 'Prime'
        WHEN credit_score >= 650 THEN 'Near Prime'
        ELSE 'Subprime / High Risk'
    END AS credit_segment,
    COUNT(customer_id) AS customer_count,
    ROUND(AVG(annual_income), 2) AS avg_annual_income,
    ROUND(AVG(total_bank_exposure), 2) AS avg_bank_exposure,
    ROUND(AVG(existing_debts_count), 1) AS avg_external_debts,
    SUM(has_defaulted) AS defaulted_customers,
    ROUND(100.0 * SUM(has_defaulted) / COUNT(customer_id), 2) AS default_propensity_pct
FROM CustomerDebtAgg
GROUP BY 
    CASE 
        WHEN credit_score >= 750 THEN 'Super Prime'
        WHEN credit_score >= 700 THEN 'Prime'
        WHEN credit_score >= 650 THEN 'Near Prime'
        ELSE 'Subprime / High Risk'
    END
ORDER BY default_propensity_pct DESC;


-- ---------------------------------------------------------------------------------
-- QUERY 11: Regional Delinquency Hotspots & Recovery Disparity
-- Business Goal: Evaluate collection efficacy across geographic operating regions.
-- ---------------------------------------------------------------------------------
WITH RegionalData AS (
    SELECT 
        c.region,
        COUNT(DISTINCT l.loan_id) AS total_loans,
        SUM(CASE WHEN r.default_flag = 1 THEN 1 ELSE 0 END) AS default_count,
        SUM(l.loan_amount - r.principal_paid) AS total_outstanding,
        SUM(CASE WHEN r.overdue_days > 30 THEN (l.loan_amount - r.principal_paid) ELSE 0 END) AS delinquent_exposure,
        COALESCE(SUM(rec.recovered_amount), 0) AS total_recovered
    FROM customers c
    INNER JOIN loans l ON c.customer_id = l.customer_id
    INNER JOIN repayments r ON l.loan_id = r.loan_id
    LEFT JOIN recovery_activities rec ON l.loan_id = rec.loan_id
    GROUP BY c.region
)
SELECT 
    region,
    total_loans,
    ROUND(100.0 * default_count / total_loans, 2) AS regional_default_rate_pct,
    ROUND(total_outstanding, 2) AS total_outstanding_balance,
    ROUND(delinquent_exposure, 2) AS delinquent_exposure,
    ROUND(total_recovered, 2) AS total_recovered_amount,
    ROUND(100.0 * total_recovered / NULLIF(delinquent_exposure, 0), 2) AS regional_recovery_rate_pct,
    DENSE_RANK() OVER (ORDER BY (100.0 * total_recovered / NULLIF(delinquent_exposure, 0)) DESC) AS recovery_efficiency_rank
FROM RegionalData
ORDER BY recovery_efficiency_rank;


-- ---------------------------------------------------------------------------------
-- QUERY 12: Origination Channel Delinquency & Risk Discrepancy
-- Business Goal: Identify if third-party sourcing agents (DSAs) produce higher default risk.
-- ---------------------------------------------------------------------------------
SELECT 
    l.origination_channel,
    COUNT(l.loan_id) AS total_loans_booked,
    ROUND(SUM(l.loan_amount), 2) AS total_volume_booked,
    SUM(r.default_flag) AS total_defaults,
    ROUND(100.0 * SUM(r.default_flag) / COUNT(l.loan_id), 2) AS channel_default_rate_pct,
    ROUND(AVG(c.credit_score), 0) AS avg_borrower_credit_score,
    ROUND(AVG(l.interest_rate), 2) AS avg_booked_interest_rate
FROM loans l
INNER JOIN customers c ON l.customer_id = c.customer_id
INNER JOIN repayments r ON l.loan_id = r.loan_id
GROUP BY l.origination_channel
ORDER BY channel_default_rate_pct DESC;


-- ---------------------------------------------------------------------------------
-- QUERY 13: Optimal Recovery Time-Window Decay Analysis
-- Business Goal: Measure how recovery rates decay as overdue duration (DPD) increases.
-- ---------------------------------------------------------------------------------
WITH AgingBuckets AS (
    SELECT 
        rec.recovery_id,
        rec.loan_id,
        r.overdue_days,
        (l.loan_amount - r.principal_paid) AS outstanding_bal,
        rec.recovered_amount,
        CASE 
            WHEN r.overdue_days <= 60 THEN '31-60 Days (Early Delinquency)'
            WHEN r.overdue_days <= 120 THEN '61-120 Days (Mid-Stage)'
            WHEN r.overdue_days <= 240 THEN '121-240 Days (Late-Stage)'
            ELSE '240+ Days (Severe Default)'
        END AS aging_band
    FROM recovery_activities rec
    INNER JOIN loans l ON rec.loan_id = l.loan_id
    INNER JOIN repayments r ON l.loan_id = r.loan_id
)
SELECT 
    aging_band,
    COUNT(*) AS total_cases,
    ROUND(SUM(outstanding_bal), 2) AS total_delinquent_balance,
    ROUND(SUM(recovered_amount), 2) AS total_recovered_amount,
    ROUND(100.0 * SUM(recovered_amount) / NULLIF(SUM(outstanding_bal), 0), 2) AS recovery_success_rate_pct,
    ROUND(AVG(recovered_amount), 2) AS avg_recovered_per_case
FROM AgingBuckets
GROUP BY aging_band
ORDER BY 
    CASE aging_band
        WHEN '31-60 Days (Early Delinquency)' THEN 1
        WHEN '61-120 Days (Mid-Stage)' THEN 2
        WHEN '121-240 Days (Late-Stage)' THEN 3
        ELSE 4
    END;


-- ---------------------------------------------------------------------------------
-- QUERY 14: Promise-to-Pay (PTP) Conversion & Broken Promise Leakage
-- Business Goal: Analyze the financial impact of customer promise-to-pay commitments.
-- ---------------------------------------------------------------------------------
SELECT 
    rec.primary_channel,
    rec.promise_to_pay_kept,
    COUNT(*) AS account_count,
    ROUND(SUM(rec.recovered_amount), 2) AS total_amount_recovered,
    ROUND(AVG(rec.recovered_amount), 2) AS avg_amount_recovered,
    ROUND(AVG(rec.contact_attempts), 1) AS avg_attempts_needed
FROM recovery_activities rec
GROUP BY rec.primary_channel, rec.promise_to_pay_kept
ORDER BY rec.primary_channel, rec.promise_to_pay_kept DESC;


-- ---------------------------------------------------------------------------------
-- QUERY 15: Prioritized Legal Action & Asset Repossession Candidate Pipeline
-- Business Goal: Generate actionable high-priority accounts for legal escalation & repossession.
-- ---------------------------------------------------------------------------------
SELECT 
    l.loan_id,
    c.customer_id,
    c.region,
    l.loan_type,
    l.collateral_type,
    l.collateral_value,
    ROUND(l.loan_amount - r.principal_paid, 2) AS outstanding_balance,
    r.overdue_days,
    rec.contact_attempts,
    rec.recovery_status,
    CASE 
        WHEN l.collateral_type != 'None' AND (l.loan_amount - r.principal_paid) > 25000 THEN 'Immediate Collateral Repossession'
        WHEN (l.loan_amount - r.principal_paid) > 50000 THEN 'High-Court / DRT Legal Summons'
        ELSE 'Third-Party Field Enforcement'
    END AS recommended_legal_action
FROM loans l
INNER JOIN customers c ON l.customer_id = c.customer_id
INNER JOIN repayments r ON l.loan_id = r.loan_id
INNER JOIN recovery_activities rec ON l.loan_id = rec.loan_id
WHERE r.default_flag = 1 
  AND rec.recovery_status IN ('In Progress', 'Legal Escalation', 'Written Off')
ORDER BY outstanding_balance DESC
LIMIT 20;
