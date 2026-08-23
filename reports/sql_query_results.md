# Advanced SQL Analytics Report: Loan Recovery & Risk Intelligence

This report provides executive intelligence extracted through production-grade SQL analytical queries, covering portfolio rollups, delinquency migration, window functions (RANK, DENSE_RANK, LAG/LEAD, NTILE), CTEs, and cohort analysis.

---

## QUERY 1: Portfolio Executive Summary & Global Loss Metrics
**Business Purpose**: *Calculate high-level KPIs across the entire lending portfolio.*

```sql
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
```

**Execution Output:**

|   total_loans_originated |   total_loan_disbursed |   total_principal_repaid |   total_outstanding_principal |   total_recovered_amount |   total_recovery_costs |   net_recovery_yield |   avg_interest_rate |   portfolio_default_rate_pct |   portfolio_overdue_rate_pct |   overall_recovery_rate_pct |
|-------------------------:|-----------------------:|-------------------------:|------------------------------:|-------------------------:|-----------------------:|---------------------:|--------------------:|-----------------------------:|-----------------------------:|----------------------------:|
|                    15000 |            1.11683e+09 |              3.36901e+08 |                   7.79933e+08 |              6.18323e+07 |             1.4471e+06 |          6.03852e+07 |                12.4 |                         7.08 |                        47.75 |                       59.62 |

---

## QUERY 2: Delinquency Roll-Forward & Migration Analysis
**Business Purpose**: *Classify delinquency stages (SMA-0 to NPA) and quantify portfolio exposure.*

```sql
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
```

**Execution Output:**

| delinquency_bucket      |   account_count |   pct_of_total_accounts |   outstanding_exposure |   avg_overdue_days |   avg_credit_score |   avg_interest_rate |
|:------------------------|----------------:|------------------------:|-----------------------:|-------------------:|-------------------:|--------------------:|
| Current                 |            7837 |                   52.25 |            5.49273e+08 |                0   |                708 |               10.62 |
| 1-30 DPD (SMA-0)        |            3391 |                   22.61 |            1.26941e+08 |               15.6 |                667 |               13.18 |
| 31-60 DPD (SMA-1)       |            1699 |                   11.33 |            4.85645e+07 |               45.7 |                645 |               14.43 |
| 61-90 DPD (SMA-2)       |            1011 |                    6.74 |            2.02401e+07 |               75.5 |                627 |               15.6  |
| 90+ DPD (NPA / Default) |            1062 |                    7.08 |            3.4914e+07  |              313.5 |                596 |               16.68 |

---

## QUERY 3: Month-over-Month (MoM) Recovery Trend using Window Functions (LAG)
**Business Purpose**: *Track monthly recovery trajectory, MoM change, and running total.*

```sql
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
```

**Execution Output:**

| recovery_month   |   recovered_accounts |   total_monthly_recovered |   prev_month_recovered |   mom_recovery_growth_pct |   cumulative_recovered_to_date |
|:-----------------|---------------------:|--------------------------:|-----------------------:|--------------------------:|-------------------------------:|
| 2024-07          |                  362 |               8.90214e+06 |          nan           |                    nan    |                    8.90214e+06 |
| 2024-08          |                  410 |               9.9695e+06  |            8.90214e+06 |                     11.99 |                    1.88716e+07 |
| 2024-09          |                  443 |               1.11951e+07 |            9.9695e+06  |                     12.29 |                    3.00667e+07 |
| 2024-10          |                  458 |               1.0265e+07  |            1.11951e+07 |                     -8.31 |                    4.03318e+07 |
| 2024-11          |                  431 |               1.11863e+07 |            1.0265e+07  |                      8.97 |                    5.15181e+07 |
| 2024-12          |                  405 |               1.03142e+07 |            1.11863e+07 |                     -7.8  |                    6.18323e+07 |

---

## QUERY 4: Recovery Officer Performance & Dense Ranking
**Business Purpose**: *Rank collection officers based on net recovery yield and productivity.*

```sql
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
```

**Execution Output:**

| recovery_officer_id   | officer_name      | officer_designation                |   assigned_cases |   resolved_cases |   resolution_rate_pct |   gross_recovered_amount |   net_recovered_amount |   avg_attempts_per_case |   net_recovery_rank |   efficiency_rank |
|:----------------------|:------------------|:-----------------------------------|-----------------:|-----------------:|----------------------:|-------------------------:|-----------------------:|------------------------:|--------------------:|------------------:|
| OFF-105               | Carlos Mendez     | Digital & Tele-Recovery Lead       |              876 |              650 |                 74.2  |              1.66955e+07 |            1.6627e+07  |                     4.8 |                   1 |                 2 |
| OFF-106               | Priya Patel       | Early Stage Collections Specialist |              823 |              611 |                 74.24 |              1.52352e+07 |            1.5172e+07  |                     4.7 |                   2 |                 1 |
| OFF-102               | Elena Rostova     | Special Asset Resolution Lead      |              325 |              156 |                 48    |              5.45736e+06 |            5.11026e+06 |                    18.8 |                   3 |                 8 |
| OFF-107               | David Kim         | Corporate Debt Recovery Executive  |              343 |              175 |                 51.02 |              5.29374e+06 |            4.93168e+06 |                    18.7 |                   4 |                 6 |
| OFF-103               | Aarav Sharma      | Field Collections Specialist       |              372 |              254 |                 68.28 |              4.98164e+06 |            4.89059e+06 |                     6.4 |                   5 |                 5 |
| OFF-108               | Zainab Al-Mansoor | Restructuring & Workout Lead       |              368 |              262 |                 71.2  |              4.95179e+06 |            4.86279e+06 |                     6.5 |                   6 |                 3 |
| OFF-104               | Sarah Jenkins     | Legal & Settlements Officer        |              327 |              164 |                 50.15 |              5.07419e+06 |            4.7289e+06  |                    18.9 |                   7 |                 7 |
| OFF-101               | Marcus Vance      | Senior Recovery Manager            |              338 |              237 |                 70.12 |              4.14292e+06 |            4.06202e+06 |                     6.4 |                   8 |                 4 |

---

## QUERY 5: Credit Score Quintile (NTILE) Risk & Default Concentration
**Business Purpose**: *Segment the portfolio into 5 equal credit risk tiers to inspect default distribution.*

```sql
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
```

**Execution Output:**

|   credit_quintile |   min_score_in_tier |   max_score_in_tier |   total_customers |   defaulted_customers |   default_rate_pct |   total_origination_amt |   total_outstanding_amt |
|------------------:|--------------------:|--------------------:|------------------:|----------------------:|-------------------:|------------------------:|------------------------:|
|                 1 |                 351 |                 631 |              3000 |                   721 |              24.03 |             1.5129e+08  |             1.09663e+08 |
|                 2 |                 631 |                 666 |              3000 |                   206 |               6.87 |             2.02929e+08 |             1.42748e+08 |
|                 3 |                 666 |                 695 |              3000 |                    92 |               3.07 |             2.29138e+08 |             1.58509e+08 |
|                 4 |                 695 |                 729 |              3000 |                    35 |               1.17 |             2.49409e+08 |             1.71322e+08 |
|                 5 |                 729 |                 850 |              3000 |                     8 |               0.27 |             2.84067e+08 |             1.97691e+08 |

---

## QUERY 6: Recovery Channel Cost-Effectiveness & ROI Multiplier
**Business Purpose**: *Identify the highest yield channels per dollar spent on debt recovery.*

```sql
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
```

**Execution Output:**

| primary_channel            |   total_cases |   fully_recovered_count |   partially_recovered_count |   written_off_count |   total_recovered_dollars |   total_cost_dollars |   net_recovered_dollars |   roi_multiple_x |   avg_contact_attempts |
|:---------------------------|--------------:|------------------------:|----------------------------:|--------------------:|--------------------------:|---------------------:|------------------------:|-----------------:|-----------------------:|
| Tele-Calling & Counseling  |          1186 |                     588 |                         283 |                   0 |               2.02418e+07 |               130878 |             2.0111e+07  |           154.66 |                    5.3 |
| AI Voice Bot & SMS         |           946 |                     460 |                         242 |                   0 |               1.79559e+07 |                52423 |             1.79035e+07 |           342.52 |                    4.8 |
| Field Visit & Face-to-Face |           722 |                     279 |                         154 |                  71 |               9.72293e+06 |               303766 |             9.41916e+06 |            32.01 |                   11.9 |
| Legal Notice & DRT         |           389 |                     133 |                          75 |                  84 |               6.87521e+06 |               353740 |             6.52147e+06 |            19.44 |                   18.7 |
| External Recovery Agency   |           298 |                      91 |                          42 |                  74 |               3.96152e+06 |               552018 |             3.40951e+06 |             7.18 |                   18.5 |
| Debt Restructuring         |           231 |                     111 |                          51 |                   0 |               3.07483e+06 |                54279 |             3.02055e+06 |            56.65 |                    6.5 |

---

## QUERY 7: Settlement Discount Tier Elasticity Analysis
**Business Purpose**: *Measure whether offering higher settlement discounts improves net dollars recovered.*

```sql
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
```

**Execution Output:**

| haircut_tier                  |   settlements_count |   avg_discount_pct |   total_recovered |   avg_recovered_per_account |   net_recovered_after_cost |
|:------------------------------|--------------------:|-------------------:|------------------:|----------------------------:|---------------------------:|
| 1% - 15% (Low Discount)       |                1662 |               7.53 |       4.80364e+07 |                     28902.7 |                4.72813e+07 |
| > 30% (Aggressive Haircut)    |                 442 |              37.48 |       7.02211e+06 |                     15887.1 |                6.88938e+06 |
| 16% - 30% (Moderate Discount) |                 405 |              22.62 |       6.77382e+06 |                     16725.5 |                6.61511e+06 |

---

## QUERY 8: Loss Given Default (LGD) and Collateral Protection
**Business Purpose**: *Quantify Loss Given Default (LGD) for secured vs unsecured loan portfolios.*

```sql
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
```

**Execution Output:**

| collateral_status   | loan_type      |   defaulted_loan_count |   total_ead_exposure |   total_recovered |    net_recovered |   loss_given_default_pct |
|:--------------------|:---------------|-----------------------:|---------------------:|------------------:|-----------------:|-------------------------:|
| Secured             | Home Loan      |                     23 |          2.10511e+06 |  907369           | 861700           |                    59.07 |
| Secured             | Auto Loan      |                     59 |     785568           |  416728           | 365115           |                    53.52 |
| Secured             | Business Loan  |                    301 |          1.93536e+07 |       1.15993e+07 |      1.11314e+07 |                    42.48 |
| Secured             | Education Loan |                      7 |     139553           |   95786.9         |  89375.6         |                    35.96 |
| Unsecured           | Education Loan |                     13 |     243872           |   66113.6         |  58139.7         |                    76.16 |
| Unsecured           | Personal Loan  |                    545 |          5.10147e+06 |       1.70962e+06 |      1.35783e+06 |                    73.38 |
| Unsecured           | Business Loan  |                    114 |          7.18479e+06 |       2.52654e+06 |      2.38716e+06 |                    66.77 |

---

## QUERY 9: Early Delinquency / Underwriting Vintage Flaw Identifier
**Business Purpose**: *Detect loans that defaulted within first 6 installments (< 180 days from origination).*

```sql
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
```

**Execution Output:**

| origination_channel        | loan_type      |   total_originated |   early_default_count |   early_default_rate_pct |   early_default_loss_exposure |
|:---------------------------|:---------------|-------------------:|----------------------:|-------------------------:|------------------------------:|
| Mobile App                 | Business Loan  |                215 |                    26 |                    12.09 |                   1.92726e+06 |
| Tele-Sales                 | Business Loan  |                 92 |                     9 |                     9.78 |              626479           |
| Branch Network             | Business Loan  |                527 |                    47 |                     8.92 |                   3.52232e+06 |
| DSA / Direct Selling Agent | Business Loan  |                244 |                    20 |                     8.2  |                   1.54077e+06 |
| Digital Portal             | Business Loan  |                718 |                    51 |                     7.1  |                   4.21364e+06 |
| Digital Portal             | Personal Loan  |               1988 |                   114 |                     5.73 |                   1.19448e+06 |
| DSA / Direct Selling Agent | Personal Loan  |                876 |                    50 |                     5.71 |              538810           |
| Branch Network             | Personal Loan  |               1585 |                    87 |                     5.49 |              962337           |
| Mobile App                 | Personal Loan  |                585 |                    28 |                     4.79 |              232349           |
| Tele-Sales                 | Personal Loan  |                193 |                     9 |                     4.66 |               48214.8         |
| DSA / Direct Selling Agent | Auto Loan      |                592 |                     6 |                     1.01 |               60741.7         |
| DSA / Direct Selling Agent | Education Loan |                204 |                     2 |                     0.98 |               26766.5         |
| Mobile App                 | Auto Loan      |                475 |                     4 |                     0.84 |               68466.4         |
| Branch Network             | Auto Loan      |               1118 |                     8 |                     0.72 |              103197           |
| Digital Portal             | Home Loan      |               1144 |                     5 |                     0.44 |              364688           |
| DSA / Direct Selling Agent | Home Loan      |                479 |                     2 |                     0.42 |              113696           |
| Mobile App                 | Home Loan      |                349 |                     1 |                     0.29 |              289930           |
| Branch Network             | Education Loan |                356 |                     1 |                     0.28 |                6245.5         |
| Digital Portal             | Auto Loan      |               1437 |                     3 |                     0.21 |               24219.6         |
| Branch Network             | Home Loan      |                915 |                     1 |                     0.11 |               93852.7         |

---

## QUERY 10: Multi-CTE Customer 360 Risk & Over-Leveraged Profile
**Business Purpose**: *Aggregate customer total exposure across debts, DTI, and delinquency.*

```sql
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
```

**Execution Output:**

| credit_segment       |   customer_count |   avg_annual_income |   avg_bank_exposure |   avg_external_debts |   defaulted_customers |   default_propensity_pct |
|:---------------------|-----------------:|--------------------:|--------------------:|---------------------:|----------------------:|-------------------------:|
| Subprime / High Risk |             4471 |             43587.3 |             39993.9 |                  1.8 |                   846 |                    18.92 |
| Near Prime           |             4930 |             63419.1 |             51679.5 |                  1.8 |                   181 |                     3.67 |
| Prime                |             3910 |             82300.3 |             58396.5 |                  1.8 |                    32 |                     0.82 |
| Super Prime          |             1689 |            114583   |             69869.5 |                  1.8 |                     3 |                     0.18 |

---

## QUERY 11: Regional Delinquency Hotspots & Recovery Disparity
**Business Purpose**: *Evaluate collection efficacy across geographic operating regions.*

```sql
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
```

**Execution Output:**

| region     |   total_loans |   regional_default_rate_pct |   total_outstanding_balance |   delinquent_exposure |   total_recovered_amount |   regional_recovery_rate_pct |   recovery_efficiency_rank |
|:-----------|--------------:|----------------------------:|----------------------------:|----------------------:|-------------------------:|-----------------------------:|---------------------------:|
| South      |          3870 |                        6.98 |                 2.00475e+08 |           2.93624e+07 |              1.85523e+07 |                        63.18 |                          1 |
| Central    |          1082 |                        7.76 |                 5.72917e+07 |           8.71976e+06 |              5.44959e+06 |                        62.5  |                          2 |
| East       |          2378 |                        7.36 |                 1.18713e+08 |           1.53561e+07 |              9.15722e+06 |                        59.63 |                          3 |
| North      |          4287 |                        7.51 |                 2.28112e+08 |           2.87301e+07 |              1.65589e+07 |                        57.64 |                          4 |
| West       |          2955 |                        6.06 |                 1.5414e+08  |           1.85863e+07 |              1.06974e+07 |                        57.56 |                          5 |
| North-East |           428 |                        7.48 |                 2.12005e+07 |           2.96396e+06 |              1.41687e+06 |                        47.8  |                          6 |

---

## QUERY 12: Origination Channel Delinquency & Risk Discrepancy
**Business Purpose**: *Identify if third-party sourcing agents (DSAs) produce higher default risk.*

```sql
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
```

**Execution Output:**

| origination_channel        |   total_loans_booked |   total_volume_booked |   total_defaults |   channel_default_rate_pct |   avg_borrower_credit_score |   avg_booked_interest_rate |
|:---------------------------|---------------------:|----------------------:|-----------------:|---------------------------:|----------------------------:|---------------------------:|
| Branch Network             |                 4501 |           3.3941e+08  |              340 |                       7.55 |                         677 |                      12.42 |
| Digital Portal             |                 5708 |           4.24531e+08 |              406 |                       7.11 |                         678 |                      12.39 |
| Mobile App                 |                 1771 |           1.31896e+08 |              121 |                       6.83 |                         681 |                      12.33 |
| Tele-Sales                 |                  625 |           4.83072e+07 |               41 |                       6.56 |                         679 |                      12.18 |
| DSA / Direct Selling Agent |                 2395 |           1.72689e+08 |              154 |                       6.43 |                         680 |                      12.47 |

---

## QUERY 13: Optimal Recovery Time-Window Decay Analysis
**Business Purpose**: *Measure how recovery rates decay as overdue duration (DPD) increases.*

```sql
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
```

**Execution Output:**

| aging_band                     |   total_cases |   total_delinquent_balance |   total_recovered_amount |   recovery_success_rate_pct |   avg_recovered_per_case |
|:-------------------------------|--------------:|---------------------------:|-------------------------:|----------------------------:|-------------------------:|
| 31-60 Days (Early Delinquency) |          1699 |                4.85645e+07 |              3.19306e+07 |                       65.75 |                  18793.8 |
| 61-120 Days (Mid-Stage)        |          1078 |                2.23728e+07 |              1.40763e+07 |                       62.92 |                  13057.8 |
| 121-240 Days (Late-Stage)      |           290 |                9.6124e+06  |              4.96482e+06 |                       51.65 |                  17120.1 |
| 240+ Days (Severe Default)     |           705 |                2.31689e+07 |              1.08605e+07 |                       46.88 |                  15404.9 |

---

## QUERY 14: Promise-to-Pay (PTP) Conversion & Broken Promise Leakage
**Business Purpose**: *Analyze the financial impact of customer promise-to-pay commitments.*

```sql
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
```

**Execution Output:**

| primary_channel            |   promise_to_pay_kept |   account_count |   total_amount_recovered |   avg_amount_recovered |   avg_attempts_needed |
|:---------------------------|----------------------:|----------------:|-------------------------:|-----------------------:|----------------------:|
| AI Voice Bot & SMS         |                     1 |             365 |              8.11792e+06 |                22240.9 |                   4.8 |
| AI Voice Bot & SMS         |                     0 |             581 |              9.83803e+06 |                16932.9 |                   4.7 |
| Debt Restructuring         |                     1 |              76 |              1.4138e+06  |                18602.6 |                   6.4 |
| Debt Restructuring         |                     0 |             155 |              1.66103e+06 |                10716.3 |                   6.5 |
| External Recovery Agency   |                     1 |              98 |              1.93352e+06 |                19729.8 |                  18.6 |
| External Recovery Agency   |                     0 |             200 |              2.02801e+06 |                10140   |                  18.5 |
| Field Visit & Face-to-Face |                     1 |             278 |              4.29812e+06 |                15460.9 |                  12.1 |
| Field Visit & Face-to-Face |                     0 |             444 |              5.4248e+06  |                12218   |                  11.7 |
| Legal Notice & DRT         |                     1 |             140 |              3.61891e+06 |                25849.3 |                  18.6 |
| Legal Notice & DRT         |                     0 |             249 |              3.2563e+06  |                13077.5 |                  18.8 |
| Tele-Calling & Counseling  |                     1 |             469 |              1.02159e+07 |                21782.3 |                   5.3 |
| Tele-Calling & Counseling  |                     0 |             717 |              1.00259e+07 |                13983.2 |                   5.3 |

---

## QUERY 15: Prioritized Legal Action & Asset Repossession Candidate Pipeline
**Business Purpose**: *Generate actionable high-priority accounts for legal escalation & repossession.*

```sql
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
```

**Execution Output:**

| loan_id   | customer_id   | region     | loan_type     | collateral_type      |   collateral_value |   outstanding_balance |   overdue_days |   contact_attempts | recovery_status   | recommended_legal_action          |
|:----------|:--------------|:-----------|:--------------|:---------------------|-------------------:|----------------------:|---------------:|-------------------:|:------------------|:----------------------------------|
| LN-213732 | CUST-113732   | Central    | Home Loan     | Residential Property |             643892 |                397571 |            181 |                 11 | In Progress       | Immediate Collateral Repossession |
| LN-212917 | CUST-112917   | West       | Home Loan     | Residential Property |             357025 |                289930 |            200 |                 11 | In Progress       | Immediate Collateral Repossession |
| LN-209975 | CUST-109975   | West       | Business Loan | Commercial Property  |             289782 |                189420 |            527 |                 27 | Written Off       | Immediate Collateral Repossession |
| LN-213952 | CUST-113952   | East       | Business Loan | Inventory            |             273265 |                184004 |            357 |                 19 | In Progress       | Immediate Collateral Repossession |
| LN-213143 | CUST-113143   | North      | Business Loan | Equipment            |             274462 |                163443 |            507 |                 27 | Written Off       | Immediate Collateral Repossession |
| LN-213090 | CUST-113090   | West       | Business Loan | Commercial Property  |             255778 |                155403 |            218 |                 11 | In Progress       | Immediate Collateral Repossession |
| LN-210229 | CUST-110229   | North      | Business Loan |                      |                  0 |                145121 |            171 |                 12 | In Progress       | High-Court / DRT Legal Summons    |
| LN-203342 | CUST-103342   | West       | Business Loan |                      |                  0 |                126971 |            129 |                 10 | In Progress       | High-Court / DRT Legal Summons    |
| LN-205292 | CUST-105292   | North-East | Business Loan |                      |                  0 |                124994 |            379 |                 18 | Written Off       | High-Court / DRT Legal Summons    |
| LN-202992 | CUST-102992   | North      | Business Loan |                      |                  0 |                117385 |            125 |                  9 | Legal Escalation  | High-Court / DRT Legal Summons    |
| LN-212914 | CUST-112914   | East       | Business Loan |                      |                  0 |                115577 |            471 |                 23 | Written Off       | High-Court / DRT Legal Summons    |
| LN-208672 | CUST-108672   | West       | Business Loan | Inventory            |             159312 |                115339 |            527 |                 27 | Written Off       | Immediate Collateral Repossession |
| LN-209115 | CUST-109115   | West       | Business Loan |                      |                  0 |                113686 |            129 |                  7 | In Progress       | High-Court / DRT Legal Summons    |
| LN-214639 | CUST-114639   | South      | Business Loan |                      |                  0 |                113397 |            446 |                 28 | Written Off       | High-Court / DRT Legal Summons    |
| LN-204745 | CUST-104745   | West       | Business Loan | Commercial Property  |             175046 |                112672 |            219 |                 13 | In Progress       | Immediate Collateral Repossession |
| LN-213157 | CUST-113157   | South      | Business Loan |                      |                  0 |                111372 |            230 |                 14 | In Progress       | High-Court / DRT Legal Summons    |
| LN-208437 | CUST-108437   | West       | Business Loan |                      |                  0 |                108681 |            519 |                 27 | Written Off       | High-Court / DRT Legal Summons    |
| LN-204815 | CUST-104815   | South      | Business Loan |                      |                  0 |                108116 |            106 |                  5 | In Progress       | High-Court / DRT Legal Summons    |
| LN-206035 | CUST-106035   | South      | Business Loan |                      |                  0 |                108116 |             93 |                  6 | In Progress       | High-Court / DRT Legal Summons    |
| LN-214138 | CUST-114138   | Central    | Business Loan | Inventory            |             165248 |                105258 |            296 |                 16 | In Progress       | Immediate Collateral Repossession |

---
