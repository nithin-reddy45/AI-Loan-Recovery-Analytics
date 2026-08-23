# Production DAX Formula Bank & KPI Repository
## Project: AI-Driven Loan Recovery & Risk Analytics

This repository contains 25+ production-grade DAX measures organized into 6 business analytical categories.

---

## Category 1: Portfolio Core Volume & Exposure Measures

### 1. Total Disbursed Loan Amount
```dax
Total Disbursed Amount = 
SUM(Fact_LoanRecovery[loan_amount])
```

### 2. Total Outstanding Principal Balance
```dax
Total Outstanding Principal = 
SUM(Fact_LoanRecovery[outstanding_principal])
```

### 3. Total Disbursed Accounts
```dax
Total Loans Count = 
DISTINCTCOUNT(Fact_LoanRecovery[loan_id])
```

### 4. Average Ticket Size
```dax
Average Loan Size = 
DIVIDE([Total Disbursed Amount], [Total Loans Count], 0)
```

---

## Category 2: Recovery Performance & Efficiency Measures

### 5. Total Gross Recovered Amount
```dax
Total Recovered Amount = 
SUM(Fact_LoanRecovery[recovered_amount])
```

### 6. Total Direct Collection Costs
```dax
Total Recovery Cost = 
SUM(Fact_LoanRecovery[recovery_cost])
```

### 7. Net Recovery Yield (Post-Cost)
```dax
Net Recovery Yield = 
[Total Recovered Amount] - [Total Recovery Cost]
```

### 8. Portfolio Recovery Rate (%)
```dax
Recovery Rate % = 
VAR DelinquentOutstanding = 
    CALCULATE(
        [Total Outstanding Principal],
        Fact_LoanRecovery[overdue_days] > 30
    )
RETURN
    DIVIDE([Total Recovered Amount], DelinquentOutstanding, 0)
```

### 9. Recovery Cost-to-Collect Ratio (Cost per Dollar Recovered)
```dax
Cost To Collect Ratio = 
DIVIDE([Total Recovery Cost], [Total Recovered Amount], 0)
```

### 10. Channel ROI Multiplier
```dax
Channel ROI Multiplier = 
DIVIDE([Total Recovered Amount], [Total Recovery Cost], 0)
```

---

## Category 3: Delinquency & Credit Risk Metrics

### 11. Default Rate (NPA %)
```dax
Default Rate % = 
VAR TotalDefaults = 
    CALCULATE(
        COUNTROWS(Fact_LoanRecovery),
        Fact_LoanRecovery[default_flag] = 1
    )
RETURN
    DIVIDE(TotalDefaults, [Total Loans Count], 0)
```

### 12. Portfolio Overdue Rate (%)
```dax
Overdue Rate % = 
VAR OverdueAccounts = 
    CALCULATE(
        COUNTROWS(Fact_LoanRecovery),
        Fact_LoanRecovery[overdue_days] > 0
    )
RETURN
    DIVIDE(OverdueAccounts, [Total Loans Count], 0)
```

### 13. Non-Performing Asset (NPA) Volume ($)
```dax
NPA Exposure Amount = 
CALCULATE(
    [Total Outstanding Principal],
    Fact_LoanRecovery[delinquency_bucket] = "90+ DPD (NPA / Default)"
)
```

### 14. Special Mention Accounts (SMA-1 & SMA-2) Exposure
```dax
Early Warning SMA Exposure = 
CALCULATE(
    [Total Outstanding Principal],
    Fact_LoanRecovery[delinquency_bucket] IN {"31-60 DPD (SMA-1)", "61-90 DPD (SMA-2)"}
)
```

### 15. Loss Given Default (LGD %)
```dax
Loss Given Default % = 
VAR DefaultExposure = 
    CALCULATE(
        [Total Outstanding Principal],
        Fact_LoanRecovery[default_flag] = 1
    )
VAR NetRecoveredOnDefaults = 
    CALCULATE(
        [Net Recovery Yield],
        Fact_LoanRecovery[default_flag] = 1
    )
RETURN
    1 - DIVIDE(NetRecoveredOnDefaults, DefaultExposure, 0)
```

---

## Category 4: Time Intelligence & Cohort Trend Measures

### 16. Monthly Recovery (Active Disbursal Date)
```dax
Monthly Disbursed Amount = 
CALCULATE(
    [Total Disbursed Amount],
    DATESMTD(Dim_Date[Date])
)
```

### 17. Monthly Recovered Amount (Using Inactive Relationship on Recovery Date)
```dax
Monthly Recovered by Recovery Date = 
CALCULATE(
    [Total Recovered Amount],
    USERELATIONSHIP(Fact_LoanRecovery[recovery_date], Dim_Date[Date]),
    DATESMTD(Dim_Date[Date])
)
```

### 18. Month-over-Month (MoM) Recovery Growth (%)
```dax
MoM Recovery Growth % = 
VAR CurrentMonth = [Monthly Recovered by Recovery Date]
VAR PriorMonth = 
    CALCULATE(
        [Monthly Recovered by Recovery Date],
        DATEADD(Dim_Date[Date], -1, MONTH)
    )
RETURN
    DIVIDE(CurrentMonth - PriorMonth, PriorMonth, 0)
```

### 19. Year-to-Date (YTD) Recovery
```dax
YTD Recovered Amount = 
CALCULATE(
    [Total Recovered Amount],
    USERELATIONSHIP(Fact_LoanRecovery[recovery_date], Dim_Date[Date]),
    DATESYTD(Dim_Date[Date])
)
```

### 20. Rolling 3-Month Average Recovery
```dax
Rolling 3M Avg Recovery = 
CALCULATE(
    [Total Recovered Amount],
    USERELATIONSHIP(Fact_LoanRecovery[recovery_date], Dim_Date[Date]),
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -3, MONTH)
) / 3
```

---

## Category 5: Agent Productivity & Collector Ranking

### 21. Resolved Cases Count
```dax
Resolved Cases = 
CALCULATE(
    COUNTROWS(Fact_LoanRecovery),
    Fact_LoanRecovery[recovery_status] IN {"Fully Recovered", "Partially Recovered"}
)
```

### 22. Collector Case Resolution Rate (%)
```dax
Collector Resolution Rate % = 
DIVIDE([Resolved Cases], COUNTROWS(Fact_LoanRecovery), 0)
```

### 23. Officer Efficiency Score (Index 0-100)
```dax
Officer Efficiency Index = 
VAR NetRecScore = NORMALIZE([Net Recovery Yield])
VAR ResRate = [Collector Resolution Rate %]
VAR AvgAtt = AVERAGE(Fact_LoanRecovery[contact_attempts])
RETURN
    (0.50 * [Collector Resolution Rate %] * 100) + 
    (0.30 * DIVIDE([Net Recovery Yield], 100000, 0)) - 
    (0.20 * AvgAtt)
```

---

## Category 6: What-If Parameter & Settlement Simulation

### 24. Simulated Net Recovery from Haircut Parameter
```dax
Simulated Recovery Amount = 
VAR TargetDiscount = SELECTEDVALUE('Settlement Parameter'[Discount %], 0.15)
VAR TargetEligibleBalance = 
    CALCULATE(
        [Total Outstanding Principal],
        Fact_LoanRecovery[overdue_days] > 60
    )
VAR EstimatedElasticityFactor = 1.25
RETURN
    TargetEligibleBalance * (1 - TargetDiscount) * EstimatedElasticityFactor * 0.65
```

### 25. High-Risk Capital Allocation Requirement (Basel III ECL)
```dax
Expected Credit Loss (ECL) = 
VAR PD = [Default Rate %]
VAR EAD = [Total Outstanding Principal]
VAR LGD = [Loss Given Default %]
RETURN
    PD * EAD * LGD
```
