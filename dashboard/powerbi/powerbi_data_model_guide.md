# Power BI Data Model Architecture Guide
## Project: AI-Driven Loan Recovery & Risk Analytics

---

## 1. Dimensional Architecture: Star Schema

The Power BI data model is architected following industry-standard Kimball dimensional modeling principles (Star Schema) to maximize DAX calculation speed, ensure clean filter propagation, and support row-level security (RLS).

```
                      +-------------------+
                      |   Dim_Date (CAL)  |
                      +-------------------+
                                | (1:*)
                                v
+------------------+  (1:*)  +--------------------+  (*:1)  +--------------------+
|   Dim_Customer   |-------->| Fact_LoanRecovery  |<--------|      Dim_Loan      |
+------------------+         +--------------------+         +--------------------+
                                ^             ^
                         (*:1)  |             | (*:1)
             +--------------------+         +--------------------+
             |    Dim_Channel     |         |   Dim_Collector    |
             +--------------------+         +--------------------+
```

---

## 2. Table Specifications & Keys

### 1. `Fact_LoanRecovery` (Central Fact Table)
- **Granularity**: 1 record per loan account with repayment & recovery status.
- **Key Columns**:
  - `Loan_Key` (FK -> `Dim_Loan.Loan_Key`)
  - `Customer_Key` (FK -> `Dim_Customer.Customer_Key`)
  - `Disbursal_Date_Key` (FK -> `Dim_Date.Date`)
  - `Recovery_Date_Key` (FK -> `Dim_Date.Date`)
  - `Channel_Key` (FK -> `Dim_Channel.Channel_Key`)
  - `Collector_Key` (FK -> `Dim_Collector.Collector_Key`)
- **Measures / Additive Metrics**:
  - `Loan_Amount`
  - `Principal_Paid`
  - `Interest_Paid`
  - `Outstanding_Principal`
  - `Recovered_Amount`
  - `Recovery_Cost`
  - `Net_Recovery_Amount`
  - `Overdue_Days`
  - `Contact_Attempts`

### 2. `Dim_Customer`
- **Granularity**: 1 row per unique borrower.
- **Attributes**: `Customer_ID`, `Age`, `Age_Group`, `Gender`, `Region`, `City_Tier`, `Employment_Status`, `Employment_Length_Years`, `Annual_Income`, `Income_Bracket`, `Credit_Score`, `Credit_Risk_Tier`, `Housing_Status`, `Marital_Status`, `Customer_Segment_Persona`.

### 3. `Dim_Loan`
- **Granularity**: 1 row per loan agreement.
- **Attributes**: `Loan_ID`, `Loan_Type`, `Interest_Rate`, `Rate_Tier`, `Loan_Term_Months`, `Collateral_Type`, `Collateral_Value`, `Is_Secured`, `Loan_Purpose`, `Origination_Channel`.

### 4. `Dim_Date` (Role-Playing Dimension)
- **Granularity**: Daily calendar dimension (2020 - 2026).
- **Attributes**: `Date`, `Year`, `Quarter`, `Month_Name`, `Month_Number`, `Year_Month`, `Fiscal_Quarter`, `Day_of_Week`, `Is_Weekend`.

### 5. `Dim_Channel`
- **Granularity**: 1 row per collection channel.
- **Attributes**: `Channel_Key`, `Primary_Channel`, `Channel_Type` (Digital / Manual / Legal), `Cost_Tier`.

### 6. `Dim_Collector`
- **Granularity**: 1 row per recovery agent / officer.
- **Attributes**: `Collector_ID`, `Officer_Name`, `Officer_Designation`, `Specialization_Team`, `Experience_Level`.

---

## 3. Relationships & Filter Propagation Rules

| From Table | To Table | From Column | To Column | Cardinality | Cross Filter Direction | Security Filtering |
|:---|:---|:---|:---|:---:|:---:|:---:|
| `Dim_Customer` | `Fact_LoanRecovery` | `Customer_ID` | `Customer_ID` | 1 to Many (1:*) | Single | Yes |
| `Dim_Loan` | `Fact_LoanRecovery` | `Loan_ID` | `Loan_ID` | 1 to Many (1:*) | Single | No |
| `Dim_Date` | `Fact_LoanRecovery` | `Date` | `Disbursement_Date` | 1 to Many (1:*) | Single (Active) | No |
| `Dim_Date` | `Fact_LoanRecovery` | `Date` | `Recovery_Date` | 1 to Many (1:*) | Inactive (`USERELATIONSHIP`) | No |
| `Dim_Channel` | `Fact_LoanRecovery` | `Primary_Channel` | `Primary_Channel` | 1 to Many (1:*) | Single | No |
| `Dim_Collector` | `Fact_LoanRecovery` | `Recovery_Officer_ID`| `Recovery_Officer_ID`| 1 to Many (1:*) | Single | Yes |

---

## 4. Row-Level Security (RLS) Implementation

To ensure compliance with banking data privacy regulations:
1. **Regional Officers Role**:
   ```dax
   [Region] = USERNAME() || [Region] IN CALCULATETABLE(VALUES(UserRegionMap[Region]), UserRegionMap[UserEmail] = USERPRINCIPALNAME())
   ```
2. **Collection Officer Role**:
   ```dax
   [Recovery_Officer_ID] = LOOKUPVALUE(Dim_Collector[Collector_ID], Dim_Collector[UserEmail], USERPRINCIPALNAME())
   ```
