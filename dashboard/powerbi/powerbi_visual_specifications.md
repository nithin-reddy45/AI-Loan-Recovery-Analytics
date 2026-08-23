# Power BI Dashboard UI/UX & Visual Layout Specifications
## Project: AI-Driven Loan Recovery & Risk Analytics

This document provides visual blueprint specifications for creating a 4-page Power BI report.

---

## Global Report Layout & Styling System
- **Theme**: Executive Modern Fintech Dark/Light Hybrid (`#0f172a` Navy Slate header, `#f8fafc` Canvas background, `#0284c7` Primary Blue, `#10b981` Emerald Green, `#ef4444` Crimson Alert).
- **Canvas Resolution**: 16:9 (1920 x 1080 px).
- **Typography**: DIN / Segoe UI Bold for KPI Callouts, Segoe UI for Data Labels.
- **Global Top Navigation Banner**: Slicers for `Region`, `Loan Type`, `Delinquency Stage`, `Date Range`.

---

## Page 1: Executive Portfolio & Recovery Command Center

### Layout Grid:
- **Top Row (Cards)**:
  1. `Total Disbursed Volume` ($1.12B) with sparkline trend.
  2. `Total Outstanding Book` ($779.9M) with % of total loan portfolio.
  3. `Total Recovered Amount` ($61.8M) with Net Yield indicator.
  4. `Portfolio Recovery Rate` (59.6%) with gauge visual (Target: 65%).
  5. `Gross NPA / Default Rate` (7.08%) with red threshold alert tag.
- **Middle Left**: **Monthly Recovery & Cost Trajectory** (Line and Clustered Column Chart).
  - *X-Axis*: `Year_Month`
  - *Column Y-Axis*: `[Total Recovered Amount]`, `[Total Recovery Cost]`
  - *Line Y-Axis*: `[MoM Recovery Growth %]`
- **Middle Right**: **Recovery Share by Product Type** (Donut Chart with Center KPI).
  - *Legend*: `Loan_Type`
  - *Values*: `[Total Recovered Amount]`
  - *Tooltips*: `[Average Loan Size]`, `[Default Rate %]`
- **Bottom Left**: **Regional Exposure & Recovery Geo Map / Treemap**.
  - *Category*: `Region` > `City_Tier`
  - *Size*: `[Total Outstanding Principal]`
  - *Color Saturation*: `[Recovery Rate %]`
- **Bottom Right**: **Origination Sourcing Channel Risk Scatter**.
  - *X-Axis*: `Average Credit Score`
  - *Y-Axis*: `Default Rate %`
  - *Bubble Size*: `Total Loan Volume`

---

## Page 2: Delinquency Roll-Forward & Credit Risk Migration

### Visual Elements:
1. **Delinquency Waterfall / Migration Matrix**:
   - *Columns*: Current -> SMA-0 (1-30 DPD) -> SMA-1 (31-60 DPD) -> SMA-2 (61-90 DPD) -> NPA (90+ DPD).
   - *Values*: Account Count, Outstanding Dollar Exposure, Delinquency Share %.
2. **Credit Risk Tier vs DTI Matrix (Decomposition Tree)**:
   - *Analyze*: `[NPA Exposure Amount]`
   - *Explain by*: `Credit_Risk_Tier` -> `Employment_Status` -> `Collateral_Type`.
3. **Overdue Severity Distribution (Violin / Box Plot Visual)**:
   - *Category*: `Loan_Type`
   - *Metric*: `Overdue_Days`
   - *Color*: `Is_Secured`

---

## Page 3: Recovery Channel Performance & Agent Leaderboard

### Visual Elements:
1. **Channel Cost-Benefit & ROI Multiplier (Bar & Ribbon Chart)**:
   - *Y-Axis*: Collection Channels (AI Voice Bot, Tele-calling, Field Visits, Legal/DRT, External Agencies).
   - *X-Axis*: `[Channel ROI Multiplier]` and `[Recovery Rate %]`.
2. **Settlement Discount Elasticity Matrix**:
   - *X-Axis*: Settlement Discount Buckets (0%, 1-15%, 16-30%, >30%).
   - *Y-Axis*: Total Dollars Recovered & Average Time to Resolve (Days).
3. **Collector Officer Productivity Matrix (Interactive Table / Matrix)**:
   - *Columns*: `Officer_Name`, `Designation`, `Assigned_Cases`, `Resolved_Cases`, `Resolution_Rate_%`, `Total_Gross_Recovered`, `Net_Recovery_Yield`, `Rank`.
   - *Conditional Formatting*: Green for Top 20% Net Recovery, Red for Low Resolution.

---

## Page 4: AI Customer Personas & What-If Settlement Optimizer

### Visual Elements:
1. **Cluster Segmentation Scatter (Python / R Visual or PCA Chart)**:
   - Visualizing the 4 AI Customer Personas across PCA coordinates.
2. **What-If Haircut & Settlement Parameter Slider**:
   - Numeric Range Slicer: 0% to 50% Haircut.
   - Dynamic DAX updates showing simulated portfolio recovery cash inflow vs capital write-off.
3. **Prioritized Action Pipeline for High-Balance Accounts**:
   - Filtered list of Top 50 delinquent accounts sorted by `Outstanding Balance DESC` with AI Recommended Action tag (Legal Summons, SARFAESI Repossession, Tele-counseling).
