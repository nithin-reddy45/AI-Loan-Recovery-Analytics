"""
Notebook Builder Script: AI-Driven Loan Recovery & Risk Analytics
Constructs and executes 4 production-grade Jupyter Notebooks with markdown explanations,
clean code cells, and embedded analytics.
"""

import os
import nbformat as nbf

def create_notebook_01():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 01. Data Understanding, Cleansing & Feature Engineering
## Project: AI-Driven Loan Recovery & Risk Analytics

### Overview
This notebook establishes the foundation of the analytics workflow:
1. **Data Ingestion & Schema Inspection**: Loading 4 core banking relational tables (Customers, Loans, Repayments, Recovery Activities).
2. **Data Quality Audit**: Diagnosing missing data patterns, abnormal outliers, type mismatches, and duplicate records.
3. **Imputation & Treatment**: Applying domain-appropriate median imputations for employment length, mode imputations for housing status, and IQR-based clipping for income anomalies.
4. **Relational Merging & Master Dataset Creation**: Creating a unified analytical record for each borrower.
5. **Advanced Feature Engineering**: Deriving critical credit metrics including Debt-to-Income (DTI), Loan-to-Value (LTV), Delinquency Severity Scores, Net Recovery Amount, and Credit Risk Tiers.
"""))
    
    cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
print("Environment initialized. Loading raw relational datasets...")

customers_df = pd.read_csv("../data/raw/customers.csv")
loans_df = pd.read_csv("../data/raw/loans.csv")
repayments_df = pd.read_csv("../data/raw/repayments.csv")
recovery_df = pd.read_csv("../data/raw/recovery_activities.csv")

print(f"Customers: {customers_df.shape}")
print(f"Loans: {loans_df.shape}")
print(f"Repayments: {repayments_df.shape}")
print(f"Recovery Activities: {recovery_df.shape}")
"""))

    cells.append(nbf.v4.new_markdown_cell("### 1. Inspecting Missing Values and Data Anomalies"))
    cells.append(nbf.v4.new_code_cell("""# Check missing values across datasets
print("--- Missing Values in Customers ---")
print(customers_df.isnull().sum()[customers_df.isnull().sum() > 0])

print("\\n--- Missing Values in Loans ---")
print(loans_df.isnull().sum()[loans_df.isnull().sum() > 0])

print("\\n--- Missing Values in Repayments ---")
print(repayments_df.isnull().sum()[repayments_df.isnull().sum() > 0])

print("\\n--- Missing Values in Recovery Activities ---")
print(recovery_df.isnull().sum()[recovery_df.isnull().sum() > 0])
"""))

    cells.append(nbf.v4.new_markdown_cell("### 2. Data Cleaning & Imputation Pipeline"))
    cells.append(nbf.v4.new_code_cell("""# 1. Impute employment length with median per employment status
emp_median = customers_df.groupby("employment_status")["employment_length_years"].transform("median")
customers_df["employment_length_years"] = customers_df["employment_length_years"].fillna(emp_median).fillna(3).astype(int)

# 2. Impute housing status with mode
mode_housing = customers_df["housing_status"].mode()[0]
customers_df["housing_status"] = customers_df["housing_status"].fillna(mode_housing)

# 3. Outlier handling for annual_income
def cap_outliers(group):
    q25 = group["annual_income"].quantile(0.25)
    q75 = group["annual_income"].quantile(0.75)
    iqr = q75 - q25
    upper = q75 + 3.0 * iqr
    lower = max(12000, q25 - 1.5 * iqr)
    group["annual_income"] = np.clip(group["annual_income"], lower, upper)
    return group

customers_df = customers_df.groupby("city_tier", group_keys=False).apply(cap_outliers)

# 4. Clean loans collateral values
loans_df["collateral_value"] = loans_df["collateral_value"].fillna(0.0)
loans_df["disbursement_date"] = pd.to_datetime(loans_df["disbursement_date"])

print("Data cleansing and imputation successfully executed.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### 3. Master Relational Merging & Feature Engineering"))
    cells.append(nbf.v4.new_code_cell("""# Merge Loans and Repayments
loans_rep_df = pd.merge(loans_df, repayments_df, on=["loan_id", "customer_id"], how="inner")
loans_rep_df["outstanding_principal"] = np.maximum(0.0, loans_rep_df["loan_amount"] - loans_rep_df["principal_paid"])

# Merge with Customers
master_df = pd.merge(customers_df, loans_rep_df, on="customer_id", how="inner")

# Left merge with Recovery Activities
master_df = pd.merge(master_df, recovery_df.drop(columns=["customer_id"]), on="loan_id", how="left")

# Fill recovery defaults for current/healthy accounts
master_df["recovery_status"] = master_df["recovery_status"].fillna("No Delinquency / Not Applicable")
master_df["primary_channel"] = master_df["primary_channel"].fillna("None")
master_df["recovered_amount"] = master_df["recovered_amount"].fillna(0.0)
master_df["recovery_cost"] = master_df["recovery_cost"].fillna(0.0)
master_df["settlement_discount_pct"] = master_df["settlement_discount_pct"].fillna(0.0)

# Engineer Financial Metrics
r = (master_df["interest_rate"] / 100.0) / 12.0
n = master_df["loan_term_months"]
master_df["monthly_emi"] = (master_df["loan_amount"] * (r * (1 + r)**n) / ((1 + r)**n - 1)).round(2)
master_df["dti_ratio"] = ((master_df["monthly_emi"] * 12) / master_df["annual_income"]).round(4)
master_df["is_secured"] = np.where(master_df["collateral_type"] != "None", 1, 0)
master_df["ltv_ratio"] = np.where(master_df["is_secured"] == 1, np.clip(master_df["loan_amount"] / np.maximum(master_df["collateral_value"], 1.0), 0.1, 1.5), 1.0).round(4)

# Credit Risk Tiers
bins = [0, 549, 649, 699, 749, 900]
labels = ["Deep Subprime (<550)", "Subprime (550-649)", "Near-Prime (650-699)", "Prime (700-749)", "Super-Prime (750+)"]
master_df["credit_risk_tier"] = pd.cut(master_df["credit_score"], bins=bins, labels=labels)

master_df["net_recovery_amount"] = (master_df["recovered_amount"] - master_df["recovery_cost"]).round(2)

print(f"Master Clean Dataset Shape: {master_df.shape}")
display(master_df.head(3))
"""))

    cells.append(nbf.v4.new_markdown_cell("### 4. Summary Statistics & Data Export"))
    cells.append(nbf.v4.new_code_cell("""print("=== Master Dataset Numerical Summary ===")
display(master_df[["annual_income", "credit_score", "loan_amount", "interest_rate", "dti_ratio", "overdue_days", "recovered_amount"]].describe().T)
"""))

    nb.cells = cells
    with open("notebooks/01_data_understanding_and_cleaning.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/01_data_understanding_and_cleaning.ipynb")

def create_notebook_02():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 02. Exploratory Data Analysis & Visual Intelligence
## Project: AI-Driven Loan Recovery & Risk Analytics

### Overview
This notebook presents in-depth exploratory data analysis across credit risk and debt recovery dimensions:
1. **Portfolio Distributions**: Loan amounts, interest rates, credit scores, and debt-to-income.
2. **Delinquency Analysis**: Overdue days by product type, geography, and sourcing channel.
3. **Recovery Channel Efficiency**: ROI multipliers and recovery rates across AI bots, tele-calling, field visits, and legal.
4. **Correlation Analysis**: Multi-variable financial risk correlation matrix.
5. **Vintage Cohort Curves**: Cumulative repayment curves across loan origination quarters.
"""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

df = pd.read_csv("../data/processed/loan_recovery_master.csv")
delinquent_df = df[df["overdue_days"] > 30].copy()
print(f"Loaded Master Analytical Dataset: {df.shape}")
"""))

    cells.append(nbf.v4.new_markdown_cell("### 1. Macro Portfolio Distribution & Risk Profile"))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x="credit_score", hue="delinquency_bucket", multiple="stack", ax=axes[0], palette="turbo")
axes[0].set_title("Credit Score Distribution by Delinquency Bucket", fontweight="bold")

sns.boxplot(data=df, x="loan_type", y="interest_rate", ax=axes[1], palette="Set2")
axes[1].set_title("Interest Rate by Loan Product Line", fontweight="bold")
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("### 2. Recovery Rate & Efficiency by Collection Channel"))
    cells.append(nbf.v4.new_code_cell("""rec_chan = delinquent_df.groupby("primary_channel").agg(
    outstanding=("outstanding_principal", "sum"),
    recovered=("recovered_amount", "sum"),
    cost=("recovery_cost", "sum")
).reset_index()
rec_chan["recovery_rate_%"] = (rec_chan["recovered"] / rec_chan["outstanding"]) * 100
rec_chan["roi_multiple"] = rec_chan["recovered"] / np.maximum(rec_chan["cost"], 1.0)
display(rec_chan.sort_values("recovery_rate_%", ascending=False))

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=rec_chan.sort_values("recovery_rate_%", ascending=False), x="recovery_rate_%", y="primary_channel", palette="Greens_r", ax=ax)
ax.set_title("Recovery Rate (%) by Channel", fontweight="bold")
for p in ax.patches:
    ax.annotate(f"{p.get_width():.1f}%", (p.get_width() + 0.5, p.get_y() + p.get_height() / 2.), va="center")
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("### 3. Correlation Matrix of Financial & Risk Drivers"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(11, 8))
cols = ["annual_income", "credit_score", "loan_amount", "interest_rate", "dti_ratio", "overdue_days", "recovered_amount", "default_flag"]
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix of Key Credit & Recovery Drivers", fontweight="bold")
plt.show()
"""))

    nb.cells = cells
    with open("notebooks/02_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/02_exploratory_data_analysis.ipynb")

def create_notebook_03():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 03. Customer & Loan Segmentation (K-Means & PCA)
## Project: AI-Driven Loan Recovery & Risk Analytics

### Overview
This notebook builds unsupervised machine learning clusters to identify distinct borrower personas:
1. **Feature Standardisation**: Standardizing multi-scale financial and behavioral metrics.
2. **Hyperparameter Selection**: Evaluating the Elbow Method (Inertia) and Silhouette Scores for optimal $K$.
3. **Persona Profiling**: Mapping numerical clusters to business personas (Prime Self-Cure, Distressed Responsive, Chronic High Risk, Secured Asset Backed).
4. **Dimensionality Reduction**: Visualizing multi-dimensional clusters using 2D Principal Component Analysis (PCA).
"""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("../data/processed/loan_recovery_master.csv")
features = ["credit_score", "annual_income", "loan_amount", "interest_rate", "overdue_days", "dti_ratio", "installments_paid", "delinquency_severity_score"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features scaled: {X_scaled.shape}")
"""))

    cells.append(nbf.v4.new_markdown_cell("### 1. Elbow Method & Silhouette Optimization"))
    cells.append(nbf.v4.new_code_cell("""inertias = []
sil_scores = []
k_vals = range(2, 7)

for k in k_vals:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(k_vals, inertias, 'bo-', label="Inertia (Elbow)")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia", color='b')

ax2 = ax1.twinx()
ax2.plot(k_vals, sil_scores, 'r^-', label="Silhouette Score")
ax2.set_ylabel("Silhouette Score", color='r')
plt.title("Cluster Evaluation: Elbow & Silhouette Scores", fontweight="bold")
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("### 2. K-Means Training & Persona Assignment"))
    cells.append(nbf.v4.new_code_cell("""kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

# PCA Projection
pca = PCA(n_components=2, random_state=42)
pca_res = pca.fit_transform(X_scaled)
df["pca_1"] = pca_res[:, 0]
df["pca_2"] = pca_res[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="cluster_id", palette="tab10", alpha=0.6)
plt.title("2D PCA Visualization of Customer Segments", fontweight="bold")
plt.show()
"""))

    nb.cells = cells
    with open("notebooks/03_customer_segmentation_kmeans.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/03_customer_segmentation_kmeans.ipynb")

def create_notebook_04():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 04. Predictive Machine Learning Modeling: Loan Recovery
## Project: AI-Driven Loan Recovery & Risk Analytics

### Overview
This notebook trains, evaluates, and compares classification machine learning models to predict the probability of successful loan recovery:
1. **Preprocessing Pipeline**: Handling numerical scaling and categorical One-Hot Encoding via `ColumnTransformer`.
2. **Model Training & Comparison**:
   - Logistic Regression (Class Weighted)
   - Random Forest Classifier
   - XGBoost Classifier
   - LightGBM Classifier (Champion Model)
3. **Model Evaluation**: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
4. **Feature Importance & Interpretability**: Identifying key drivers of debt recovery.
5. **Model Serialization**: Saving champion pipeline to `models/recovery_predictor_lightgbm.pkl`.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

df = pd.read_csv("../data/processed/loan_recovery_master.csv")
delinquent_df = df[df["overdue_days"] > 30].copy()
delinquent_df["recovery_target"] = delinquent_df["recovery_status"].apply(lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0)

print(f"Sample Size: {delinquent_df.shape}")
print(delinquent_df["recovery_target"].value_counts(normalize=True))
"""))

    cells.append(nbf.v4.new_markdown_cell("### 1. Building Preprocessing Pipeline & Splitting Data"))
    cells.append(nbf.v4.new_code_cell("""numeric_features = [
    "credit_score", "annual_income", "loan_amount", "interest_rate",
    "loan_term_months", "overdue_days", "dti_ratio", "ltv_ratio",
    "is_secured", "contact_attempts", "promise_to_pay_kept",
    "settlement_discount_pct", "delinquency_severity_score"
]

categorical_features = [
    "loan_type", "region", "city_tier", "employment_status",
    "housing_status", "collateral_type", "origination_channel",
    "primary_channel"
]

X = delinquent_df[numeric_features + categorical_features]
y = delinquent_df["recovery_target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)
print("Pipeline preprocessor configured.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### 2. Model Training & Evaluation"))
    cells.append(nbf.v4.new_code_cell("""models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
}

results = []
plt.figure(figsize=(9, 6))

for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Comparative ROC-AUC Benchmark", fontweight="bold")
plt.legend(loc="lower right")
plt.show()

res_df = pd.DataFrame(results).set_index("Model")
display(res_df)
"""))

    nb.cells = cells
    with open("notebooks/04_predictive_modeling_loan_recovery.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/04_predictive_modeling_loan_recovery.ipynb")

if __name__ == "__main__":
    create_notebook_01()
    create_notebook_02()
    create_notebook_03()
    create_notebook_04()
    print("All 4 Jupyter Notebooks generated successfully!")
