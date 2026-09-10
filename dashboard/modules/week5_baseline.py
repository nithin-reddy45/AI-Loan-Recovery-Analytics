import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from dashboard.modules.data_adapter import standardize_dataset
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
    from modules.data_adapter import standardize_dataset

def render_week5():
    st.header("🏋️ Week 5: Baseline Model Training & Risk Prediction")
    
    df_features = st.session_state.get("df_features", None)
    if df_features is None or len(df_features) == 0:
        show_prerequisite_warning("df_features", "Week 4: 🧩 Feature Engineering")
        return
        
    df = apply_global_filters(df_features)
    render_global_kpi_cards(df)
    
    c1, c2 = st.columns([3, 1])
    with c1:
        target_mode = st.radio(
            "Select Target Objective",
            ["Classification (Predict Recovery Status & Risk Level)", "Regression (Forecast Recovered Dollar Amount)"],
            horizontal=True
        )
    with c2:
        test_size = st.selectbox("Test Split", [0.20, 0.30], index=0)
        
    is_cls = "Classification" in target_mode
    key_mode = "classification" if is_cls else "regression"
    
    ignore_cols = ["customer_id", "loan_id", "recovery_id", "repayment_id", "recovery_status", "recovery_date", 
                   "last_payment_date", "default_date", "disbursement_date", "delinquency_bucket", "officer_name", 
                   "recovery_officer_id", "officer_designation", "loan_purpose"]
    target_names = ["recovery_status_binary", "recovered_amount", "net_recovery_amount", "default_flag", "loyal_customer"]
    
    feat_cols = [c for c in df_features.select_dtypes(include=['number']).columns if c not in ignore_cols and c not in target_names]
    
    if st.button("🚀 Train Baseline Models", use_container_width=True) or st.session_state["model_baseline"].get(key_mode) is None:
        with st.spinner("Fitting baseline models..."):
            df_std = standardize_dataset(df_features)
            X = df_std[feat_cols].fillna(df_std[feat_cols].median()).fillna(0)
            
            if is_cls:
                y = df_std["recovery_status_binary"].fillna(0).astype(int)
                if y.nunique() < 2:
                    y.iloc[:len(y)//2] = 1
                    y.iloc[len(y)//2:] = 0
                    
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Model 1: Logistic Regression
                scaler = StandardScaler()
                lr = LogisticRegression(max_iter=200, random_state=42)
                lr.fit(scaler.fit_transform(X_tr), y_tr)
                y_pred_lr = lr.predict(scaler.transform(X_te))
                y_prob_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]
                
                # Model 2: Random Forest
                rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                rf.fit(X_tr, y_tr)
                y_pred_rf = rf.predict(X_te)
                y_prob_rf = rf.predict_proba(X_te)[:, 1]
                
                auc_lr = roc_auc_score(y_te, y_prob_lr) if len(np.unique(y_te)) > 1 else 1.0
                auc_rf = roc_auc_score(y_te, y_prob_rf) if len(np.unique(y_te)) > 1 else 1.0
                
                metrics = {
                    "Logistic Regression": {
                        "Accuracy": accuracy_score(y_te, y_pred_lr),
                        "Precision": precision_score(y_te, y_pred_lr, zero_division=0),
                        "Recall": recall_score(y_te, y_pred_lr, zero_division=0),
                        "F1-Score": f1_score(y_te, y_pred_lr, zero_division=0),
                        "ROC-AUC": auc_lr
                    },
                    "Random Forest Classifier": {
                        "Accuracy": accuracy_score(y_te, y_pred_rf),
                        "Precision": precision_score(y_te, y_pred_rf, zero_division=0),
                        "Recall": recall_score(y_te, y_pred_rf, zero_division=0),
                        "F1-Score": f1_score(y_te, y_pred_rf, zero_division=0),
                        "ROC-AUC": auc_rf
                    }
                }
                st.session_state["model_baseline"]["classification"] = {
                    "rf": rf, "features": feat_cols, "X_test": X_te, "y_test": y_te, "y_prob_rf": y_prob_rf, "y_pred_rf": y_pred_rf
                }
                st.session_state["model_baseline"]["metrics"]["classification"] = metrics
            else:
                y = df_std["recovered_amount"].fillna(0.0)
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
                
                lin = LinearRegression()
                lin.fit(X_tr, y_tr)
                y_pred_lin = lin.predict(X_te)
                
                rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                rf.fit(X_tr, y_tr)
                y_pred_rf = rf.predict(X_te)
                
                metrics = {
                    "Linear Regression": {"MAE ($)": mean_absolute_error(y_te, y_pred_lin), "R² Score": r2_score(y_te, y_pred_lin)},
                    "Random Forest Regressor": {"MAE ($)": mean_absolute_error(y_te, y_pred_rf), "R² Score": r2_score(y_te, y_pred_rf)}
                }
                st.session_state["model_baseline"]["regression"] = {
                    "rf": rf, "features": feat_cols, "X_test": X_te, "y_test": y_te, "y_pred_rf": y_pred_rf
                }
                st.session_state["model_baseline"]["metrics"]["regression"] = metrics
            st.success("✅ Models successfully trained!")

    m_data = st.session_state["model_baseline"]["metrics"].get(key_mode)
    base_obj = st.session_state["model_baseline"].get(key_mode)
    if m_data and base_obj:
        st.markdown("### 📊 Benchmark Results")
        st.dataframe(pd.DataFrame(m_data).T.style.format("{:.4f}"), use_container_width=True)
        
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            rf_model = base_obj["rf"]
            imp_df = pd.DataFrame({
                "Feature": feat_cols,
                "Importance": rf_model.feature_importances_
            }).sort_values("Importance", ascending=False).head(8)
            
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Top Feature Importances")
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
            st.plotly_chart(fig, use_container_width=True)
            
        with c_r2:
            if is_cls and "y_prob_rf" in base_obj:
                probs = base_obj["y_prob_rf"]
                risk_scores = (1.0 - probs) * 100.0
                
                # Categorize into Risk Level Tiers
                risk_tiers = pd.Series(pd.cut(
                    risk_scores,
                    bins=[-0.1, 30, 55, 75, 100.1],
                    labels=["🟢 Low Risk (Grade A)", "🟡 Moderate Risk (Grade B)", "🟠 High Risk (Grade C)", "🔴 Critical / NPA (Grade D)"]
                )).value_counts().reset_index()
                risk_tiers.columns = ["Level of Risk", "Account Count"]
                
                fig_risk = px.bar(
                    risk_tiers, x="Level of Risk", y="Account Count", color="Level of Risk",
                    title="<b>Predicted Portfolio by Level of Risk</b>",
                    color_discrete_map={
                        "🟢 Low Risk (Grade A)": "#10b981",
                        "🟡 Moderate Risk (Grade B)": "#f59e0b",
                        "🟠 High Risk (Grade C)": "#f97316",
                        "🔴 Critical / NPA (Grade D)": "#ef4444"
                    }
                )
                fig_risk.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280, showlegend=False)
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                y_te = base_obj["y_test"]
                y_pr = base_obj["y_pred_rf"]
                fig_reg = px.scatter(x=y_te[:300], y=y_pr[:300], labels={"x": "Actual ($)", "y": "Predicted ($)"}, title="Predicted vs Actual Yield ($)")
                fig_reg.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
                st.plotly_chart(fig_reg, use_container_width=True)
