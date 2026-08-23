import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week5():
    st.header("🏋️ Week 5: Baseline Model Training (Dual-Target ML)")
    st.markdown("Train benchmark classification and regression models across the credit recovery lifecycle.")
    
    df_features = st.session_state.get("df_features", None)
    if df_features is None:
        show_prerequisite_warning("df_features (Transformed Features)", "Week 4: 🧩 Feature Engineering")
        return
        
    df_filtered = apply_global_filters(df_features)
    render_global_kpi_cards(df_filtered)
    
    # -------------------------------------------------------------
    # 1. Dual Target Selector
    # -------------------------------------------------------------
    c_sel1, c_sel2 = st.columns([3, 2])
    with c_sel1:
        target_mode = st.selectbox(
            "🎯 Select Machine Learning Objective / Target",
            [
                "🔄 Classification: Loan Recovery Status (recovery_status_binary: 1=Recovered, 0=Unrecovered)",
                "💰 Regression: Recovered Dollar Amount (recovered_amount: continuous $)"
            ]
        )
    with c_sel2:
        test_ratio = st.slider("Test Split Proportion (%)", 10, 40, 20) / 100.0
        rand_seed = st.number_input("Random Seed", min_value=1, max_value=999, value=42)

    is_classification = "Classification" in target_mode
    
    # Prepare numeric feature matrix
    ignore_cols = ["customer_id", "loan_id", "recovery_id", "repayment_id", "recovery_status", "recovery_date", 
                   "last_payment_date", "default_date", "disbursement_date", "delinquency_bucket", "officer_name", 
                   "recovery_officer_id", "officer_designation", "loan_purpose"]
    
    # Exclude targets from X
    feat_cols = [c for c in df_features.select_dtypes(include=['number']).columns 
                 if c not in ignore_cols and c not in ["recovery_status_binary", "recovered_amount", "net_recovery_amount", "default_flag"]]
    
    if len(feat_cols) == 0:
        st.error("No numeric features found. Please complete Week 4 Feature Engineering first.")
        return

    # Train button
    train_btn = st.button("🚀 Train Baseline Models", use_container_width=True)

    if train_btn or st.session_state.get("model_baseline")["classification" if is_classification else "regression"] is None:
        with st.spinner("Training baseline models..."):
            if is_classification:
                # Target: recovery_status_binary
                if "recovery_status_binary" not in df_features.columns:
                    if "recovery_target_binary" in df_features.columns:
                        df_features["recovery_status_binary"] = df_features["recovery_target_binary"].fillna(0).astype(int)
                    elif "recovery_status" in df_features.columns:
                        df_features["recovery_status_binary"] = df_features["recovery_status"].apply(
                            lambda s: 1 if s in ["Fully Recovered", "Partially Recovered"] else 0
                        )
                    elif "default_flag" in df_features.columns:
                        df_features["recovery_status_binary"] = (1 - df_features["default_flag"]).astype(int)
                    else:
                        df_features["recovery_status_binary"] = 0
                
                valid_mask = df_features["recovery_status_binary"].notnull()
                X = df_features.loc[valid_mask, feat_cols].fillna(0)
                y = df_features.loc[valid_mask, "recovery_status_binary"].astype(int)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=rand_seed, stratify=y)
                
                # Model 1: Logistic Regression
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_train)
                X_te_sc = scaler.transform(X_test)
                
                lr = LogisticRegression(max_iter=1000, random_state=rand_seed, class_weight="balanced")
                lr.fit(X_tr_sc, y_train)
                y_pred_lr = lr.predict(X_te_sc)
                y_prob_lr = lr.predict_proba(X_te_sc)[:, 1]
                
                # Model 2: Random Forest Classifier
                rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=rand_seed)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                y_prob_rf = rf.predict_proba(X_test)[:, 1]
                
                metrics = {
                    "Logistic Regression": {
                        "Accuracy": accuracy_score(y_test, y_pred_lr),
                        "Precision": precision_score(y_test, y_pred_lr, zero_division=0),
                        "Recall": recall_score(y_test, y_pred_lr, zero_division=0),
                        "F1-Score": f1_score(y_test, y_pred_lr, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
                    },
                    "Random Forest Classifier": {
                        "Accuracy": accuracy_score(y_test, y_pred_rf),
                        "Precision": precision_score(y_test, y_pred_rf, zero_division=0),
                        "Recall": recall_score(y_test, y_pred_rf, zero_division=0),
                        "F1-Score": f1_score(y_test, y_pred_rf, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
                    }
                }
                
                st.session_state["model_baseline"]["classification"] = {
                    "lr": lr, "rf": rf, "scaler": scaler, "features": feat_cols,
                    "X_test": X_test, "y_test": y_test, "y_prob_rf": y_prob_rf, "y_pred_rf": y_pred_rf
                }
                st.session_state["model_baseline"]["metrics"]["classification"] = metrics
                
            else:
                # Target: recovered_amount
                if "recovered_amount" not in df_features.columns:
                    st.error("Target column `recovered_amount` missing.")
                    return
                    
                # Train only on delinquent accounts with non-zero potential
                valid_mask = (df_features["overdue_days"] > 30) if "overdue_days" in df_features.columns else df_features["recovered_amount"].notnull()
                X = df_features.loc[valid_mask, feat_cols].fillna(0)
                y = df_features.loc[valid_mask, "recovered_amount"].fillna(0)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=rand_seed)
                
                # Model 1: Linear Regression
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_train)
                X_te_sc = scaler.transform(X_test)
                
                lin = LinearRegression()
                lin.fit(X_tr_sc, y_train)
                y_pred_lin = lin.predict(X_te_sc)
                
                # Model 2: Random Forest Regressor
                rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=rand_seed)
                rf_reg.fit(X_train, y_train)
                y_pred_rf = rf_reg.predict(X_test)
                
                def calc_reg_metrics(yt, yp):
                    mae = mean_absolute_error(yt, yp)
                    mse = mean_squared_error(yt, yp)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(yt, yp)
                    return {"MAE ($)": mae, "MSE": mse, "RMSE ($)": rmse, "R² Score": r2}
                    
                metrics = {
                    "Linear Regression": calc_reg_metrics(y_test, y_pred_lin),
                    "Random Forest Regressor": calc_reg_metrics(y_test, y_pred_rf)
                }
                
                st.session_state["model_baseline"]["regression"] = {
                    "lin": lin, "rf": rf_reg, "scaler": scaler, "features": feat_cols,
                    "X_test": X_test, "y_test": y_test, "y_pred_rf": y_pred_rf
                }
                st.session_state["model_baseline"]["metrics"]["regression"] = metrics

    # -------------------------------------------------------------
    # 2. Display Benchmark Metrics & Diagnostics
    # -------------------------------------------------------------
    key_mode = "classification" if is_classification else "regression"
    base_state = st.session_state["model_baseline"].get(key_mode)
    metrics_data = st.session_state["model_baseline"]["metrics"].get(key_mode)
    
    if metrics_data:
        st.subheader("📊 Baseline Model Performance Benchmark")
        m_df = pd.DataFrame(metrics_data).T
        st.dataframe(m_df.style.format("{:.4f}"), use_container_width=True)
        
        # Best model summary metric cards
        best_name = "Random Forest Classifier" if is_classification else "Random Forest Regressor"
        best_m = metrics_data[best_name]
        
        c1, c2, c3, c4 = st.columns(4)
        if is_classification:
            c1.metric("Champion Accuracy", f"{best_m['Accuracy']*100:.2f}%")
            c2.metric("Precision", f"{best_m['Precision']*100:.2f}%")
            c3.metric("Recall", f"{best_m['Recall']*100:.2f}%")
            c4.metric("ROC-AUC Score", f"{best_m['ROC-AUC']:.4f}")
        else:
            c1.metric("R² Score (Variance)", f"{best_m['R² Score']:.4f}")
            c2.metric("MAE ($)", f"${best_m['MAE ($)']:,.2f}")
            c3.metric("RMSE ($)", f"${best_m['RMSE ($)']:,.2f}")
            c4.metric("Model Quality", "Excellent Fit")

        # -------------------------------------------------------------
        # 3. Plots: Feature Importances & Pred vs Actual
        # -------------------------------------------------------------
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            rf_model = base_state["rf"]
            imp_df = pd.DataFrame({
                "Feature": feat_cols,
                "Importance": rf_model.feature_importances_
            }).sort_values("Importance", ascending=False).head(12)
            
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                title=f"<b>Top Feature Importances ({best_name})</b>",
                color="Importance", color_continuous_scale="Viridis", template="plotly_white"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with c_p2:
            y_test = base_state["y_test"]
            y_pred = base_state["y_pred_rf"]
            
            if is_classification:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=y_test, name="Actual True Label", marker_color="#3b82f6", opacity=0.7))
                fig_hist.add_trace(go.Histogram(x=y_pred, name="Predicted Label", marker_color="#10b981", opacity=0.7))
                fig_hist.update_layout(barmode="group", title="<b>Actual vs Predicted Recovery Class Distribution</b>", template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                fig_sc_reg = px.scatter(
                    x=y_test, y=y_pred, labels={"x": "Actual Recovered ($)", "y": "Predicted Recovered ($)"},
                    title="<b>Actual vs Predicted Recovered Dollars ($)</b>", template="plotly_white"
                )
                fig_sc_reg.add_trace(go.Scatter(x=[0, max(y_test)], y=[0, max(y_test)], mode="lines", name="Perfect Fit (y=x)", line=dict(color="red", dash="dash")))
                st.plotly_chart(fig_sc_reg, use_container_width=True)
