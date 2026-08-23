import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, mean_absolute_error, mean_squared_error, r2_score
)

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week7():
    st.header("📊 Week 7: Model Diagnostics & Baseline vs Tuned Benchmarking")
    st.markdown("Inspect diagnostic curves, confusion matrices, residual distributions, and evaluate performance lifts.")
    
    df_features = st.session_state.get("df_features", None)
    base_state = st.session_state.get("model_baseline", {})
    if df_features is None:
        show_prerequisite_warning("df_features (Feature Matrix)", "Week 4: 🧩 Feature Engineering")
        return
        
    df_filtered = apply_global_filters(df_features)
    render_global_kpi_cards(df_filtered)
    
    # Target Mode
    target_mode = st.selectbox(
        "🎯 Select Model Evaluation Domain",
        [
            "🔄 Classification: Loan Recovery Status (recovery_status_binary)",
            "💰 Regression: Recovered Dollar Amount (recovered_amount)"
        ]
    )
    is_classification = "Classification" in target_mode
    key_mode = "classification" if is_classification else "regression"
    
    base_obj = base_state.get(key_mode)
    tuned_obj = st.session_state.get("model_tuned", {}).get(key_mode)
    
    if base_obj is None:
        show_prerequisite_warning(f"Baseline {key_mode.capitalize()} Model", "Week 5: 🏋️ Baseline Training")
        return

    # Extract test data
    X_test = base_obj["X_test"]
    y_test = base_obj["y_test"]
    
    # -------------------------------------------------------------
    # 1. Side-by-Side Performance Lift Comparison
    # -------------------------------------------------------------
    st.subheader("⚔️ Baseline vs Tuned Model Performance Lift")
    
    c_l1, c_l2 = st.columns(2)
    
    with c_l1:
        st.markdown("#### 🔹 Baseline Model")
        if is_classification:
            base_pred = base_obj["rf"].predict(X_test)
            base_prob = base_obj["rf"].predict_proba(X_test)[:, 1]
            base_auc = roc_auc_score(y_test, base_prob)
            st.metric("Baseline ROC-AUC", f"{base_auc:.4f}")
            st.metric("Baseline Accuracy", f"{(base_pred == y_test).mean()*100:.2f}%")
        else:
            base_pred = base_obj["rf"].predict(X_test)
            base_r2 = r2_score(y_test, base_pred)
            base_mae = mean_absolute_error(y_test, base_pred)
            st.metric("Baseline R² Score", f"{base_r2:.4f}")
            st.metric("Baseline MAE ($)", f"${base_mae:,.2f}")
            
    with c_l2:
        st.markdown("#### 🌟 Tuned Champion Model")
        if tuned_obj is not None:
            if is_classification:
                tuned_pred = tuned_obj.predict(X_test)
                tuned_prob = tuned_obj.predict_proba(X_test)[:, 1]
                tuned_auc = roc_auc_score(y_test, tuned_prob)
                lift_auc = tuned_auc - base_auc
                st.metric("Tuned ROC-AUC", f"{tuned_auc:.4f}", delta=f"{lift_auc:+.4f} lift")
                st.metric("Tuned Accuracy", f"{(tuned_pred == y_test).mean()*100:.2f}%", delta=f"{((tuned_pred == y_test).mean() - (base_pred == y_test).mean())*100:+.2f}%")
            else:
                tuned_pred = tuned_obj.predict(X_test)
                tuned_r2 = r2_score(y_test, tuned_pred)
                tuned_mae = mean_absolute_error(y_test, tuned_pred)
                st.metric("Tuned R² Score", f"{tuned_r2:.4f}", delta=f"{tuned_r2 - base_r2:+.4f} lift")
                st.metric("Tuned MAE ($)", f"${tuned_mae:,.2f}", delta=f"-${base_mae - tuned_mae:,.2f}", delta_color="inverse")
        else:
            st.info("ℹ️ Tune hyperparameters in **Week 6** to unlock comparative lift analysis.")
            tuned_pred, tuned_prob = base_pred, (base_prob if is_classification else None)

    # -------------------------------------------------------------
    # 2. Detailed Diagnostic Curves & Plots
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Statistical Diagnostic Curves")
    
    if is_classification:
        tab_cm, tab_roc, tab_pr = st.tabs(["🔲 Confusion Matrix", "📉 ROC-AUC Curve", "🎯 Precision-Recall Curve"])
        
        with tab_cm:
            c_cm1, c_cm2 = st.columns(2)
            with c_cm1:
                cm = confusion_matrix(y_test, tuned_pred)
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    x=["Predicted Unrecovered (0)", "Predicted Recovered (1)"],
                    y=["Actual Unrecovered (0)", "Actual Recovered (1)"],
                    color_continuous_scale="Blues", title="<b>Tuned Model Confusion Matrix</b>"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            with c_cm2:
                tn, fp, fn, tp = cm.ravel()
                st.markdown("#### 🎯 Classification Decision Analysis")
                st.write(f"- **True Positives (Recoveries Identified)**: {tp:,}")
                st.write(f"- **True Negatives (Correctly Identified Defaults)**: {tn:,}")
                st.write(f"- **False Positives (Misclassified as Recoverable)**: {fp:,}")
                st.write(f"- **False Negatives (Missed Recovery Opportunities)**: {fn:,}")
                st.info(f"✨ **Recovery Recall Rate**: **{(tp / (tp + fn) * 100):.1f}%** of all recoverable dollars successfully captured.")
                
        with tab_roc:
            fpr, tpr, _ = roc_curve(y_test, tuned_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Tuned Model (AUC = {roc_auc_score(y_test, tuned_prob):.4f})", line=dict(color="#10b981", width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Random Guess"))
            fig_roc.update_layout(title="<b>Receiver Operating Characteristic (ROC) Curve</b>", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white")
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with tab_pr:
            prec_c, rec_c, _ = precision_recall_curve(y_test, tuned_prob)
            pr_auc = average_precision_score(y_test, tuned_prob)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec_c, y=prec_c, name=f"PR Curve (Avg Prec = {pr_auc:.4f})", line=dict(color="#3b82f6", width=3)))
            fig_pr.update_layout(title="<b>Precision-Recall Curve</b>", xaxis_title="Recall", yaxis_title="Precision", template="plotly_white")
            st.plotly_chart(fig_pr, use_container_width=True)
            
    else:
        # Regression Diagnostic Plots
        tab_act, tab_res, tab_fit = st.tabs(["🎯 Actual vs Predicted", "📊 Residuals Distribution", "📉 Residuals vs Fitted"])
        
        with tab_act:
            fig_act = px.scatter(
                x=y_test, y=tuned_pred,
                labels={"x": "Actual Recovered ($)", "y": "Predicted Recovered ($)"},
                title="<b>Actual vs Predicted Recovered Dollar Amount</b>", template="plotly_white"
            )
            fig_act.add_trace(go.Scatter(x=[0, max(y_test)], y=[0, max(y_test)], mode="lines", name="Perfect Fit (y=x)", line=dict(color="red", dash="dash")))
            st.plotly_chart(fig_act, use_container_width=True)
            
        with tab_res:
            residuals = y_test - tuned_pred
            fig_res = px.histogram(
                residuals, nbins=40,
                title="<b>Model Residuals (Errors) Distribution ($)</b>",
                color_discrete_sequence=["#8b5cf6"], template="plotly_white"
            )
            st.plotly_chart(fig_res, use_container_width=True)
            
        with tab_fit:
            residuals = y_test - tuned_pred
            fig_rf = px.scatter(
                x=tuned_pred, y=residuals,
                labels={"x": "Fitted Values ($)", "y": "Residuals ($)"},
                title="<b>Residuals vs Fitted Values (Heteroscedasticity Check)</b>", template="plotly_white"
            )
            fig_rf.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_rf, use_container_width=True)
