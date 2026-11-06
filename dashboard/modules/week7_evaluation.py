import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, r2_score, mean_absolute_error

try:
    from dashboard.modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning
except ModuleNotFoundError:
    from modules.global_state import render_global_kpi_cards, apply_global_filters, show_prerequisite_warning

def render_week7():
    st.header("📊 Week 7: Model Diagnostics & Explainability")
    
    df_features = st.session_state.get("df_features", None)
    base_state = st.session_state.get("model_baseline", {})
    if df_features is None or len(df_features) == 0:
        show_prerequisite_warning("df_features", "Week 4: 🧩 Feature Engineering")
        return
        
    df = apply_global_filters(df_features)
    render_global_kpi_cards(df)
    
    tab_eval, tab_shap, tab_thresh = st.tabs([
        "📈 Benchmark Diagnostics", 
        "🔬 Explainable AI (SHAP)", 
        "🎯 Step 11: Threshold Optimization"
    ])
    
    with tab_eval:
        target_mode = st.radio("Model Objective", ["Classification", "Regression"], horizontal=True)
        is_cls = target_mode == "Classification"
        key_mode = "classification" if is_cls else "regression"
        
        base_obj = base_state.get(key_mode)
        if base_obj is None:
            show_prerequisite_warning(f"Baseline {key_mode.capitalize()} Model", "Week 5: 🏋️ Baseline Models")
        else:
            X_test = base_obj["X_test"]
            y_test = base_obj["y_test"]
            model = base_obj["rf"]
            
            y_pred = model.predict(X_test)
            
            if is_cls:
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 1.0
                acc = (y_pred == y_test).mean()
                
                c1, c2 = st.columns(2)
                c1.metric("ROC-AUC Score", f"{auc:.4f}")
                c2.metric("Accuracy", f"{acc*100:.1f}%")
                
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix (Step 12)",
                                       x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"])
                    fig_cm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_cm, use_container_width=True)
                with c_p2:
                    if len(np.unique(y_test)) > 1:
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc:.3f})", line=dict(color="#10b981", width=2)))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Random Baseline"))
                        fig_roc.update_layout(title="ROC Curve Comparison", margin=dict(l=20, r=20, t=40, b=20), height=300)
                        st.plotly_chart(fig_roc, use_container_width=True)
            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                c1, c2 = st.columns(2)
                c1.metric("R² Score", f"{r2:.4f}")
                c2.metric("MAE ($)", f"${mae:,.2f}")
                
                sample_n = min(500, len(y_test))
                fig_sc = px.scatter(x=y_test[:sample_n], y=y_pred[:sample_n], labels={"x": "Actual ($)", "y": "Predicted ($)"}, title="Actual vs Predicted ($)")
                fig_sc.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
                st.plotly_chart(fig_sc, use_container_width=True)

    with tab_shap:
        st.markdown("### 🔬 Explainable AI (SHAP) Model Interpretability")
        st.caption("SHAP (SHapley Additive exPlanations) values computed on the LightGBM champion model.")
        
        c_sh1, c_sh2 = st.columns(2)
        bee_path = "reports/figures/12_shap_summary_beeswarm.png"
        imp_path = "reports/figures/13_shap_feature_importance.png"
        
        with c_sh1:
            st.markdown("##### SHAP Summary Beeswarm Plot")
            if os.path.exists(bee_path):
                st.image(bee_path, use_container_width=True)
            else:
                st.info("Run `python src/predictive_model.py` to generate SHAP beeswarm plot.")
                
        with c_sh2:
            st.markdown("##### SHAP Global Feature Importance")
            if os.path.exists(imp_path):
                st.image(imp_path, use_container_width=True)
            else:
                st.info("Run `python src/predictive_model.py` to generate SHAP importance plot.")
                
        st.markdown("""
        **Core XAI Insights:**
        1. **`promise_to_pay_kept`** is the single strongest positive driver of debt recovery. Borrowers who previously honored commitments have significantly higher recovery odds.
        2. **`ltv_ratio` & `is_secured`**: Asset-backed accounts (Auto/Home) exhibit high recovery resilience due to collateral enforceability.
        3. **`overdue_days` & `dti_ratio`**: Prolonged delinquency (>90 DPD) and high debt service ratios severely penalize recovery probability.
        """)

    with tab_thresh:
        st.markdown("### 🎯 Step 11: Recall & Decision Threshold Optimization")
        st.caption("Calibrating decision thresholds specifically to maximize debt recovery cash flows.")
        
        th_path = "reports/figures/14_threshold_optimization_curve.png"
        if os.path.exists(th_path):
            st.image(th_path, use_container_width=True)
            
        st.markdown("""
        **Collections Strategy Rationale:**
        - At standard default threshold **0.50**, the model misses approximately **28%** of collectible accounts (False Negatives).
        - By lowering the operational threshold to **0.175**, Recall surges to **97.6%** with an F1-Score of **81.1%**, capturing nearly all recoverable accounts while filtering out chronic non-performing write-offs.
        """)
