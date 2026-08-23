import sys
import os

# Ensure project root and dashboard directory are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI Loan Recovery & Risk Analytics (9-Week ML)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for High-Contrast Premium Fintech Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Streamlit Native Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
    }
    
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
        color: #f8fafc !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] div,
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Import Global State Manager & Weekly Modules
try:
    from dashboard.modules.global_state import init_session_state, render_global_filters
    from dashboard.modules.week1_preview import render_week1
    from dashboard.modules.week2_eda import render_week2
    from dashboard.modules.week3_cleaning import render_week3
    from dashboard.modules.week4_features import render_week4
    from dashboard.modules.week5_baseline import render_week5
    from dashboard.modules.week6_tuning import render_week6
    from dashboard.modules.week7_evaluation import render_week7
    from dashboard.modules.week8_pipeline import render_week8
    from dashboard.modules.week9_dashboard import render_week9
except ModuleNotFoundError:
    from modules.global_state import init_session_state, render_global_filters
    from modules.week1_preview import render_week1
    from modules.week2_eda import render_week2
    from modules.week3_cleaning import render_week3
    from modules.week4_features import render_week4
    from modules.week5_baseline import render_week5
    from modules.week6_tuning import render_week6
    from modules.week7_evaluation import render_week7
    from modules.week8_pipeline import render_week8
    from modules.week9_dashboard import render_week9

# Initialize Persistent Session State
init_session_state()

# ==============================================================================
# SIDEBAR NAVIGATION & GLOBAL FILTERS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("AI Loan Recovery")
st.sidebar.caption("9-Week Full-Lifecycle ML Platform")

# 9-Week Navigation List
WEEK_PAGES = [
    "Week 9: 🏠 Project Dashboard",
    "Week 1: 📂 Dataset Preview",
    "Week 2: 🔎 EDA Viewer",
    "Week 3: 🧼 Data Cleaning",
    "Week 4: 🧩 Feature Engineering",
    "Week 5: 🏋️ Baseline Models",
    "Week 6: ⚙️ Model Tuning",
    "Week 7: 📊 Model Diagnostics",
    "Week 8: 🚀 Pipeline Runner"
]

# Check if navigated via quick link
if "nav_selection" in st.session_state and st.session_state["nav_selection"] in WEEK_PAGES:
    selected_page_index = WEEK_PAGES.index(st.session_state["nav_selection"])
else:
    selected_page_index = 0

selected_page = st.sidebar.selectbox(
    "🧭 Select Weekly ML Stage",
    WEEK_PAGES,
    index=selected_page_index
)

# Store active page
st.session_state["nav_selection"] = selected_page

st.sidebar.markdown("---")

# Render Persistent Global Filters in Sidebar
render_global_filters()

st.sidebar.markdown("---")
st.sidebar.caption("🏦 AI-Driven Loan Recovery & Risk Analytics | Dual Target ML Lifecycle")

# ==============================================================================
# PAGE ROUTER
# ==============================================================================
if selected_page == "Week 9: 🏠 Project Dashboard":
    render_week9()
elif selected_page == "Week 1: 📂 Dataset Preview":
    render_week1()
elif selected_page == "Week 2: 🔎 EDA Viewer":
    render_week2()
elif selected_page == "Week 3: 🧼 Data Cleaning":
    render_week3()
elif selected_page == "Week 4: 🧩 Feature Engineering":
    render_week4()
elif selected_page == "Week 5: 🏋️ Baseline Models":
    render_week5()
elif selected_page == "Week 6: ⚙️ Model Tuning":
    render_week6()
elif selected_page == "Week 7: 📊 Model Diagnostics":
    render_week7()
elif selected_page == "Week 8: 🚀 Pipeline Runner":
    render_week8()
