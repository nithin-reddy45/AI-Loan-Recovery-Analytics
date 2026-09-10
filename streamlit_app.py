"""
Universal Cloud & Local Deployment Entrypoint: AI-Driven Loan Recovery & Risk Analytics
Supports Streamlit Community Cloud, HuggingFace Spaces, Render, AWS, and local execution.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

# Delegate to the dashboard app
from dashboard.app import *
