from __future__ import annotations

import streamlit as st

APP_TITLE = "Solar Inspection AI"

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "REVIEW"]


def risk_color(risk: str) -> str:
    risk = (risk or "").upper()
    return {
        "LOW": "#22c55e",
        "MEDIUM": "#f59e0b",
        "HIGH": "#ef4444",
        "REVIEW": "#a78bfa",
    }.get(risk, "#94a3b8")


def inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
          [data-testid="stSidebar"] { border-right: 1px solid #ffffff14; }
          .stDataFrame { border: 1px solid #ffffff14; border-radius: 12px; overflow: hidden; }
          .small-muted { font-size: 12px; opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True,
    )
