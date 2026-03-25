from __future__ import annotations

import pandas as pd
import streamlit as st

from ..theme import RISK_ORDER


def risk_distribution(df_all: pd.DataFrame):
    st.subheader("Risk Distribution")
    if df_all.empty:
        st.info("No data yet.")
        return

    counts = df_all["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    chart_df = pd.DataFrame({"risk": counts.index, "count": counts.values})
    st.bar_chart(chart_df.set_index("risk"))
