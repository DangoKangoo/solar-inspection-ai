from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional

from .theme import APP_TITLE, RISK_ORDER
from .io_results import list_runs
from .components.cards import kpi_card
from .components.charts import risk_distribution
from .components.panel_view import panel_detail as panel_detail_component


def run_to_df(run: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for p in run.get("panels", []):
        rows.append(
            {
                "image_id": p.get("image_id"),
                "risk_level": (p.get("risk_level") or "").upper(),
                "predicted_class": p.get("predicted_class"),
                "fault_probability": p.get("fault_probability", 0.0),
                "confidence": p.get("confidence", 0.0),
                "flags": ", ".join(p.get("flags", [])) if p.get("flags") else "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["risk_level"] = pd.Categorical(df["risk_level"], categories=RISK_ORDER, ordered=True)
    return df.sort_values(["risk_level", "fault_probability"], ascending=[False, False])


def sidebar_controls(df: pd.DataFrame) -> Dict[str, Any]:
    st.sidebar.markdown(f"### {APP_TITLE}")
    st.sidebar.markdown('<div class="small-muted">Inspection console • decision support</div>', unsafe_allow_html=True)
    st.sidebar.divider()

    runs = list_runs()
    run_id = st.sidebar.selectbox("Run", runs, index=0)

    st.sidebar.subheader("Filters")
    risk_filter = st.sidebar.multiselect(
        "Risk",
        options=RISK_ORDER,
        default=["HIGH", "MEDIUM", "LOW", "REVIEW"],
    )

    classes = sorted([c for c in df["predicted_class"].dropna().unique()]) if not df.empty else []
    cls_filter = st.sidebar.multiselect("Class", options=classes, default=classes)

    min_prob = st.sidebar.slider("Min fault probability", 0.0, 1.0, 0.0, 0.01)
    min_conf = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.01)

    flagged_only = st.sidebar.checkbox("Flagged only", value=False)

    st.sidebar.divider()
    st.sidebar.subheader("Display")
    show_logs = st.sidebar.checkbox("Show logs panel", value=True)

    return {
        "run_id": run_id,
        "risk_filter": risk_filter,
        "cls_filter": cls_filter,
        "min_prob": min_prob,
        "min_conf": min_conf,
        "flagged_only": flagged_only,
        "show_logs": show_logs,
    }


def apply_filters(df: pd.DataFrame, ctl: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out[out["risk_level"].isin(ctl["risk_filter"])]

    if ctl["cls_filter"]:
        out = out[out["predicted_class"].isin(ctl["cls_filter"])]

    out = out[out["fault_probability"] >= ctl["min_prob"]]
    out = out[out["confidence"] >= ctl["min_conf"]]

    if ctl["flagged_only"]:
        out = out[out["flags"].astype(str).str.len() > 0]

    return out


def overview_row(run: Dict[str, Any], df_all: pd.DataFrame, df_view: pd.DataFrame):
    total = len(df_all)
    high = int((df_all["risk_level"] == "HIGH").sum()) if total else 0
    review = int((df_all["risk_level"] == "REVIEW").sum()) if total else 0
    avg_conf = float(df_all["confidence"].mean()) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("PANELS", f"{total}", "Total inspected in run", accent="#60a5fa")
    with c2:
        kpi_card("HIGH RISK", f"{high}", "Maintenance priority", accent="#ef4444")
    with c3:
        kpi_card("REVIEW", f"{review}", "Low confidence / flagged", accent="#a78bfa")
    with c4:
        kpi_card("AVG CONF", f"{avg_conf:.2f}", "Model confidence avg", accent="#22c55e")

    st.markdown("")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Inspection Queue")
        st.caption("Filtered view drives the table + drilldown below.")
        st.dataframe(
            df_view,
            use_container_width=True,
            height=420,
            hide_index=True,
            column_config={
                "risk_level": st.column_config.TextColumn("Risk"),
                "fault_probability": st.column_config.ProgressColumn(
                    "Fault Prob", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

    with right:
        risk_distribution(df_all)
        st.markdown("")
        st.subheader("Model / Run")
        model = run.get("model", {})
        st.write(
            f"**Run:** `{run.get('run_id','-')}`  \n"
            f"**Model:** `{model.get('name','-')}`  \n"
            f"**Framework:** `{model.get('framework','-')}`  \n"
            f"**Version:** `{model.get('version','-')}`"
        )


def panel_detail(run: Dict[str, Any], df_view: pd.DataFrame):
    panel_detail_component(run, df_view)


def logs_panel(run: Dict[str, Any]):
    st.divider()
    st.subheader("Pipeline Logs")
    logs = run.get("logs", [])
    if not logs:
        st.info("No logs yet.")
        return

    for item in logs:
        level = (item.get("level") or "INFO").upper()
        msg = item.get("msg") or ""
        if level == "ERROR":
            st.error(msg)
        elif level in ("WARN", "WARNING"):
            st.warning(msg)
        else:
            st.info(msg)
