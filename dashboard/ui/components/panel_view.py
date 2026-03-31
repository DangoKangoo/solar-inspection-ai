from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from ..io_results import RESULTS_ROOT
from ..theme import risk_color
from .cards import pill, score_bar


def panel_detail(run: Dict[str, Any], df_view: pd.DataFrame):
    st.divider()
    st.subheader("Panel Drilldown")

    if df_view.empty:
        st.info("No panels match your filters.")
        return

    ids = df_view["image_id"].tolist()
    selected = st.selectbox("Select panel", ids, index=0)

    panel = next((p for p in run.get("panels", []) if p.get("image_id") == selected), None)
    if not panel:
        st.warning("Panel not found in run data.")
        return

    risk = (panel.get("risk_level") or "REVIEW").upper()
    st.markdown(pill(f"RISK: {risk}", risk_color(risk)), unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        score_bar("Fault probability", float(panel.get("fault_probability", 0.0)))
    with m2:
        score_bar("Confidence", float(panel.get("confidence", 0.0)))
    with m3:
        st.markdown("**Predicted class**")
        st.write(f"`{panel.get('predicted_class', '-')}`")

    tabs = st.tabs(["Raw", "Processed", "Grad-CAM", "Decision + Audit"])
    artifacts = panel.get("artifacts", {}) if isinstance(panel.get("artifacts"), dict) else {}

    def show_img(path: Optional[str], empty_msg: str):
        if not path:
            st.info(empty_msg)
            return

        p = Path(path)
        if not p.is_absolute():
            run_id = run.get("run_id")
            if run_id:
                p = RESULTS_ROOT / run_id / p

        if not p.exists():
            st.warning(f"Missing artifact: {p}")
            return

        st.image(str(p), use_column_width=True)

    with tabs[0]:
        show_img(artifacts.get("raw"), "Raw image will appear here once backend saves it.")

    with tabs[1]:
        show_img(
            artifacts.get("processed"),
            "A 224x224 RGB preview of the model input will appear here once generated.",
        )

    with tabs[2]:
        show_img(artifacts.get("heatmap"), "Grad-CAM heatmap will appear here once generated.")

    with tabs[3]:
        st.markdown("### Decision Logic")
        st.write(
            "Current rule-based mapping used for this run:\n\n"
            "- **HIGH**: fault probability > 0.78\n"
            "- **MEDIUM**: 0.45-0.78\n"
            "- **LOW**: < 0.45\n"
            "- **REVIEW**: low confidence or explainability failure\n"
        )

        flags = panel.get("flags", []) or []
        st.markdown("### Flags / Exceptions")
        if flags:
            for flag in flags:
                st.error(flag)
        else:
            st.success("No flags")

        st.markdown("### Audit Metadata")
        st.code(
            json.dumps(
                {
                    "image_id": panel.get("image_id"),
                    "predicted_class": panel.get("predicted_class"),
                    "fault_probability": panel.get("fault_probability"),
                    "confidence": panel.get("confidence"),
                    "risk_level": panel.get("risk_level"),
                    "model": run.get("model", {}),
                    "timestamp": run.get("timestamp", None),
                },
                indent=2,
            ),
            language="json",
        )
