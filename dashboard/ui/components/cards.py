from __future__ import annotations

import streamlit as st


def pill(text: str, color: str) -> str:
    return f"""
    <span style="
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        background:{color}20;
        border:1px solid {color}55;
        color:{color};
        font-weight:600;
        font-size:12px;
        letter-spacing:0.2px;">
        {text}
    </span>
    """


def score_bar(label: str, value: float):
    value = float(max(0.0, min(1.0, value)))
    st.markdown(
        f"""
        <div style="margin: 6px 0 10px 0;">
          <div style="display:flex; justify-content:space-between; font-size:13px; opacity:0.9;">
            <div>{label}</div>
            <div style="font-variant-numeric: tabular-nums;">{value:.2f}</div>
          </div>
          <div style="height:10px; border-radius:999px; background:#ffffff14; overflow:hidden; margin-top:6px;">
            <div style="height:100%; width:{int(value*100)}%; background:#22c55e66;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str, sub: str = "", accent: str = "#22c55e"):
    st.markdown(
        f"""
        <div style="
          border:1px solid #ffffff14;
          background: linear-gradient(180deg, #ffffff08, #00000000);
          border-radius:16px;
          padding:14px 14px 12px 14px;
          box-shadow: 0 10px 25px rgba(0,0,0,0.25);
          position:relative;
          overflow:hidden;">

          <!-- bigger + softer glow -->
          <div style="
            position:absolute;
            left:-80px;
            top:-80px;
            width:220px;
            height:220px;
            background:{accent}33;
            border-radius:999px;
            filter: blur(10px);
          "></div>

          <!-- optional thin accent line -->
          <div style="
            position:absolute;
            left:0;
            top:0;
            width:100%;
            height:3px;
            background: linear-gradient(90deg, {accent}aa, transparent);
          "></div>

          <div style="font-size:12px; opacity:0.75; letter-spacing:0.3px;">{title}</div>

          <div style="
            font-size:28px;
            font-weight:850;
            margin-top:6px;
            font-variant-numeric: tabular-nums;
            color: {accent};
            text-shadow: 0 0 12px {accent}33;
          ">{value}</div>

          <div style="font-size:12px; opacity:0.65; margin-top:6px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

