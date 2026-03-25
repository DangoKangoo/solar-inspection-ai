from __future__ import annotations

import json
import streamlit as st

from ui.theme import APP_TITLE, inject_css
from ui.io_results import list_runs, load_run, RESULTS_ROOT
from ui.layout import (
    sidebar_controls,
    apply_filters,
    overview_row,
    panel_detail,
    logs_panel,
    run_to_df,
)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()

    # Load run (mock if none)
    run_id_guess = list_runs()[0]
    run_dir = RESULTS_ROOT / run_id_guess
    run = load_run(run_dir)

    df_all = run_to_df(run)
    ctl = sidebar_controls(df_all)

    # Reload based on sidebar run selection
    run_dir = RESULTS_ROOT / ctl["run_id"]
    run = load_run(run_dir)
    df_all = run_to_df(run)
    df_view = apply_filters(df_all, ctl)

    st.title("Solar Inspection AI Dashboard")
    st.caption("Human-in-the-loop PV inspection • explainable outputs • auditable decision logic")

    overview_row(run, df_all, df_view)
    panel_detail(run, df_view)

    if ctl["show_logs"]:
        logs_panel(run)

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Export filtered table (CSV)",
            data=df_view.to_csv(index=False).encode("utf-8"),
            file_name="inspection_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Export run JSON",
            data=json.dumps(run, indent=2).encode("utf-8"),
            file_name="run_results.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
