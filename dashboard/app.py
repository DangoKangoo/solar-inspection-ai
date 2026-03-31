from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
for candidate in (REPO_ROOT, APP_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.app.pipeline.inference import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LABELS_PATH,
    analyze_uploaded_panel,
    validate_model_artifacts,
)
from ui.theme import APP_TITLE, inject_css  # noqa: E402
from ui.io_results import list_runs, load_run, RESULTS_ROOT  # noqa: E402
from ui.layout import (  # noqa: E402
    apply_filters,
    logs_panel,
    overview_row,
    panel_detail,
    run_to_df,
    sidebar_controls,
)


def _path_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def cached_model_artifact_issues(
    checkpoint_path_str: str,
    labels_path_str: str,
    checkpoint_mtime_ns: int | None,
    labels_mtime_ns: int | None,
) -> list[str]:
    del checkpoint_mtime_ns, labels_mtime_ns
    return validate_model_artifacts(
        checkpoint_path=Path(checkpoint_path_str),
        labels_path=Path(labels_path_str),
    )


def render_upload_panel() -> None:
    st.subheader("Analyze a Panel")
    st.caption("Upload a single cropped panel image to generate a local quality and risk assessment.")

    issues = cached_model_artifact_issues(
        str(DEFAULT_CHECKPOINT_PATH),
        str(DEFAULT_LABELS_PATH),
        _path_mtime_ns(DEFAULT_CHECKPOINT_PATH),
        _path_mtime_ns(DEFAULT_LABELS_PATH),
    )
    uploaded_file = st.file_uploader(
        "Panel image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="uploaded_panel_image",
    )

    if issues:
        st.warning("Model files are required before analysis can run.\n\n" + "\n".join(f"- {item}" for item in issues))
    else:
        st.success(
            f"Model ready for inference.\n\n- Checkpoint: `{DEFAULT_CHECKPOINT_PATH}`\n- Labels: `{DEFAULT_LABELS_PATH}`"
        )

    analyze_disabled = bool(issues) or uploaded_file is None
    if st.button("Analyze panel", use_container_width=True, disabled=analyze_disabled):
        try:
            with st.spinner("Running model inference..."):
                run_id, _, _ = analyze_uploaded_panel(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
                    labels_path=DEFAULT_LABELS_PATH,
                )
            st.session_state["_pending_run_id"] = run_id
            st.rerun()
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    st.divider()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()

    runs = list_runs()
    pending = st.session_state.pop("_pending_run_id", None)
    if pending and pending in runs:
        st.session_state["selected_run_id"] = pending

    st.title("Solar Inspection AI Dashboard")
    st.caption("Human-in-the-loop PV inspection | explainable outputs | auditable decision logic")

    render_upload_panel()

    if not runs:
        st.info("No inspection runs yet. Upload a panel image above to get started.")
        return

    run_id_guess = st.session_state.get("selected_run_id", runs[0])
    if run_id_guess not in runs:
        run_id_guess = runs[0]
        st.session_state["selected_run_id"] = run_id_guess

    run_dir = RESULTS_ROOT / run_id_guess
    run = load_run(run_dir)

    df_all = run_to_df(run)
    ctl = sidebar_controls(df_all)

    run_dir = RESULTS_ROOT / ctl["run_id"]
    run = load_run(run_dir)
    df_all = run_to_df(run)
    df_view = apply_filters(df_all, ctl)

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
