from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "runs"

EMPTY_RUN: Dict[str, Any] = {
    "run_id": "",
    "model": {},
    "preprocess": {"status": "N/A"},
    "timestamp": "",
    "panels": [],
    "logs": [],
}


def list_runs() -> List[str]:
    if not RESULTS_ROOT.exists():
        return []
    runs = [p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()]
    return sorted(runs, reverse=True)


def load_run(run_dir: Path) -> Dict[str, Any]:
    results_file = run_dir / "results.json"

    if not results_file.exists():
        return EMPTY_RUN

    try:
        raw = results_file.read_text(encoding="utf-8").strip()
        if not raw:
            return EMPTY_RUN
        return json.loads(raw)
    except Exception:
        return EMPTY_RUN
