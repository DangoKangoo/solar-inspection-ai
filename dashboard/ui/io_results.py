from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .mock_data import mock_results

# later: backend writes here
RESULTS_ROOT = Path("dashboard/results/runs")


def list_runs() -> List[str]:
    if not RESULTS_ROOT.exists():
        return ["demo_run"]
    runs = [p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()]
    return sorted(runs) if runs else ["demo_run"]


def load_run(run_dir: Path) -> Dict[str, Any]:
    results_file = run_dir / "results.json"

    if not results_file.exists():
        return mock_results()

    try:
        raw = results_file.read_text(encoding="utf-8").strip()
        if not raw:
            return mock_results()
        return json.loads(raw)
    except Exception:
        # corrupted/partial JSON -> don't crash dashboard
        return mock_results()
