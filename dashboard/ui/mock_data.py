from __future__ import annotations

from typing import Any, Dict


def mock_results() -> Dict[str, Any]:
    # Matches what your partner should output eventually.
    panels = []
    for i in range(1, 26):
        prob = min(0.98, max(0.02, (i % 10) / 10 + (0.05 if i % 7 == 0 else 0.0)))
        conf = min(0.99, max(0.05, 0.55 + (i % 8) / 20))
        cls = ["normal", "micro_crack", "finger_defect", "hotspot"][i % 4]
        if prob > 0.78:
            risk = "HIGH"
        elif prob > 0.45:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        flags = []
        if conf < 0.55:
            flags.append("LOW_CONFIDENCE")
            risk = "REVIEW"
        if i % 11 == 0:
            flags.append("EXPLAINABILITY_MISALIGNED")

        panels.append(
            {
                "image_id": f"panel_{i:03d}",
                "predicted_class": cls,
                "fault_probability": float(prob),
                "confidence": float(conf),
                "risk_level": risk,
                "flags": flags,
                "artifacts": {"raw": None, "processed": None, "heatmap": None},
            }
        )

    return {
        "run_id": "demo_run",
        "model": {"name": "resnet50", "framework": "pytorch", "version": "dev-mock"},
        "preprocess": {"status": "OK"},
        "timestamp": "LOCAL-DEMO",
        "panels": panels,
        "logs": [
            {"level": "INFO", "msg": "Loaded 25 images"},
            {"level": "INFO", "msg": "Preprocessing: contour extraction + normalization"},
            {"level": "WARN", "msg": "2 panels flagged: explainability misaligned"},
        ],
    }
