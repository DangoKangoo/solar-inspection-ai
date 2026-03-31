from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .gradcam import save_gradcam
from .model import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LABELS_PATH,
    PredictionResult,
    load_model,
    predict_image,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "dashboard" / "results" / "runs"
DEFAULT_UPLOADS_ROOT = REPO_ROOT / "dashboard" / "uploads"
ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPLAINABILITY_FAILURE_FLAG = "EXPLAINABILITY_FAILED"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-_").lower()
    return slug or "panel"


def create_run_id(prefix: str = "panel") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(prefix)}-{timestamp}"


def ensure_unique_run_id(run_id: str, *roots: Path) -> str:
    candidate = run_id
    suffix = 1
    while any((root / candidate).exists() for root in roots):
        candidate = f"{run_id}-{suffix:02d}"
        suffix += 1
    return candidate


def build_dashboard_result(
    *,
    run_id: str,
    run_dir: Path,
    image_id: str,
    prediction: PredictionResult,
    checkpoint_path: Path,
    architecture: str,
    raw_artifact_path: Path,
    processed_artifact_path: Path | None,
    heatmap_artifact_path: Path | None = None,
    extra_logs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    logs = [
        {"level": "INFO", "msg": f"Loaded model checkpoint: {checkpoint_path.name}"},
        {"level": "INFO", "msg": f"Analyzed panel: {image_id}"},
        {"level": "INFO", "msg": f"Predicted class: {prediction.predicted_class}"},
    ]
    if processed_artifact_path is None:
        logs.append({"level": "WARN", "msg": "Processed preview was not generated."})
    if heatmap_artifact_path is not None:
        logs.append({"level": "INFO", "msg": "Grad-CAM heatmap generated."})
    if extra_logs:
        logs.extend(extra_logs)

    return {
        "run_id": run_id,
        "model": {
            "name": architecture,
            "framework": "pytorch",
            "version": checkpoint_path.name,
        },
        "preprocess": {"status": "OK"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "panels": [
            {
                "image_id": image_id,
                "predicted_class": prediction.predicted_class,
                "fault_probability": round(prediction.fault_probability, 4),
                "confidence": round(prediction.confidence, 4),
                "risk_level": prediction.risk_level,
                "flags": prediction.flags,
                "artifacts": {
                    "raw": raw_artifact_path.relative_to(run_dir).as_posix(),
                    "processed": (
                        processed_artifact_path.relative_to(run_dir).as_posix()
                        if processed_artifact_path
                        else None
                    ),
                    "heatmap": (
                        heatmap_artifact_path.relative_to(run_dir).as_posix()
                        if heatmap_artifact_path
                        else None
                    ),
                },
                "class_probabilities": {
                    label: round(probability, 6)
                    for label, probability in prediction.probabilities.items()
                },
            }
        ],
        "logs": logs,
    }


def _save_processed_preview(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path) as image:
        processed = image.convert("RGB").resize((224, 224))
        processed.save(target_path)


def _write_run_result(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "results.json"
    results_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_saved_image(
    image_path: Path,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    run_id: str | None = None,
    device: str | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    loaded_model = load_model(
        checkpoint_path=checkpoint_path,
        labels_path=labels_path,
        device=device,
    )
    prediction = predict_image(
        image_path=image_path,
        model=loaded_model.model,
        labels=loaded_model.labels,
        device=loaded_model.device,
    )

    resolved_run_id = run_id or create_run_id()
    resolved_run_id = ensure_unique_run_id(resolved_run_id, results_root)
    image_id = slugify(image_path.stem)
    run_dir = results_root / resolved_run_id
    artifact_dir = run_dir / "artifacts"
    raw_target = artifact_dir / f"{image_id}-raw{image_path.suffix.lower()}"
    processed_target = artifact_dir / f"{image_id}-processed.png"
    heatmap_target = artifact_dir / f"{image_id}-gradcam.png"
    processed_artifact_path: Path | None = None
    heatmap_artifact_path: Path | None = None
    extra_logs: list[dict[str, str]] = []

    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, raw_target)

        try:
            _save_processed_preview(image_path, processed_target)
            processed_artifact_path = processed_target
        except Exception as exc:
            processed_target.unlink(missing_ok=True)
            extra_logs.append({"level": "WARN", "msg": f"Processed preview unavailable: {exc}"})

        try:
            save_gradcam(
                image_path,
                loaded_model.model,
                heatmap_target,
                device=loaded_model.device,
            )
            heatmap_artifact_path = heatmap_target
        except Exception as exc:
            heatmap_target.unlink(missing_ok=True)
            if EXPLAINABILITY_FAILURE_FLAG not in prediction.flags:
                prediction.flags = [*prediction.flags, EXPLAINABILITY_FAILURE_FLAG]
            prediction.risk_level = "REVIEW"
            extra_logs.append({"level": "WARN", "msg": f"Grad-CAM generation failed: {exc}"})

        payload = build_dashboard_result(
            run_id=resolved_run_id,
            run_dir=run_dir,
            image_id=image_id,
            prediction=prediction,
            checkpoint_path=checkpoint_path,
            architecture=loaded_model.architecture,
            raw_artifact_path=raw_target,
            processed_artifact_path=processed_artifact_path,
            heatmap_artifact_path=heatmap_artifact_path,
            extra_logs=extra_logs,
        )
        _write_run_result(run_dir, payload)
        return resolved_run_id, run_dir, payload
    except Exception:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def analyze_uploaded_panel(
    file_bytes: bytes,
    filename: str,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
    uploads_root: Path = DEFAULT_UPLOADS_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    device: str | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    run_id = create_run_id(Path(filename).stem or "panel")
    run_id = ensure_unique_run_id(run_id, uploads_root, results_root)
    upload_dir = uploads_root / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{slugify(Path(filename).stem)}{suffix}"

    try:
        upload_path.write_bytes(file_bytes)
        return analyze_saved_image(
            upload_path,
            checkpoint_path=checkpoint_path,
            labels_path=labels_path,
            results_root=results_root,
            run_id=run_id,
            device=device,
        )
    except Exception:
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise
