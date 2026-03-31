from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.pipeline.inference import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_RESULTS_ROOT,
    analyze_saved_image,
    build_dashboard_result as _build_dashboard_result,
    load_model as _load_model,
    predict_image as _predict_image,
    prepare_image as _prepare_image,
)


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
):
    return _load_model(checkpoint_path=checkpoint_path, labels_path=labels_path)


def prepare_image(path_to_image: str | Path):
    return _prepare_image(path_to_image)


def predict_image(path_to_image: str | Path, model, labels, device=None):
    return _predict_image(path_to_image, model=model, labels=labels, device=device)


def build_dashboard_result(**kwargs):
    return _build_dashboard_result(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run solar panel inference on a single image.")
    parser.add_argument("image_path", type=Path, help="Path to a cropped panel image.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id, run_dir, payload = analyze_saved_image(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        labels_path=args.labels,
        results_root=args.results_root,
        run_id=args.run_id,
        device=args.device,
    )
    panel = payload["panels"][0]
    print(f"Run saved to {run_dir}")
    print(f"Run id: {run_id}")
    print(f"Predicted class: {panel['predicted_class']}")
    print(f"Confidence: {panel['confidence']:.4f}")
    print(f"Fault probability: {panel['fault_probability']:.4f}")
    print(f"Risk level: {panel['risk_level']}")


if __name__ == "__main__":
    main()
