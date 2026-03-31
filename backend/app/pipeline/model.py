from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "TestModel.pth"
DEFAULT_LABELS_PATH = REPO_ROOT / "models" / "checkpoints" / "classes.txt"
NORMAL_LABEL_ALIASES = {"normal", "healthy", "ok", "no_defect", "no-defect", "clean"}


class TransferLearningResNet50(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        features = self.flatten(features)
        return self.classifier(features)


@dataclass
class LoadedModel:
    model: nn.Module
    labels: list[str]
    device: torch.device
    architecture: str


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    fault_probability: float
    risk_level: str
    flags: list[str]
    probabilities: dict[str, float]


def choose_device(device: str | None = None) -> torch.device:
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def validate_model_artifacts(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
) -> list[str]:
    issues: list[str] = []
    resolved_labels: list[str] | None = None
    state_dict: dict[str, Any] | None = None

    if not checkpoint_path.exists():
        issues.append(f"Missing checkpoint: {checkpoint_path}")
    if not labels_path.exists():
        issues.append(f"Missing labels file: {labels_path}")

    if labels_path.exists():
        try:
            resolved_labels = load_labels(labels_path)
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            issues.append(f"Invalid labels file: {reason}")

    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            issues.append(f"Invalid checkpoint: {reason}")

    if state_dict is not None and resolved_labels is not None:
        try:
            _load_state_dict_with_fallbacks(state_dict, len(resolved_labels))
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            issues.append(
                f"Checkpoint is not compatible with {len(resolved_labels)} labels: {reason}"
            )

    return issues


def load_labels(labels_path: Path = DEFAULT_LABELS_PATH) -> list[str]:
    labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not labels:
        raise ValueError(f"No labels found in {labels_path}")
    return labels


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return dict(checkpoint["state_dict"])
    if isinstance(checkpoint, dict):
        return dict(checkpoint)
    raise TypeError("Checkpoint must be a state dict or a dict with a state_dict key.")


def _build_transfer_model(num_classes: int) -> nn.Module:
    return TransferLearningResNet50(num_classes)


def _build_plain_resnet_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_state_dict_with_fallbacks(state_dict: dict[str, Any], num_classes: int) -> tuple[nn.Module, str]:
    builders = (
        ("resnet50_transfer", _build_transfer_model),
        ("resnet50_fc", _build_plain_resnet_model),
    )
    errors: list[str] = []
    for name, builder in builders:
        candidate = builder(num_classes)
        try:
            candidate.load_state_dict(state_dict)
            return candidate, name
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Could not load checkpoint with known model layouts.\n" + "\n".join(errors))


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    labels: Sequence[str] | None = None,
    labels_path: Path = DEFAULT_LABELS_PATH,
    device: str | None = None,
) -> LoadedModel:
    issues = validate_model_artifacts(checkpoint_path=checkpoint_path, labels_path=labels_path)
    if issues:
        message = "\n".join(issues)
        if all(issue.startswith("Missing ") for issue in issues):
            raise FileNotFoundError(message)
        raise ValueError(message)

    resolved_labels = list(labels) if labels is not None else load_labels(labels_path)
    resolved_device = choose_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    state_dict = _extract_state_dict(checkpoint)
    model, architecture = _load_state_dict_with_fallbacks(state_dict, len(resolved_labels))
    model.to(resolved_device)
    model.eval()
    return LoadedModel(
        model=model,
        labels=resolved_labels,
        device=resolved_device,
        architecture=architecture,
    )


def prepare_image(image_path: str | Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return transform(rgb_image)


def _fault_probability(probabilities: dict[str, float]) -> float:
    normal_probability = 0.0
    for label, probability in probabilities.items():
        normalized = label.strip().lower().replace(" ", "_")
        if normalized in NORMAL_LABEL_ALIASES:
            normal_probability += probability
    if normal_probability > 0.0:
        return max(0.0, min(1.0, 1.0 - normal_probability))
    return max(probabilities.values())


def _risk_level(fault_probability: float, confidence: float) -> tuple[str, list[str]]:
    flags: list[str] = []
    if fault_probability > 0.78:
        risk_level = "HIGH"
    elif fault_probability > 0.45:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if confidence < 0.55:
        flags.append("LOW_CONFIDENCE")
        risk_level = "REVIEW"

    return risk_level, flags


def predict_image(
    image_path: str | Path,
    model: nn.Module,
    labels: Sequence[str],
    device: str | torch.device | None = None,
) -> PredictionResult:
    resolved_device = device if isinstance(device, torch.device) else choose_device(str(device or "auto"))
    inputs = prepare_image(image_path).unsqueeze(0).to(resolved_device)

    with torch.no_grad():
        logits = model(inputs)
        scores = torch.softmax(logits, dim=1).squeeze(0).cpu()

    probabilities = {
        label: float(scores[index].item())
        for index, label in enumerate(labels)
    }
    top_index = int(torch.argmax(scores).item())
    confidence = float(scores[top_index].item())
    predicted_class = labels[top_index]
    fault_probability = _fault_probability(probabilities)
    risk_level, flags = _risk_level(fault_probability, confidence)

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        fault_probability=fault_probability,
        risk_level=risk_level,
        flags=flags,
        probabilities=probabilities,
    )
