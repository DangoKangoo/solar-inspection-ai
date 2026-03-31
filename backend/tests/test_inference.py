from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import backend.app.pipeline.artifacts as artifacts_module
from backend.app.pipeline.inference import (
    TransferLearningResNet50,
    analyze_saved_image,
    validate_model_artifacts,
)


class InferencePipelineTests(unittest.TestCase):
    def test_validate_model_artifacts_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            issues = validate_model_artifacts(
                checkpoint_path=tmp_path / "missing.pth",
                labels_path=tmp_path / "missing.txt",
            )
            self.assertEqual(len(issues), 2)

    def test_validate_model_artifacts_reports_incompatible_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "TestModel.pth"
            labels_path = tmp_path / "classes.txt"

            torch.save(TransferLearningResNet50(num_classes=2).state_dict(), checkpoint_path)
            labels_path.write_text("normal\nhotspot\ncrack\n", encoding="utf-8")

            issues = validate_model_artifacts(
                checkpoint_path=checkpoint_path,
                labels_path=labels_path,
            )

            self.assertTrue(
                any("Checkpoint is not compatible with 3 labels" in issue for issue in issues)
            )

    def test_validate_model_artifacts_reports_empty_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "TestModel.pth"
            labels_path = tmp_path / "classes.txt"

            torch.save(TransferLearningResNet50(num_classes=2).state_dict(), checkpoint_path)
            labels_path.write_text("", encoding="utf-8")

            issues = validate_model_artifacts(
                checkpoint_path=checkpoint_path,
                labels_path=labels_path,
            )

            self.assertTrue(any("Invalid labels file" in issue for issue in issues))

    def test_analyze_saved_image_writes_dashboard_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "TestModel.pth"
            labels_path = tmp_path / "classes.txt"
            image_path = tmp_path / "panel.png"
            results_root = tmp_path / "results"

            labels_path.write_text("normal\nhotspot\n", encoding="utf-8")
            model = TransferLearningResNet50(num_classes=2)
            torch.save(model.state_dict(), checkpoint_path)
            Image.new("RGB", (320, 240), color=(128, 96, 64)).save(image_path)

            run_id, run_dir, payload = analyze_saved_image(
                image_path=image_path,
                checkpoint_path=checkpoint_path,
                labels_path=labels_path,
                results_root=results_root,
                run_id="panel-test-run",
                device="cpu",
            )

            self.assertEqual(run_id, "panel-test-run")
            self.assertTrue((run_dir / "results.json").exists())

            saved_payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["run_id"], "panel-test-run")
            self.assertEqual(len(saved_payload["panels"]), 1)
            self.assertIn(saved_payload["panels"][0]["predicted_class"], {"normal", "hotspot"})
            self.assertEqual(saved_payload["panels"][0]["artifacts"]["raw"], "artifacts/panel-raw.png")
            self.assertEqual(
                saved_payload["panels"][0]["artifacts"]["processed"],
                "artifacts/panel-processed.png",
            )
            self.assertEqual(
                saved_payload["panels"][0]["artifacts"]["heatmap"],
                "artifacts/panel-gradcam.png",
            )
            self.assertTrue((run_dir / saved_payload["panels"][0]["artifacts"]["raw"]).exists())
            self.assertTrue((run_dir / saved_payload["panels"][0]["artifacts"]["processed"]).exists())
            self.assertTrue((run_dir / saved_payload["panels"][0]["artifacts"]["heatmap"]).exists())
            self.assertEqual(payload["run_id"], "panel-test-run")

    def test_analyze_saved_image_keeps_run_when_gradcam_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "TestModel.pth"
            labels_path = tmp_path / "classes.txt"
            image_path = tmp_path / "panel.png"
            results_root = tmp_path / "results"

            labels_path.write_text("normal\nhotspot\n", encoding="utf-8")
            torch.save(TransferLearningResNet50(num_classes=2).state_dict(), checkpoint_path)
            Image.new("RGB", (320, 240), color=(64, 96, 128)).save(image_path)

            with patch.object(artifacts_module, "save_gradcam", side_effect=RuntimeError("gradcam failed")):
                run_id, run_dir, payload = analyze_saved_image(
                    image_path=image_path,
                    checkpoint_path=checkpoint_path,
                    labels_path=labels_path,
                    results_root=results_root,
                    run_id="panel-gradcam-failure",
                    device="cpu",
                )

            self.assertEqual(run_id, "panel-gradcam-failure")
            self.assertTrue((run_dir / "results.json").exists())
            panel = payload["panels"][0]
            self.assertEqual(panel["risk_level"], "REVIEW")
            self.assertIn("EXPLAINABILITY_FAILED", panel["flags"])
            self.assertIsNone(panel["artifacts"]["heatmap"])
            self.assertTrue(any("Grad-CAM generation failed" in item["msg"] for item in payload["logs"]))


if __name__ == "__main__":
    unittest.main()
