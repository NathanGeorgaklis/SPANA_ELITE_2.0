"""Detector evaluation tool and the adapter constructor for the trained model. No network (fake client)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spana.artifacts import sha256_doc
from spana.dataset.config import PolicyError
from spana.detection.roboflow_backend import RoboflowDetector, RoboflowError
from tests.test_roboflow_prep import img, make_cfg
from tools import detector_eval as de

SECRET = "rf_TEST_SECRET_never_leaks_789"


def cfg_with_model(**over):
    return make_cfg(lambda d: (d["detector"].update(model_id="proj/2", dataset_version=2),
                               d["detector_eval"].update(over)))


def box(x, s=20.0):
    return [x, 100.0, x + s, 100.0 + s]


def frame(k, gt, pred):
    return {"name": f"M2_f{k}.jpg", "id": f"id{k:03d}", "gt": [{"class": c, "box": box(x)} for c, x in gt],
            "pred": [{"class": c, "box": box(x), "score": s} for c, x, s in pred]}


class FakeClient:
    def __init__(self, preds):
        self.preds, self.calls = preds, []

    def infer(self, image_jpeg, *, model_id, confidence, iou):
        self.calls.append((model_id, confidence, iou))
        return {"image": {"width": 64, "height": 36}, "predictions": self.preds}


class AdapterTests(unittest.TestCase):
    def test_trained_model_adapter_uses_detector_config(self):
        cfg = cfg_with_model()
        client = FakeClient([{"x": 10, "y": 10, "width": 4, "height": 4, "confidence": 0.9, "class": "ball"},
                             {"x": 30, "y": 10, "width": 4, "height": 8, "confidence": 0.2, "class": "referee"},
                             {"x": 30, "y": 20, "width": 4, "height": 8, "confidence": 0.9, "class": "linesman"}])
        det = RoboflowDetector.from_detector_config(cfg, SECRET, client=client)
        out = det.detect(np.zeros((36, 64, 3), np.uint8))
        self.assertEqual(client.calls, [("proj/2", cfg.detector.confidence, cfg.detector.iou)])
        self.assertEqual([det.class_names[d.class_id] for d in out], ["ball"])   # below confidence / unknown dropped
        self.assertEqual(dict(det.dropped_labels), {"linesman": 1})
        low = RoboflowDetector.from_detector_config(cfg, SECRET, client=client, confidence=0.01)
        self.assertEqual(len(low.detect(np.zeros((36, 64, 3), np.uint8))), 2)

    def test_no_trained_model_configured(self):
        with self.assertRaisesRegex(RoboflowError, "model_id is empty"):
            RoboflowDetector.from_detector_config(make_cfg(), SECRET, client=FakeClient([]))


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_with_model(render_frames=3, worst_per_class=1)
        self.frames = [frame(0, [("player", 0), ("ball", 100)], [("player", 0, 0.9), ("ball", 100, 0.9)]),
                       frame(1, [("player", 0), ("ball", 100)], [("referee", 0, 0.9), ("ball", 300, 0.9)]),
                       frame(2, [("player", 0)], [("player", 0, 0.9), ("player", 200, 0.9), ("player", 300, 0.2)]),
                       frame(3, [("player", 0)], [("player", 0, 0.9)])]

    def test_metrics_and_size_recall(self):
        r = de.evaluate(self.cfg, self.frames)
        self.assertEqual(r["confusion"]["player"]["referee"], 1)
        self.assertEqual(r["per_class"]["ball"]["recall"], 0.5)
        self.assertEqual(r["per_class"]["player"]["fp"], 1)                     # the 0.2 box is below confidence
        self.assertEqual(r["recall_by_short_side_px"]["iou_0.5"], {"16-24": {"labels": 2, "found": 1, "recall": 0.5}})
        self.assertEqual(r["frames"], 4)

    def test_worst_frames_first_then_samples(self):
        picks = de.pick_frames(self.cfg, self.frames)
        self.assertEqual([(p["index"], p["why"]) for p in picks[:2]],
                         [(1, "worst-ball-2-errors"), (2, "worst-player-1-errors")])   # frame 1 is not picked twice
        self.assertEqual(len(picks), 3)
        self.assertEqual(picks[2]["why"], "sample")

    def test_labels_must_be_the_frozen_snapshot_of_the_trained_version(self):
        doc = {"frames": [{"split": "valid", "name": "a"}, {"split": "train", "name": "b"}, {"split": "test", "name": "c"}]}
        with tempfile.TemporaryDirectory() as td:
            vdir = Path(td) / self.cfg.paths.version_dir
            vdir.mkdir(parents=True)
            labels = vdir / "labels.json"
            labels.write_text(json.dumps(doc))
            with mock.patch.object(de, "REPO", Path(td)):
                (vdir / "version_2.json").write_text(json.dumps({"labels_sha256": sha256_doc(doc)}))
                self.assertEqual([e["name"] for e in de.load_frozen_labels(self.cfg, labels, 2)], ["a"])
                (vdir / "version_2.json").write_text(json.dumps({"labels_sha256": "0" * 64}))
                with self.assertRaisesRegex(PolicyError, "not the label snapshot frozen"):
                    de.load_frozen_labels(self.cfg, labels, 2)
                (vdir / "version_1.json").write_text(json.dumps({"labels_sha256": sha256_doc(doc)}))
                with self.assertRaisesRegex(PolicyError, "not detector.dataset_version"):
                    de.load_frozen_labels(self.cfg, labels, 1)

    def test_predict_applies_the_licence_gate_and_frozen_ids(self):
        det = RoboflowDetector.from_detector_config(self.cfg, SECRET, client=FakeClient([]))
        entry = {"id": "id001", "name": "M_OTHER_f0000001.jpg", "boxes": []}
        with self.assertRaisesRegex(PolicyError, "allow-list"):
            de.predict(self.cfg, None, det, [entry], {"id001": img(1, mid="M_OTHER")}, Path("unused"))
        with self.assertRaisesRegex(PolicyError, "no longer in the project"):
            de.predict(self.cfg, None, det, [entry], {}, Path("unused"))

    def test_config_refuses_the_test_split(self):
        from spana.config import ConfigError
        with self.assertRaises(ConfigError):
            cfg_with_model(split="test")


if __name__ == "__main__":
    unittest.main()
