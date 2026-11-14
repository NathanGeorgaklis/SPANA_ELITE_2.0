"""Detection -> tracking interface (docs/INTERFACES.md, "Detection -> tracking"): the trained detector's adapter,
through the existing per-frame record builder, gives per-frame detections with frame index, class, score and box,
and the confidence threshold comes from config. No network (fake inference client)."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from spana.artifacts import DETECTIONS_SCHEMA, detection_frame, detections_from_frame, validate_detections
from spana.config import load_config
from spana.detection.roboflow_backend import RoboflowDetector
from tests.test_roboflow_prep import make_cfg

REPO = Path(__file__).resolve().parent.parent
W, H = 192, 108


class FakeClient:
    def __init__(self):
        self.calls = []

    def infer(self, image_jpeg, *, model_id, confidence, iou):
        self.calls.append((model_id, confidence, iou))
        return {"image": {"width": W, "height": H}, "predictions": [
            {"x": 20, "y": 50, "width": 10, "height": 30, "confidence": 0.91, "class": "player"},
            {"x": 60, "y": 50, "width": 10, "height": 30, "confidence": 0.77, "class": "goalkeeper"},
            {"x": 100, "y": 50, "width": 10, "height": 30, "confidence": 0.66, "class": "referee"},
            {"x": 140, "y": 90, "width": 4, "height": 4, "confidence": 0.58, "class": "ball"},
            {"x": 180, "y": 50, "width": 10, "height": 30, "confidence": 0.20, "class": "player"},   # below config
            {"x": 188, "y": 50, "width": 20, "height": 30, "confidence": 0.95, "class": "player"},   # cut at edge
        ]}


class DetectionTrackingInterfaceTests(unittest.TestCase):
    def test_adapter_output_matches_the_interface(self):
        cfg = make_cfg(lambda d: d["detector"].update(model_id="proj/2"))
        oc = load_config(REPO / "config" / "default.yaml").output
        client = FakeClient()
        det = RoboflowDetector.from_detector_config(cfg, "k", client=client)
        frames = [detection_frame(fi, fi / 25.0, det.detect(np.zeros((H, W, 3), np.uint8)), oc) for fi in (0, 5)]

        # threshold: taken from config, sent with the request and applied to what comes back
        self.assertEqual(client.calls, [("proj/2", cfg.detector.confidence, cfg.detector.iou)] * 2)
        doc = {"schema": DETECTIONS_SCHEMA[0], "schema_version": DETECTIONS_SCHEMA[1], "provenance": {},
               "video": {}, "model": det.model_id, "detection_config": {}, "frame_stride": 5, "frames": frames}
        validate_detections(doc)                                   # the artifact the tracker reads accepts it
        self.assertEqual(doc["model"]["confidence"], cfg.detector.confidence)

        for fr in frames:
            self.assertIsInstance(fr["frame"], int)                # frame index (source video frame)
            self.assertEqual(set(fr), {"frame", "t", "detections"})
            dets = detections_from_frame(fr)
            names = [det.class_names[d.class_id] for d in dets]
            self.assertEqual(sorted(names), ["ball", "goalkeeper", "player", "player", "referee"])
            self.assertEqual(det.class_names, tuple(cfg.classes))  # class_id indexes the configured class list
            for d, row in zip(dets, fr["detections"]):
                self.assertEqual(len(row), 7)                      # x1 y1 x2 y2 score class_id multi_body
                self.assertGreaterEqual(d.score, cfg.detector.confidence)
                x1, y1, x2, y2 = d.box                             # source pixels, inside the frame
                self.assertTrue(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H)
                self.assertFalse(d.multi_body)                     # this backend never sets the flag
        self.assertEqual([f["frame"] for f in frames], [0, 5])


if __name__ == "__main__":
    unittest.main()
