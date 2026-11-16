"""Football detector -> tracking: person view (class filter, class-agnostic NMS, score_low request), detections
artifact class_names, licence gate on `detect --backend roboflow`, and track on the result. No network."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from spana import cli
from spana.artifacts import ArtifactError, read_json, validate_detections, validate_tracks
from spana.config import load_config
from spana.dataset.config import PolicyError
from spana.detection.person_view import PersonView, PersonViewError
from spana.records import Detection
from tests.synth import CONFIG_PATH
from tests.test_cli_e2e import write_video
from tests.test_roboflow_prep import raw as raw_dataset_cfg

CFG = load_config(CONFIG_PATH)
NAMES = ("player", "goalkeeper", "referee", "ball")


class FakeDetector:
    class_names = NAMES

    def __init__(self, dets, confidence=0.10):
        self._dets, self._conf = dets, confidence

    @property
    def model_id(self):
        return {"backend": "fake", "confidence": self._conf}

    def detect(self, frame):
        return list(self._dets)


class PersonViewTests(unittest.TestCase):
    frame = np.zeros((180, 320, 3), np.uint8)

    def view(self, dets, **kw):
        return PersonView(FakeDetector(dets, **kw), ("player", "goalkeeper", "referee"), CFG.detection)

    def test_ball_dropped_and_cross_class_duplicate_merged(self):
        dets = [Detection((100.0, 50.0, 120.0, 100.0), 0.40, 2),     # same person as referee, lower score
                Detection((101.0, 51.0, 121.0, 101.0), 0.70, 0),     # player: kept, class hint kept
                Detection((200.0, 90.0, 204.0, 94.0), 0.95, 3),      # ball: not a person
                Detection((250.0, 50.0, 270.0, 100.0), 0.05, 1),     # below detection.score_low
                Detection((10.0, 10.0, 14.0, 11.0), 0.90, 0),        # shorter than min_box_height_frac
                Detection((40.0, 50.0, 60.0, 100.0), 0.20, 1)]       # weak goalkeeper: kept for stage 2
        out = self.view(dets).detect(self.frame)
        self.assertEqual([(d.class_id, d.score) for d in out], [(0, 0.70), (1, 0.20)])

    def test_refusals(self):
        with self.assertRaisesRegex(PersonViewError, "score_low"):
            self.view([], confidence=0.50)                            # would starve the low-score round
        with self.assertRaisesRegex(PersonViewError, "not all in the detector's classes"):
            PersonView(FakeDetector([]), ("player", "linesman"), CFG.detection)

    def test_class_names_required_in_artifact(self):
        doc = {"schema": "spana.detections", "schema_version": "1.4.0", "provenance": {}, "video": {}, "model": {},
               "class_names": ["person"], "detection_config": {}, "frame_stride": 1,
               "frames": [{"frame": 0, "t": 0.0, "detections": [[1.0, 1.0, 5.0, 9.0, 0.9, 1, 0]]}]}
        with self.assertRaisesRegex(ArtifactError, "no entry in class_names"):
            validate_detections(doc)
        for bad in ([], ["person", "person"], None):
            with self.assertRaises(ArtifactError):
                validate_detections({**doc, "class_names": bad})


class FakeClient:
    """Hosted inference stand-in: one player, the same player as a weaker referee, and a ball, every frame."""
    calls = []

    def __init__(self, *a, **k):
        pass

    def infer(self, image_jpeg, *, model_id, confidence, iou):
        FakeClient.calls.append(confidence)
        return {"image": {"width": 320, "height": 180}, "predictions": [
            {"x": 160, "y": 90, "width": 20, "height": 50, "confidence": 0.9, "class": "player"},
            {"x": 161, "y": 91, "width": 20, "height": 50, "confidence": 0.3, "class": "referee"},
            {"x": 60, "y": 150, "width": 6, "height": 6, "confidence": 0.8, "class": "ball"}]}


class RoboflowBackendCli(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "clip.avi"
        write_video(self.video, 8, 25.0)
        d = raw_dataset_cfg()
        d["licensed_matches"] = [{"match_id": "M_DEV", "split": "dev", "permission": "test",
                                  "third_party_processing_confirmed": True},
                                 {"match_id": "M_EVAL", "split": "eval", "permission": "test",
                                  "third_party_processing_confirmed": True}]
        d["preflight"]["split_matches"] = ["train:M_X", "valid:M_Y"]
        self.dcfg = self.td / "roboflow.yaml"
        self.dcfg.write_text(yaml.safe_dump(d))
        FakeClient.calls = []

    def detect(self, match_id, out):
        with mock.patch("spana.detection.roboflow_backend.HttpInferenceClient", FakeClient), \
                mock.patch("spana.secrets.load_secret", return_value="k"):
            return cli.main(["detect", "--backend", "roboflow", "--video", str(self.video), "--config",
                             str(CONFIG_PATH), "--dataset-config", str(self.dcfg), "--match-id", match_id,
                             "--out", str(out)])

    def test_detect_then_track(self):
        dpath, tpath = self.td / "d.json", self.td / "t.json"
        self.assertEqual(self.detect("M_DEV", dpath), 0)
        ddoc = read_json(dpath)
        validate_detections(ddoc)
        self.assertEqual(ddoc["class_names"], list(NAMES))
        self.assertEqual(ddoc["model"]["person_view"], {"person_classes": ["player", "goalkeeper", "referee"]})
        self.assertEqual(set(FakeClient.calls), {CFG.detection.score_low})       # asked down to score_low
        for fr in ddoc["frames"]:                                                 # ball gone, duplicate merged
            self.assertEqual([(r[5], r[4]) for r in fr["detections"]], [(0, 0.9)])
        self.assertEqual(cli.main(["track", "--video", str(self.video), "--config", str(CONFIG_PATH),
                                   "--detections", str(dpath), "--out", str(tpath)]), 0)
        validate_tracks(read_json(tpath), ddoc)

    def test_licence_gate(self):
        with self.assertRaisesRegex(PolicyError, "registered for split 'eval'"):
            self.detect("M_EVAL", self.td / "e.json")
        with self.assertRaisesRegex(PolicyError, "not in licensed_matches"):
            self.detect("GAME_NOT_LISTED", self.td / "g.json")
        self.assertEqual(FakeClient.calls, [])                                    # nothing was sent


if __name__ == "__main__":
    unittest.main()
