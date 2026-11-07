"""Config, geometry, Kalman and artifact contract tests."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from spana.artifacts import ArtifactError, tracks_frame, validate_tracks, write_json_atomic, TRACKS_SCHEMA
from spana.config import ConfigError, config_from_dict, frames_for, load_config
from spana.geometry import iou_matrix
from spana.tracking.kalman import BoxKalman
from tests.synth import CONFIG_PATH, Player, run


def _raw():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


class ConfigTests(unittest.TestCase):
    def test_default_loads_and_hash_is_stable(self):
        self.assertEqual(load_config(CONFIG_PATH).sha256(), config_from_dict(_raw()).sha256())

    def test_missing_unknown_and_bad_values_fail_closed(self):
        cases = []
        d = _raw(); del d["tracking"]["lost_max_s"]; cases.append(d)
        d = _raw(); d["tracking"]["extra"] = 1; cases.append(d)
        d = _raw(); d["detection"]["score_high"] = True; cases.append(d)
        d = _raw(); d["detection"]["score_low"] = 0.9; cases.append(d)       # low > high
        d = _raw(); d["tracking"]["predict_max_s"] = 9.0; cases.append(d)    # predict > lost
        d = _raw(); d["schema_version"] = "0.0.1"; cases.append(d)
        d = _raw(); d["camera"]["max_features"] = 1.5; cases.append(d)
        d = _raw(); d["kalman"]["pos_std_h"] = float("nan"); cases.append(d)
        for c in cases:
            with self.assertRaises(ConfigError):
                config_from_dict(c)

    def test_frames_for(self):
        self.assertEqual(frames_for(0.5, 30), 15)
        self.assertEqual(frames_for(0.001, 25), 1)
        with self.assertRaises(ConfigError):
            frames_for(1.0, 0)


class GeometryKalman(unittest.TestCase):
    def test_iou(self):
        m = iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]))
        np.testing.assert_allclose(m, [[1.0, 1 / 3, 0.0]])

    def test_kalman_learns_velocity_independent_of_fps(self):
        cfg = load_config(CONFIG_PATH).kalman
        for fps in (25.0, 60.0):
            kf = BoxKalman(np.array([0.0, 0.0, 32.0, 80.0]), cfg)
            for i in range(1, int(fps)):
                kf.predict(1 / fps)
                kf.update(np.array([200.0 * i / fps, 0.0, 32.0, 80.0]))
            self.assertAlmostEqual(kf.x[4], 200.0, delta=10.0)

    def test_camera_mapping_of_state(self):
        cfg = load_config(CONFIG_PATH).kalman
        kf = BoxKalman(np.array([100.0, 50.0, 20.0, 40.0]), cfg)
        kf.apply_camera(np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0]]))
        np.testing.assert_allclose(kf.x[:4], [210.0, 95.0, 40.0, 80.0])


class ArtifactTests(unittest.TestCase):
    def _doc(self):
        res = run([Player("A", (300, 500), (100, 0)), Player("B", (330, 500), (0, 0))], fps=30, duration_s=1)
        assert len(res.frames) > 1
        oc = load_config(CONFIG_PATH).output
        return {"schema": TRACKS_SCHEMA[0], "schema_version": TRACKS_SCHEMA[1], "provenance": {}, "video": {},
                "detections_sha256": "0" * 64, "frames": [tracks_frame(ft, oc) for ft in res.frames]}

    def test_tracker_output_validates_and_is_json(self):
        doc = self._doc()
        validate_tracks(doc)
        json.dumps(doc, allow_nan=False)

    def test_contract_violations_rejected(self):
        base = self._doc()
        tr = next(fr for fr in base["frames"][1:] if fr["tracks"])
        idx = base["frames"].index(tr)
        mutations = [
            lambda d: d.update(schema_version="9.9.9"),
            lambda d: d["frames"][idx]["tracks"][0].update(source="GUESSED"),
            lambda d: d["frames"][idx]["tracks"][0].update(detection_index=None),
            lambda d: d["frames"][idx]["tracks"][0].update(new_key=1),
            lambda d: d["frames"][idx]["tracks"].append(copy.deepcopy(d["frames"][idx]["tracks"][0])),
            lambda d: d["frames"].reverse(),
            lambda d: d["frames"][idx]["camera"].update(status="MAYBE"),
            lambda d: d["frames"][idx]["tracks"][0].update(overlap_with=None, max_overlap_iou=0.5),
            lambda d: d["frames"][idx]["tracks"][0].update(state="CONFIRMED", source="PREDICTED"),
            lambda d: d["frames"][idx]["tracks"][0].update(box=[10, 10, 5, 20]),
            lambda d: d["frames"][idx]["tracks"][0].update(box=["a", 1, 2, 3]),
            lambda d: d["frames"][idx]["tracks"][0].update(detection_index=-1),
            lambda d: d["frames"][idx]["tracks"][0].update(gaps_bridged=1, last_gap_s=0.0),
            lambda d: d["frames"][idx]["tracks"][0].update(epoch=-1),
            lambda d: d["frames"][idx]["camera"].update(affine=[[1, 0]]),
            lambda d: d["frames"][idx]["camera"].pop("inliers"),
            lambda d: d["frames"][idx].update(t=-1.0),
            lambda d: d["frames"][idx]["ended"].append(["T999999", "LOST_EXPIRED"]),
            lambda d: d["frames"][idx]["camera"].update(status="FIRST"),
        ]
        for m in mutations:
            d = copy.deepcopy(base)
            m(d)
            with self.assertRaises(ArtifactError):
                validate_tracks(d)

    def test_atomic_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            write_json_atomic(p, {"a": 1})
            with self.assertRaises(ArtifactError):
                write_json_atomic(p, {"a": 2})
            self.assertEqual(json.loads(p.read_text()), {"a": 1})


if __name__ == "__main__":
    unittest.main()
