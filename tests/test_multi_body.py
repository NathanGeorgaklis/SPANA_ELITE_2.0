"""Regression for the real DEV finding (FRESH_02): two players in contact produce one merged box next to one
player's own box; the merged box started competing tracks (fragmentation) and could contaminate appearance.
Merged boxes are FLAGGED, never deleted: no spawn, no appearance update, any match through one is contested."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spana.artifacts import ArtifactError, detection_frame, detections_from_frame, validate_detections
from spana.config import load_config
from spana.detection.postprocess import filter_detections, flag_multi_body
from spana.records import CameraObservation, CameraStatus, Detection
from spana.tracking.tracker import MultiPlayerTracker

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
PLAYER = [500, 300, 540, 420]                 # one player's own box (h 120, w 40)
MERGED = [495, 295, 590, 425]                 # that player plus a second body beside them (w 95)


def _flags(boxes, scores):
    return flag_multi_body(np.array(boxes, float), np.array(scores, float), CFG.detection)


class MultiBodyFlag(unittest.TestCase):
    def test_box_extending_beyond_confident_member_is_flagged_not_deleted(self):
        out = filter_detections(np.array([PLAYER, MERGED], float), np.array([0.9, 0.8]), np.zeros(2),
                                frame_w=1920, frame_h=1080, cfg=CFG.detection, keep_class=0)
        self.assertEqual({d.box: d.multi_body for d in out},
                         {tuple(map(float, PLAYER)): False, tuple(map(float, MERGED)): True})

    def test_full_body_around_low_score_partial_is_not_flagged(self):
        upper = [502, 300, 538, 360]
        self.assertEqual(_flags([PLAYER, upper], [0.9, 0.3]), [False, False])

    def test_loose_box_on_same_body_is_not_flagged(self):
        loose = [497, 297, 543, 425]          # < multibody_min_extent in both axes
        self.assertEqual(_flags([PLAYER, loose], [0.9, 0.8]), [False, False])

    def test_separate_players_are_not_flagged_and_order_does_not_matter(self):
        other = [700, 300, 740, 420]
        self.assertEqual(_flags([PLAYER, other], [0.9, 0.9]), [False, False])
        self.assertEqual(_flags([MERGED, PLAYER], [0.8, 0.9]), [True, False])


class MultiBodyArtifact(unittest.TestCase):
    def test_flag_round_trips_and_validator_is_strict(self):
        fr = detection_frame(0, 0.0, [Detection(tuple(map(float, MERGED)), 0.8, 0, True)], CFG.output)
        self.assertEqual(fr["detections"][0][6], 1)
        self.assertTrue(detections_from_frame(fr)[0].multi_body)


class MultiBodyTracking(unittest.TestCase):
    def _tracker(self):
        return MultiPlayerTracker(CFG, 1920, 1080)

    @staticmethod
    def _cam(i):
        return CameraObservation.identity(CameraStatus.FIRST if i == 0 else CameraStatus.OK)

    def test_flagged_box_cannot_start_a_track(self):
        t = self._tracker()
        ids = set()
        for i in range(10):
            dets = [Detection(tuple(map(float, PLAYER)), 0.9, 0)]
            if i >= 3:
                dets.append(Detection(tuple(map(float, MERGED)), 0.9, 0, True))
            ft = t.update(i, i / 30, dets, self._cam(i))
            ids |= {o.track_id for o in ft.tracks}
            if i >= 3:
                self.assertIn({"kind": "spawn_refused", "detection": 1, "reason": "MULTI_BODY_DETECTION"},
                              ft.decisions)
        self.assertEqual(len(ids), 1)

    def test_match_through_flagged_box_is_contested_and_keeps_appearance(self):
        t = self._tracker()
        own, other = np.full(4, 0.25), np.array([0.7, 0.1, 0.1, 0.1])
        for i in range(6):
            t.update(i, i / 30, [Detection(tuple(map(float, PLAYER)), 0.9, 0)], self._cam(i), appearance=[own])
        before = t._tracks[0].appearance.copy()
        box = tuple(float(v) for v in np.array(PLAYER) + [1, 0, 1, 0])
        ft = t.update(6, 6 / 30, [Detection(box, 0.9, 0, True)], self._cam(6), appearance=[other])
        (obs,) = ft.tracks
        self.assertEqual(obs.detection_index, 0)
        self.assertTrue(obs.contested)
        np.testing.assert_array_equal(t._tracks[0].appearance, before)
        self.assertTrue(any(d["kind"] == "contested_multi_body" for d in ft.decisions))


class RefilterMigration(unittest.TestCase):
    def test_refilter_upgrades_1_2_0_and_computes_flags(self):
        cfg_sec = {k: getattr(CFG.detection, k) for k in CFG.detection.__dataclass_fields__}
        old = {k: v for k, v in cfg_sec.items() if not k.startswith("multibody_")}
        src = {"schema": "spana.detections", "schema_version": "1.2.0", "provenance": {}, "frame_stride": 1,
               "video": {"sha256": "0" * 64}, "model": {}, "detection_config": old,
               "frames": [{"frame": 0, "t": 0.0, "detections": [[*map(float, PLAYER), 0.9, 0],
                                                                [*map(float, MERGED), 0.8, 0]]}]}
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / "in.json", Path(d) / "out.json"
            inp.write_text(json.dumps(src))
            r = subprocess.run([sys.executable, "-m", "tools.refilter_detections", "--detections", str(inp),
                                "--config", str(ROOT / "config" / "default.yaml"), "--out", str(out)],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            doc = json.loads(out.read_text())
        self.assertEqual(doc["schema_version"], "1.3.0")
        self.assertEqual([row[6] for row in doc["frames"][0]["detections"]], [0, 1])
        validate_detections(doc)
        for bad in ([*doc["frames"][0]["detections"][0][:6]], [*doc["frames"][0]["detections"][0][:6], True]):
            broken = {**doc, "frames": [{**doc["frames"][0], "detections": [bad]}]}
            with self.assertRaises(ArtifactError):
                validate_detections(broken)


if __name__ == "__main__":
    unittest.main()
