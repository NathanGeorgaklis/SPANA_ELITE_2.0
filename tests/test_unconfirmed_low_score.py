"""Regression for the real DEV finding (FRESH_02): the low-score round ran for unconfirmed tracks before their
high-score round, so a new track took its player's low-score partial box and the high-score full-body box then
started a duplicate track. Unconfirmed tracks now only match confident detections."""
from __future__ import annotations

import unittest
from pathlib import Path

from spana.config import load_config
from spana.records import CameraObservation, CameraStatus, Detection
from spana.tracking.tracker import MultiPlayerTracker

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
FULL = (500.0, 300.0, 540.0, 420.0)            # full body, h 120
PARTIAL = (501.0, 300.0, 539.0, 372.0)         # same player's upper body (IoU with FULL 0.6), low score


def _cam(i):
    return CameraObservation.identity(CameraStatus.FIRST if i == 0 else CameraStatus.OK, inliers=100)


class UnconfirmedTrackLowScore(unittest.TestCase):
    def test_new_track_uses_full_body_box_and_no_duplicate_starts(self):
        self.assertLess(0.3, CFG.detection.score_high)
        t = MultiPlayerTracker(CFG, 1920, 1080)
        t.update(0, 0.0, [Detection(FULL, 0.9, 0)], _cam(0))
        ft = t.update(1, 1 / 30, [Detection(PARTIAL, 0.3, 0), Detection(FULL, 0.85, 0)], _cam(1))
        (obs,) = ft.tracks
        self.assertFalse(obs.confirmed)
        self.assertEqual(obs.detection_index, 1)
        self.assertFalse(any(d["kind"] == "spawn" for d in ft.decisions))

    def test_confirmed_track_still_continues_through_a_low_score_frame(self):
        t = MultiPlayerTracker(CFG, 1920, 1080)
        for i in range(10):
            ft = t.update(i, i / 30, [Detection(FULL, 0.9, 0)], _cam(i))
        self.assertTrue(ft.tracks[0].confirmed)
        ft = t.update(10, 10 / 30, [Detection(FULL, 0.3, 0)], _cam(10))
        (obs,) = ft.tracks
        self.assertEqual((obs.detection_index, obs.track_id), (0, ft.tracks[0].track_id))
        self.assertTrue(any(d["kind"] == "match" and d["stage"] == "S2" for d in ft.decisions))


if __name__ == "__main__":
    unittest.main()
