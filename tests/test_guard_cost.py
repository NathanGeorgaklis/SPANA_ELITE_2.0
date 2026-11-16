"""tools/guard_cost.py: guard timing (threshold, minimum duration, hold) and per-slice bookkeeping."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spana.artifacts import validate_tracks, write_json_atomic
from tests.test_reid_headroom import _labels
from tests.test_target_follow import CFG, A, N, tracks_doc
from tools.guard_cost import analyse_clip, guarded_frames, pooled


def _answers_and_tracks(overlaps, states=None):
    answers, frames = [], {}
    for n, ov in enumerate(overlaps):
        answers.append({"frame": n, "t": n / 10, "state": (states or {}).get(n, "TARGET")})
        frames[n] = {"tracks": [{"track_id": "t1", "max_overlap_iou": ov}]}
    return answers, frames


class GuardTimingTest(unittest.TestCase):
    OV = [0, 0, .5, .5, .5, 0, 0, 0, 0, 0]

    def test_immediate(self):
        a, f = _answers_and_tracks(self.OV)
        self.assertEqual(guarded_frames(a, f, "t1", 0.3, 0.0, 0.0), {2, 3, 4})

    def test_minimum_duration(self):
        a, f = _answers_and_tracks(self.OV)
        self.assertEqual(guarded_frames(a, f, "t1", 0.3, 0.1, 0.0), {3, 4})
        self.assertEqual(guarded_frames(a, f, "t1", 0.3, 0.5, 0.0), set())    # overlap never lasts 0.5 s

    def test_hold_after_overlap(self):
        a, f = _answers_and_tracks(self.OV)
        self.assertEqual(guarded_frames(a, f, "t1", 0.3, 0.0, 0.2), {2, 3, 4, 5, 6})

    def test_threshold_and_only_target_answers(self):
        a, f = _answers_and_tracks(self.OV, states={3: "ABSTAIN"})
        self.assertEqual(guarded_frames(a, f, "t1", 0.6, 0.0, 0.0), set())
        self.assertEqual(guarded_frames(a, f, "t1", 0.3, 0.0, 0.0), {2, 4})


class GuardCostClipTest(unittest.TestCase):
    def test_slices(self):
        tracks = tracks_doc([{"t1": N, "t2": N}] + [{"t1": A, "t2": A}] * 5)
        for fr in tracks["frames"][2:3]:                       # t1 and t2 overlap on frame 2
            for t in fr["tracks"]:
                t["max_overlap_iou"], t["overlap_with"] = 0.5, ("t2" if t["track_id"] == "t1" else "t1")
        validate_tracks(tracks)
        rows = []
        for f in (1, 2, 4, 5):                               # the frames _labels marks as labelled
            rows += [f"{f + 1},1,0,100,40,100,1,1,1", f"{f + 1},2,100,100,40,100,1,1,1"]
        ball = {"video": {"sha256": "v" * 64},
                "frames": [{"frame": n, "t": n / 10, "boxes": [[5.0, 195.0, 15.0, 205.0, 0.9]] if n == 2 else []}
                           for n in range(6)]}
        with tempfile.TemporaryDirectory() as tmp:
            _labels(Path(tmp) / "lab", rows)
            trk, bp = Path(tmp) / "trk.json", Path(tmp) / "ball.json"
            write_json_atomic(trk, tracks)
            write_json_atomic(bp, ball)
            rep = analyse_clip(Path(tmp) / "lab", trk, bp, CFG, [(0.3, 0.0, 0.0)], [1.0])
        v = rep["variants"][0]
        self.assertEqual((v["all"]["v1_correct"], v["all"]["correct_lost"]), (8, 2))      # both players, frame 2
        self.assertEqual((v["near_ball_1h"]["v1_correct"], v["near_ball_1h"]["correct_lost"]), (1, 1))
        self.assertEqual(v["contested"]["v1_correct"], 0)
        self.assertEqual(pooled([rep])[0]["all"]["correct_lost_share"], 0.25)


if __name__ == "__main__":
    unittest.main()
