"""After-match review list (target.review_min_gap_s, spana.identity.follower.review_list).

The review shows only gaps without a TARGET answer lasting at least review_min_gap_s; the spana.target document
keeps every gap. Motivation: records/resume_after_doubt_2026-11-18 (the simulated reviewer tapped short gaps that
would have resolved on their own)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import write_json_atomic
from spana.config import ConfigError
from spana.identity.follower import gap_duration_s, review_list
from tests.test_target_recovery import CFG, dropped_offer


def with_min(s: float):
    return dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, review_min_gap_s=s))


def gap(t0: float, t1: float) -> dict:
    return {"start_frame": int(t0 * 10), "end_frame": int(t1 * 10), "start_t": t0, "end_t": t1, "frames": 1,
            "states": ["LOST"], "offers": []}


class ReviewList(unittest.TestCase):
    def test_zero_lists_every_gap(self):
        gaps = [gap(0.0, 0.0), gap(1.0, 1.2), gap(3.0, 5.0)]
        self.assertEqual(review_list(gaps, with_min(0.0)), gaps)

    def test_short_gaps_are_not_listed(self):
        gaps = [gap(0.0, 0.0), gap(1.0, 1.2), gap(3.0, 5.0)]
        self.assertEqual(review_list(gaps, with_min(0.2)), gaps[1:])        # boundary included
        self.assertEqual(review_list(gaps, with_min(1.0)), gaps[2:])
        self.assertEqual(gap_duration_s(gaps[2]), 2.0)

    def test_negative_minimum_is_refused(self):
        with self.assertRaises(ConfigError):
            dataclasses.replace(CFG.target, review_min_gap_s=-1.0).validate()


class ReviewerUsesTheList(unittest.TestCase):
    def _score(self, cfg):
        from tools.target_score import score_clip
        tracks = dropped_offer()
        rows = [f"{f + 1},1,{x},100,40,100,1,1,1" for f, x in ((1, 0), (2, 300), (4, 200), (5, 200))]
        gt = ("\n".join(rows) + "\n").encode()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "lab"
            d.mkdir()
            (d / "gt.txt").write_bytes(gt)
            (d / "FROZEN.json").write_text(json.dumps({
                "split": "DEV", "gt_sha256": hashlib.sha256(gt).hexdigest(), "frame_step": 1,
                "clip": {"name": "synthetic", "width": 640, "height": 360, "sha256": "v" * 64}}))
            (d / "labels_frozen.json").write_text(json.dumps({"frames_labelled": [1, 2, 4, 5]}))
            trk = Path(tmp) / "trk.json"
            write_json_atomic(trk, tracks)
            return score_clip(d, trk, cfg, None, simulate_reviewer=True)["per_player"][0]

    def test_a_gap_shorter_than_the_minimum_is_left_alone(self):
        everything = self._score(with_min(0.0))
        self.assertGreaterEqual(sum(everything["actions"].values()), 1)
        nothing = self._score(with_min(100.0))                              # longer than the whole clip
        self.assertEqual(sum(nothing["actions"].values()), 0)
        self.assertLessEqual(nothing["correct"], everything["correct"])


if __name__ == "__main__":
    unittest.main()
