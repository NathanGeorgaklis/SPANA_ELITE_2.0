"""tracking.rival_eligible_only (records/rival_eligible_2026-11-19; Nathan's FRESH_08 review, player 10).

S4's RIVAL_TRACK refusal and S3c's exclusivity check counted every track whose gate held the detection - also a stale
track that could no longer take any detection (not observed on the previous frame and past its recovery window). On
DEV such a track (the player's own old track) blocked an extent link and a recovery and split one player three
times. With the flag, only a track that could take the detection on this frame counts. A still-recoverable rival
refuses exactly as before."""
from __future__ import annotations

import dataclasses
import unittest

from tests.synth import Player, cfg, kit, run
from tests.test_tracker import H, assert_safe


def config(on: bool):
    c = cfg()
    return dataclasses.replace(c, tracking=dataclasses.replace(c.tracking, rival_eligible_only=on))


def scene(b_gone_at: float, on: bool, dx_h: float = 0.5):
    """A stands still and is hidden for 0.1 s from t = 1.0 (a short gap); B, dx_h body heights away, vanishes at
    b_gone_at and never returns."""
    b = Player("B", (800, 500), (0, 0), appearance=kit(1), visible=lambda t: t < b_gone_at)
    a = Player("A", (800 + dx_h * H, 500), (0, 0), appearance=kit(2), visible=lambda t: not (1.0 <= t < 1.1))
    return run([a, b], fps=30, duration_s=1.5, config=config(on))


def recoveries(res):
    """(track, kind, stage, reason, rivals) of every recovery decision (S3c bridged extent link or S4)."""
    return [(d["track_id"], d["kind"], d.get("stage"), d.get("reason"), d.get("rival_tracks")) for ft in res.frames
            for d in ft.decisions if d["kind"] in ("recovery", "recovery_refused")]


def track_ids(res):
    return sorted({t.track_id for ft in res.frames for t in ft.tracks})


class Stale(unittest.TestCase):
    def test_a_stale_track_no_longer_blocks_a_recovery(self):
        off = scene(0.5, False)                       # B unobserved 0.6 s at A's return: past its 0.2 s window
        self.assertIn(("T000001", "recovery_refused", "S4", "RIVAL_TRACK", ["T000002"]), recoveries(off))
        self.assertEqual(track_ids(off), ["T000001", "T000002", "T000003"])      # A split: a new track spawned
        on = scene(0.5, True)                         # S3c's exclusivity no longer counts B either: A's bridged
        self.assertEqual(recoveries(on), [("T000001", "recovery", "S3c", None, None)])   # gap closes in S3c
        self.assertEqual(track_ids(on), ["T000001", "T000002"])                  # A keeps its track
        assert_safe(self, on)

    def test_a_recoverable_rival_still_refuses(self):
        runs = [scene(0.95, on, dx_h=0.2) for on in (False, True)]   # B unobserved 0.15 s: recoverable (<= 0.2 s)
        for res in runs:
            self.assertIn(("T000001", "recovery_refused", "S4", "RIVAL_TRACK", ["T000002"]), recoveries(res))
            self.assertEqual(track_ids(res), ["T000001", "T000002", "T000003"])
            assert_safe(self, res)
        self.assertEqual(recoveries(runs[0]), recoveries(runs[1]))                # identical with the flag

    def test_on_by_default(self):
        self.assertTrue(cfg().tracking.rival_eligible_only)       # met the agreed tracking bar on DEV


if __name__ == "__main__":
    unittest.main()
