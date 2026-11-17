"""Tracker: a confirmed track that leaves the picture is held for tracking.exit_hold_s and recovered when its player
steps back in at that edge, under the long-gap rules (appearance must match and be distinct); otherwise a new track.

Real DEV finding (records/window_f05_2026-11-18): on a 10-minute window of a full DEV match most confirmed tracks
ended LEFT_FRAME (camera pans, players at the edge of the shot), e.g. a player left at the left edge and stepped
back in there 1.2 s later on a new track."""
from __future__ import annotations

import unittest

import numpy as np

from spana.records import EndReason
from tests.synth import Player, cfg, kit, run
from tests.test_tracker import assert_safe

FPS_SET = (10, 25, 60)
W = 1920


def held(hold_s=1.5):
    """The default config is off (exit_hold_s 0, decision record exit_hold_2026-11-18): tests set the hold."""
    c = cfg()
    return c.__class__(**{**c.__dict__, "tracking": c.tracking.__class__(**{**c.tracking.__dict__,
                                                                            "exit_hold_s": hold_s})})


class OutAndBack(Player):
    """Walks right out of the picture, turns at t_turn, walks back in (300 px/s each way)."""

    def __init__(self, gid, t_turn, appearance, x0=1820.0):
        super().__init__(gid, (x0, 500), (300, 0), appearance=appearance,
                         visible=lambda t: self.x(t) < W - 10)
        self.t_turn, self.x0 = t_turn, x0

    def x(self, t):
        return self.x0 + 300 * t if t <= self.t_turn else self.x0 + 300 * self.t_turn - 300 * (t - self.t_turn)

    def box_at(self, t):
        h, w = self.height, self.height * self.aspect
        return np.array([self.x(t) - w / 2, 500 - h / 2, self.x(t) + w / 2, 500 + h / 2])


def tracks_of(res, gid):
    ids = []
    for ft, gids in zip(res.frames, res.gt_of_detection):
        for o in ft.tracks:
            if o.detection_index is not None and gids[o.detection_index] == gid and o.track_id not in ids:
                ids.append(o.track_id)
    return ids


class ExitHoldTest(unittest.TestCase):
    def test_back_within_the_hold_keeps_the_track(self):
        for fps in FPS_SET:
            res = run([OutAndBack("A", 0.8, kit(1))], fps=fps, duration_s=2.5, config=held())   # out ~1.0 s
            assert_safe(self, res, fps)
            self.assertEqual(len(tracks_of(res, "A")), 1, fps)
            self.assertNotIn(EndReason.LEFT_FRAME, [r for ft in res.frames for _, r in ft.ended], fps)
            self.assertTrue(any(d.get("kind") == "recovery" for ft in res.frames for d in ft.decisions), fps)

    def test_back_after_the_hold_is_a_new_track(self):
        for fps in FPS_SET:
            res = run([OutAndBack("A", 1.4, kit(1))], fps=fps, duration_s=3.5, config=held())   # out ~2.2 s
            assert_safe(self, res, fps)
            self.assertEqual(len(tracks_of(res, "A")), 2, fps)
            self.assertIn(EndReason.LEFT_FRAME, [r for ft in res.frames for _, r in ft.ended], fps)

    def test_same_kit_player_near_the_edge_blocks_the_recovery(self):
        for fps in FPS_SET:
            b = Player("B", (1760, 700), (0, 0), appearance=kit(1))                # same kit, in view, another row
            res = run([OutAndBack("A", 0.8, kit(1)), b], fps=fps, duration_s=2.5, config=held())
            assert_safe(self, res, fps)
            self.assertEqual(len(tracks_of(res, "A")), 2, fps)                     # never a guess: new track
            self.assertEqual(len(tracks_of(res, "B")), 1, fps)
            refused = [d for ft in res.frames for d in ft.decisions
                       if d.get("kind") == "recovery_refused" and d.get("reason") == "APPEARANCE_NOT_DISTINCT"]
            self.assertTrue(refused, fps)

    def test_default_is_off_and_restores_the_old_behaviour(self):
        self.assertEqual(cfg().tracking.exit_hold_s, 0.0)
        res = run([OutAndBack("A", 0.8, kit(1))], fps=25, duration_s=2.5)
        self.assertEqual(len(tracks_of(res, "A")), 2)


if __name__ == "__main__":
    unittest.main()
