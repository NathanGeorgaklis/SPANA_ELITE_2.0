"""Silent identity transfers on DEV (records/silent_transfer_fix_2026-11-20): two triggers of IDENTITY_UNCERTAIN.

1. Crossing without a contest (tracking.crossing_uncertain). Real DEV finding, FRESH_07 players 14 / 15, frames
   63-72: the two crossing tracks had established velocities, so each Kalman gate held only its own box and no
   contest was raised. One track took a merged box that held the other's (observed) body, and the labels saw
   player 15 under it at frame 66. The fixture holds the exact tracker inputs (ROI, frames 40-90) and the frozen
   DEV labels of the two players.
2. Ambiguous spawn (tracking.ambiguous_spawn_uncertain). A track spawned naming >= 2 possible_predecessors starts
   IDENTITY_UNCERTAIN, its predecessors as rivals (FRESH_02 T000045; the synthetic h1 same-look case at 60 fps).
"""
from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

import numpy as np

from spana.config import load_config
from spana.records import CameraObservation, CameraStatus, Detection, FrameTracks, IdentityState
from spana.tracking.tracker import MultiPlayerTracker
from tests.synth import Player, kit, run
from tests.test_identity_uncertain import score

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
FIXTURE = ROOT / "tests" / "data" / "fresh07_crossing_f40_90.json"


def with_tracking(**kw):
    return dataclasses.replace(CFG, tracking=dataclasses.replace(CFG.tracking, **kw))


def replay(cfg):
    doc = json.loads(FIXTURE.read_text())
    trk = MultiPlayerTracker(cfg, 1920, 1080)
    out = {}
    for k, fr in enumerate(doc["frames"]):
        c = fr["camera"]
        cam = CameraObservation(CameraStatus(c["status"]), tuple(tuple(r) for r in c["affine"]), c["inliers"],
                                c["hist_distance"], c["motion_std_px"]) if k else \
            CameraObservation.identity(CameraStatus.FIRST)
        dets = [Detection(tuple(r[:4]), r[4], r[5], bool(r[6])) for r in fr["detections"]]
        desc = [None if d is None else np.array(d) for d in fr["descriptors"]]
        out[fr["frame"]] = trk.update(fr["frame"], fr["t"], dets, cam, desc)
    return doc, out


def rules(out):
    return [(f, d) for f, ft in sorted(out.items()) for d in ft.decisions if d["kind"] == "identity_uncertain"]


class RealCrossingWithoutContest(unittest.TestCase):
    def test_fixture_reproduces_the_silent_transfer_without_the_rule(self):
        doc, out = replay(with_tracking(crossing_uncertain=False))
        r = score(doc, out, with_identity=True)
        self.assertGreaterEqual(r.unflagged_transfers, 1)
        self.assertFalse(any(d.get("rule") == "CROSSING" for _, d in rules(out)))

    def test_crossing_marks_both_tracks_at_the_swap_and_no_transfer_is_silent(self):
        doc, out = replay(CFG)
        before = score(doc, replay(with_tracking(crossing_uncertain=False))[1], with_identity=True)
        r = score(doc, out, with_identity=True)
        self.assertEqual(r.identity_transfers, before.identity_transfers)      # same tracking, now flagged
        self.assertEqual(r.unflagged_transfers, 0)
        crossing = [(f, d) for f, d in rules(out) if d.get("rule") == "CROSSING"]
        self.assertTrue(crossing)
        self.assertTrue(all(d["rival_observed"] and d["rival_inside"] >= CFG.tracking.crossing_min_ioa
                            for _, d in crossing))
        swapped = {e["track_id"] for e in r.transfer_events}
        marked = {d["track_id"] for _, d in crossing} | {d["rival"] for _, d in crossing}
        self.assertTrue(swapped <= marked)
        first = min(f for f, _ in crossing)
        swap = min(e["frame"] for e in r.transfer_events)
        self.assertLessEqual(first, swap)                 # uncertain no later than the swap frame
        for f, d in crossing:                             # both tracks uncertain, naming each other
            ids = {o.track_id: o for o in out[f].tracks}
            self.assertIn(d["rival"], ids[d["track_id"]].identity_rivals)
            self.assertIn(d["track_id"], ids[d["rival"]].identity_rivals)


def _cam(f):
    return CameraObservation.identity(CameraStatus.FIRST if f == 0 else CameraStatus.OK, inliers=100)


def _two_confirmed_tracks(cfg, look_b):
    """A at x 100-140 and B at x 150-190 (not overlapping), observed apart for 0.4 s: both confirmed."""
    trk = MultiPlayerTracker(cfg, 1920, 1080)
    for f in range(10):
        dets = [Detection((100.0, 100.0, 140.0, 180.0), 0.9, 0), Detection((150.0, 100.0, 190.0, 180.0), 0.9, 0)]
        trk.update(f, f / 25, dets, _cam(f), [kit(0), look_b])
    assert all(t.confirmed for t in trk._tracks) and len(trk._tracks) == 2
    return trk


def _nested_frame(trk, b_observed, desc_a=None):
    """A's detection holds B's whole box; B on its own nested detection (observed) or unobserved."""
    dets = [Detection((95.0, 95.0, 195.0, 185.0), 0.9, 0), Detection((152.0, 102.0, 188.0, 178.0), 0.4, 0)]
    matched = {0: (0, False)}
    if b_observed:
        matched[1] = (1, False)
    out = FrameTracks(10, 10 / 25, _cam(10))
    trk._crossings(matched, dets, [desc_a, None], 10 / 25, out)
    return out


class CrossingRule(unittest.TestCase):
    def test_similar_looking_nested_pair_is_marked_both_ways(self):
        trk = _two_confirmed_tracks(CFG, kit(0))
        out = _nested_frame(trk, b_observed=True)
        a, b = trk._tracks
        self.assertIn(b.track_id, a.identity_rivals)
        self.assertIn(a.track_id, b.identity_rivals)
        self.assertEqual([d["rule"] for d in out.decisions], ["CROSSING"])

    def test_clearly_different_looks_are_not_marked(self):
        trk = _two_confirmed_tracks(CFG, kit(1))
        out = _nested_frame(trk, b_observed=True)
        self.assertEqual(out.decisions, [])
        self.assertFalse(any(t.identity_rivals for t in trk._tracks))

    def test_a_crop_matching_both_tracks_equally_does_not_separate_them(self):
        trk = _two_confirmed_tracks(CFG, kit(0))
        _nested_frame(trk, b_observed=True, desc_a=kit(0))
        self.assertTrue(trk._tracks[0].identity_rivals)

    def test_hidden_rival_only_with_crossing_hidden_rivals(self):
        for hidden, expect in ((False, False), (True, True)):
            trk = _two_confirmed_tracks(with_tracking(crossing_hidden_rivals=hidden), kit(0))
            _nested_frame(trk, b_observed=False)
            self.assertEqual(bool(trk._tracks[0].identity_rivals), expect, hidden)

    def test_switch_off(self):
        trk = _two_confirmed_tracks(with_tracking(crossing_uncertain=False), kit(0))
        dets = [Detection((95.0, 95.0, 195.0, 185.0), 0.9, 0), Detection((152.0, 102.0, 188.0, 178.0), 0.4, 0)]
        out = FrameTracks(10, 10 / 25, _cam(10))
        trk._identity_ambiguity({0: (0, False), 1: (1, False)}, dets, [None, None], 10 / 25, out)
        self.assertEqual(out.decisions, [])


def _h1(fps, look, cfg=None):
    a = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0), visible=lambda t: not (1.2 <= t < 1.2 + 1.5 / 25))
    b = Player("B", (560, 495), (110, 0), appearance=kit(look), visible=lambda t: not (0.9 <= t < 1.2))
    return run([a, b], fps=fps, duration_s=2, config=cfg)


class AmbiguousSpawn(unittest.TestCase):
    def test_spawn_with_two_predecessors_starts_uncertain_naming_them(self):
        for fps in (10, 25, 60):
            res = _h1(fps, 0)
            born = [o for ft in res.frames for o in ft.tracks if o.age_frames == 0 and len(o.possible_predecessors) >= 2]
            self.assertTrue(born, fps)
            for o in born:
                self.assertEqual(o.identity, IdentityState.IDENTITY_UNCERTAIN, fps)
                self.assertEqual(set(o.identity_rivals), set(o.possible_predecessors), fps)
                self.assertEqual(o.uncertain_episodes, 1, fps)

    def test_switch_off_starts_certain(self):
        res = _h1(60, 0, with_tracking(ambiguous_spawn_uncertain=False))
        born = [o for ft in res.frames for o in ft.tracks if o.age_frames == 0 and len(o.possible_predecessors) >= 2]
        self.assertTrue(born)
        self.assertTrue(all(o.identity == IdentityState.CERTAIN for o in born))

    def test_one_predecessor_is_not_ambiguous(self):
        # A player missed for longer than the recovery allows comes back as a new track naming only him: certain.
        p = Player("A", (500, 500), (0, 0), visible=lambda t: not (1.0 <= t < 1.5))
        res = run([p], fps=25, duration_s=2)
        born = [o for ft in res.frames for o in ft.tracks if o.age_frames == 0 and o.possible_predecessors]
        self.assertTrue(born)
        self.assertTrue(all(len(o.possible_predecessors) == 1 and o.identity == IdentityState.CERTAIN for o in born))

    def test_distinct_predecessors_resolve_same_look_never(self):
        resolved = {}
        for look in (1, 0):
            res = _h1(25, look)
            resolved[look] = [d for ft in res.frames for d in ft.decisions if d["kind"] == "identity_resolved"
                              and d["rivals"].get("(predecessors)") == "PREDECESSOR"]
        self.assertTrue(resolved[1])       # the crop rules out one of two clearly different predecessors
        self.assertFalse(resolved[0])      # two same-looking predecessors can never be told apart: fail closed


if __name__ == "__main__":
    unittest.main()
