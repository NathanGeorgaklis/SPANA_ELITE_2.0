"""Re-ID resolution of same-look IDENTITY_UNCERTAIN episodes (config 2.14.0, records/reid_episode_resolution_2026-11-20).

Re-ID has its own run: it resolves an episode (resolved_by REID) only when it cleared every rival by
reid_resolve.margin - for an ambiguous spawn, the same predecessor beat every other one by the margin - on
reid_resolve.min_evidence_frames clean frames spanning tracking.identity_resolve_s. Colour that cannot tell the
players apart no longer blocks it; colour clearly favouring the rival does. Never one frame; every evidence frame
and every refusal is logged with its scores / reason; off = no change and no log rows.
"""
from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np

from spana.config import ConfigError, load_config
from spana.records import FrameTracks, IdentityState
from spana.tracking.tracker import MultiPlayerTracker, _Rival
from tests.synth import Player, kit, run
from tests.test_reid_resolve import crossing, same_identity_throughout, unit

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
ON = dataclasses.replace(CFG, reid_resolve=dataclasses.replace(CFG.reid_resolve, enabled=True))
DISTINCT = {"A": unit(1, 0.05), "B": unit(0.05, 1)}


def decisions(res, kind):
    return [d for f in res.frames for d in f.decisions if d["kind"] == kind]


def same_kit_crossing(cfg, reid):
    """Two players in the SAME kit cross: colour sees them alike (its TWO_SIDED rule says False every frame)."""
    a = Player("A", (300, 500), (240, 0), appearance=kit(0))
    b = Player("B", (700, 500), (-240, 0), appearance=kit(0))
    return run([a, b], fps=30, duration_s=3.0, config=cfg, reid=reid)


class SameLookCrossing(unittest.TestCase):
    def test_off_logs_nothing(self):
        res = same_kit_crossing(CFG, DISTINCT)
        self.assertFalse(decisions(res, "reid_evidence"))
        self.assertFalse([d for d in decisions(res, "identity_resolved") if d["resolved_by"] == "REID"])

    def test_colour_that_cannot_tell_does_not_block_reid(self):
        off = same_kit_crossing(CFG, DISTINCT)
        self.assertTrue(any(o.identity != IdentityState.CERTAIN for o in off.frames[-1].tracks))
        res = same_kit_crossing(ON, DISTINCT)
        rr = [d for d in decisions(res, "identity_resolved") if d["resolved_by"] == "REID"]
        self.assertTrue(rr)
        for d in rr:
            self.assertGreaterEqual(d["run_frames"], ON.reid_resolve.min_evidence_frames)
            self.assertGreaterEqual(d["run_s"], ON.tracking.identity_resolve_s - 1e-9)
            self.assertEqual(d["reid_margin"], ON.reid_resolve.margin)
            for r in d["reid"].values():                       # every rival cleared by the margin, logged
                self.assertEqual(r["verdict"], "CLEARED")
                self.assertGreaterEqual(r["margin"], ON.reid_resolve.margin)
        self.assertTrue(all(o.identity == IdentityState.CERTAIN for o in res.frames[-1].tracks))
        self.assertTrue(same_identity_throughout(res))

    def test_no_clear_margin_stays_uncertain_and_logs_why(self):
        for reid in ({"A": unit(1, 1), "B": unit(1, 1)}, {"A": unit(1, 0.6), "B": unit(0.6, 1)}):
            res = same_kit_crossing(ON, reid)
            self.assertFalse([d for d in decisions(res, "identity_resolved") if d["resolved_by"] == "REID"])
            ev = decisions(res, "reid_evidence")
            self.assertTrue(ev)
            refused = {r.get("refused") for d in ev for r in d["rivals"].values()}
            self.assertIn("BELOW_MARGIN", refused)
            self.assertTrue({d["status"] for d in ev} & {"NOT_CLEAN", "CONTESTED"})   # overlapping crops refused
            for d in ev:
                for r in d["rivals"].values():
                    if r.get("refused") == "BELOW_MARGIN":
                        self.assertLess(abs(r["margin"]), ON.reid_resolve.margin)

    def test_no_colour_crossing_never_switches_identity(self):
        res = crossing(ON, reid=DISTINCT)
        self.assertTrue([d for d in decisions(res, "identity_resolved") if d["resolved_by"] == "REID"])
        self.assertTrue(same_identity_throughout(res))


def _uncertain_pair(cfg=ON):
    """A tracker holding track t (own reference) uncertain against rival r (reference snapshot at the contest)."""
    from spana.geometry import box_to_cxcywh
    from spana.tracking.kalman import BoxKalman
    from spana.tracking.tracker import _Track
    trk = MultiPlayerTracker(cfg, 1920, 1080)
    mk = lambda tid: _Track(tid, BoxKalman(box_to_cxcywh(np.array([0, 0, 10, 30.0])), cfg.kalman), 0, 0, 0.0, 0.0,
                            0, 0.0)
    t, r = mk("T1"), mk("T2")
    t.reid_ref, r.reid_ref = unit(1, 0), unit(0, 1)
    t.appearance = r.appearance = None
    t.identity_rivals["T2"] = _Rival(None, 0.0, r.reid_ref.copy())
    return trk, t


class RunRules(unittest.TestCase):
    def test_against_resets_and_below_margin_holds(self):
        trk, t = _uncertain_pair()
        out = FrameTracks(0, 0.0, None)
        own, rival, between = unit(1, 0), unit(0, 1), unit(1, 1)
        ev = trk._reid_evidence(t, 0, own, None, 0.1, out)
        self.assertTrue(ev["cleared"])
        self.assertFalse(trk._advance_reid_run(t, 0, ev, None, 0.1, out))
        self.assertEqual(t.reid_run_frames, 1)
        ev = trk._reid_evidence(t, 0, between, None, 0.2, out)           # no clear margin: hold
        self.assertFalse(ev["cleared"] or ev["against"])
        trk._advance_reid_run(t, 0, ev, None, 0.2, out)
        self.assertEqual((t.reid_run_since_t, t.reid_run_frames), (0.1, 1))
        ev = trk._reid_evidence(t, 0, rival, None, 0.3, out)             # nearer the rival: reset
        self.assertEqual(ev["against"], ["T2"])
        trk._advance_reid_run(t, 0, ev, None, 0.3, out)
        self.assertEqual((t.reid_run_since_t, t.reid_run_frames), (None, 0))
        self.assertTrue([d for d in out.decisions if d["kind"] == "reid_run_reset"])

    def test_never_one_frame_and_needs_the_time_span(self):
        cfg = dataclasses.replace(ON, tracking=dataclasses.replace(ON.tracking, identity_resolve_s=0.01))
        trk, t = _uncertain_pair(cfg)
        out = FrameTracks(0, 0.0, None)
        n = cfg.reid_resolve.min_evidence_frames
        for k in range(n):
            ev = trk._reid_evidence(t, 0, unit(1, 0), None, 0.1 * (k + 1), out)
            done = trk._advance_reid_run(t, 0, ev, None, 0.1 * (k + 1), out)
            self.assertEqual(done, k == n - 1)
        self.assertEqual(t.resolved_by, "REID")
        self.assertFalse(t.identity_rivals)
        trk, t = _uncertain_pair()                                          # enough frames, too short a span
        for k in range(n + 2):
            ev = trk._reid_evidence(t, 0, unit(1, 0), None, 0.1 + 0.01 * k, out)
            self.assertFalse(trk._advance_reid_run(t, 0, ev, None, 0.1 + 0.01 * k, out))
        with self.assertRaises(ConfigError):
            dataclasses.replace(ON.reid_resolve, min_evidence_frames=1).validate()

    def test_colour_clearly_favouring_the_rival_vetoes(self):
        trk, t = _uncertain_pair()
        t.appearance = np.array([1.0, 0, 0, 0])
        t.identity_rivals["T2"] = _Rival(np.array([0, 0, 0, 1.0]), 0.0, unit(0, 1))
        out = FrameTracks(0, 0.0, None)
        ev = trk._reid_evidence(t, 0, unit(1, 0), np.array([0, 0, 0, 1.0]), 0.1, out)   # crop has the rival's colour
        self.assertEqual(ev["against"], ["T2"])
        self.assertEqual(ev["rivals"]["T2"]["verdict"], "COLOUR_CONTRARY")
        ev = trk._reid_evidence(t, 0, unit(1, 0), np.array([0.5, 0, 0, 0.5]), 0.2, out)  # colour cannot tell
        self.assertTrue(ev["cleared"])

    def test_missing_references_are_refusals(self):
        trk, t = _uncertain_pair()
        out = FrameTracks(0, 0.0, None)
        t.identity_rivals["T2"] = _Rival(None, 0.0, None)
        self.assertEqual(trk._reid_evidence(t, 0, unit(1, 0), None, 0.1, out)["rivals"]["T2"]["refused"],
                         "NO_RIVAL_REFERENCE")
        t.reid_ref = None
        self.assertEqual(trk._reid_evidence(t, 0, unit(1, 0), None, 0.1, out)["status"], "NO_OWN_REFERENCE")
        self.assertEqual(trk._reid_evidence(t, 0, None, None, 0.1, out)["status"], "NO_EMBEDDING")


class AmbiguousSpawnByReid(unittest.TestCase):
    def _pred_case(self, preds):
        trk, t = _uncertain_pair()
        t.identity_rivals = {p: _Rival(None, 0.0, ref, predecessor=True) for p, ref in preds.items()}
        t.reid_ref = None                                                   # born uncertain: no own reference
        return trk, t

    def test_one_predecessor_by_the_margin_resolves_with_the_same_pick(self):
        trk, t = self._pred_case({"P1": unit(1, 0), "P2": unit(0, 1)})
        out = FrameTracks(0, 0.0, None)
        n = ON.reid_resolve.min_evidence_frames
        done = False
        for k in range(n + 6):
            ev = trk._reid_evidence(t, 0, unit(1, 0.02), None, 0.1 * (k + 1), out)
            self.assertEqual(ev["rivals"]["(predecessors)"]["best"], "P1")
            done = trk._advance_reid_run(t, 0, ev, None, 0.1 * (k + 1), out)
            if done:
                break
        self.assertTrue(done)
        d = [x for x in out.decisions if x["kind"] == "identity_resolved"][-1]
        self.assertEqual((d["resolved_by"], d["predecessor"]), ("REID", "P1"))

    def test_pick_change_resets_and_same_look_predecessors_never_resolve(self):
        trk, t = self._pred_case({"P1": unit(1, 0), "P2": unit(0, 1)})
        out = FrameTracks(0, 0.0, None)
        trk._advance_reid_run(t, 0, trk._reid_evidence(t, 0, unit(1, 0), None, 0.1, out), None, 0.1, out)
        self.assertEqual(t.reid_run_pred, "P1")
        ev = trk._reid_evidence(t, 0, unit(0, 1), None, 0.2, out)
        self.assertEqual(ev["rivals"]["(predecessors)"]["verdict"], "PICK_CHANGED")
        trk._advance_reid_run(t, 0, ev, None, 0.2, out)
        self.assertIsNone(t.reid_run_since_t)
        trk, t = self._pred_case({"P1": unit(1, 0.1), "P2": unit(1, 0.12)})
        ev = trk._reid_evidence(t, 0, unit(1, 0.11), None, 0.1, out)
        self.assertEqual(ev["rivals"]["(predecessors)"]["refused"], "BELOW_MARGIN")
        trk, t = self._pred_case({"P1": unit(1, 0), "P2": None})
        ev = trk._reid_evidence(t, 0, unit(1, 0), None, 0.1, out)
        self.assertEqual(ev["rivals"]["(predecessors)"]["refused"], "NO_PREDECESSOR_REFERENCE")

    def test_h1_same_look_spawn_end_to_end(self):
        """The synthetic same-look ambiguous spawn (h1, both in the same kit: colour can never pick): with distinct
        embeddings re-ID resolves it to the predecessor that carried the same player; off it stays uncertain."""
        for fps in (10, 25, 60):
            base = _h1_reid(fps, CFG)
            self.assertFalse([d for d in decisions(base, "identity_resolved") if d["resolved_by"] == "REID"])
            res = _h1_reid(fps, ON)
            own = players_of(res)
            rr = [(f.frame_index, d) for f in res.frames for d in f.decisions
                  if d["kind"] == "identity_resolved" and d.get("predecessor")]
            self.assertTrue(rr, fps)
            for frame, d in rr:
                self.assertEqual(set(own[d["predecessor"]].values()), {own[d["track_id"]][frame]}, fps)
            same = _h1_reid(fps, ON, {"A": unit(1, 1), "B": unit(1, 1)})
            self.assertFalse([d for d in decisions(same, "identity_resolved") if d["resolved_by"] == "REID"])


def _h1_reid(fps, cfg, reid=DISTINCT):
    a = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0), visible=lambda t: not (1.2 <= t < 1.2 + 1.5 / 25))
    b = Player("B", (560, 495), (110, 0), appearance=kit(0), visible=lambda t: not (0.9 <= t < 1.2))
    return run([a, b], fps=fps, duration_s=3, config=cfg, reid=reid)


def players_of(res):
    own = {}
    for ft, g in zip(res.frames, res.gt_of_detection):
        for o in ft.tracks:
            if o.detection_index is not None:
                own.setdefault(o.track_id, {})[ft.frame_index] = g[o.detection_index]
    return own


if __name__ == "__main__":
    unittest.main()


class FollowerAnchor(unittest.TestCase):
    """Re-ID re-confirms the identity a track had before its episode. The follower resumes after a REID resolution
    only on a track it attached to while CERTAIN; a selection made during the episode has no such anchor."""
    def answers(self, select_frame):
        from spana.identity.follower import Selection, follow
        from tests.test_reid_resolve import with_resolution
        from tests.test_target_follow import AU, A, N, tracks_doc
        frames = [{"t1": N, "t2": N}, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": A, "t2": A}, {"t1": A, "t2": A}]
        cfg = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, resume_after_reid=True))
        out, _ = follow(with_resolution(tracks_doc(frames), "REID"), Selection.track_at_frame("t1", select_frame), cfg)
        return [(f["state"], f["reason"]) for f in out]

    def test_selected_while_certain_resumes(self):
        self.assertEqual(self.answers(1)[-2:], [("TARGET", "TRACK_OBSERVED")] * 2)

    def test_selected_during_the_episode_never_resumes(self):
        self.assertEqual(self.answers(2)[-2:], [("ABSTAIN", "AFTER_UNCERTAIN")] * 2)
