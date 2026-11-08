"""IDENTITY_UNCERTAIN: a lasting identity-ambiguity state.

Regression for the real DEV finding (FRESH_02, players crossing, frames 166-174): with no usable appearance, two
tracks fitted the same boxes, one track moved to the other player, and only per-frame contest flags were raised,
so the identity swap was silent. The fixture holds the exact tracker inputs of that crossing (ROI, frames 120-204)
and the frozen DEV labels of the two players."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from spana.config import load_config
from spana.evaluation import NESTED_V2, EvalFrame, TrackBox, evaluate
from spana.records import CameraObservation, CameraStatus, Detection, IdentityState
from spana.tracking.tracker import MultiPlayerTracker
from tests.synth import Player, kit, run

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
FIXTURE = ROOT / "tests" / "data" / "fresh02_crossing_f120_204.json"


def replay_fixture():
    doc = json.loads(FIXTURE.read_text())
    trk = MultiPlayerTracker(CFG, 1920, 1080)
    out = {}
    for k, fr in enumerate(doc["frames"]):
        c = fr["camera"]
        status = CameraStatus.FIRST if k == 0 else CameraStatus(c["status"])
        cam = CameraObservation(status, tuple(tuple(r) for r in c["affine"]), c["inliers"], c["hist_distance"],
                                c["motion_std_px"]) if k else CameraObservation.identity(CameraStatus.FIRST)
        dets = [Detection(tuple(r[:4]), r[4], r[5], bool(r[6])) for r in fr["detections"]]
        desc = [None if d is None else np.array(d) for d in fr["descriptors"]]
        out[fr["frame"]] = trk.update(fr["frame"], fr["t"], dets, cam, desc)
    return doc, out


def score(doc, out, with_identity: bool):
    frames = []
    for f, gtb in sorted(((int(f), g) for f, g in doc["labels"].items())):
        obs = [TrackBox(o.track_id, o.box, o.confirmed, bool(o.possible_predecessors) or o.recoveries > 0,
                        o.detection_index, with_identity and o.identity != IdentityState.CERTAIN,
                        o.uncertain_episodes if with_identity else 0)
               for o in out[f].tracks if o.source.value == "OBSERVED"]
        frames.append(EvalFrame(f, gtb, [], obs, [], frozenset(doc["occluded"][str(f)])))
    return evaluate(frames, rule=NESTED_V2)


class RealCrossing(unittest.TestCase):
    def test_swap_in_the_crossing_is_flagged_not_silent(self):
        doc, out = replay_fixture()
        without = score(doc, out, with_identity=False)
        self.assertGreaterEqual(without.unflagged_transfers, 1)       # the fixture reproduces the silent swap
        r = score(doc, out, with_identity=True)
        self.assertEqual(r.identity_transfers, without.identity_transfers)   # same tracking, now flagged
        self.assertEqual(r.unflagged_transfers, 0)
        swapped = {e["track_id"] for e in r.transfer_events}
        uncertain = {o.track_id for ft in out.values() for o in ft.tracks if o.identity != IdentityState.CERTAIN}
        self.assertTrue(swapped <= uncertain)


class SyntheticIdentity(unittest.TestCase):
    def _crossing(self, appearance):
        # Two players run through each other along the same line (same depth): only motion separates them.
        a = Player("A", (300, 500), (240, 0), appearance=appearance[0])
        b = Player("B", (700, 500), (-240, 0), appearance=appearance[1])
        return run([a, b], fps=30, duration_s=3.0, use_appearance=appearance[0] is not None)

    def test_crossing_without_appearance_ends_uncertain(self):
        res = self._crossing((None, None))
        last = res.frames[-1].tracks
        self.assertTrue(any(o.identity == IdentityState.IDENTITY_UNCERTAIN for o in last))
        self.assertTrue(all(o.uncertain_episodes >= 1 for o in last if o.identity != IdentityState.CERTAIN))

    def test_distinct_appearance_resolves_or_never_triggers(self):
        res = self._crossing((kit(0), kit(1)))
        self.assertTrue(all(o.identity == IdentityState.CERTAIN for o in res.frames[-1].tracks))

    def test_separate_players_never_become_uncertain(self):
        ps = [Player(f"P{k}", (300 + 300 * k, 500), (60, 0)) for k in range(4)]
        res = run(ps, fps=30, duration_s=3.0)
        self.assertFalse(any(o.identity != IdentityState.CERTAIN for ft in res.frames for o in ft.tracks))


class Resolution(unittest.TestCase):
    """Two side-by-side tracks share one undescribed box for a frame; afterwards clean appearance either
    confirms each track's own appearance (resolved after identity_resolve_s) or contradicts it (stays uncertain)."""
    A, B = (500.0, 300.0, 540.0, 420.0), (545.0, 300.0, 585.0, 420.0)
    MID = (522.0, 300.0, 562.0, 420.0)
    DA, DB = np.full(4, 0.25), np.array([0.7, 0.1, 0.1, 0.1])

    def _run(self, swapped_after: bool):
        trk = MultiPlayerTracker(CFG, 1920, 1080)
        cam = lambda i: CameraObservation.identity(CameraStatus.FIRST if i == 0 else CameraStatus.OK, inliers=100)  # noqa
        fts = []
        for i in range(40):
            if i == 10:
                dets, desc = [Detection(self.MID, 0.9, 0)], [None]
            else:
                da, db = (self.DB, self.DA) if swapped_after and i > 10 else (self.DA, self.DB)
                dets, desc = [Detection(self.A, 0.9, 0), Detection(self.B, 0.9, 0)], [da, db]
            fts.append(trk.update(i, i / 30, dets, cam(i), desc))
        return fts

    def test_clean_own_appearance_resolves(self):
        fts = self._run(swapped_after=False)
        self.assertTrue(any(o.identity != IdentityState.CERTAIN for o in fts[10].tracks))
        kinds = [d["kind"] for ft in fts for d in ft.decisions]
        self.assertIn("identity_resolved", kinds)
        self.assertTrue(all(o.identity == IdentityState.CERTAIN for o in fts[-1].tracks))

    def test_contradicting_appearance_never_resolves(self):
        fts = self._run(swapped_after=True)
        self.assertTrue(any(o.uncertain_episodes for o in fts[-1].tracks))
        self.assertTrue(all(o.identity == IdentityState.IDENTITY_UNCERTAIN
                            for o in fts[-1].tracks if o.uncertain_episodes))
        self.assertNotIn("identity_resolved", [d["kind"] for ft in fts for d in ft.decisions])


if __name__ == "__main__":
    unittest.main()
