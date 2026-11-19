"""Appearance doubt link (tracking.appearance_doubt_link, S3b).

Real DEV finding (records/continuity_attribution_2026-11-18): the torso-colour veto ended continuing tracks on
their own player's next box (IoU 0.46-0.99, one frame later) and a new track was spawned on the same box. With the
doubt link the track continues on an EXCLUSIVE motion link, but is IDENTITY_UNCERTAIN (identity_doubt) until clean
crops match its pre-doubt appearance again. Never a silent identity transfer."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
import yaml

from spana.artifacts import TRACKS_SCHEMA, ArtifactError, tracklet_rows, tracks_frame, validate_tracks
from spana.config import config_from_dict
from spana.records import CameraObservation, CameraStatus, Detection, IdentityState

ROOT = Path(__file__).resolve().parents[1]
RAW = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
FPS = 30.0


def cfg(**tracking):
    """default.yaml with the doubt link ON (it ships off; these tests exercise the mechanism)."""
    raw = copy.deepcopy(RAW)
    raw["tracking"].update({"appearance_doubt_link": True, **tracking})
    return config_from_dict(raw)


def hist(k: int, n: int = 128) -> np.ndarray:
    """A normalised histogram concentrated on bin block k: blocks far apart are dissimilar (Bhattacharyya ~0)."""
    h = np.full(n, 1e-3)
    h[k * 8:(k + 1) * 8] = 1.0
    return h / h.sum()


def mix(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    h = (1 - w) * a + w * b
    return h / h.sum()


def run(config, players, frames: int):
    """players: list of callables frame -> (box, descriptor) or None (not detected)."""
    from spana.tracking.tracker import MultiPlayerTracker
    trk = MultiPlayerTracker(config, 1920, 1080)
    out = []
    for f in range(frames):
        dets, desc = [], []
        for p in players:
            r = p(f)
            if r is not None:
                dets.append(Detection(tuple(float(v) for v in r[0]), 0.9, 0))
                desc.append(r[1])
        cam = CameraObservation.identity(CameraStatus.FIRST if f == 0 else CameraStatus.OK, inliers=100)
        out.append(trk.update(f, f / FPS, dets, cam, desc))
    return trk, out


def walker(x0: float, vx: float, descs):
    """A 150 px tall player walking right; descs(f) gives the torso descriptor."""
    return lambda f: ((x0 + vx * f, 400.0, x0 + vx * f + 60.0, 550.0), descs(f))


A, B = hist(0), hist(8)
CHANGE = 20     # frame where the torso descriptor jumps from A to B (pose / light / partial crop)


def ids_of(out):
    return [{o.track_id for o in ft.tracks if o.source.value == "OBSERVED"} for ft in out]


class DoubtLink(unittest.TestCase):
    def test_off_reproduces_the_break(self):
        _, out = run(cfg(appearance_doubt_link=False), [walker(100, 3, lambda f: A if f < CHANGE else B)], 40)
        self.assertNotEqual(ids_of(out)[CHANGE - 1], ids_of(out)[CHANGE])     # new track on the same body

    def test_on_continues_flagged(self):
        _, out = run(cfg(), [walker(100, 3, lambda f: A if f < CHANGE else B)], 40)
        ids = ids_of(out)
        self.assertEqual(len(set.union(*ids)), 1)                            # one track throughout
        before = out[CHANGE - 1].tracks[0]
        self.assertEqual(before.identity, IdentityState.CERTAIN)
        for ft in out[CHANGE:]:
            o = ft.tracks[0]
            self.assertEqual(o.identity, IdentityState.IDENTITY_UNCERTAIN)    # appearance keeps disagreeing
            self.assertTrue(o.identity_doubt)
            self.assertEqual(o.identity_rivals, ())
            self.assertEqual(o.uncertain_episodes, 1)
        kinds = [d["kind"] for ft in out for d in ft.decisions]
        self.assertIn("doubt_link", kinds)

    def test_doubt_resolves_only_on_the_old_appearance(self):
        c = cfg()
        back = CHANGE + 3                                                    # a 3-frame appearance glitch
        _, out = run(c, [walker(100, 3, lambda f: B if CHANGE <= f < back else A)], 60)
        self.assertEqual(len(set.union(*ids_of(out))), 1)
        need = int(np.ceil(c.tracking.identity_resolve_s * FPS))
        self.assertEqual(out[back].tracks[0].identity, IdentityState.IDENTITY_UNCERTAIN)
        self.assertEqual(out[back + need + 1].tracks[0].identity, IdentityState.CERTAIN)
        self.assertFalse(out[-1].tracks[0].identity_doubt)

    def test_frozen_appearance_is_not_learned_from_the_doubted_crops(self):
        trk, out = run(cfg(), [walker(100, 3, lambda f: A if f < CHANGE else B)], 90)
        t = trk._tracks[0]
        from spana.appearance import similarity
        self.assertGreater(similarity(t.appearance, A), 0.9)                 # still the pre-doubt player
        self.assertEqual(out[-1].tracks[0].identity, IdentityState.IDENTITY_UNCERTAIN)

    def test_no_link_when_another_track_could_take_the_detection(self):
        # A second player walks alongside, close enough that both tracks gate the changed box: not exclusive.
        p1 = walker(100, 3, lambda f: A if f < CHANGE else B)
        p2 = walker(125, 3, lambda f: hist(4))
        _, out = run(cfg(), [p1, p2], 30)
        self.assertEqual(len(ids_of(out)[CHANGE - 1]), 2)                    # two live tracks side by side
        kinds = [d["kind"] for ft in out for d in ft.decisions]
        self.assertNotIn("doubt_link", kinds)                                # fail closed: the old break stays

    def test_no_link_on_a_loose_overlap(self):
        # The vetoed box jumps away within the motion gate but overlaps the prediction by less than the minimum IoU.
        def p(f):
            x = 100 + 3 * f + (25 if f >= CHANGE else 0)
            return (x, 400.0, x + 60.0, 550.0), (A if f < CHANGE else B)
        _, out = run(cfg(appearance_doubt_min_iou=0.9), [p], 30)
        self.assertNotIn("doubt_link", [d["kind"] for ft in out for d in ft.decisions])

    def test_gradual_change_is_still_learned(self):
        # Similar consecutive crops never trigger the veto, so no doubt: the ordinary appearance update applies.
        _, out = run(cfg(), [walker(100, 3, lambda f: mix(A, B, min(1.0, f / 60)))], 60)
        self.assertTrue(all(ft.tracks[0].identity == IdentityState.CERTAIN for ft in out))


class SelfConsistentResolution(unittest.TestCase):
    """tracking.doubt_resolve_self_consistent (option (a)): off by default."""

    def test_off_never_resolves_a_lasting_change(self):
        _, out = run(cfg(), [walker(100, 3, lambda f: A if f < CHANGE else B)], 90)
        self.assertEqual(out[-1].tracks[0].identity, IdentityState.IDENTITY_UNCERTAIN)

    def test_on_resolves_consistent_crops_and_adopts_them(self):
        c = cfg(doubt_resolve_self_consistent=True)
        trk, out = run(c, [walker(100, 3, lambda f: A if f < CHANGE else B)], 90)
        self.assertEqual(len(set.union(*ids_of(out))), 1)
        need = int(np.ceil(c.tracking.identity_resolve_s * FPS))
        self.assertEqual(out[CHANGE + 1].tracks[0].identity, IdentityState.IDENTITY_UNCERTAIN)
        done = next(n for n in range(CHANGE, 90) if out[n].tracks[0].identity == IdentityState.CERTAIN)
        self.assertGreaterEqual(done - CHANGE, need)
        self.assertTrue(all(ft.tracks[0].identity == IdentityState.CERTAIN for ft in out[done:]))
        from spana.appearance import similarity
        self.assertGreater(similarity(trk._tracks[0].appearance, B), 0.9)          # adopted the new look
        rules = [d.get("rivals") for ft in out for d in ft.decisions if d["kind"] == "identity_resolved"]
        self.assertEqual(rules, [{"(appearance_doubt)": "SELF_CONSISTENT"}])

    def test_on_inconsistent_crops_never_resolve(self):
        flicker = lambda f: A if f < CHANGE else (B if f % 2 else hist(12))      # noqa: E731
        _, out = run(cfg(doubt_resolve_self_consistent=True), [walker(100, 3, flicker)], 90)
        self.assertEqual(len(set.union(*ids_of(out))), 1)
        self.assertTrue(all(ft.tracks[0].identity == IdentityState.IDENTITY_UNCERTAIN for ft in out[CHANGE:]))


class Artifact(unittest.TestCase):
    def _doc(self, version=TRACKS_SCHEMA[1]):
        c = cfg()
        trk, out = run(c, [walker(100, 3, lambda f: A if f < CHANGE else B)], 30)
        return {"schema": TRACKS_SCHEMA[0], "schema_version": version, "provenance": {}, "video": {},
                "detections_sha256": "0" * 64, "frames": [tracks_frame(ft, c.output) for ft in out],
                "tracklets": tracklet_rows(trk.summaries())}

    def test_doubt_validates(self):
        doc = self._doc()
        validate_tracks(doc)
        self.assertTrue(any(t["identity_doubt"] and not t["identity_rivals"] and
                            t["identity"] == IdentityState.IDENTITY_UNCERTAIN.value
                            for fr in doc["frames"] for t in fr["tracks"]))

    def test_uncertain_without_rival_or_doubt_is_refused(self):
        doc = self._doc()
        t = next(t for fr in doc["frames"] for t in fr["tracks"] if t["identity_doubt"])
        t["identity_doubt"] = False
        with self.assertRaises(ArtifactError):
            validate_tracks(doc)

    def test_2_1_0_documents_still_validate_and_keep_their_rule(self):
        doc = self._doc()
        for fr in doc["frames"]:
            for t in fr["tracks"]:
                doubt = t.pop("identity_doubt")
                t.pop("resolved_by")                                    # tracks 2.3.0 field
                if doubt:
                    t["identity"], t["uncertain_episodes"] = IdentityState.CERTAIN.value, t["uncertain_episodes"]
        doc["schema_version"] = "2.1.0"
        validate_tracks(doc)
        doc["frames"][-1]["tracks"][0]["identity"] = IdentityState.IDENTITY_UNCERTAIN.value   # no rival in 2.1.0
        with self.assertRaises(ArtifactError):
            validate_tracks(doc)


if __name__ == "__main__":
    unittest.main()
