"""Target identity (layer 3): the minimal follower and the spana.target 1.0.0 artifact, on synthetic tracks.

Each scenario is a hand-built, validated spana.tracks document: per frame, the live tracks with their lifecycle state,
confirmation and identity state. Generic ids only; no clip, player or frame knowledge.
"""
from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spana.artifacts import ArtifactError, read_json, sha256_doc, validate_target, validate_tracks, write_json_atomic
from spana.config import load_config
from spana.identity.follower import Selection, SelectionError, follow, target_document

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
FPS = 10.0

# per-track spec on a frame: (state, confirmed, uncertain); state A=ACTIVE, O=OCCLUDED, L=LOST
Spec = Tuple[str, bool, bool]


def tracks_doc(frames: List[Dict[str, Spec]], ended: Optional[Dict[int, List[Tuple[str, str]]]] = None,
               cut_at: Optional[int] = None, preds: Optional[Dict[str, List[str]]] = None) -> dict:
    """A valid spana.tracks 2.1.0 document. frames[n] = {track_id: spec}; ended[n] = [(track_id, EndReason)].
    A track's first frame must be a fresh unconfirmed observation (N), as the tracker emits it."""
    ended = ended or {}
    out, state = [], {}
    epoch = 0
    for n, live in enumerate(frames):
        cam = "FIRST" if n == 0 else ("CUT" if n == cut_at else "OK")
        if cam == "CUT":
            epoch += 1
        trs = []
        for k, (tid, (st, conf, unc)) in enumerate(sorted(live.items())):
            if tid not in state and (st, conf, unc) != N:
                raise ValueError(f"track {tid} must start as N")
            s = state.setdefault(tid, {"age": -1, "obs": 0, "miss": 0, "eps": 0, "unc": False, "epoch": epoch})
            s["age"] += 1
            observed = st == "A"
            s["obs"] += 1 if observed else 0
            s["miss"] = 0 if observed else s["miss"] + 1
            if unc and not s["unc"]:
                s["eps"] += 1
            s["unc"] = unc
            x = 100.0 * k
            occluder = None
            if st == "O":
                occluder = next(o for o in sorted(live) if o != tid)
            rival = next((o for o in sorted(live) if o != tid), "ghost")
            trs.append({
                "track_id": tid, "state": {"A": "ACTIVE", "O": "OCCLUDED", "L": "LOST"}[st], "confirmed": conf,
                "source": "OBSERVED" if observed else "PREDICTED", "box": [x, 100.0, x + 40.0, 200.0],
                "detection_index": k if observed else None, "detection_score": 0.9 if observed else None,
                "confidence": 0.9, "contested": False, "age_frames": s["age"], "missed_frames": s["miss"],
                "time_since_observed_s": 0 if observed else s["miss"] / FPS, "observed_frames": max(1, s["obs"]),
                "epoch": s["epoch"], "occluder": occluder, "max_overlap_iou": 0.0, "overlap_with": None,
                "recoveries": 0, "last_gap_s": 0, "possible_predecessors": list((preds or {}).get(tid, [])),
                "identity": "IDENTITY_UNCERTAIN" if unc else "CERTAIN", "identity_rivals": [rival] if unc else [],
                "uncertain_episodes": s["eps"]})
        out.append({"frame": n, "t": n / FPS,
                    "camera": {"status": cam, "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "inliers": 0,
                               "hist_distance": None},
                    "tracks": trs, "ended": [list(e) for e in ended.get(n, [])]})
    last_live = set(frames[-1]) if frames else set()
    ended_reason = {tid: r for es in ended.values() for tid, r in es}
    tracklets = []
    for tid in sorted(state):
        seen = [n for n, live in enumerate(frames) if tid in live]
        tracklets.append({"track_id": tid, "epoch": state[tid]["epoch"], "first_frame": seen[0],
                          "last_frame": seen[-1], "last_observed_frame": seen[-1], "observed_frames": state[tid]["obs"],
                          "confirmed": True, "end_reason": None if tid in last_live else ended_reason[tid],
                          "recoveries": 0, "possible_predecessors": list((preds or {}).get(tid, []))})
    doc = {"schema": "spana.tracks", "schema_version": "2.1.0", "provenance": {}, "video": {"sha256": "v" * 64},
           "detections_sha256": "d" * 64, "frames": out, "tracklets": tracklets}
    validate_tracks(doc)
    return doc


A, AU, O, L, N = ("A", True, False), ("A", True, True), ("O", True, False), ("L", True, False), ("A", False, False)
W = {"t1": N, "t2": N}   # warm-up frame: both tracks fresh (unconfirmed)


def states(doc: dict, sel: Selection, cfg=CFG) -> List[Tuple[str, str]]:
    frames, _ = follow(doc, sel, cfg)
    return [(f["state"], f["reason"]) for f in frames]


class FollowerTest(unittest.TestCase):
    def test_follows_confident_track_and_never_switches(self):
        doc = tracks_doc([W] + [{"t1": A, "t2": A}] * 4)
        frames, info = follow(doc, Selection.track_at_frame("t1", 2), CFG)
        self.assertEqual([f["frame"] for f in frames], [2, 3, 4])          # from the selection on, never earlier
        self.assertTrue(all(f["state"] == "TARGET" and f["track_id"] == "t1" for f in frames))
        self.assertTrue(all(f["box"] == [0.0, 100.0, 40.0, 200.0] for f in frames))
        self.assertEqual(info, {"track_ends": [], "relinks": [], "reentry_log": []})

    def test_track_end_is_lost_for_the_rest(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": A, "t2": A}, {"t2": A}, {"t2": A}],
                         ended={3: [("t1", "LOST_EXPIRED")]})
        frames, info = follow(doc, Selection.track_at_frame("t1", 1), CFG)
        self.assertEqual([(f["state"], f["reason"]) for f in frames],
                         [("TARGET", "TRACK_OBSERVED")] * 2 + [("LOST", "TRACK_ENDED")] * 2)
        self.assertEqual(info["track_ends"], [{"track_id": "t1", "frame": 3, "reason": "LOST_EXPIRED"}])
        self.assertTrue(all(f["box"] is None and f["track_id"] == "t1" for f in frames[2:]))   # 2.0.0: names t1

    def test_camera_cut_ends_following_even_if_a_new_track_sits_on_the_same_place(self):
        doc = tracks_doc([{"t1": N}, {"t1": A}, {"t1": A}, {"t9": N}, {"t9": A}], ended={3: [("t1", "CAMERA_CUT")]},
                         cut_at=3)
        frames, info = follow(doc, Selection.track_at_frame("t1", 1), CFG)
        self.assertEqual([f["state"] for f in frames], ["TARGET", "TARGET", "LOST", "LOST"])
        self.assertEqual(info["track_ends"][0]["reason"], "CAMERA_CUT")
        self.assertTrue(all(f["track_id"] == "t1" for f in frames))   # never t9

    def test_identity_uncertain_abstains_and_stays_closed(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": A, "t2": A}, {"t1": A, "t2": A}])
        self.assertEqual(states(doc, Selection.track_at_frame("t1", 1)),
                         [("TARGET", "TRACK_OBSERVED"), ("ABSTAIN", "IDENTITY_UNCERTAIN"),
                          ("ABSTAIN", "AFTER_UNCERTAIN"), ("ABSTAIN", "AFTER_UNCERTAIN")])

    def test_resume_after_uncertain_is_a_config_policy(self):
        cfg = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, resume_after_uncertain=True))
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": A, "t2": A}])
        self.assertEqual([s for s, _ in states(doc, Selection.track_at_frame("t1", 1), cfg)],
                         ["TARGET", "ABSTAIN", "TARGET"])

    def test_selection_on_an_uncertain_track_abstains(self):
        doc = tracks_doc([W, {"t1": AU, "t2": AU}, {"t1": A, "t2": A}])
        self.assertEqual([s for s, _ in states(doc, Selection.track_at_frame("t1", 1))], ["ABSTAIN", "ABSTAIN"])

    def test_occluded_abstains_lost_is_lost_and_tracker_recovery_is_followed(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": O, "t2": A}, {"t1": L, "t2": A}, {"t1": A, "t2": A}])
        self.assertEqual(states(doc, Selection.track_at_frame("t1", 1)),
                         [("TARGET", "TRACK_OBSERVED"), ("ABSTAIN", "TRACK_OCCLUDED"), ("LOST", "TRACK_LOST"),
                          ("TARGET", "TRACK_OBSERVED")])

    def test_unconfirmed_track_abstains_until_confirmed(self):
        doc = tracks_doc([{"t1": N}, {"t1": A}])
        self.assertEqual([s for s, _ in states(doc, Selection.track_at_frame("t1", 0))], ["ABSTAIN", "TARGET"])

    def test_bad_selections_are_refused(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": O, "t2": A}])
        for sel in (Selection.track_at_frame("nope", 1),          # non-existent track
                    Selection.track_at_frame("t1", 7),            # frame not in the artifact
                    Selection.track_at_frame("t1", 2),            # not an observed box on that frame
                    Selection("shirt_number", "1.0.0", 1, "t1")):  # selection kind not supported yet
            with self.assertRaises(SelectionError):
                follow(doc, sel, CFG)


class TargetArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tracks = tracks_doc([W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t2": A}],
                                 ended={3: [("t1", "LEFT_FRAME")]})
        self.doc = target_document(self.tracks, Selection.track_at_frame("t1", 1), CFG, ["test"])

    def test_document_is_valid_and_bound_to_its_tracks(self):
        validate_target(self.doc, self.tracks)
        self.assertEqual(self.doc["tracks_sha256"], sha256_doc(self.tracks))
        self.assertEqual(self.doc["selection"], {"kind": "track_at_frame", "version": "1.0.0", "frame": 1,
                                                 "track_id": "t1"})
        other = copy.deepcopy(self.tracks)
        other["detections_sha256"] = "e" * 64
        with self.assertRaises(ArtifactError):
            validate_target(self.doc, other)

    def test_validator_refuses_inconsistent_answers(self):
        def broken(edit):
            d = copy.deepcopy(self.doc)
            edit(d)
            return d
        cases = [
            lambda d: d["frames"][1].update(state="TARGET", reason="TRACK_OBSERVED", track_id="t1",
                                            box=[100.0, 100.0, 140.0, 200.0]),           # TARGET while uncertain
            lambda d: d["frames"][0].update(track_id="t2"),                              # another track
            lambda d: d["frames"][0].update(state="CANDIDATE"),                          # CANDIDATE needs its reason
            lambda d: d["frames"][2].update(state="ABSTAIN"),                            # ended -> must be LOST
            lambda d: d["frames"][1].update(box=[0.0, 0.0, 1.0, 1.0]),                   # box without an answer
            lambda d: d.update(track_ends=[]),                                           # end not recorded
            lambda d: d["frames"].pop(0),                                                # does not start at selection
        ]
        for n, edit in enumerate(cases):
            with self.subTest(case=n), self.assertRaises(ArtifactError):
                validate_target(broken(edit), self.tracks)

    def test_written_once_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "target.json"
            write_json_atomic(p, self.doc)
            with self.assertRaises(ArtifactError):
                write_json_atomic(p, self.doc)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            trk, out = Path(tmp) / "trk.json", Path(tmp) / "target.json"
            write_json_atomic(trk, self.tracks)
            cmd = [sys.executable, "-m", "spana.cli", "target", "--tracks", str(trk), "--config",
                   str(ROOT / "config" / "default.yaml"), "--select-track", "t1", "--select-frame", "1", "--out"]
            r = subprocess.run(cmd + [str(out)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            doc = read_json(out)
            validate_target(doc, self.tracks)
            self.assertEqual([f["state"] for f in doc["frames"]], ["TARGET", "ABSTAIN", "LOST"])
            bad = subprocess.run(cmd[:-5] + ["--select-track", "missing", "--select-frame", "1", "--out",
                                             str(Path(tmp) / "x.json")], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("SelectionError", bad.stderr)
            self.assertFalse((Path(tmp) / "x.json").exists())


if __name__ == "__main__":
    unittest.main()


class TargetScoreTest(unittest.TestCase):
    """tools/target_score.py on synthetic frozen labels: correct / wrong / no-answer bookkeeping."""

    def test_counts(self):
        import hashlib
        import json
        from tools.target_score import score_clip
        tracks = tracks_doc([W] + [{"t1": A, "t2": A}] * 4)        # t1 box x 0..40, t2 box x 100..140
        lines = []
        for f in (1, 2, 3, 4):
            p1x, p2x = (0, 100) if f <= 2 else (300, 0)         # from frame 3 the t1 box shows player 2
            lines += [f"{f + 1},1,{p1x},100,40,100,1,1,1", f"{f + 1},2,{p2x},100,40,100,1,1,1"]
        gt = ("\n".join(lines) + "\n").encode()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "labels"
            d.mkdir()
            (d / "gt.txt").write_bytes(gt)
            (d / "FROZEN.json").write_text(json.dumps({
                "split": "DEV", "gt_sha256": hashlib.sha256(gt).hexdigest(), "frame_step": 1,
                "clip": {"name": "synthetic", "width": 640, "height": 360, "sha256": "v" * 64}}))
            (d / "labels_frozen.json").write_text(json.dumps({"frames_labelled": [1, 2, 3, 4]}))
            trk = Path(tmp) / "trk.json"
            write_json_atomic(trk, tracks)
            rep = score_clip(d, trk, CFG, None)
        p1 = next(p for p in rep["per_player"] if p["player"] == "1")
        self.assertEqual(p1["selection"]["track_id"], "t1")
        self.assertEqual((p1["correct"], p1["wrong"], p1["abstain"], p1["lost"]), (2, 2, 0, 0))
        self.assertEqual(p1["wrong_time"], 0.5)
        self.assertEqual(rep["pooled"]["wrong"], 2)
