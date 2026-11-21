"""Target relink (spana.target 2.0.0): CANDIDATE proposals from the tracker's unique predecessor link, the user's
confirmation, automatic promotion behind a flag, and the 2.0.0 / 1.0.0 validators. Synthetic tracks, 10 fps."""
from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import ArtifactError, read_json, validate_target, write_json_atomic
from spana.identity.follower import (ConfirmationError, Selection, confirm_candidate, follow, target_document)
from tests.test_target_follow import CFG, ROOT, A, AU, L, N, tracks_doc

W = {"t1": N, "t2": N}
SEL = Selection.track_at_frame("t1", 1)
AUTO = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, relink_auto_confirm=True))
NO_HOLD = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, offer_hold_s=0.0))   # drop at once


def lost_then_new(extra_new=None, gap_frames=1, more=2):
    """t1 followed, lost after frame 1; t3 born gap_frames later naming t1, confirmed one frame after its birth."""
    frames = [W, {"t1": A, "t2": A}]
    frames += [{"t1": L, "t2": A}] * (gap_frames - 1)
    born = {"t1": L, "t2": A, "t3": N, **({k: N for k in extra_new} if extra_new else {})}
    frames.append(born)
    frames += [{"t1": L, "t2": A, "t3": A, **({k: A for k in extra_new} if extra_new else {})}] * more
    preds = {"t3": ["t1"], **({k: ["t1"] for k in extra_new} if extra_new else {})}
    return tracks_doc(frames, preds=preds)


def states(frames):
    return [(f["state"], f["track_id"]) for f in frames]


class RelinkTest(unittest.TestCase):
    def test_unique_predecessor_within_gap_is_a_candidate_after_it_confirms(self):
        doc = lost_then_new()
        frames, info = follow(doc, SEL, CFG)
        self.assertEqual(states(frames), [("TARGET", "t1"), ("LOST", "t1"), ("CANDIDATE", "t3"), ("CANDIDATE", "t3")])
        self.assertEqual(frames[2]["reason"], "RELINK_PREDECESSOR")
        self.assertIsNotNone(frames[2]["box"])
        (r,) = info["relinks"]
        self.assertEqual((r["from_track"], r["to_track"], r["lost_frame"], r["proposed_frame"], r["outcome"]),
                         ("t1", "t3", 1, 3, "PENDING"))
        self.assertAlmostEqual(r["gap_s"], 0.1)
        self.assertIsNone(r["confirmation"])

    def test_two_candidates_naming_the_lost_track_stay_lost(self):
        frames, info = follow(lost_then_new(extra_new=["t4"]), SEL, CFG)
        self.assertEqual([s for s, _ in states(frames)], ["TARGET", "LOST", "LOST", "LOST"])
        self.assertEqual(info["relinks"], [])

    def test_gap_too_long_stays_lost(self):
        frames, info = follow(lost_then_new(gap_frames=6), SEL, CFG)      # born 0.6 s after the last observation
        self.assertNotIn("CANDIDATE", [f["state"] for f in frames])
        self.assertEqual(info["relinks"], [])

    def test_camera_cut_between_loss_and_proposal_stays_lost(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t3": N}, {"t3": A}],
                         ended={2: [("t1", "CAMERA_CUT"), ("t2", "CAMERA_CUT")]}, cut_at=2, preds={"t3": ["t1"]})
        frames, info = follow(doc, SEL, CFG)
        self.assertEqual(states(frames), [("TARGET", "t1"), ("LOST", "t1"), ("LOST", "t1")])
        self.assertEqual(info["relinks"], [])

    def test_lost_track_uncertain_before_its_loss_is_not_relinked(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": L, "t2": A, "t3": N},
                          {"t1": L, "t2": A, "t3": A}], preds={"t3": ["t1"]})
        frames, _ = follow(doc, SEL, CFG)
        self.assertNotIn("CANDIDATE", [f["state"] for f in frames])

    def test_candidate_dropped_when_the_original_returns(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A, "t3": N}, {"t1": L, "t2": A, "t3": A},
                          {"t1": A, "t2": A, "t3": A}], preds={"t3": ["t1"]})
        frames, info = follow(doc, SEL, CFG)
        self.assertEqual(states(frames)[-2:], [("CANDIDATE", "t3"), ("TARGET", "t1")])
        self.assertEqual((info["relinks"][0]["outcome"], info["relinks"][0]["dropped_reason"]),
                         ("DROPPED", "ORIGINAL_OBSERVED"))

    def test_candidate_dropped_when_it_becomes_unusable(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A, "t3": N}, {"t1": L, "t2": A, "t3": A},
                          {"t1": L, "t2": A, "t3": L}, {"t1": L, "t2": A, "t3": A}], preds={"t3": ["t1"]})
        frames, info = follow(doc, SEL, NO_HOLD)          # with the default offer hold: tests/test_offer_hold.py
        self.assertEqual([s for s, _ in states(frames)], ["TARGET", "LOST", "CANDIDATE", "LOST", "LOST"])
        self.assertEqual(info["relinks"][0]["dropped_reason"], "CANDIDATE_NOT_USABLE")

    def test_confirmation_makes_the_candidate_the_target(self):
        doc = lost_then_new()
        frames, info = follow(doc, SEL, CFG, [confirm_candidate("t3", 3)])
        self.assertEqual(states(frames), [("TARGET", "t1"), ("LOST", "t1"), ("TARGET", "t3"), ("TARGET", "t3")])
        self.assertEqual(info["relinks"][0]["outcome"], "CONFIRMED")
        later, _ = follow(doc, SEL, CFG, [confirm_candidate("t3", 4)])     # confirmed on a later candidate frame
        self.assertEqual([s for s, _ in states(later)], ["TARGET", "LOST", "CANDIDATE", "TARGET"])

    def test_confirmations_that_name_no_candidate_are_refused(self):
        doc = lost_then_new()
        for bad in (confirm_candidate("t2", 3),                              # another track
                    confirm_candidate("t3", 2),                              # t3 not yet a candidate (unconfirmed)
                    {"kind": "confirm_candidate", "version": "9.9.9", "frame": 3, "track_id": "t3"}):
            with self.subTest(bad=bad), self.assertRaises(ConfirmationError):
                follow(doc, SEL, CFG, [bad])

    def test_auto_confirm_flag_promotes_immediately(self):
        self.assertFalse(CFG.target.relink_auto_confirm)                       # off by default
        frames, info = follow(lost_then_new(), SEL, AUTO)
        self.assertEqual(states(frames)[2:], [("TARGET", "t3"), ("TARGET", "t3")])
        self.assertEqual(info["relinks"][0]["confirmation"], {"kind": "auto_confirm", "frame": 3})

    def test_chain_of_two_relinks(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A, "t3": N}, {"t1": L, "t2": A, "t3": A},
                          {"t2": A, "t3": L, "t5": N}, {"t2": A, "t3": L, "t5": A}, {"t2": A, "t3": L, "t5": A}],
                         ended={4: [("t1", "LOST_EXPIRED")]}, preds={"t3": ["t1"], "t5": ["t3"]})
        frames, info = follow(doc, SEL, CFG, [confirm_candidate("t3", 3), confirm_candidate("t5", 5)])
        self.assertEqual(states(frames), [("TARGET", "t1"), ("LOST", "t1"), ("TARGET", "t3"), ("LOST", "t3"),
                                          ("TARGET", "t5"), ("TARGET", "t5")])
        self.assertEqual([(r["from_track"], r["to_track"], r["outcome"]) for r in info["relinks"]],
                         [("t1", "t3", "CONFIRMED"), ("t3", "t5", "CONFIRMED")])
        auto, _ = follow(doc, SEL, AUTO)
        self.assertEqual(states(auto), states(frames))

    def test_v1_behaviour_without_any_relink_is_unchanged(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A}, {"t1": A, "t2": A}])
        frames, info = follow(doc, SEL, AUTO)
        self.assertEqual([f["state"] for f in frames], ["TARGET", "LOST", "TARGET"])
        self.assertEqual(info["relinks"], [])


class RelinkArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tracks = lost_then_new()
        self.doc = target_document(self.tracks, SEL, CFG, ["test"], [confirm_candidate("t3", 4)])

    def test_valid(self):
        validate_target(self.doc, self.tracks)
        self.assertEqual(self.doc["schema_version"], "2.9.0")

    def test_validator_refuses_inconsistent_documents(self):
        def broken(edit):
            d = copy.deepcopy(self.doc)
            edit(d)
            return d
        cases = [
            lambda d: d["frames"][2].update(state="TARGET", reason="TRACK_OBSERVED"),     # TARGET before confirmation
            lambda d: d["frames"][2].update(track_id="t2"),                              # CANDIDATE never proposed
            lambda d: d["relinks"][0].update(gap_s=0.9),                                 # gap over the limit
            lambda d: d["relinks"][0].update(confirmation=None, outcome="PENDING"),      # unconsumed confirmation
            lambda d: d["user_actions"].append(confirm_candidate("t3", 9)),             # confirms nothing
            lambda d: d["relinks"][0]["evidence"].update(possible_predecessors=[]),      # no evidence
            lambda d: d["frames"][1].update(track_id=None),                              # every frame names a track
            lambda d: d["relinks"][0]["confirmation"].update(frame=3),                   # not the given confirmation
        ]
        for n, edit in enumerate(cases):
            with self.subTest(case=n), self.assertRaises(ArtifactError):
                validate_target(broken(edit), self.tracks)
        rival = lost_then_new(extra_new=["t4"])       # with a second lister the relink is not supported
        with self.assertRaises(ArtifactError):
            validate_target(self.doc, rival)

    def test_auto_confirmation_requires_the_flag(self):
        doc = target_document(self.tracks, SEL, AUTO, ["test"])
        validate_target(doc, self.tracks)
        doc["target_config"]["relink_auto_confirm"] = False
        with self.assertRaises(ArtifactError):
            validate_target(doc, self.tracks)

    def test_v1_documents_still_validate(self):
        tracks = tracks_doc([W, {"t1": A, "t2": A}, {"t1": A, "t2": A}, {"t2": A}], ended={3: [("t1", "LEFT_FRAME")]})
        d2 = target_document(tracks, SEL, CFG, ["test"])
        v1 = {"schema": "spana.target", "schema_version": "1.0.0", "provenance": d2["provenance"],
              "video": d2["video"], "tracks_sha256": d2["tracks_sha256"], "selection": d2["selection"],
              "target_config": {"resume_after_uncertain": False},
              "track_end": {"frame": 3, "reason": "LEFT_FRAME"},
              "frames": [{**f, "track_id": f["track_id"] if f["state"] == "TARGET" else None} for f in d2["frames"]]}
        validate_target(v1, tracks)
        v1["frames"][0].update(state="CANDIDATE", reason="RELINK_PREDECESSOR")
        with self.assertRaises(ArtifactError):
            validate_target(v1, tracks)


class RelinkScoreTest(unittest.TestCase):
    """tools/target_score.py: candidate frames counted apart from answers; the simulated user confirms right ones."""

    def _score(self, cfg, simulate_user, rows=None, simulate_reviewer=False):
        import hashlib
        from tools.target_score import score_clip
        tracks = lost_then_new(more=3)          # t3 born frame 3, confirmed from frame 4; frames 0..6
        # live tracks are drawn at x = 100 * (index among the frame's sorted ids): t1 x0, t2 x100, t3 x200
        if rows is None:
            rows = []
            for f in (1, 2, 4, 5):
                rows += [f"{f + 1},1,{0 if f == 1 else 200},100,40,100,1,1,1", f"{f + 1},2,100,100,40,100,1,1,1"]
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
            rep = score_clip(d, trk, cfg, None, simulate_user, simulate_reviewer)
        return next(p for p in rep["per_player"] if p["player"] == "1"), rep

    def test_candidates_never_count_as_correct_time(self):
        p1, rep = self._score(CFG, False)
        self.assertEqual((p1["correct"], p1["candidate_correct"], p1["lost"]), (1, 2, 1))
        self.assertEqual(p1["relinks"][0]["verdict"], "right")
        self.assertEqual(rep["pooled"]["relinks"]["proposed"], 1)

    def test_an_offer_of_another_player_is_wrong_while_ours_is_out_of_the_picture(self):
        """Player 1 leaves after frame 2; the offered t3 is player 3. Before 2026-11-19 this was "undecided"
        because player 1 is not labelled on the offer's frames."""
        rows = [f"{f + 1},1,0,100,40,100,1,1,1" for f in (1, 2)]
        rows += [f"{f + 1},2,100,100,40,100,1,1,1" for f in (1, 2, 4, 5)]
        rows += [f"{f + 1},3,200,100,40,100,1,1,1" for f in (4, 5)]
        every_gap = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, review_min_gap_s=0.0))
        p1, rep = self._score(every_gap, False, rows)
        r = p1["relinks"][0]
        self.assertEqual((r["verdict"], r["verdict_player"]), ("wrong", "3"))
        self.assertEqual((p1["candidate_wrong"], p1["candidates_while_absent_on_other_player"]), (0, 2))
        self.assertEqual(rep["pooled"]["relinks"]["wrong"], 1)
        rev, _ = self._score(every_gap, False, rows, simulate_reviewer=True)
        self.assertEqual((rev["actions"]["reject_candidate"], rev["actions"]["confirm_relink"]), (1, 0))
        self.assertEqual(rev["wrong"], 0)

    def test_simulated_user_confirms_the_right_candidate(self):
        p1, rep = self._score(CFG, True)
        self.assertEqual((p1["correct"], p1["candidate_correct"], p1["confirmations"]), (3, 0, 1))
        self.assertEqual(rep["pooled"]["relinks"]["confirmed"], 1)
        auto, _ = self._score(AUTO, False)
        self.assertEqual(auto["correct"], 3)


class RelinkCliTest(unittest.TestCase):
    def test_confirm_file(self):
        tracks = lost_then_new()
        with tempfile.TemporaryDirectory() as tmp:
            trk, conf, out = Path(tmp) / "trk.json", Path(tmp) / "confirm.json", Path(tmp) / "target.json"
            write_json_atomic(trk, tracks)
            conf.write_text(json.dumps([confirm_candidate("t3", 3)]))
            cmd = [sys.executable, "-m", "spana.cli", "target", "--tracks", str(trk), "--config",
                   str(ROOT / "config" / "default.yaml"), "--select-track", "t1", "--select-frame", "1"]
            r = subprocess.run(cmd + ["--confirm", str(conf), "--out", str(out)], cwd=ROOT, capture_output=True,
                               text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            doc = read_json(out)
            validate_target(doc, tracks)
            self.assertEqual([f["state"] for f in doc["frames"]], ["TARGET", "LOST", "TARGET", "TARGET"])
            conf2 = Path(tmp) / "bad.json"
            conf2.write_text(json.dumps([confirm_candidate("t2", 3)]))
            bad = subprocess.run(cmd + ["--confirm", str(conf2), "--out", str(Path(tmp) / "x.json")], cwd=ROOT,
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("UserActionError", bad.stderr)
            self.assertFalse((Path(tmp) / "x.json").exists())


if __name__ == "__main__":
    unittest.main()
