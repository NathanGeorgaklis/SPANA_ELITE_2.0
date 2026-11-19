"""Target recovery inputs (spana.target 2.1.0): confirm an offer by its relink entry (also after it was dropped),
reject an offer, re-select by a tap mid-clip, the gap list, 2.0.0 compatibility, and the simulated after-match
reviewer of tools/target_score.py. Synthetic tracks at 10 fps."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import ArtifactError, gaps_of, validate_target, write_json_atomic
from spana.identity.follower import (Selection, UserActionError, confirm_candidate, confirm_relink, follow,
                                     reject_candidate, reselect, target_document)
from tests.test_target_follow import CFG, A, L, N, tracks_doc
from tests.test_target_relink import SEL, W, lost_then_new


def states(frames):
    return [(f["state"], f["track_id"]) for f in frames]


def dropped_offer():
    """t1 lost after frame 1; t3 (naming t1) offered on frame 3, dropped on frame 4 (briefly LOST), back on 5-6."""
    return tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A, "t3": N}, {"t1": L, "t2": A, "t3": A},
                       {"t1": L, "t2": A, "t3": L}, {"t1": L, "t2": A, "t3": A}, {"t1": L, "t2": A, "t3": A}],
                      preds={"t3": ["t1"]})


class RecoveryActionTest(unittest.TestCase):
    def test_confirm_an_offer_after_it_was_dropped(self):
        doc = dropped_offer()
        frames, info = follow(doc, SEL, CFG)
        self.assertEqual([s for s, _ in states(frames)], ["TARGET", "LOST", "CANDIDATE", "LOST", "LOST", "LOST"])
        (r,) = info["relinks"]
        self.assertEqual(r["outcome"], "DROPPED")
        later, info2 = follow(doc, SEL, CFG, [confirm_relink(r, 5)])          # reviewer confirms it at frame 5
        self.assertEqual(states(later)[4:], [("TARGET", "t3"), ("TARGET", "t3")])
        r2 = info2["relinks"][0]
        self.assertEqual((r2["outcome"], r2["dropped_reason"]), ("CONFIRMED", "CANDIDATE_NOT_USABLE"))

    def test_confirm_relink_refused_when_the_offered_track_is_gone_or_unknown(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A, "t3": N}, {"t1": L, "t2": A, "t3": A},
                          {"t1": L, "t2": A}], ended={4: [("t3", "LOST_EXPIRED")]}, preds={"t3": ["t1"]})
        r = follow(doc, SEL, CFG)[1]["relinks"][0]
        with self.assertRaises(UserActionError):
            follow(doc, SEL, CFG, [confirm_relink(r, 4)])                         # t3 has ended
        with self.assertRaises(UserActionError):
            follow(doc, SEL, CFG, [confirm_relink({**r, "to_track": "t2"}, 3)])   # no such offer
        with self.assertRaises(UserActionError):
            follow(doc, SEL, CFG, [confirm_relink(r, 2)])                         # before the offer was made

    def test_reject_withdraws_the_offer(self):
        doc = lost_then_new()
        r = follow(doc, SEL, CFG)[1]["relinks"][0]
        frames, info = follow(doc, SEL, CFG, [reject_candidate(r, 3)])
        self.assertEqual([s for s, _ in states(frames)], ["TARGET", "LOST", "LOST", "LOST"])
        r2 = info["relinks"][0]
        self.assertEqual((r2["outcome"], r2["dropped_reason"], r2["rejection"]["kind"]),
                         ("DROPPED", "REJECTED_BY_USER", "reject_candidate"))
        with self.assertRaises(UserActionError):                              # a rejected offer cannot be confirmed
            follow(doc, SEL, CFG, [reject_candidate(r, 3), confirm_relink(r, 4)])

    def test_reselect_by_tap(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A}, {"t1": L, "t2": A}, {"t1": L, "t2": A}])
        frames, _ = follow(doc, SEL, CFG, [reselect("t2", 3)])
        self.assertEqual(states(frames), [("TARGET", "t1"), ("LOST", "t1"), ("TARGET", "t2"), ("TARGET", "t2")])
        for bad in (reselect("t1", 3),        # not an observed box on that frame
                    reselect("t9", 3),        # no such track
                    reselect("t2", 0)):       # before the selection
            with self.subTest(bad=bad), self.assertRaises(UserActionError):
                follow(doc, SEL, CFG, [bad])

    def test_reselect_drops_a_pending_offer_and_clears_taint(self):
        doc = lost_then_new()
        frames, info = follow(doc, SEL, CFG, [reselect("t2", 3)])
        self.assertEqual(states(frames)[2:], [("TARGET", "t2"), ("TARGET", "t2")])
        self.assertEqual(info["relinks"], [])                    # the offer of frame 3 is never made: t2 answers

    def test_one_track_changing_action_per_frame(self):
        doc = lost_then_new()
        r = follow(doc, SEL, CFG)[1]["relinks"][0]
        with self.assertRaises(UserActionError):
            follow(doc, SEL, CFG, [reselect("t2", 3), confirm_relink(r, 3)])


class GapsAndValidatorTest(unittest.TestCase):
    def test_gaps_list_runs_without_a_target_answer(self):
        doc = dropped_offer()
        frames, info = follow(doc, SEL, CFG)
        (g,) = gaps_of(frames, info["relinks"])
        self.assertEqual((g["start_frame"], g["end_frame"], g["frames"]), (2, 6, 5))
        self.assertEqual(g["states"], ["CANDIDATE", "LOST"])
        self.assertEqual(g["offers"], [["t1", "t3", 3]])
        frames2, info2 = follow(doc, SEL, CFG, [confirm_relink(info["relinks"][0], 5)])
        self.assertEqual([(x["start_frame"], x["end_frame"]) for x in gaps_of(frames2, info2["relinks"])], [(2, 4)])

    def test_document_2_6_0_and_2_5_0(self):
        tracks = dropped_offer()
        r = follow(tracks, SEL, CFG)[1]["relinks"][0]
        doc = target_document(tracks, SEL, CFG, ["test"], [confirm_relink(r, 5)])
        self.assertEqual(doc["schema_version"], "2.6.0")
        validate_target(doc, tracks)
        from tests.target_versions import older
        v25 = older(doc, "2.5.0")                                          # 2.5.0 = without resume_after_reid
        self.assertNotIn("resume_after_reid", v25["target_config"])
        validate_target(v25, tracks)
        with self.assertRaises(ArtifactError):
            validate_target(dict(v25, target_config=doc["target_config"]), tracks)

        def broken(edit):
            d = copy.deepcopy(doc)
            edit(d)
            return d
        cases = [
            lambda d: d["gaps"].pop(),                                               # gaps disagree with frames
            lambda d: d["user_actions"].append(reselect("t2", 2)),                   # action applies to nothing
            lambda d: d["relinks"][0].update(confirmation=confirm_relink(r, 6)),     # not among the user actions
            lambda d: d["relinks"][0].update(rejection=reject_candidate(r, 3)),      # rejection not given
        ]
        for n, edit in enumerate(cases):
            with self.subTest(case=n), self.assertRaises(ArtifactError):
                validate_target(broken(edit), tracks)

    def test_reselection_is_checked_against_the_tracks(self):
        tracks = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": A}, {"t1": L, "t2": A}])
        doc = target_document(tracks, SEL, CFG, ["test"], [reselect("t2", 3)])
        validate_target(doc, tracks)
        d = copy.deepcopy(doc)
        d["user_actions"][0]["track_id"] = "t1"                              # t1 is LOST on frame 3
        d["frames"][2].update(track_id="t1")
        with self.assertRaises(ArtifactError):
            validate_target(d, tracks)

    def test_2_0_0_documents_still_validate(self):
        tracks = lost_then_new()
        d = target_document(tracks, SEL, CFG, ["test"], [confirm_candidate("t3", 4)])
        from tests.target_versions import older
        d = older(d, "2.0.0")
        v2 = {k: v for k, v in d.items() if k not in ("user_actions", "gaps")}
        v2.update(confirmations=d["user_actions"],
                  relinks=[{k: v for k, v in r.items() if k != "rejection"} for r in d["relinks"]])
        validate_target(v2, tracks)
        v2["confirmations"].append(reselect("t2", 3))                       # 2.0.0 knows confirmations only
        with self.assertRaises(ArtifactError):
            validate_target(v2, tracks)


class SimulatedReviewerTest(unittest.TestCase):
    def test_reviewer_confirms_a_dropped_right_offer_and_taps_elsewhere(self):
        from tools.target_score import score_clip
        tracks = dropped_offer()
        # live tracks are drawn at x = 100 * (index among the frame's sorted ids): t1 x0, t2 x100, t3 x200
        rows = []
        for f, x in ((1, 0), (2, 300), (4, 200), (5, 200)):
            rows += [f"{f + 1},1,{x},100,40,100,1,1,1"]
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
            every_gap = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, review_min_gap_s=0.0))
            base = score_clip(d, trk, every_gap, None)
            rev = score_clip(d, trk, every_gap, None, simulate_reviewer=True)   # synthetic gaps are < 1 s
        p0, p1 = base["per_player"][0], rev["per_player"][0]
        self.assertEqual((p0["correct"], p0["lost"]), (1, 3))       # dropped offer is not re-offered: frame 5 LOST
        self.assertEqual(p1["actions"]["confirm_relink"], 1)                   # the dropped offer, confirmed
        self.assertEqual((p1["correct"], p1["wrong"]), (2, 0))                 # frame 5 on t3; frame 4 t3 LOST
        self.assertEqual(rev["pooled"]["actions"]["confirm_relink"], 1)


if __name__ == "__main__":
    unittest.main()
