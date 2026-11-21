"""Player confirmations of target identity episodes (spana.target 2.10.0, spana.identity.confirmation).

Synthetic tracks at 10 fps (tests.test_target_follow.tracks_doc: an uncertain track's rival is the next live track).
Generic ids only. Covers: IS_ME resolves the episode for the target only; scope (a later episode is uncertain
again); NOT_ME (negative evidence, offers excluded); every conflict rule; the gallery (clean crops only, tagged);
undo (outputs identical to no confirmation); refusals; gallery re-ID of a later episode (off by default);
the validator; the simulated reviewer's confirmation taps.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from spana.artifacts import ArtifactError, validate_target
from spana.identity.confirmation import (Crop, CropFeatures, appearance_conflicts, claims_from_targets,
                                         gallery_frame_verdict, player_confirmation, revoke_confirmation)
from spana.identity.follower import (Selection, UserActionError, follow, reconfirmation_list, reselect,
                                     target_document)
from tests.target_versions import older
from tests.test_target_follow import CFG, A, AU, L, N, W, tracks_doc
from tests.test_target_relink import lost_then_new

SEL = Selection.track_at_frame("t1", 1)
OUTPUTS = ("track_ends", "relinks", "reentry_log", "gallery", "gallery_reid_log")
EPISODE = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, confirmation_scope="episode"))
GALLERY_ON = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, gallery_reid_resolve=True),
                                 reid_resolve=dataclasses.replace(CFG.reid_resolve, enabled=True))


def unit(*v) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32)
    return x / np.linalg.norm(x)


E1, E2, E3 = unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)
C1, C2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])     # torso colour histograms (normalised)


def crops(entries) -> CropFeatures:
    return CropFeatures({k: Crop(*v) for k, v in entries.items()}, {"schema": "spana.crop_features",
                                                                   "schema_version": "1.0.0", "tracks_sha256": ""})


def is_me(cid, tid, f, f_end=None):
    return player_confirmation(cid, tid, f, "IS_ME", f_end)


def not_me(cid, tid, f, f_end=None):
    return player_confirmation(cid, tid, f, "NOT_ME", f_end)


def answers(frames):
    return [(f["state"], f["reason"], f["track_id"]) for f in frames]


def crossing():
    """t1 / t2 confirmed (frame 1), a crossing makes both uncertain on frames 2-3 (episode 1), CERTAIN again 4-5."""
    return tracks_doc([W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": AU, "t2": AU},
                       {"t1": A, "t2": A}, {"t1": A, "t2": A}])


class IsMeTest(unittest.TestCase):
    def test_is_me_answers_the_confirmed_frames_only(self):
        doc = crossing()
        frames, info = follow(doc, SEL, CFG, [is_me("c1", "t1", 2)])
        # frame 2 confirmed; frame 3 is not (the track may change player inside the episode); the tracker's later
        # resolution re-confirms the pre-episode player, so the episode taints the rest of the track
        self.assertEqual(answers(frames), [("TARGET", "TRACK_OBSERVED", "t1"), ("TARGET", "PLAYER_CONFIRMED", "t1"),
                                           ("ABSTAIN", "IDENTITY_UNCERTAIN", "t1"), ("ABSTAIN", "AFTER_UNCERTAIN", "t1"),
                                           ("ABSTAIN", "AFTER_UNCERTAIN", "t1")])
        validate_target(target_document(doc, SEL, CFG, ["test"], [is_me("c1", "t1", 2)]), doc)

    def test_a_range_ending_on_a_certain_frame_continues_like_a_tap(self):
        doc = crossing()
        frames, _ = follow(doc, SEL, CFG, [is_me("c1", "t1", 2, 4)])
        self.assertEqual([r for _, r, _ in answers(frames)],
                         ["TRACK_OBSERVED", "PLAYER_CONFIRMED", "PLAYER_CONFIRMED", "TRACK_OBSERVED", "TRACK_OBSERVED"])
        validate_target(target_document(doc, SEL, CFG, ["test"], [is_me("c1", "t1", 2, 4)]), doc)

    def test_episode_scope_resolves_the_whole_episode_for_the_target_only(self):
        doc = crossing()
        base = answers(follow(doc, SEL, CFG)[0])
        self.assertEqual([s for s, _, _ in base], ["TARGET", "ABSTAIN", "ABSTAIN", "ABSTAIN", "ABSTAIN"])
        frames, info = follow(doc, SEL, EPISODE, [is_me("c1", "t1", 2)])
        self.assertEqual(answers(frames), [("TARGET", "TRACK_OBSERVED", "t1"), ("TARGET", "PLAYER_CONFIRMED", "t1"),
                                           ("TARGET", "PLAYER_CONFIRMED", "t1"), ("TARGET", "TRACK_OBSERVED", "t1"),
                                           ("TARGET", "TRACK_OBSERVED", "t1")])
        (e,) = info["confirmation_log"]
        self.assertEqual((e["status"], e["episode"], e["track_identity"], e["track_rivals"]),
                         ("APPLIED", 1, "IDENTITY_UNCERTAIN", ["t2"]))
        # the rival stays unresolved: following t2 is unchanged, and the tracks document is never modified
        sel2 = Selection.track_at_frame("t2", 1)
        self.assertEqual(follow(doc, sel2, CFG)[0], follow(crossing(), sel2, CFG)[0])
        self.assertEqual(doc, crossing())
        d = target_document(doc, SEL, EPISODE, ["test"], [is_me("c1", "t1", 2)])
        validate_target(d, doc)
        self.assertEqual(d["schema_version"], "2.10.0")
        with self.assertRaises(ArtifactError):           # the same answers are not backed under range scope
            validate_target({**d, "target_config": {**d["target_config"], "confirmation_scope": "range"}}, doc)

    def test_scope_a_later_episode_is_uncertain_again(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": A, "t2": A},
                          {"t1": AU, "t2": AU}, {"t1": A, "t2": A}])
        frames, _ = follow(doc, SEL, EPISODE, [is_me("c1", "t1", 2)])
        self.assertEqual([r for _, r, _ in answers(frames)],
                         ["TRACK_OBSERVED", "PLAYER_CONFIRMED", "TRACK_OBSERVED", "IDENTITY_UNCERTAIN",
                          "AFTER_UNCERTAIN"])

    def test_is_me_on_another_uncertain_track_moves_the_answer_there(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": L, "t2": AU}, {"t1": L, "t2": AU}, {"t1": L, "t2": A}])
        frames, _ = follow(doc, SEL, CFG, [is_me("c1", "t2", 3, 4)])
        self.assertEqual(answers(frames)[2:], [("TARGET", "PLAYER_CONFIRMED", "t2"), ("TARGET", "TRACK_OBSERVED", "t2")])
        validate_target(target_document(doc, SEL, CFG, ["t"], [is_me("c1", "t2", 3, 4)]), doc)


class NotMeTest(unittest.TestCase):
    def test_not_me_on_the_answered_uncertain_track(self):
        doc = crossing()
        frames, info = follow(doc, SEL, CFG, [not_me("n1", "t1", 2)])
        self.assertEqual([r for _, r, _ in answers(frames)],
                         ["TRACK_OBSERVED", "PLAYER_REJECTED", "PLAYER_REJECTED", "PLAYER_REJECTED",
                          "PLAYER_REJECTED"])          # same episode count until a new episode starts
        self.assertEqual(info["confirmation_log"][0]["status"], "APPLIED")
        self.assertEqual(info["gallery"]["human_confirmed"], [])
        validate_target(target_document(doc, SEL, CFG, ["t"], [not_me("n1", "t1", 2)]), doc)

    def test_not_me_removes_a_track_from_offers(self):
        doc = lost_then_new()
        frames, info = follow(doc, SEL, CFG)
        self.assertIn("CANDIDATE", [s for s, _, _ in answers(frames)])
        frames, info = follow(doc, SEL, CFG, [not_me("n1", "t3", 2)])
        self.assertNotIn("CANDIDATE", [s for s, _, _ in answers(frames)])
        self.assertEqual(info["relinks"], [])


class ConflictTest(unittest.TestCase):
    def test_track_certain_as_another_identity(self):
        doc = crossing()
        claims = {"other": {2: "t1", 3: "t1"}}
        frames, info = follow(doc, SEL, CFG, [is_me("c1", "t1", 2)], claims=claims)
        self.assertEqual(frames, follow(doc, SEL, CFG)[0])              # not applied: the track stays uncertain
        (e,) = info["confirmation_log"]
        self.assertEqual((e["status"], [c["reason"] for c in e["conflicts"]]), ("CONFLICT", ["OTHER_IDENTITY_CERTAIN"]))
        d = target_document(doc, SEL, CFG, ["t"], [is_me("c1", "t1", 2)], claims=claims)
        self.assertEqual(reconfirmation_list(d), [{"confirmation_id": "c1", "track_id": "t1", "frame": 2,
                                                   "verdict": "IS_ME", "reasons": ["OTHER_IDENTITY_CERTAIN"]}])

    def test_target_certain_elsewhere(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": A, "t2": AU}, {"t1": A, "t2": AU}])
        frames, info = follow(doc, SEL, CFG, [is_me("c1", "t2", 2)])
        self.assertEqual(info["confirmation_log"][0]["conflicts"], [{"reason": "TARGET_CERTAIN_ELSEWHERE", "track": "t1"}])
        self.assertTrue(all(f["track_id"] == "t1" and f["state"] == "TARGET" for f in frames))

    def test_not_me_against_the_targets_certain_track_holds_it_uncertain(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": A, "t2": A}, {"t1": A, "t2": A}])
        frames, info = follow(doc, SEL, CFG, [not_me("n1", "t1", 2)])
        self.assertEqual(info["confirmation_log"][0]["conflicts"], [{"reason": "TARGET_CERTAIN", "track": "t1"}])
        self.assertEqual([r for _, r, _ in answers(frames)],
                         ["TRACK_OBSERVED", "CONFIRMATION_CONFLICT", "CONFIRMATION_CONFLICT"])
        validate_target(target_document(doc, SEL, CFG, ["t"], [not_me("n1", "t1", 2)]), doc)

    def test_player_contradicting_themselves(self):
        doc = crossing()
        _, info = follow(doc, SEL, CFG, [not_me("n1", "t1", 2), is_me("c1", "t1", 3)])
        self.assertEqual([e["status"] for e in info["confirmation_log"]], ["APPLIED", "CONFLICT"])
        self.assertEqual(info["confirmation_log"][1]["conflicts"][0]["reason"], "PLAYER_CONTRADICTION")

    def test_reid_and_colour_contradiction(self):
        doc = crossing()
        # the target looked like E1 / C1 while answered (frame 1); another identity is E2 / C2 on t2 at frame 1;
        # the confirmed track t1 at frame 2 looks like E2 / C2: re-ID and colour both contradict the IS_ME
        feats = crops({(1, "t1"): (C1, E1), (1, "t2"): (C2, E2), (2, "t1"): (C2, E2)})
        claims = {"other": {1: "t2"}}
        frames, info = follow(doc, SEL, CFG, [is_me("c1", "t1", 2)], crops=feats, claims=claims)
        (e,) = info["confirmation_log"]
        self.assertEqual(sorted(c["reason"] for c in e["conflicts"]), ["COLOUR_CONTRARY", "REID_CONTRARY"])
        self.assertEqual(info["gallery"]["human_confirmed"], [])
        # the same crop looking like the target: no conflict, applied, one gallery crop
        feats = crops({(1, "t1"): (C1, E1), (1, "t2"): (C2, E2), (2, "t1"): (C1, E1)})
        _, info = follow(doc, SEL, CFG, [is_me("c1", "t1", 2)], crops=feats, claims=claims)
        self.assertEqual(info["confirmation_log"][0]["status"], "APPLIED")

    def test_appearance_rule_needs_every_crop(self):
        own, other = {"reid": E1, "colour": None}, {"q": {"reid": E2, "colour": None}}
        cs = [Crop(None, E2), Crop(None, E1)]                   # one crop looks like the target: no contradiction
        self.assertEqual(appearance_conflicts(cs, own, other, CFG)[0], [])
        self.assertEqual(appearance_conflicts([Crop(None, E2)], own, other, CFG)[0][0]["reason"], "REID_CONTRARY")
        self.assertEqual(appearance_conflicts([Crop(None, E2)], {"reid": None, "colour": None}, other, CFG)[0], [])


class GalleryTest(unittest.TestCase):
    def test_clean_crops_of_is_me_frames_only_tagged_with_the_confirmation(self):
        doc = crossing()
        feats = crops({(2, "t1"): (C1, E1), (2, "t2"): (C2, E2)})      # frame 3 of t1 is not clean: no entry
        _, info = follow(doc, SEL, CFG, [is_me("c1", "t1", 2, 3)], crops=feats)
        self.assertEqual(info["gallery"]["human_confirmed"],
                         [{"confirmation_id": "c1", "track_id": "t1", "frame": 2, "box": [0.0, 100.0, 40.0, 200.0]}])
        self.assertEqual(info["confirmation_log"][0]["gallery_crops"], 1)
        d = target_document(doc, SEL, CFG, ["t"], [is_me("c1", "t1", 2, 3)], crops=feats)
        validate_target(d, doc)
        bad = {**d, "gallery": {"human_confirmed": d["gallery"]["human_confirmed"] * 2}}
        with self.assertRaises(ArtifactError):
            validate_target(bad, doc)


class UndoTest(unittest.TestCase):
    def test_revoke_restores_every_output_exactly(self):
        doc = crossing()
        feats = crops({(2, "t1"): (C1, E1)})
        for acts in ([is_me("c1", "t1", 2)], [not_me("n1", "t1", 2)], [not_me("n1", "t1", 4)]):
            with self.subTest(acts=acts):
                cid = acts[0]["confirmation_id"]
                none_f, none_i = follow(doc, SEL, CFG, [], crops=feats)
                undo_f, undo_i = follow(doc, SEL, CFG, acts + [revoke_confirmation(cid)], crops=feats)
                self.assertEqual(undo_f, none_f)
                self.assertEqual({k: undo_i[k] for k in OUTPUTS}, {k: none_i[k] for k in OUTPUTS})
                (e,) = undo_i["confirmation_log"]
                self.assertEqual((e["status"], e["revocation"]["confirmation_id"]), ("REVOKED", cid))
                d0 = target_document(doc, SEL, CFG, ["t"], [], crops=feats)
                d1 = target_document(doc, SEL, CFG, ["t"], acts + [revoke_confirmation(cid)], crops=feats)
                validate_target(d1, doc)
                self.assertEqual(d1["frames"], d0["frames"])
                self.assertEqual(d1["gaps"], d0["gaps"])

    def test_revoke_with_taps_around_it(self):
        doc = crossing()
        acts = [reselect("t2", 4)]
        conf = [is_me("c1", "t1", 2)]
        self.assertEqual(follow(doc, SEL, CFG, acts + conf + [revoke_confirmation("c1")])[0],
                         follow(doc, SEL, CFG, acts)[0])


class RefusalTest(unittest.TestCase):
    def test_refused(self):
        doc = tracks_doc([W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": A, "t2": A}, {"t1": AU, "t2": AU},
                          {"t1": L, "t2": A}])
        for bad in ([is_me("c1", "t1", 2, 4)],                       # spans two episodes
                    [is_me("c1", "t1", 5)],                           # not observed on its first frame
                    [is_me("c1", "t9", 2)],                           # no such track
                    [is_me("c1", "t1", 0)],                           # before the selection
                    [revoke_confirmation("zz")],                      # names no confirmation
                    [is_me("c1", "t1", 2), is_me("c1", "t2", 2)],     # duplicate id
                    [is_me("c1", "t1", 2), revoke_confirmation("c1"), revoke_confirmation("c1", "player")],
                    [{**is_me("c1", "t1", 2), "verdict": "MAYBE"}],
                    [{**is_me("c1", "t1", 2), "source": "coach"}]):
            with self.subTest(bad=bad), self.assertRaises(UserActionError):
                follow(doc, SEL, CFG, bad)


class GalleryReidTest(unittest.TestCase):
    def doc(self):
        """t1 / t2: episode 1 on frame 2 (the player confirms t1), episode 2 on frames 4-11, CERTAIN on 12."""
        fr = [W, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": A, "t2": A}]
        fr += [{"t1": AU, "t2": AU}] * 8 + [{"t1": A, "t2": A}]
        return tracks_doc(fr)

    def feats(self, ep2=E1):
        e = {(1, "t2"): (C2, E2), (3, "t2"): (C2, E2), (2, "t1"): (C1, E1)}
        e.update({(f, "t1"): (C1, ep2) for f in range(4, 12)})
        return crops(e)

    def test_off_by_default(self):
        frames, info = follow(self.doc(), SEL, CFG, [is_me("c1", "t1", 2, 3)], crops=self.feats())
        self.assertEqual(answers(frames)[3][1], "IDENTITY_UNCERTAIN")
        self.assertEqual(info["gallery_reid_log"], [])

    def test_gallery_clears_a_later_episode_after_a_run(self):
        doc = self.doc()
        frames, info = follow(doc, SEL, GALLERY_ON, [is_me("c1", "t1", 2, 3)], crops=self.feats())
        reasons = [r for _, r, _ in answers(frames)]
        # frames 4-8 run (t 0.4 .. 0.8, 5 clears), qualified at frame 9 (t 0.9 = start + identity_resolve_s); each
        # answer needs its own clearing crop; nothing carries past the episode (the tracker's resolution taints)
        self.assertEqual(reasons[3:], ["IDENTITY_UNCERTAIN"] * 5 + ["GALLERY_REID"] * 3 + ["AFTER_UNCERTAIN"])
        (g,) = info["gallery_reid_log"]
        self.assertEqual((g["track_id"], g["episode"], g["resolved_frame"], g["against"], g["answered"]),
                         ("t1", 2, 9, 0, 3))

    def test_a_frame_without_a_clearing_crop_is_not_answered(self):
        doc = self.doc()
        f = self.feats()
        del f.crops[(10, "t1")]                         # frame 10: no clean crop -> no answer there, run held
        frames, _ = follow(doc, SEL, GALLERY_ON, [is_me("c1", "t1", 2, 3)], crops=f)
        self.assertEqual([r for _, r, _ in answers(frames)][8:11], ["GALLERY_REID", "IDENTITY_UNCERTAIN", "GALLERY_REID"])
        validate_target(target_document(doc, SEL, GALLERY_ON, ["t"], [is_me("c1", "t1", 2, 3)], crops=self.feats()), doc)

    def test_no_gallery_or_unlike_crops_never_clear(self):
        doc = self.doc()
        frames, _ = follow(doc, SEL, GALLERY_ON, [], crops=self.feats())                  # no confirmation, no gallery
        self.assertNotIn("GALLERY_REID", [r for _, r, _ in answers(frames)])
        frames, info = follow(doc, SEL, GALLERY_ON, [is_me("c1", "t1", 2, 3)], crops=self.feats(ep2=E2))
        self.assertNotIn("GALLERY_REID", [r for _, r, _ in answers(frames)])
        self.assertGreater(info["gallery_reid_log"][0]["against"], 0)

    def test_rival_without_reference_holds(self):
        st, det = gallery_frame_verdict(Crop(None, E1), E1, {"r": None}, 0.517)
        self.assertEqual((st, det["reason"]), ("HOLD", "NO_RIVAL_REFERENCE"))
        self.assertEqual(gallery_frame_verdict(None, E1, {"r": E2}, 0.517)[1]["reason"], "NO_CLEAN_CROP")
        self.assertEqual(gallery_frame_verdict(Crop(None, E1), E1, {"r": E2}, 0.517)[0], "CLEAR")


class ValidatorTest(unittest.TestCase):
    def test_tampering_and_old_versions(self):
        doc = crossing()
        d = target_document(doc, SEL, CFG, ["t"], [is_me("c1", "t1", 2)])
        validate_target(older(target_document(doc, SEL, CFG, ["t"]), "2.9.0"), doc)
        no_conf = {**d, "user_actions": [], "confirmation_log": []}
        with self.assertRaises(ArtifactError):
            validate_target(no_conf, doc)                     # PLAYER_CONFIRMED with nothing behind it
        forged = {**d, "confirmation_log": [{**d["confirmation_log"][0], "status": "CONFLICT"}]}
        with self.assertRaises(ArtifactError):
            validate_target(forged, doc)                      # CONFLICT without conflicts listed

    def test_claims_from_targets_takes_only_machine_certain_answers(self):
        doc = crossing()
        d = target_document(doc, SEL, CFG, ["t"], [is_me("c1", "t1", 2, 4)])
        self.assertEqual(claims_from_targets([("p", d)]), {"p": {1: "t1", 4: "t1", 5: "t1"}})


if __name__ == "__main__":
    unittest.main()
