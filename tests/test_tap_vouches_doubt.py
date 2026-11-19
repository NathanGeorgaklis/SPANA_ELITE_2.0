"""User-vouched doubt (target.tap_vouches_doubt, spana.target 2.2.0).

A track uncertain ONLY by an appearance doubt (tracks 2.2.0 identity_doubt, no rival) answers TARGET (USER_VOUCHED)
after the user taps it, for the rest of that uncertainty episode. A rival, a new episode, or a tap on a CERTAIN or
rival-uncertain track never vouches. Real DEV finding (records/continuity_attribution_2026-11-18): with the doubt link
on, the follower abstained on doubted tracks even after the reviewer's tap, so correct time fell 0.778 -> 0.698."""
from __future__ import annotations

import copy
import dataclasses
import unittest
from typing import Dict, List, Tuple

from spana.artifacts import ArtifactError, validate_target, validate_tracks
from spana.identity.follower import Selection, follow, reselect, target_document
from tests.test_target_follow import CFG, N, tracks_doc

ON = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, tap_vouches_doubt=True))
OFF = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, tap_vouches_doubt=False))
C, D, R = ("A", True, "C"), ("A", True, "D"), ("A", True, "R")    # certain / doubt only / rival ambiguity
LD = ("L", True, "D")                                             # lost while doubted


def doc22(frames: List[Dict[str, tuple]]) -> dict:
    """A valid spana.tracks 2.2.0 document; kind C / D / R per track and frame (N = fresh)."""
    base = tracks_doc([{t: (s if s == N else (s[0], s[1], s[2] != "C")) for t, s in fr.items()} for fr in frames])
    for fr, spec in zip(base["frames"], frames):
        for t in fr["tracks"]:
            kind = "C" if spec[t["track_id"]] == N else spec[t["track_id"]][2]
            t["identity_doubt"] = kind == "D"
            if kind == "D":
                t["identity_rivals"] = []
    base["schema_version"] = "2.2.0"
    validate_tracks(base)
    return base


def answers(doc, sel, cfg, actions=()) -> List[Tuple[str, str]]:
    return [(f["state"], f["reason"]) for f in follow(doc, sel, cfg, actions)[0]]


TV, TO = ("TARGET", "USER_VOUCHED"), ("TARGET", "TRACK_OBSERVED")
AU, AA = ("ABSTAIN", "IDENTITY_UNCERTAIN"), ("ABSTAIN", "AFTER_UNCERTAIN")


class Vouch(unittest.TestCase):
    def test_tap_on_a_doubted_track_answers_until_the_doubt_clears(self):
        doc = doc22([{"t1": N}, {"t1": C}, {"t1": D}, {"t1": D}, {"t1": D}, {"t1": C}])
        sel = Selection.track_at_frame("t1", 1)
        self.assertEqual(answers(doc, sel, ON)[1:], [AU, AU, AU, AA])              # no tap: as before
        tap = [reselect("t1", 3)]
        self.assertEqual(answers(doc, sel, OFF, tap)[1:], [AU, AU, AU, AA])        # policy off: as before
        self.assertEqual(answers(doc, sel, ON, tap)[1:], [AU, TV, TV, TO])         # vouched from the tap on

    def test_selection_on_a_doubted_track_vouches(self):
        doc = doc22([{"t1": N}, {"t1": D}, {"t1": D}])
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 1), ON), [TV, TV])

    def test_a_rival_ends_the_vouch_fail_closed(self):
        doc = doc22([{"t1": N, "t2": N}, {"t1": D, "t2": C}, {"t1": D, "t2": C}, {"t1": R, "t2": R},
                     {"t1": C, "t2": C}])
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 1), ON), [TV, TV, AU, AA])

    def test_a_new_episode_is_not_vouched(self):
        doc = doc22([{"t1": N}, {"t1": D}, {"t1": C}, {"t1": D}, {"t1": C}])
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 1), ON), [TV, TO, AU, AA])

    def test_tap_on_a_rival_ambiguity_or_a_certain_track_vouches_nothing(self):
        doc = doc22([{"t1": N, "t2": N}, {"t1": R, "t2": R}, {"t1": R, "t2": R}])
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 1), ON), [AU, AU])
        doc = doc22([{"t1": N}, {"t1": C}, {"t1": D}])
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 1), ON), [TO, AU])

    def test_vouch_survives_a_short_loss_of_the_same_track(self):
        doc = doc22([{"t1": N}, {"t1": D}, {"t1": LD}, {"t1": D}])
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 1), ON),
                         [TV, ("LOST", "TRACK_LOST"), TV])

    def test_retap_elsewhere_drops_the_vouch(self):
        doc = doc22([{"t1": N, "t2": N}, {"t1": D, "t2": C}, {"t1": D, "t2": C}, {"t1": D, "t2": C}])
        acts = [reselect("t2", 2), reselect("t1", 3)]
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 1), ON, acts), [TV, TO, TV])


class Artifact(unittest.TestCase):
    def setUp(self):
        self.tracks = doc22([{"t1": N}, {"t1": C}, {"t1": D}, {"t1": D}, {"t1": C}])
        self.sel = Selection.track_at_frame("t1", 1)
        self.doc = target_document(self.tracks, self.sel, ON, ["test"], [reselect("t1", 2)])

    def test_valid(self):
        validate_target(self.doc, self.tracks)
        self.assertEqual(self.doc["schema_version"], "2.6.0")
        self.assertTrue(self.doc["target_config"]["tap_vouches_doubt"])
        self.assertIn("USER_VOUCHED", [f["reason"] for f in self.doc["frames"]])

    def _refused(self, doc, tracks=None):
        with self.assertRaises(ArtifactError):
            validate_target(doc, tracks or self.tracks)

    def test_vouched_answer_needs_the_policy(self):
        d = copy.deepcopy(self.doc)
        d["target_config"]["tap_vouches_doubt"] = False
        self._refused(d)

    def test_vouched_answer_needs_a_tap_on_that_track_during_the_doubt(self):
        d = copy.deepcopy(self.doc)
        d["user_actions"] = []                       # no tap: frame 2 would be a re-selection without an action
        self._refused(d)
        t = copy.deepcopy(self.tracks)               # the track gains a rival after the tap
        tr = t["frames"][3]["tracks"][0]
        tr["identity_doubt"], tr["identity_rivals"] = False, ["ghost"]
        self._refused(self.doc, t)

    def test_track_observed_on_a_doubted_track_is_still_refused(self):
        d = copy.deepcopy(self.doc)
        f = next(f for f in d["frames"] if f["reason"] == "USER_VOUCHED")
        f["reason"] = "TRACK_OBSERVED"
        self._refused(d)

    def test_2_1_0_target_documents_cannot_carry_user_vouched(self):
        from tests.target_versions import older
        d = older(self.doc, "2.1.0")
        self._refused(d)
        f = next(f for f in d["frames"] if f["reason"] == "USER_VOUCHED")
        f.update(state="ABSTAIN", reason="IDENTITY_UNCERTAIN", box=None)
        for g in d["frames"]:
            if g["reason"] == "USER_VOUCHED":
                g.update(state="ABSTAIN", reason="IDENTITY_UNCERTAIN", box=None)
        from spana.artifacts import gaps_of
        d["gaps"] = gaps_of(d["frames"], d["relinks"])
        validate_target(d, self.tracks)                  # without the vouched answers it is a valid 2.1.0 document


if __name__ == "__main__":
    unittest.main()
