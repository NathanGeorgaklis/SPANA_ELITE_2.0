"""Resume after a no-rival doubt (target.resume_after_doubt_resolved, spana.target 2.3.0).

An uncertainty episode that was only ever a doubt with no rival (tracks 2.2.0 identity_doubt) does not taint the
followed track: ABSTAIN while it lasts, TARGET again once the tracker resolves it. Any rival frame in the episode
taints as before (target.resume_after_uncertain false). DEV motivation: records/doubt_resolution_2026-11-18 (each
doubt still cost one review tap because a resolved doubt left the track AFTER_UNCERTAIN)."""
from __future__ import annotations

import dataclasses
import unittest

from spana.artifacts import validate_target
from spana.identity.follower import Selection, target_document
from tests.test_tap_vouches_doubt import AA, AU, C, CFG, D, R, TO, TV, answers, doc22
from tests.test_target_follow import N

T = CFG.target
RESUME = dataclasses.replace(CFG, target=dataclasses.replace(T, resume_after_doubt_resolved=True,
                                                             tap_vouches_doubt=False))
BOTH = dataclasses.replace(CFG, target=dataclasses.replace(T, resume_after_doubt_resolved=True,
                                                           tap_vouches_doubt=True))
OFF = dataclasses.replace(CFG, target=dataclasses.replace(T, resume_after_doubt_resolved=False,
                                                          tap_vouches_doubt=False))
SEL = Selection.track_at_frame("t1", 1)


class Resume(unittest.TestCase):
    def test_resolved_doubt_answers_again(self):
        doc = doc22([{"t1": N}, {"t1": C}, {"t1": D}, {"t1": D}, {"t1": C}, {"t1": C}])
        self.assertEqual(answers(doc, SEL, OFF), [TO, AU, AU, AA, AA])          # today: tainted for good
        self.assertEqual(answers(doc, SEL, RESUME), [TO, AU, AU, TO, TO])

    def test_a_rival_in_the_episode_still_taints(self):
        doc = doc22([{"t1": N, "t2": N}, {"t1": C, "t2": C}, {"t1": D, "t2": C}, {"t1": R, "t2": R},
                     {"t1": C, "t2": C}])
        self.assertEqual(answers(doc, SEL, RESUME), [TO, AU, AU, AA])

    def test_repeated_doubts_each_resume(self):
        doc = doc22([{"t1": N}, {"t1": C}, {"t1": D}, {"t1": C}, {"t1": D}, {"t1": C}])
        self.assertEqual(answers(doc, SEL, RESUME), [TO, AU, TO, AU, TO])

    def test_selection_during_a_doubt_resumes_after_it(self):
        doc = doc22([{"t1": N}, {"t1": D}, {"t1": D}, {"t1": C}])
        self.assertEqual(answers(doc, SEL, RESUME), [AU, AU, TO])

    def test_with_the_tap_vouch(self):
        doc = doc22([{"t1": N}, {"t1": C}, {"t1": D}, {"t1": D}, {"t1": C}])
        self.assertEqual(answers(doc, SEL, BOTH), [TO, AU, AU, TO])
        self.assertEqual(answers(doc, Selection.track_at_frame("t1", 2), BOTH), [TV, TV, TO])

    def test_document(self):
        doc = doc22([{"t1": N}, {"t1": C}, {"t1": D}, {"t1": C}])
        d = target_document(doc, SEL, RESUME, ["test"])
        self.assertEqual(d["schema_version"], "2.5.0")
        self.assertTrue(d["target_config"]["resume_after_doubt_resolved"])
        validate_target(d, doc)


if __name__ == "__main__":
    unittest.main()
