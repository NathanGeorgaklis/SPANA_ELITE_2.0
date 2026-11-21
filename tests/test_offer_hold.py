"""Offer hold (target.offer_hold_s, spana.target 2.8.0; records/offer_hold_2026-11-19).

Nathan's FRESH_08 review, player 10: a right offer vanished for good when its track was hidden for a single frame.
An offer whose track is briefly hidden (live, confirmed, CERTAIN, OCCLUDED / LOST) is now kept for up to
offer_hold_s after its last usable frame; no CANDIDATE is shown while held, the same offer is shown again when the
track is back. Uncertain or hidden too long: dropped as before. Synthetic tracks at 10 fps."""
from __future__ import annotations

import dataclasses
import unittest

from spana.artifacts import validate_target
from spana.identity.follower import follow, target_document
from tests.test_reentry_offer import ME, ME_AGAIN, cfg, emb_doc
from tests.test_target_follow import CFG, A, L, N, tracks_doc
from tests.test_target_relink import SEL, W


def hold(s, base=CFG):
    return dataclasses.replace(base, target=dataclasses.replace(base.target, offer_hold_s=s))


def hidden(frames_hidden: int, kind=L):
    """t1 lost after frame 1; t3 (naming t1) born on frame 2, offered on frame 3, hidden `frames_hidden` frames from
    frame 4, then observed again for 2 frames."""
    rows = [W, {"t1": A, "t2": A}, {"t1": L, "t2": A, "t3": N}, {"t1": L, "t2": A, "t3": A}]
    rows += [{"t1": L, "t2": A, "t3": kind}] * frames_hidden
    rows += [{"t1": L, "t2": A, "t3": A}] * 2
    return tracks_doc(rows, preds={"t3": ["t1"]})


def states(frames):
    return [f["state"] for f in frames]


class Predecessor(unittest.TestCase):
    def test_a_one_frame_hide_keeps_the_offer(self):
        doc = hidden(1)
        frames, info = follow(doc, SEL, hold(0.5))
        self.assertEqual(states(frames), ["TARGET", "LOST", "CANDIDATE", "LOST", "CANDIDATE", "CANDIDATE"])
        (r,) = info["relinks"]
        self.assertEqual((r["outcome"], r["dropped_reason"]), ("PENDING", None))
        d = target_document(doc, SEL, hold(0.5), ["test"])
        self.assertEqual(d["schema_version"], "2.10.0")
        validate_target(d, doc)
        self.assertEqual([g["offers"] for g in d["gaps"]], [[["t1", "t3", 3]]])     # one offer, shown twice
        no_hold = follow(doc, SEL, hold(0.0))
        self.assertEqual(states(no_hold[0]), ["TARGET", "LOST", "CANDIDATE", "LOST", "LOST", "LOST"])
        self.assertEqual(no_hold[1]["relinks"][0]["dropped_reason"], "CANDIDATE_NOT_USABLE")

    def test_hidden_longer_than_the_hold_is_dropped(self):
        doc = hidden(6)                                                  # 0.6 s hidden > 0.5 s hold
        frames, info = follow(doc, SEL, hold(0.5))
        self.assertEqual(info["relinks"][0]["dropped_reason"], "CANDIDATE_NOT_USABLE")
        self.assertEqual(states(frames)[-2:], ["LOST", "LOST"])          # not shown again once dropped
        validate_target(target_document(doc, SEL, hold(0.5), ["test"]), doc)

    def test_uncertain_while_hidden_is_dropped(self):
        doc = hidden(1, kind=("L", True, True))
        frames, info = follow(doc, SEL, hold(0.5))
        self.assertEqual(info["relinks"][0]["dropped_reason"], "CANDIDATE_NOT_USABLE")
        self.assertEqual(states(frames)[-2:], ["LOST", "LOST"])


class Reid(unittest.TestCase):
    def test_a_one_frame_hide_keeps_a_reid_offer(self):
        rows = [{"t1": N}, {"t1": A}, {"t1": A}, {"t1": A}, {}, {}, {}, {}, {"t2": N}, {"t2": A}, {"t2": L},
                {"t2": A}, {"t2": A}]
        tr = tracks_doc(rows, ended={4: [("t1", "LEFT_FRAME")]})
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN)})
        sel = SEL.__class__.track_at_frame("t1", 1)
        on = cfg(offer_hold_s=0.5)
        frames, info = follow(tr, sel, on, (), emb)
        self.assertEqual([f["state"] for f in frames][-4:], ["CANDIDATE", "LOST", "CANDIDATE", "CANDIDATE"])
        self.assertEqual(len(info["relinks"]), 1)
        self.assertEqual(info["relinks"][0]["outcome"], "PENDING")
        validate_target(target_document(tr, sel, on, ["test"], (), emb), tr, emb)
        off = follow(tr, sel, cfg(offer_hold_s=0.0), (), emb)
        self.assertEqual([f["state"] for f in off[0]][-3:], ["LOST", "LOST", "LOST"])


if __name__ == "__main__":
    unittest.main()
