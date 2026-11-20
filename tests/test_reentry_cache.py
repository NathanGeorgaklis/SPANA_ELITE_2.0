"""ReentryContext memo (records/reentry_cache_2026-11-20): exit_side / position_fit / appearance are memoised per
context because the follower re-runs the re-entry rule on every frame of a loss. The memo must be exact: every decision
with one shared context equals the decision with a fresh context per call."""
from __future__ import annotations

import unittest

from spana.identity.reid import ReentryContext, reentry_choice
from tests.test_reentry_offer import CFG, ME, ME_AGAIN, SOMEONE, scene, tc, unit


class Memo(unittest.TestCase):
    def test_shared_context_decides_exactly_like_fresh_ones(self):
        docs = [scene({"t2": (5, 102.0), "t3": (9, 0.0), "t4": (12, 900.0)}, left_frame=lf, pan_px=p)
                for lf in (False, True) for p in (0.0, 60.0)]
        sigs = {"t1": ME, "t2": unit(0.8, 0.6, 0), "t3": ME_AGAIN, "t4": SOMEONE}
        sig = {k: {"first": v, "last": v} for k, v in sigs.items()}
        variants = [tc(), tc(reentry_mutual=True), tc(reentry_exit_edge=True, reentry_exit_edge_max_frac=0.05),
                    tc(reentry_position_weight=0.0), tc(reentry_position_weight=1.0, reentry_position_max_gap_s=0.2)]
        for doc in docs:
            shared = ReentryContext.of(doc)
            for now in range(4, len(doc["frames"])):
                for v in variants:
                    a = reentry_choice("t1", 3, now, shared, sig, v, CFG.tracking)
                    b = reentry_choice("t1", 3, now, ReentryContext.of(doc), sig, v, CFG.tracking)
                    self.assertEqual(a, b)
            self.assertTrue(shared.memo)                          # the shared context did memoise

    def test_a_changed_threshold_misses_the_memo(self):
        """Every config value the memoised calculations read is in their key: changing one on the same context
        recomputes (equals a fresh context), it never serves the old value."""
        doc = scene({"t2": (5, 102.0), "t3": (6, 300.0)})
        sig = {k: {"first": v, "last": v} for k, v in {"t1": ME, "t2": unit(0.8, 0.6, 0), "t3": ME_AGAIN}.items()}
        shared = ReentryContext.of(doc)
        for kw in ({}, {"reentry_position_base_h": 2.0}, {"reentry_position_speed_h_per_s": 0.5},
                   {"reentry_position_time_scale_s": 5.0}, {"reentry_position_max_gap_s": 0.05},
                   {"reentry_position_max_pan_frac": 0.0}, {"reentry_position_weight": 2.0},
                   {"reentry_signature": "mean"}):
            a = reentry_choice("t1", 3, 10, shared, sig, tc(**kw), CFG.tracking)
            b = reentry_choice("t1", 3, 10, ReentryContext.of(doc), sig, tc(**kw), CFG.tracking)
            self.assertEqual(a, b, kw)
        base = reentry_choice("t1", 3, 10, shared, sig, tc(), CFG.tracking)["candidates"]["t2"]["position"]
        wider = reentry_choice("t1", 3, 10, shared, sig, tc(reentry_position_base_h=2.0), CFG.tracking)
        self.assertNotEqual(base, wider["candidates"]["t2"]["position"])          # a real miss, not a no-op

    def test_memo_does_not_change_context_equality(self):
        doc = scene({"t2": (5, 102.0)})
        a, b = ReentryContext.of(doc), ReentryContext.of(doc)
        reentry_choice("t1", 3, 10, a, {"t1": {"first": ME, "last": ME}, "t2": {"first": ME, "last": ME}}, tc(),
                       CFG.tracking)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
