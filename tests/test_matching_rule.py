"""Frozen matching rule NESTED_V1: tolerant to box style, strict about identity of the body."""
from __future__ import annotations

import unittest

import numpy as np

from spana.evaluation import IOU_050, NESTED_V1, EvalFrame, eligible_matrix, evaluate

LOOSE = [100.0, 100.0, 160.0, 240.0]          # label drawn with margins (h 140, w 60)
TIGHT = [112.0, 115.0, 148.0, 230.0]          # detector box on the same body (h 115, w 36)


def ok(g, d, rule=NESTED_V1) -> bool:
    return bool(eligible_matrix(np.array([g], float), np.array([d], float), rule)[0][0, 0])


class NestedV1(unittest.TestCase):
    def test_same_body_different_style_matches(self):
        self.assertFalse(ok(LOOSE, TIGHT, IOU_050))          # the official IoU rule misses it (IoU ~0.49)
        self.assertTrue(ok(LOOSE, TIGHT))
        self.assertTrue(ok(TIGHT, LOOSE))                     # symmetric: tight label, loose detector

    def test_superset_of_iou_rule(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            g = rng.uniform(0, 500, 2); gs = rng.uniform(20, 200, 2)
            d = g + rng.uniform(-40, 40, 2); ds = gs * rng.uniform(0.5, 1.5, 2)
            gb, db = [*g, *(g + gs)], [*d, *(d + ds)]
            if ok(gb, db, IOU_050):
                self.assertTrue(ok(gb, db))

    def test_rejects_neighbour_partial_body_and_lagging_label(self):
        neighbour = [165.0, 110.0, 205.0, 235.0]              # next player, touching the loose label
        upper_half = [112.0, 115.0, 148.0, 170.0]             # head/torso only
        legs = [114.0, 175.0, 146.0, 230.0]
        lagging = [150.0, 100.0, 210.0, 240.0]                # copied label box not moved onto the player
        for d in (neighbour, upper_half, legs):
            self.assertFalse(ok(LOOSE, d), d)
        self.assertFalse(ok(lagging, TIGHT))

    def test_two_player_union_does_not_match_either_player(self):
        a, b = [100.0, 100.0, 136.0, 200.0], [140.0, 110.0, 176.0, 210.0]
        union = [98.0, 98.0, 178.0, 212.0]
        self.assertFalse(ok(a, union)); self.assertFalse(ok(b, union))

    def test_one_to_one_assignment(self):
        # Two detections on one loose label: one match, the other is a duplicate detection.
        f = EvalFrame(0, {"1": LOOSE}, [TIGHT, [110.0, 112.0, 150.0, 235.0]], [], [])
        r = evaluate([f], rule=NESTED_V1)
        self.assertEqual((r.missed_players, r.duplicate_dets), (0, 1))


if __name__ == "__main__":
    unittest.main()
