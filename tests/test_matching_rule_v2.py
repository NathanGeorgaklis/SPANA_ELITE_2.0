"""Frozen matching rule NESTED_V2: overlapping labelled players are told apart by foot point; when they cannot be,
the attribution is AMBIGUOUS (coverage, but no identity change). Regression for the DEV finding (FRESH_07 /
FRESH_02): IoU-only attribution flipped a correctly tracked player to the occluded player behind him."""
from __future__ import annotations

import unittest

from spana.evaluation import NESTED_V1, NESTED_V2, EvalFrame, TrackBox, evaluate

FRONT = [1838.0, 438.0, 1864.0, 555.0]       # visible player (feet lower in the image)
BEHIND = [1832.0, 411.0, 1862.0, 526.0]      # occluded player behind him (labels overlap: IoU ~0.5)
TRACK_ON_FRONT = [1826.0, 422.0, 1874.0, 548.0]   # tracker box on the front player (IoU larger with BEHIND)
APART_FRONT, APART_BEHIND = [1900.0, 438.0, 1926.0, 555.0], [1700.0, 411.0, 1730.0, 526.0]


def frame(i, gt, track_box, flagged=False, occluded=()):
    return EvalFrame(i, gt, [], [TrackBox("T1", track_box, True, flagged, None)], [], frozenset(occluded))


class NestedV2(unittest.TestCase):
    def seq(self):
        return [frame(0, {"F": APART_FRONT, "B": APART_BEHIND}, [1898.0, 437.0, 1928.0, 556.0]),
                frame(1, {"F": FRONT, "B": BEHIND}, TRACK_ON_FRONT)]

    def test_foot_point_keeps_the_front_player(self):
        self.assertEqual(evaluate(self.seq(), rule=NESTED_V1).identity_transfers, 1)   # IoU-only flips
        r = evaluate(self.seq(), rule=NESTED_V2)
        self.assertEqual((r.identity_transfers, r.id_switches, r.ambiguous_attributions, r.covered), (0, 0, 0, 2))

    def test_occluded_label_makes_attribution_ambiguous_not_a_swap(self):
        """DEV finding (FRESH_02): an occluded player's label is drawn to estimated feet hidden behind the player in
        front, so its foot point is not evidence; deciding by it attributed a correct track to the wrong player."""
        fs = [self.seq()[0], frame(1, {"F": FRONT, "B": BEHIND}, TRACK_ON_FRONT, occluded=("B",))]
        r = evaluate(fs, rule=NESTED_V2)
        self.assertEqual((r.identity_transfers, r.ambiguous_attributions, r.covered), (0, 1, 2))

    def test_unseparable_feet_are_ambiguous_not_a_swap(self):
        a, b = [100.0, 100.0, 140.0, 220.0], [110.0, 100.0, 150.0, 220.0]   # side by side, same depth
        fs = [frame(0, {"A": [100.0, 100.0, 140.0, 220.0], "B": [300.0, 100.0, 340.0, 220.0]},
                    [100.0, 100.0, 140.0, 220.0]),
              frame(1, {"A": a, "B": b}, [105.0, 100.0, 145.0, 220.0])]
        r = evaluate(fs, rule=NESTED_V2)
        self.assertEqual((r.identity_transfers, r.ambiguous_attributions, r.covered), (0, 1, 2))
        self.assertEqual(r.ambiguous_events[0]["last_clear"], "A")

    def test_clear_swap_after_ambiguity_is_still_counted(self):
        a, b = [100.0, 100.0, 140.0, 220.0], [110.0, 100.0, 150.0, 220.0]
        fs = [frame(0, {"A": [100.0, 100.0, 140.0, 220.0], "B": [300.0, 100.0, 340.0, 220.0]},
                    [100.0, 100.0, 140.0, 220.0]),
              frame(1, {"A": a, "B": b}, [105.0, 100.0, 145.0, 220.0]),
              frame(2, {"A": [0.0, 100.0, 40.0, 220.0], "B": [300.0, 100.0, 340.0, 220.0]},
                    [300.0, 100.0, 340.0, 220.0])]
        self.assertEqual(evaluate(fs, rule=NESTED_V2).identity_transfers, 1)

    def test_identical_to_v1_when_players_do_not_overlap(self):
        fs = [frame(0, {"A": [100.0, 100.0, 140.0, 220.0], "B": [300.0, 100.0, 340.0, 220.0]},
                    [100.0, 100.0, 140.0, 220.0]),
              frame(1, {"A": [0.0, 100.0, 40.0, 220.0], "B": [300.0, 100.0, 340.0, 220.0]},
                    [300.0, 100.0, 340.0, 220.0])]
        self.assertEqual(evaluate(fs, rule=NESTED_V1).summary(), evaluate(fs, rule=NESTED_V2).summary())


if __name__ == "__main__":
    unittest.main()
