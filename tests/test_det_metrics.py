"""Detection metrics on hand-built frames (expected values worked out by hand)."""
from __future__ import annotations

import unittest

from spana.dataset import det_metrics as m


def g(cls, x, s=10.0):
    return {"class": cls, "box": [x, 0.0, x + s, s]}


def p(cls, x, score, s=10.0):
    return {"class": cls, "box": [x, 0.0, x + s, s], "score": score}


class MetricTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(m.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(m.iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)
        self.assertAlmostEqual(m.iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_average_precision(self):
        frames = [{"gt": [g("a", 0), g("a", 100)], "pred": [p("a", 0, 0.9), p("a", 300, 0.8), p("a", 100, 0.7)]}]
        # ranked: hit, miss, hit -> precision envelope 1.0 up to recall 0.5, then 2/3 up to recall 1.0
        want = (51 * 1.0 + 50 * (2 / 3)) / 101
        self.assertAlmostEqual(m.average_precision(frames, "a", 0.5), want)
        self.assertIsNone(m.average_precision(frames, "b", 0.5))                 # no labels: no AP
        self.assertEqual(m.average_precision([{"gt": [g("a", 0)], "pred": []}], "a", 0.5), 0.0)
        # one label is matched once: the duplicate is a false positive
        dup = [{"gt": [g("a", 0)], "pred": [p("a", 0, 0.9), p("a", 1, 0.8)]}]
        self.assertEqual(m.precision_recall(dup, "a", 0.5)["fp"], 1)
        # a looser box passes IoU 0.5 and fails 0.75
        loose = [{"gt": [g("a", 0)], "pred": [p("a", 3, 0.9)]}]                  # IoU 7/13
        self.assertEqual(m.average_precision(loose, "a", 0.5), 1.0)
        self.assertEqual(m.average_precision(loose, "a", 0.75), 0.0)

    def test_precision_recall_at_confidence(self):
        frames = [{"gt": [g("a", 0), g("a", 100)], "pred": [p("a", 0, 0.9), p("a", 300, 0.8), p("a", 100, 0.3)]}]
        out = m.precision_recall(m.at_confidence(frames, 0.5), "a", 0.5)
        self.assertEqual((out["tp"], out["fp"], out["fn"], out["precision"], out["recall"]), (1, 1, 1, 0.5, 0.5))
        self.assertIsNone(m.precision_recall([{"gt": [], "pred": []}], "a", 0.5)["precision"])

    def test_confusion_ignores_class_when_matching(self):
        frames = [{"gt": [g("player", 0), g("referee", 100), g("ball", 200)],
                   "pred": [p("player", 0, 0.9), p("player", 100, 0.9), p("goalkeeper", 400, 0.9)]}]
        c = m.confusion(frames, ("player", "goalkeeper", "referee", "ball"), 0.5)
        self.assertEqual(c["player"]["player"], 1)
        self.assertEqual(c["referee"]["player"], 1)                              # referee boxed as player
        self.assertEqual(c["ball"][m.MISSED], 1)
        self.assertEqual(c[m.BACKGROUND]["goalkeeper"], 1)
        self.assertEqual(sum(sum(r.values()) for r in c.values()), 4)

    def test_recall_by_size_and_frame_errors(self):
        frames = [{"gt": [g("ball", 0, 6.0), g("ball", 100, 20.0)], "pred": [p("ball", 100, 0.9, 20.0)]}]
        out = m.recall_by_size(frames, "ball", 0.5, lambda s: "small" if s < 10 else "large")
        self.assertEqual(out, {"small": {"labels": 1, "found": 0, "recall": 0.0},
                               "large": {"labels": 1, "found": 1, "recall": 1.0}})
        self.assertEqual(m.frame_errors(frames[0], "ball", 0.5), 1)
        self.assertEqual(m.frame_errors({"gt": [], "pred": [p("ball", 0, 0.9)]}, "ball", 0.5), 1)

    def test_summary(self):
        frames = [{"gt": [g("a", 0)], "pred": [p("a", 0, 0.9)]}, {"gt": [g("b", 0)], "pred": []}]
        s = m.summarize(frames, ("a", "b", "c"), 0.5, 0.5)
        self.assertEqual((s["per_class"]["a"]["ap50"], s["per_class"]["a"]["ap50_95"]), (1.0, 1.0))
        self.assertEqual(s["per_class"]["b"]["ap50"], 0.0)
        self.assertIsNone(s["per_class"]["c"]["ap50"])
        self.assertEqual(s["map50"], 0.5)                                        # mean over classes with labels


if __name__ == "__main__":
    unittest.main()
