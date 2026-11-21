"""Re-ID v4: teammate-only triplet term, leave-one-TRAIN-match-out evaluation / calibration, DEV pair replay."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import reid_heldout as H
from tools.reid_crops import load_train_config


def _rows():
    # match M: A1, A2 (side A), B1 (side B); crops at t = 0, 2, 4 s
    rows = []
    for p, side in (("A1", "A"), ("A2", "A"), ("B1", "B")):
        for t in (0.0, 2.0, 4.0):
            rows.append({"player": p, "identity": f"M/{p}", "team": f"M/{side}", "t": t, "overlap": 0.0})
    return rows


class TeammateTriplet(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_negatives_are_teammates_only(self):
        import torch
        from tools.reid_train import teammate_triplet
        # anchor 0 (id 0, team 0); teammate id 1 far away, opponent id 2 right on top of the anchor
        z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        y = torch.tensor([0, 0, 1, 1, 2, 2])
        team = torch.tensor([0, 0, 0, 0, 1, 1])
        # ids 0 / 1: positives at distance 0, nearest teammate at sqrt(2) > margin -> 0; the opponent (id 2) on the
        # anchor is ignored; id 2 has no teammate and is left out
        self.assertAlmostEqual(float(teammate_triplet(z, y, team, 0.3)), 0.0, places=6)
        team_all = torch.tensor([0, 0, 0, 0, 0, 0])   # now id 2 is a teammate on top of id 0: loss > 0
        self.assertGreater(float(teammate_triplet(z, y, team_all, 0.3)), 0.0)

    def test_no_teammate_in_batch_gives_zero_with_graph(self):
        import torch
        from tools.reid_train import teammate_triplet
        z = torch.randn(4, 3, requires_grad=True)
        out = teammate_triplet(z, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]), 0.3)
        self.assertEqual(float(out.detach()), 0.0)
        out.backward()                                   # keeps the graph


class Config(unittest.TestCase):
    def test_v4_keys(self):
        c = load_train_config("config/reid_train.yaml")
        self.assertIn("teammate_triplet_weight", c["train"])
        s = H._settings("config/reid_train.yaml")
        self.assertEqual(s["margin_floor"], "all_tests")
        self.assertEqual(s["clean_max_iou"], 0.05)


class Heldout(unittest.TestCase):
    def test_rival_tests_teammates_gap_and_clean(self):
        rows = _rows()
        rows[1]["overlap"] = 0.5                          # A1 at t=2 overlaps someone
        z = np.eye(3, dtype=np.float32)[[0, 0, 0, 1, 1, 1, 2, 2, 2]]
        rv = H.rival_tests(z, rows, gap=1.0, clean_iou=0.05)
        # queries at t=2 / 4 of A1 and A2 against their one teammate; never against B1
        self.assertEqual(sorted(set(rv["pair"])), ["A1|A2"])
        self.assertEqual(len(rv["margin"]), 4)
        self.assertTrue(all(m == 1.0 for m in rv["margin"]))
        # dirty: A1's query at 2; A1's query at 4 (own reference at 2); A2's query at 4 (rival reference A1 at 2)
        self.assertEqual(rv["clean"].count(False), 3)

    def test_retrieval_and_aucs(self):
        rows = _rows()
        z = np.eye(3, dtype=np.float32)[[0, 0, 0, 1, 1, 1, 2, 2, 2]]
        ident = np.array([r["identity"] for r in rows])
        t = np.array([r["t"] for r in rows])
        r = H.retrieval(z, ident, t, gap=1.0)
        self.assertEqual((r["queries"], r["rank1"], r["mAP"]), (9, 1.0, 1.0))
        a = H.pair_aucs(z, ident, np.array([r["team"] for r in rows]), t, gap=1.0, cap=50)
        self.assertEqual(a["auc_same_vs_teammate"], 1.0)

    def test_fit_floor_and_fail_closed(self):
        s = {"target_false_rate": 0.005, "confidence": 0.95, "margin_floor": "all_tests"}
        rng = np.random.default_rng(0)
        m = rng.uniform(0.0, 0.5, 3000)                   # clean tests, all right
        m_all = np.concatenate([m, np.full(30, -0.4)])    # 30 wrong answers, all on non-clean tests
        c_all = np.concatenate([np.ones(3000, bool), np.zeros(30, bool)])
        f = H._fit([m_all], [c_all], s)
        self.assertLess(f["clean_margin"], 0.1)           # clean tests alone would resolve almost everything
        self.assertGreater(f["all_tests_margin"], 0.4)    # the dirty errors stop the all-tests scan at 0.4
        self.assertEqual(f["margin"], f["all_tests_margin"])   # floored: max(clean, all tests)
        none = H._fit([np.array([0.5, -0.5])], [np.array([True, True])], s)
        self.assertIsNone(none["margin"])

    def test_calibrate_leaves_each_fold_out(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for k in range(3):
                m = rng.uniform(0.05, 1.0, 2000).tolist()
                f = {"match_id": f"M{k}", "model_sha256": "x", "rival_tests": {"margin": m, "clean": [True] * len(m),
                                                                              "pair": ["a|b"] * len(m)}}
                p = Path(d) / f"f{k}.json"
                p.write_text(json.dumps(f))
                paths.append(str(p))
            out = Path(d) / "cal.json"
            H.main(["calibrate", "--folds", *paths, "--out", str(out)])
            cal = json.loads(out.read_text())
        self.assertEqual(cal["final"]["label"], "borrowed")
        self.assertEqual(sorted(cal["held_out"]["per_fold"]), ["M0", "M1", "M2"])
        self.assertEqual(cal["held_out"]["pooled"]["wrong"], 0)
        self.assertTrue(cal["held_out"]["pooled"]["passes"])


class DevPairReplay(unittest.TestCase):
    def test_replayed_pairs_and_missing_clip_refused(self):
        from tools.dev.appearance_cue_eval import replayed_rival_pairs
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps({"clips": {"C": {"uncertain_rival_pairs": [["8", "13"], ["4", "5"]]}}}))
            self.assertEqual(replayed_rival_pairs(p, "C"), {("13", "8"), ("4", "5")})
            with self.assertRaises(SystemExit):
                replayed_rival_pairs(p, "D")


if __name__ == "__main__":
    unittest.main()
