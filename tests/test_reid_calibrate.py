"""DEV margin calibration (tools/dev/reid_calibrate.py): the margin rule meets the bound and stops at the first
failure, a held-out clip never influences its own fit, the combined margin formula, determinism."""
import copy
import json
import unittest

import numpy as np

from tools.dev import cue_fusion as F
from tools.dev import reid_calibrate as C
from tools.reid_cluster import upper_bound

CFG = C.load_cfg("config/reid_calibration.yaml")
FCFG = F.load_fusion_config("config/reid_fusion.yaml")


def table(name, seed, n_rival=1500, sep=1.0):
    """JSON-like DEV table: cue scores of verification pairs and rival tests; the query is nearer A by `sep`."""
    rng = np.random.default_rng(seed)
    lst = lambda x: [float(v) for v in x]                                # noqa: E731
    pairs = {k: {s: {"scores": {"hist": lst(0.5 + 0.1 * rng.normal(size=300) + (0.1 if s == "same" else 0)),
                                "spana": lst(0.3 + 0.2 * rng.normal(size=300) + (0.2 if s == "same" else 0))}}
                 for s in ("same", "diff")} for k in ("all", "hard")}
    qa = {"hist": lst(0.6 + 0.08 * rng.normal(size=n_rival)), "spana": lst(0.5 + 0.15 * rng.normal(size=n_rival))}
    qb = {"hist": lst(0.6 - 0.1 * sep + 0.08 * rng.normal(size=n_rival)),
          "spana": lst(0.5 - 0.2 * sep + 0.15 * rng.normal(size=n_rival))}
    return {"match": name, "pairs": pairs,
            "rival": {"qa": qa, "qb": qb, "height": lst(rng.uniform(50, 200, n_rival)), "pair": ["1|2"] * n_rival}}


class Margin(unittest.TestCase):
    def test_choose_margin_meets_the_bound_and_stops_at_the_first_failure(self):
        rng = np.random.default_rng(0)
        m = np.concatenate([rng.uniform(0.05, 1.0, 3000), -rng.uniform(0.0, 0.15, 30)])
        t = C.choose_margin(m, 0.005, 0.95)
        a = C.apply(m, t, 0.95)
        self.assertLessEqual(a["upper_bound"], 0.005)
        lower = np.unique(np.abs(m[np.abs(m) < t]))
        b = C.apply(m, float(lower.max()), 0.95)                       # the next lower margin fails
        self.assertGreater(b["upper_bound"], 0.005)

    def test_meets_equals_the_bound_rule(self):
        for k, n in ((0, 600), (0, 590), (1, 950), (2, 1300), (3, 4500)):
            self.assertEqual(C.meets(k, n, 0.005, 0.95), upper_bound(k, n, 0.95) <= 0.005, (k, n))

    def test_too_few_tests_never_resolve(self):
        self.assertIsNone(C.choose_margin(np.linspace(0.1, 1, 100), 0.005, 0.95))   # 0 / 100: bound 3 % > 0.5 %
        self.assertEqual(C.apply(np.ones(5), None, 0.95)["resolved"], 0)

    def test_tie_at_zero_is_never_resolved(self):
        m = np.concatenate([np.zeros(10), np.linspace(0.1, 1, 2000)])
        self.assertGreater(C.choose_margin(m, 0.005, 0.95), 0)


class Fit(unittest.TestCase):
    def setUp(self):
        self.doc = {"tables": [table(f"M{i}", i) for i in range(4)]}
        self.tables = [C.to_arrays(t) for t in self.doc["tables"]]

    def test_combined_margin_formula(self):
        f = C.fit(self.tables, "combined", CFG, FCFG)
        st, w = f["_stats"], f["weight"]
        t = self.tables[0]
        z = lambda c, x: F.normalise(x, st[c])                           # noqa: E731
        want = w * (z("hist", t["rival"]["qa"]["hist"]) - z("hist", t["rival"]["qb"]["hist"])) + \
            (1 - w) * (z("spana", t["rival"]["qa"]["spana"]) - z("spana", t["rival"]["qb"]["spana"]))
        got = C.margins([t], "combined", {"cues": ["hist", "spana"], "stats": st, "weight": w})
        self.assertTrue(np.allclose(got, want))

    def test_held_out_clip_never_fits_itself(self):
        r1 = C.held_out(self.tables, "combined", CFG, FCFG)
        t2 = copy.deepcopy(self.tables)
        t2[2] = C.to_arrays(table("M2", 99, sep=0.1))                   # replace the held-out clip entirely
        r2 = C.held_out(t2, "combined", CFG, FCFG)
        a, b = r1["per_clip"]["M2"], r2["per_clip"]["M2"]
        self.assertEqual((a["margin"], a["weight"], a["fit_on"]), (b["margin"], b["weight"], b["fit_on"]))
        self.assertNotIn("M2", a["fit_on"])
        self.assertNotEqual(r1["per_clip"]["M0"]["margin"], r2["per_clip"]["M0"]["margin"])   # M0's fit uses M2

    def test_calibrate_is_deterministic_and_reports_every_option(self):
        a = C.calibrate(self.doc, CFG, FCFG)
        b = C.calibrate(json.loads(json.dumps(self.doc)), CFG, FCFG)
        self.assertEqual(json.dumps(a, default=str, sort_keys=True), json.dumps(b, default=str, sort_keys=True))
        self.assertEqual(set(a["options"]), {"hist", "spana", "combined"})
        fin = a["options"]["combined"]["final"]
        self.assertLessEqual(fin["in_sample"]["upper_bound"], CFG["target_false_rate"])
        pooled = a["options"]["combined"]["held_out"]["pooled"]
        self.assertEqual(pooled["tests"], 4 * 1500)
        if pooled["resolved"]:
            self.assertAlmostEqual(pooled["upper_bound"], upper_bound(pooled["wrong"], pooled["resolved"], 0.95))


class Clean(unittest.TestCase):
    """Clean subset: only tests whose 3 crops are clean (tracker's rule), fail closed without the fields."""
    def with_clean(self, t, seed):
        rng = np.random.default_rng(seed)
        n = len(t["rival"]["pair"])
        t["rival"]["max_overlap"] = [float(v) for v in rng.choice([0.0, 0.02, 0.3, 0.7], n)]
        t["rival"]["any_occluded"] = [bool(v) for v in rng.random(n) < 0.2]
        return t

    def test_subset_keeps_only_clean_tests(self):
        t = C.to_arrays(self.with_clean(table("M0", 0, n_rival=200), 1))
        s = C.clean_subset(t, 0.05, True)
        r = t["rival"]
        want = np.flatnonzero((r["max_overlap"] <= 0.05) & ~r["any_occluded"])
        self.assertEqual(len(s["rival"]["pair"]), len(want))
        self.assertTrue(np.array_equal(s["rival"]["qa"]["spana"], r["qa"]["spana"][want]))
        self.assertTrue((s["rival"]["max_overlap"] <= 0.05).all() and not s["rival"]["any_occluded"].any())
        s2 = C.clean_subset(t, 0.05, False)                                # occluded allowed: more tests
        self.assertGreater(len(s2["rival"]["pair"]), len(want))

    def test_without_fields_clean_is_unavailable_not_all(self):
        self.assertIsNone(C.clean_subset(C.to_arrays(table("M0", 0, n_rival=50)), 0.05, True))
        doc = {"tables": [table(f"M{i}", i, n_rival=300) for i in range(3)]}
        res = C.calibrate(doc, CFG, FCFG)
        self.assertFalse(res["clean"]["available"])

    def test_threshold_comes_from_the_tracker_config(self):
        import yaml
        want = yaml.safe_load(open(CFG["clean_subset"]["max_iou_from"]))["appearance"]["clean_max_iou"]
        self.assertEqual(C.clean_max_iou(CFG), want)
        doc = {"tables": [self.with_clean(table(f"M{i}", i, n_rival=600), 10 + i) for i in range(3)]}
        res = C.calibrate(doc, CFG, FCFG)
        self.assertTrue(res["clean"]["available"])
        self.assertEqual(res["clean"]["max_iou"], want)
        n_clean = sum(v["clean"] for v in res["clean"]["tests_per_clip"].values())
        self.assertEqual(res["clean"]["options"]["spana"]["held_out"]["pooled"]["tests"], n_clean)
        self.assertEqual(res["options"]["spana"]["held_out"]["pooled"]["tests"], 3 * 600)   # all: unchanged


class EvalTableFields(unittest.TestCase):
    def test_fusion_table_records_cleanliness_of_all_three_crops(self):
        from tools.dev.appearance_cue_eval import fusion_table
        crops = [{"player": "1", "t": 0.0, "h": 100.0, "occluded": False, "max_iou": 0.0},
                 {"player": "2", "t": 0.0, "h": 100.0, "occluded": True, "max_iou": 0.4},
                 {"player": "1", "t": 2.0, "h": 100.0, "occluded": False, "max_iou": 0.01},
                 {"player": "2", "t": 2.0, "h": 100.0, "occluded": False, "max_iou": 0.0}]
        tri = [(2, 0, 1, ("1", "2")), (3, 1, 0, ("1", "2"))]
        t = fusion_table("X", crops, ["hist"], lambda c, a, b: 0.5, [], [], [], [], tri)
        self.assertEqual(t["rival"]["any_occluded"], [True, True])
        self.assertEqual(t["rival"]["max_overlap"], [0.4, 0.4])
        tri = [(2, 0, 3, ("1", "2"))]
        t = fusion_table("X", crops, ["hist"], lambda c, a, b: 0.5, [], [], [], [], tri)
        self.assertEqual((t["rival"]["any_occluded"], t["rival"]["max_overlap"]), ([False], [0.01]))


if __name__ == "__main__":
    unittest.main()
