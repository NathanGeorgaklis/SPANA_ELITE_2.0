"""DEV cue fusion (tools/dev/cue_fusion.py): normalisation, combination rule, leave-one-match-out isolation,
abstention on a missing cue, single-cue equivalence with the eval's own metrics, determinism, decision rule."""
import copy
import json
import unittest

import numpy as np

from tools.dev import cue_fusion as F

CFG = F.load_fusion_config("config/reid_fusion.yaml")


def table(name, seed, sep1=1.0, sep2=1.0, n=400, nr=200):
    """Synthetic match: cue 'hist' and 'spana' scores, same-player pairs higher by sep."""
    rng = np.random.default_rng(seed)
    t = F.make_table(name)
    for k in F.KINDS:
        for s in ("same", "diff"):
            base = (1.0 if s == "same" else 0.0)
            t["pairs"][k][s]["scores"] = {"hist": 0.5 + 0.1 * (base * sep1 + rng.normal(0, 1, n)),
                                          "spana": 0.3 + 0.2 * (base * sep2 + rng.normal(0, 1, n))}
            t["pairs"][k][s]["occluded"] = rng.random(n) < 0.2
            t["pairs"][k][s]["players"] = ["1|1" if s == "same" else ("1|2" if i % 2 else "3|4") for i in range(n)]
    t["rival"]["qa"] = {"hist": 0.6 + 0.1 * rng.normal(0, 1, nr), "spana": 0.5 + 0.2 * rng.normal(0, 1, nr)}
    t["rival"]["qb"] = {"hist": 0.5 + 0.1 * rng.normal(0, 1, nr), "spana": 0.3 + 0.2 * rng.normal(0, 1, nr)}
    t["rival"]["occluded"] = rng.random(nr) < 0.2
    t["rival"]["pair"] = ["1|2" if i % 2 else "3|4" for i in range(nr)]
    return t


def small_cfg(**kw):
    c = copy.deepcopy(CFG)
    c["report"]["bootstrap"] = 30
    c.update(kw)
    return c


class NormaliseCombine(unittest.TestCase):
    def test_zscore_and_rank(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        z = F.normalise(ref, F.norm_stats(ref, "zscore"))
        self.assertAlmostEqual(float(np.nanmean(z)), 0.0)
        self.assertAlmostEqual(float(np.nanstd(z)), 1.0)
        self.assertTrue(np.isnan(z[-1]))
        r = F.normalise(np.array([0.0, 2.0, 9.0, np.nan]), F.norm_stats(ref, "rank"))
        self.assertEqual(r[:3].tolist(), [0.0, 0.5, 1.0])
        self.assertTrue(np.isnan(r[3]))
        c = F.normalise(np.array([5.0]), F.norm_stats(np.array([5.0, 5.0]), "zscore"))   # zero spread: no div by 0
        self.assertEqual(c.tolist(), [0.0])

    def test_combination_rule_and_missing(self):
        a, b = np.array([1.0, np.nan, 2.0]), np.array([3.0, 4.0, np.nan])
        out = F.combine(a, b, 0.25, "abstain")
        self.assertAlmostEqual(out[0], 0.25 * 1 + 0.75 * 3)
        self.assertTrue(np.isnan(out[1]) and np.isnan(out[2]))                 # abstain, never filled in
        out = F.combine(a, b, 0.25, "single_cue")
        self.assertEqual(out[1:].tolist(), [4.0, 2.0])

    def test_tar_matches_eval_formula_and_missing_never_accepted(self):
        rng = np.random.default_rng(0)
        S, D = rng.normal(1, 1, 500), rng.normal(0, 1, 700)
        thr = float(np.percentile(D, 99))
        self.assertEqual(F.tar_at_far(S, D, 0.01), float(np.mean(S > thr)))   # the eval's tpr_at_1pct_far_hard
        S2 = S.copy()
        S2[:100] = np.nan
        self.assertLessEqual(F.tar_at_far(S2, D, 0.01), F.tar_at_far(S, D, 0.01))
        D2 = D.copy()
        D2[:] = np.nan                                                          # every negative abstains: no false accepts
        self.assertEqual(F.tar_at_far(np.full(3, np.nan), D2, 0.01), 0.0)
        r = F.rival_stats(np.array([1.0, np.nan, 0.0]), np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(r["acc"], 1 / 3)
        self.assertAlmostEqual(r["acc_answered"], 0.5)
        self.assertAlmostEqual(r["abstain"], 1 / 3)


class LeaveOneMatchOut(unittest.TestCase):
    def test_held_out_match_never_used(self):
        tables = [table(f"M{i}", i, sep1=1 + i, sep2=2) for i in range(4)]
        opts, info, name = F.options(tables, small_cfg())
        for i, inf in enumerate(info[name]):
            self.assertNotIn(f"M{i}", inf["stats_from"])
            self.assertNotIn(f"M{i}", inf["fit"]["fit_on"])
        # replace match 2's scores entirely: its own weight and normalised scores-per-raw-score must not change
        t2 = copy.deepcopy(tables)
        t2[2] = table("M2", 99, sep1=0.0, sep2=5.0)
        _, info2, _ = F.options(t2, small_cfg())
        self.assertEqual(info[name][2]["weight_first_cue"], info2[name][2]["weight_first_cue"])
        self.assertEqual(info[name][2]["fit"]["objective"], info2[name][2]["fit"]["objective"])
        # while a match that does use match 2 for fitting / normalisation sees the change
        self.assertNotEqual(json.dumps(info[name][0], default=str), json.dumps(info2[name][0], default=str))

    def test_fit_prefers_the_informative_cue(self):
        tables = [table(f"M{i}", i, sep1=0.0, sep2=3.0) for i in range(4)]    # colour carries no signal
        _, info, name = F.options(tables, small_cfg())
        self.assertTrue(all(inf["weight_first_cue"] <= 0.2 for inf in info[name]))


class ReportTests(unittest.TestCase):
    def test_abstention_reasons(self):
        tables = [table(f"M{i}", i) for i in range(4)]
        tables[1]["pairs"]["all"]["same"]["scores"]["hist"][:5] = np.nan
        tables[1]["rival"]["qa"]["hist"][:3] = np.nan
        rep = F.report(tables, small_cfg())
        self.assertEqual(rep["abstentions"]["M1"]["reasons"]["no_hist_score"], 8)
        self.assertEqual(rep["abstentions"]["M0"]["reasons"]["no_hist_score"], 0)
        p = rep["groups"]["M1"]["all"]["point"]
        self.assertAlmostEqual(p[name := rep["combined"]]["rival_abstain"], 3 / 200)
        self.assertEqual(p["spana"]["rival_abstain"], 0.0)
        self.assertAlmostEqual(p["hist"]["rival_abstain"], 3 / 200)             # single cue: same item, no score

    def test_deterministic_for_seed(self):
        tables = [table(f"M{i}", i) for i in range(4)]
        a = json.dumps(F.report(tables, small_cfg()), default=str, sort_keys=True)
        b = json.dumps(F.report(copy.deepcopy(tables), small_cfg()), default=str, sort_keys=True)
        self.assertEqual(a, b)

    def test_single_cue_points_are_the_eval_numbers(self):
        tables = [table(f"M{i}", i) for i in range(4)]
        rep = F.report(tables, small_cfg())
        t = tables[0]
        for c in ("hist", "spana"):
            p = rep["groups"]["M0"]["all"]["point"][c]
            m = t["rival"]["qa"][c] - t["rival"]["qb"][c]
            self.assertEqual(p["rival_acc"], float(np.mean([x > 0 for x in m])))
            Sh, Dh = t["pairs"]["hard"]["same"]["scores"][c], t["pairs"]["hard"]["diff"]["scores"][c]
            self.assertEqual(p["tar@0.01_hard"], float(np.mean(np.asarray(Sh) > float(np.percentile(Dh, 99)))))

    def test_subsets_and_verdict(self):
        tables = [table(f"M{i}", i, sep1=2.0, sep2=2.0) for i in range(4)]   # independent noise: fusion helps
        rep = F.report(tables, small_cfg())
        self.assertIn("query_occluded", rep["groups"]["pooled"])
        self.assertIn("colour_hard_pairs", rep["groups"]["M0"])
        v = rep["verdict"]
        self.assertEqual(v["metric"], "tar@0.005_hard")
        pooled = v["pooled"]
        self.assertEqual(v["promising"], pooled["hist+spana"] > max(pooled["hist"], pooled["spana"])
                         and not v["matches_below"])
        bad = [table(f"M{i}", i, sep1=0.0, sep2=0.0) for i in range(4)]       # no signal anywhere
        bad[0]["pairs"]["hard"]["same"]["scores"]["spana"] += 10                # one cue perfect on one match
        self.assertFalse(F.report(bad, small_cfg())["verdict"]["promising"])


if __name__ == "__main__":
    unittest.main()
