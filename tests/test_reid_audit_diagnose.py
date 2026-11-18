"""Audit diagnosis (tools/dev/reid_audit_diagnose.py): a stale cluster decision left behind by a split is counted by
the tool but is not a real conflict; a real player-vs-player conflict is; unsure answers are categorised."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_reid_cluster import TRACKLETS, colour_embed, make_pack, write_cfg
from tools import reid_cluster as RC
from tools.dev.reid_audit_diagnose import diagnose


class Diagnose(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.pack = make_pack(self.td)
        self.cfg = write_cfg(self.td, min_tracklets=3)
        self.R = RC.build(self.pack, self.td / "review", self.cfg, embed=colour_embed)
        # the diagnosis is for 1.0.0 reviews, whose audit sample was drawn at build time: give it one
        self.R["audit"] = ["T000001-1", "T000005-1", "T000006-1"]
        (self.td / "review" / "review.json").write_text(json.dumps(self.R))

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def state(self):
        members = {k: list(c["tracklets"]) for k, c in self.R["clusters"].items()}
        players = {"A": {"side": "A", "shirt_number": "7"}, "B": {"side": "A", "shirt_number": "9"},
                   "C": {"side": "B", "shirt_number": "4"}}
        cdec = {}
        for k, m in members.items():
            who = {TRACKLETS[t][0] for t in m}
            cdec[k] = {"decision": "player", "player": who.pop()} if len(who) == 1 else {"decision": "unsure", "player": None}
        audit = {t: {"decision": "player", "player": TRACKLETS[t][0]} for t in self.R["audit"]}
        return {"players": players, "members": members, "cdec": cdec, "overrides": {}, "audit": audit, "next": 99}

    def test_stale_split_real_conflict_and_unsure(self):
        S = self.state()
        audit = self.R["audit"]
        cl = {t: k for k, m in S["members"].items() for t in m}
        # 1) stale: the audit tracklet's cluster was assigned, then all other members were split away and relabelled
        stale_t = next(t for t in audit if len(S["members"][cl[t]]) > 1)
        k = cl[stale_t]
        rest = [m for m in S["members"][k] if m != stale_t]
        S["members"]["C900s"] = rest
        S["members"][k] = [stale_t]
        S["cdec"]["C900s"] = dict(S["cdec"][k])
        S["cdec"][k] = {"decision": "player", "player": "B" if TRACKLETS[stale_t][0] != "B" else "A"}   # stale, wrong
        # 2) unsure answer on another audit tracklet
        others = [t for t in audit if t != stale_t]
        S["audit"][others[0]] = {"decision": "unsure", "player": None}
        res = diagnose(self.td / "review", self.pack, S, self.td / "diag", light="night", cfg_path=self.cfg)
        rows = {r["tracklet"]: r for r in res["all"]}
        r = rows[stale_t]
        self.assertTrue(r["counted_by_tool_as_disagreement"])
        self.assertIsNone(r["post_edit_label"])                         # no live mates left
        self.assertTrue(r["category"].startswith("other: stale cluster label"))
        self.assertIsNone(rows[others[0]]["counted_by_tool_as_disagreement"])   # unsure: not counted by the tool
        self.assertEqual(res["rate_b_as_tool_counts"]["disagreements"], 1)
        self.assertEqual(res["rate_a_real_conflicts"]["disagreements"], 0)     # the stale one is no real conflict
        self.assertTrue((self.td / "diag" / f"sheet_{stale_t}.jpg").exists())
        self.assertTrue((self.td / "diag" / "index.html").exists())
        self.assertEqual(r["light"], "night")

    def test_real_conflict_same_team(self):
        S = self.state()
        t = next(t for t in self.R["audit"] if TRACKLETS[t][0] in ("A", "B") and
                 len([m for k, mm in S["members"].items() if t in mm for m in mm]) > 1)
        wrong = "B" if TRACKLETS[t][0] == "A" else "A"
        S["audit"][t] = {"decision": "player", "player": wrong}
        res = diagnose(self.td / "review", self.pack, S, self.td / "diag2", cfg_path=self.cfg)
        r = {x["tracklet"]: x for x in res["all"]}[t]
        self.assertTrue(r["real_conflict"])
        self.assertEqual(r["category"], "wrong cluster (same team)")
        self.assertEqual(res["rate_a_real_conflicts"]["disagreements"], 1)


if __name__ == "__main__":
    unittest.main()
