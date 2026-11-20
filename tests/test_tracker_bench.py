"""DEV tracker bench summary (tools/dev/tracker_bench.py): pooling as in records/bytetrack_compare_2026-11-18 —
flagged = transfers - unflagged, shares over all scored player-frames (CANDIDATE frames counted, never correct),
no answer = abstain + lost, actions per followed player-minute; swapped default / reviewer reports refused."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.dev import tracker_bench as TB


def bench(clip, frag, idsw, tr, unfl):
    return {"labels": {"dir": f"/x/data/labels/{clip}_G1_abc/v1"},
            "summary": {"fragments_total": frag, "id_switches": idsw, "identity_transfers": tr,
                        "unflagged_transfers": unfl, "track_coverage": 0.9}}


def target(reviewer, clips):
    return {"simulate_reviewer": reviewer, "clips": [
        {"clip": f"{c}_G1_abc", "pooled": {**counts, "actions": {"track_at_frame": acts, "confirm_relink": 0},
                                            "followed_s": fs}} for c, counts, acts, fs in clips]}


def counts(correct, wrong, abstain, lost, cand=0):
    return {"correct": correct, "wrong": wrong, "unverified": 0, "abstain": abstain, "lost": lost,
            "candidate_correct": cand, "candidate_wrong": 0, "candidate_unverified": 0}


class Summary(unittest.TestCase):
    def test_pooling(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "b1.json").write_text(json.dumps(bench("FRESH_02", 10, 11, 3, 1)))
            (td / "b2.json").write_text(json.dumps(bench("FRESH_05", 5, 6, 2, 2)))
            (td / "d.json").write_text(json.dumps(target(False, [("FRESH_02", counts(60, 2, 10, 20, 8), 0, 60.0),
                                                                  ("FRESH_05", counts(40, 0, 0, 60), 0, 60.0)])))
            (td / "r.json").write_text(json.dumps(target(True, [("FRESH_02", counts(90, 3, 5, 2), 4, 60.0),
                                                                 ("FRESH_05", counts(90, 0, 5, 5), 2, 60.0)])))
            doc = TB.summary({"x": {"default": td / "d.json", "reviewer": td / "r.json",
                                    "benches": [td / "b1.json", td / "b2.json"]}})["x"]
            tp = doc["tracking"]["pooled"]
            self.assertEqual((tp["fragments_total"], tp["identity_transfers"], tp["unflagged_transfers"],
                              tp["flagged_transfers"]), (15, 5, 3, 2))
            self.assertEqual(doc["tracking"]["clips"]["FRESH_02"]["flagged_transfers"], 2)
            fd = doc["following_default"]["pooled"]
            self.assertEqual(fd["scored"], 200)
            self.assertEqual((fd["correct_share"], fd["wrong_share"], fd["no_answer_share"]), (0.5, 0.01, 0.45))
            fr = doc["following_reviewer"]["pooled"]
            self.assertEqual(fr["actions_per_followed_min"], 3.0)
            with self.assertRaises(SystemExit):
                TB.summary({"x": {"default": td / "r.json", "reviewer": td / "d.json", "benches": [td / "b1.json"]}})


if __name__ == "__main__":
    unittest.main()
