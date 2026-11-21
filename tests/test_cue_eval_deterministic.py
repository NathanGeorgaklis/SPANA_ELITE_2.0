"""tools/dev/appearance_cue_eval.py: rival tests come out in the same order whatever the hash seed, so reruns write
byte-identical --dump-tables files (task evaluation/cue-eval-deterministic)."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Synthetic clip: 12 string player ids, 3 crops each, 1.5 s apart; every pair of players is a hard pair (a set of
# string tuples, whose iteration order follows PYTHONHASHSEED). Prints the dumped rival table as the tool writes it.
SCRIPT = r'''
import collections, json, math
from tools.dev.appearance_cue_eval import fusion_table, rival_triplets
players = [f"P{k}" for k in range(12)]
crops = [{"player": p, "t": 1.5 * j, "h": 80.0 + j, "occluded": j == 1, "max_iou": 0.1 * j}
         for j in range(3) for p in players]
by_player = collections.defaultdict(list)
for k, c in enumerate(crops):
    by_player[c["player"]].append(k)
hard = {tuple(sorted((a, b))) for a in players for b in players if a != b}
tri = rival_triplets(hard, by_player, crops)
sim = lambda c, a, b: math.cos(a * 7.0 + b * 3.0)
t = fusion_table("X", crops, ["hist"], sim, [], [], [], [], tri)
print(json.dumps({"triplets": tri, "rival": {k: (v if not isinstance(v, dict) else {c: list(map(float, a)) for c, a in v.items()})
                                             for k, v in t["rival"].items() if k != "occluded"}}))
'''


def run(seed: str) -> bytes:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    return subprocess.run([sys.executable, "-c", SCRIPT], cwd=REPO, env=env, capture_output=True, check=True).stdout


class DeterministicRivalTests(unittest.TestCase):
    def test_two_runs_give_identical_output(self):
        outs = {run(s) for s in ("0", "1", "2", "12345")}
        self.assertEqual(len(outs), 1)
        self.assertIn(b'"P0|P1"', next(iter(outs)))

    def test_order_is_sorted_pairs(self):
        import collections
        from tools.dev.appearance_cue_eval import rival_triplets
        crops = [{"player": p, "t": t} for t in (0.0, 2.0) for p in ("b", "a", "c")]
        by_player = collections.defaultdict(list)
        for k, c in enumerate(crops):
            by_player[c["player"]].append(k)
        tri = rival_triplets({("b", "c"), ("a", "c"), ("a", "b")}, by_player, crops)
        self.assertEqual([t[3] for t in tri], [("a", "b")] * 2 + [("a", "c")] * 2 + [("b", "c")] * 2)


if __name__ == "__main__":
    unittest.main()
