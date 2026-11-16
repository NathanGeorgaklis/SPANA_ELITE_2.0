"""tools/reid_headroom.py on synthetic tracks + frozen labels: handover extraction, clue ranks, ceiling."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import write_json_atomic
from tests.test_target_follow import CFG, A, L, N, tracks_doc
from tools.reid_headroom import _rank, analyse_clip


def _labels(d: Path, rows) -> None:
    gt = ("\n".join(rows) + "\n").encode()
    d.mkdir()
    (d / "gt.txt").write_bytes(gt)
    (d / "FROZEN.json").write_text(json.dumps({
        "split": "DEV", "gt_sha256": hashlib.sha256(gt).hexdigest(), "frame_step": 1,
        "clip": {"name": "synthetic", "width": 640, "height": 360, "sha256": "v" * 64}}))
    (d / "labels_frozen.json").write_text(json.dumps({"frames_labelled": [1, 2, 4, 5]}))


class HeadroomTest(unittest.TestCase):
    def test_loss_handover_ranks_and_ceiling(self):
        # t1 (player 1) is lost after frame 2 and ends on frame 4; t3 is born on frame 4 and carries player 1.
        tracks = tracks_doc([{"t1": N, "t2": N}, {"t1": A, "t2": A}, {"t1": A, "t2": A}, {"t1": L, "t2": A},
                             {"t2": A, "t3": N}, {"t2": A, "t3": A}],
                            ended={4: [("t1", "LOST_EXPIRED")]})
        # live tracks are drawn at x = 100 * (index among the frame's sorted ids)
        rows = []
        for f, (p1, p2) in {1: (0, 100), 2: (0, 100), 4: (100, 0), 5: (100, 0)}.items():
            rows += [f"{f + 1},1,{p1},100,40,100,1,1,1", f"{f + 1},2,{p2},100,40,100,1,1,1"]
        with tempfile.TemporaryDirectory() as tmp:
            _labels(Path(tmp) / "lab", rows)
            trk = Path(tmp) / "trk.json"
            write_json_atomic(trk, tracks)
            rep = analyse_clip(Path(tmp) / "lab", trk, None, CFG)
        ev = [e for e in rep["handovers"] if e["player"] == "1"]
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertEqual((e["from_track"], e["to_track"], e["kind"]), ("t1", "t3", "LOSS"))
        self.assertEqual((e["a_last_observed_frame"], e["b_first_frame"], e["a_end"]), (2, 4, "LOST_EXPIRED"))
        self.assertEqual(e["candidates"], 1)
        self.assertEqual(e["position"], {"b_distance_h": 1.0, "rank": 1})
        self.assertIsNone(e["appearance"])
        p1 = next(r for r in rep["ceiling"]["per_player"] if r["player"] == "1")
        self.assertEqual((p1["v1_correct"], p1["ceiling_correct"]), (2, 3))   # t3 unconfirmed on its first frame

    def test_rank_ties_count_against(self):
        self.assertEqual(_rank({"b": 1.0, "x": 1.0, "y": 2.0}, "b", higher_better=False), 2)
        self.assertEqual(_rank({"b": 0.9, "x": None}, "b", higher_better=True), 1)
        self.assertIsNone(_rank({"b": None}, "b", higher_better=True))


if __name__ == "__main__":
    unittest.main()
