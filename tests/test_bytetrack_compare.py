"""DEV ByteTrack comparison (tools/dev/bytetrack_compare.py): ByteTrack's output becomes a valid spana.tracks artifact
on the same detections, with ByteTrack's own timing (lost pool, removal after max_time_lost), and the real
supervision ByteTrack (if supervision 0.25.1 is importable) runs end to end."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from spana import cli
from spana.artifacts import read_json, validate_tracks
from tools.dev import bytetrack_compare as BC


def fixture():
    """Reuse the end-to-end fixture: a synthetic clip, one detection per frame, our detections + tracks artifacts."""
    from tests.test_cli_e2e import EndToEnd               # imported here so pytest does not collect it twice
    e = EndToEnd("test_full_pipeline")
    e.setUp()
    d, t = e.td / "det.json", e.td / "trk.json"
    e._detect(d)
    assert cli.main(["track", "--video", str(e.video), "--config", e.config, "--detections", str(d), "--out", str(t)]) == 0
    return e.td, read_json(d), read_json(t)


class Convert(unittest.TestCase):
    def setUp(self):
        self.td, self.dets, self.ref = fixture()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_lost_pool_and_expiry_follow_bytetrack(self):
        frames = [fr["frame"] for fr in self.dets["frames"]]
        assoc = {f: [] for f in frames}
        for f in list(range(0, 6)) + list(range(9, 13)):
            assoc[f].append((7, 0))                              # ByteTrack id 7: seen, lost 6-8, seen again, gone
        for f in range(14, 20):
            assoc[f].append((9, 0))                              # a new id takes the same detection later
        doc = BC.to_tracks(self.dets, self.ref, assoc, max_lost=3, provenance={})
        validate_tracks(doc, self.dets)
        rows = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in doc["frames"]}
        self.assertEqual(rows[0]["T000001"]["confirmed"], False)              # fresh start
        self.assertEqual(rows[1]["T000001"]["confirmed"], True)
        self.assertEqual((rows[7]["T000001"]["state"], rows[7]["T000001"]["source"]), ("LOST", "HELD"))
        self.assertEqual(rows[9]["T000001"]["state"], "ACTIVE")                # same id back after the gap
        self.assertIn("T000001", rows[15])                                    # still in the lost pool (12 + 3)
        ended = {fr["frame"]: fr["ended"] for fr in doc["frames"]}
        self.assertEqual(ended[16], [["T000001", "LOST_EXPIRED"]])            # removed when ByteTrack removes it
        t1 = next(t for t in doc["tracklets"] if t["track_id"] == "T000001")
        self.assertEqual((t1["last_observed_frame"], t1["last_frame"], t1["observed_frames"]), (12, 15, 10))
        t2 = next(t for t in doc["tracklets"] if t["track_id"] == "T000002")
        self.assertIsNone(t2["end_reason"])                                    # live at the end of the clip
        lf = BC.life(doc)
        self.assertEqual(lf["confirmed_tracks"], 2)
        self.assertEqual(lf["uncertain_track_frames"], 0)                      # ByteTrack never flags identity

    def test_cut_refused(self):
        bad = json.loads(json.dumps(self.ref))
        bad["frames"][5]["camera"]["status"] = "CUT"
        with self.assertRaises(SystemExit):
            BC.to_tracks(self.dets, bad, {fr["frame"]: [] for fr in self.dets["frames"]}, 3, {})

    def test_max_time_lost_matches_supervision(self):
        self.assertEqual(BC.max_time_lost(30.0, 30), 30)
        self.assertEqual(BC.max_time_lost(60.0, 30), 60)
        self.assertEqual(BC.max_time_lost(25.0, 30), 25)


def _sv():
    try:
        import supervision as sv
        return sv.__version__ == BC.load_cfg()["supervision_version"] and hasattr(sv, "ByteTrack")
    except Exception:
        return False


@unittest.skipUnless(_sv(), "supervision 0.25.1 not importable (run with PYTHONPATH pointing at it)")
class RealByteTrack(unittest.TestCase):
    def test_run_end_to_end(self):
        td, dets, ref = fixture()
        try:
            (td / "d.json").write_text(json.dumps(dets))
            (td / "r.json").write_text(json.dumps(ref))
            out = td / "bt.trk.json"
            self.assertEqual(BC.main(["run", "--detections", str(td / "d.json"), "--ref-tracks", str(td / "r.json"),
                                      "--out", str(out)]), 0)
            doc = read_json(out)
            validate_tracks(doc, dets)
            self.assertIn("git", doc["provenance"])                   # tools/benchmark.py reads it
            self.assertTrue(any(t["state"] == "ACTIVE" for fr in doc["frames"] for t in fr["tracks"]))
            self.assertEqual((td / "bt.trk.decisions.jsonl").read_text(), "")
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
