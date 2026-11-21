"""DEV BoT-SORT comparison (tools/dev/botsort_compare.py, tracker bench v1): BoT-SORT's output becomes a valid
spana.tracks artifact on the same detections, -1 (unconfirmed) rows are not tracks, a track lives exactly as long as
BoT-SORT holds it, the frame reaches update() (camera motion compensation), and the real trackers BoTSORTTracker (if
trackers 2.7.0 is importable) runs end to end. The ByteTrack conversion stays unchanged without removed_at."""
from __future__ import annotations

import json
import shutil
import unittest
from types import SimpleNamespace

import numpy as np

from spana.artifacts import read_json, validate_tracks
from tests.test_bytetrack_compare import fixture
from tools.dev import botsort_compare as BS
from tools.dev import bytetrack_compare as BC


class FakeTracker:
    """Scripted stand-in: per call, (output [(id, det idx)], ids held in the track list); records the frames given."""

    def __init__(self, script):
        self.script, self.k, self.tracks, self.frames_seen = script, 0, [], []

    def update(self, det, frame=None):
        self.frames_seen.append(frame)
        outs, held = self.script[self.k] if self.k < len(self.script) else ([], [])
        self.k += 1
        self.tracks = [SimpleNamespace(tracker_id=t) for t in held]
        ids = np.array([t for t, _ in outs], int)
        idx = np.array([i for _, i in outs], int)
        return _Res(ids, idx)


class _Res:
    def __init__(self, ids, idx):
        self.tracker_id, self.data = ids, {"idx": idx}

    def __len__(self):
        return len(self.tracker_id)


class Convert(unittest.TestCase):
    def setUp(self):
        self.td, self.dets, self.ref = fixture()
        self.frames = [fr["frame"] for fr in self.dets["frames"]]

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_unconfirmed_rows_lost_pool_and_observed_removal(self):
        n = len(self.frames)
        script = []
        for k in range(n):
            if k == 0:
                script.append(([(-1, 0)], [-1]))                     # unconfirmed first sighting: not a track
            elif k <= 5 or 9 <= k <= 12:
                script.append(([(3, 0)], [3]))                       # id 3 seen
            elif 6 <= k <= 8 or 13 <= k <= 16:
                script.append(([], [3]))                             # id 3 in the lost pool
            else:
                script.append(([], []))                              # removed on frame 17
        fake = FakeTracker(script)
        imgs = [(f, np.full((2, 2, 3), f, np.uint8)) for f in self.frames]
        assoc, removed = BS.run_tracker(fake, self.dets, iter(imgs))
        self.assertEqual(assoc[self.frames[0]], [])
        self.assertEqual(removed, {3: self.frames[17]})
        self.assertTrue(all(img is not None for img in fake.frames_seen))     # the frame reaches update() (CMC)
        doc = BC.to_tracks(self.dets, self.ref, assoc, max_lost=0, provenance={}, removed_at=removed)
        validate_tracks(doc, self.dets)
        rows = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in doc["frames"]}
        f = self.frames
        self.assertNotIn("T000001", rows[f[0]])
        self.assertEqual((rows[f[7]]["T000001"]["state"], rows[f[7]]["T000001"]["source"]), ("LOST", "HELD"))
        self.assertIn("T000001", rows[f[16]])
        self.assertNotIn("T000001", rows[f[17]])
        ended = {fr["frame"]: fr["ended"] for fr in doc["frames"]}
        self.assertEqual(ended[f[17]], [["T000001", "LOST_EXPIRED"]])
        self.assertEqual(BC.life(doc)["uncertain_track_frames"], 0)           # BoT-SORT never flags identity

    def test_held_at_clip_end_has_no_end(self):
        assoc = {f: [(5, 0)] if k < 3 else [] for k, f in enumerate(self.frames)}
        doc = BC.to_tracks(self.dets, self.ref, assoc, max_lost=0, provenance={}, removed_at={})
        t = doc["tracklets"][0]
        self.assertIsNone(t["end_reason"])
        self.assertEqual(t["last_frame"], self.frames[-1])

    def test_removal_before_last_output_refused(self):
        assoc = {f: [(5, 0)] if k < 5 else [] for k, f in enumerate(self.frames)}
        with self.assertRaises(SystemExit):
            BC.to_tracks(self.dets, self.ref, assoc, 0, {}, removed_at={5: self.frames[2]})

    def test_frame_mismatch_refused(self):
        fake = FakeTracker([])
        with self.assertRaises(SystemExit):
            BS.run_tracker(fake, self.dets, iter([(self.frames[0] + 1, None)]))

    def test_bytetrack_conversion_unchanged_without_removed_at(self):
        assoc = {f: [(7, 0)] if k < 4 else [] for k, f in enumerate(self.frames)}
        a = BC.to_tracks(self.dets, self.ref, assoc, max_lost=3, provenance={})
        self.assertEqual(a["tracklets"][0]["last_frame"], self.frames[6])
        self.assertEqual(a["tracklets"][0]["end_reason"], "LOST_EXPIRED")

    def test_config_is_published_defaults(self):
        import inspect
        cfg = BS.load_cfg()
        try:
            from trackers import BoTSORTTracker
        except Exception:
            self.skipTest("trackers not importable")
        sig = inspect.signature(BoTSORTTracker.__init__).parameters
        for k, v in cfg["botsort"].items():
            self.assertEqual(sig[k].default, v, k)


def _trackers():
    try:
        from importlib.metadata import version
        import trackers  # noqa: F401
        return version("trackers") == BS.load_cfg()["trackers_version"]
    except Exception:
        return False


@unittest.skipUnless(_trackers(), "trackers 2.7.0 not importable (run with PYTHONPATH pointing at it)")
class RealBoTSORT(unittest.TestCase):
    def test_run_end_to_end(self):
        td, dets, ref = fixture()
        try:
            video = next(td.glob("*.mp4"), None) or next(td.glob("*.avi"))
            (td / "d.json").write_text(json.dumps(dets))
            (td / "r.json").write_text(json.dumps(ref))
            out = td / "bs.trk.json"
            self.assertEqual(BS.main(["run", "--video", str(video), "--detections", str(td / "d.json"),
                                      "--ref-tracks", str(td / "r.json"), "--out", str(out)]), 0)
            doc = read_json(out)
            validate_tracks(doc, dets)
            self.assertIn("git", doc["provenance"])
            self.assertTrue(any(t["state"] == "ACTIVE" for fr in doc["frames"] for t in fr["tracks"]))
            self.assertEqual((td / "bs.trk.decisions.jsonl").read_text(), "")
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
