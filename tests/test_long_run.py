"""tools/long_run.py: locate a clip in a longer video, subsample detections, select a labelled player in a window,
summarise a long target document. Synthetic data only."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spana.artifacts import validate_detections
from spana.config import load_config
from spana.identity.follower import target_document
from tests.test_target_follow import A, N, tracks_doc
from tools.long_run import locate, overlap, select, subsample, summary

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")


def _write(path, frames, fps):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (128, 72))
    if not w.isOpened():
        raise unittest.SkipTest("no video writer")
    for f in frames:
        w.write(f)
    w.release()


class LocateTest(unittest.TestCase):
    def test_finds_the_clip_start(self):
        rng = np.random.default_rng(3)
        base = [cv2.resize(rng.integers(0, 255, (9, 16, 3), dtype=np.uint8), (128, 72), interpolation=cv2.INTER_NEAREST)
                for _ in range(120)]
        with tempfile.TemporaryDirectory() as tmp:
            _write(Path(tmp) / "match.mp4", base, 10.0)
            _write(Path(tmp) / "clip.mp4", base[37:], 10.0)
            r = locate(str(Path(tmp) / "clip.mp4"), str(Path(tmp) / "match.mp4"), [4.0], 3.0)
        self.assertEqual(r["frame"], 37)
        self.assertLess(r["confirm_score"], r["median_score_in_windows"])


class SubsampleTest(unittest.TestCase):
    def test_every_nth_frame(self):
        det = {"schema": "spana.detections", "schema_version": "1.4.0", "provenance": {"config_sha256": "c"},
               "video": {"sha256": "v"}, "model": {}, "class_names": ["person"], "detection_config": {},
               "frame_stride": 1,
               "frames": [{"frame": i, "t": i / 10, "detections": [[0, 0, 10, 20, 0.9, 0, 0]]} for i in range(10)]}
        out = subsample(det, 3, ["test"], "s" * 64)
        validate_detections(out)
        self.assertEqual([f["frame"] for f in out["frames"]], [0, 3, 6, 9])
        self.assertEqual(out["frame_stride"], 3)
        self.assertEqual(out["provenance"]["derived_from"]["stride"], 3)


class SelectAndSummaryTest(unittest.TestCase):
    def test_select_maps_clip_frames_and_scales_boxes(self):
        # window: 10 fps, 640 x 360; t1 box x 0..40 (tracks_doc draws live tracks at x = 100 * index)
        tracks = tracks_doc([{"t1": N, "t2": N}] + [{"t1": A, "t2": A}] * 6)
        tracks["video"].update(width=640, height=360, fps=10.0)
        # clip: 20 fps, 1280 x 720 (boxes x2); clip frames 0 and 4 labelled -> window frames 2 + 0 and 2 + 2
        gt = "1,7,200,200,80,200,1,1,1\n5,7,0,200,80,200,1,1,1\n".encode()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "gt.txt").write_bytes(gt)
            (d / "FROZEN.json").write_text(json.dumps({"split": "DEV", "gt_sha256": hashlib.sha256(gt).hexdigest(),
                                                       "frame_step": 4, "clip": {"name": "c", "width": 1280,
                                                                                 "height": 720, "sha256": "x"}}))
            (d / "labels_frozen.json").write_text(json.dumps({"frames_labelled": [0, 4]}))
            r = select(d, 20.0, 2, tracks, "7", "NESTED_V2")
        # clip frame 0 -> window frame 2; the label (x 200..280 at 1280 px) scales to x 100..140 = t2's box
        self.assertEqual((r["frame"], r["track_id"], r["clip_frame"]), (2, "t2", 0))

    def test_overlap_scores_answers_on_the_mapped_labels(self):
        from spana.identity.follower import Selection
        tracks = tracks_doc([{"t1": N, "t2": N}] + [{"t1": A, "t2": A}] * 6)
        tracks["video"].update(width=640, height=360, fps=10.0)
        # clip 20 fps at 1280 x 720, offset 2: clip frames 2 and 4 -> window 3 and 4; player 7 is on t1 (x 0..40)
        gt = "3,7,0,200,80,200,1,1,1\n5,7,200,200,80,200,1,1,1\n5,8,0,200,80,200,1,1,1\n".encode()   # 8 on t1
        doc = target_document(tracks, Selection.track_at_frame("t1", 2), CFG, ["test"])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "gt.txt").write_bytes(gt)
            (d / "FROZEN.json").write_text(json.dumps({"split": "DEV", "gt_sha256": hashlib.sha256(gt).hexdigest(),
                                                       "frame_step": 2, "clip": {"name": "c", "width": 1280,
                                                                                 "height": 720, "sha256": "x"}}))
            (d / "labels_frozen.json").write_text(json.dumps({"frames_labelled": [2, 4]}))
            r = overlap(d, 20.0, 2, tracks, doc, "7", "NESTED_V2")
        self.assertEqual((r["counts"]["correct"], r["counts"]["wrong"]), (1, 1))     # frame 4: t1 shows player 8
        self.assertEqual(r["events"][0]["window_frame"], 4)

    def test_summary(self):
        tracks = tracks_doc([{"t1": N, "t2": N}] + [{"t1": A, "t2": A}] * 4)
        doc = target_document(tracks, __import__("spana.identity.follower", fromlist=["Selection"]).Selection
                              .track_at_frame("t1", 0), CFG, ["test"])
        s = summary(doc)
        self.assertEqual(s["frames"], 5)
        self.assertEqual(s["gaps"], 1)                      # the unconfirmed first frame
        self.assertAlmostEqual(s["shares"]["TARGET"], 0.8)


if __name__ == "__main__":
    unittest.main()
