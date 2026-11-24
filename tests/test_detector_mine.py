"""tools/detector_mine.py: mining candidates for detector retraining from TRAIN footage only (proposes frames for a
person to review; labels, uploads and trains nothing)."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spana.artifacts import sha256_doc
from spana.ball.path import ball_path_document
from spana.config import load_config as load_spana_config
from tests.synth import cfg as spana_cfg
from tests.test_ball_path import FPS, H, W, ball_doc
from tests.test_ball_tiling_fp import BlobDetector
from tests.test_ball_tracker import ball
from tools import detector_mine as dm
from tools.diagnostics import ball_tiling_fp as fp

ROOT = Path(__file__).resolve().parents[1]
CFG = dm.load_config()
ON = dataclasses.replace(spana_cfg().ball, tile_size_frac=1.0)
PC = spana_cfg().ball_path
LABELS = ROOT / "records" / "roboflow_version" / "labels_20261114T082808Z.json"
FLAGS = ROOT / "records" / "roboflow_review" / "review_flags_20261114T052034Z.json"
TILING_CONFIG = "records/ball_tiling_v1_2026-11-23/tiling_on.config.yaml"
REAL = lambda i: (100 + 6 * i, 300)        # noqa: E731
CONE = (640, 500)


def untiled(i):                             # the full frame: the real ball, missed on frames 20-27
    return [] if 20 <= i <= 27 else [ball(*REAL(i))]


def tiled(i):                               # the tiles also see the ball on 20-27, and a steady look-alike on 10-50
    return [ball(*REAL(i))] + ([ball(*CONE, score=0.5)] if 10 <= i <= 50 else [])


def paths():
    u_doc, t_doc = ball_doc([untiled(i) for i in range(60)], ball_cfg=ON), ball_doc([tiled(i) for i in range(60)],
                                                                                    ball_cfg=ON)
    return (ball_path_document(u_doc, PC, {}, sha256_doc(u_doc)), ball_path_document(t_doc, PC, {}, sha256_doc(t_doc)),
            t_doc)


class FrozenLabels(unittest.TestCase):
    """Read-only facts of the frozen detector-v1 training labels."""

    def test_scale_report(self):
        rep = dm.scale_report(json.loads(LABELS.read_text()))
        self.assertEqual((rep["frames"], rep["frame_sizes"]), (875, [(1280, 720)]))
        self.assertEqual({k: v["ball_boxes"] for k, v in rep["splits"].items()}, {"test": 2, "train": 414, "valid": 225})
        self.assertEqual((rep["splits"]["train"]["p50"], rep["splits"]["valid"]["p50"]), (13.0, 13.0))

    def test_spare_ball_audit_candidates(self):
        rows = dm.audit_candidates(json.loads(LABELS.read_text()), json.loads(FLAGS.read_text()), CFG)
        self.assertEqual((len(rows), len({r["name"] for r in rows})), (31, 30))
        why = [w for r in rows for w in r["why"]]
        self.assertEqual({w: why.count(w) for w in set(why)}, {"ball_static": 15, "ball_offpitch": 14,
                                                               "several_ball_boxes": 2})
        self.assertEqual([r["n"] for r in rows], list(range(1, 32)))
        two = [r for r in rows if "several_ball_boxes" in r["why"]]
        self.assertEqual({r["name"] for r in two}, {"TRAIN_M001_MORETON_CITY_f0020508.jpg"})
        self.assertEqual(two[0]["frame"], 20508)


class Windows(unittest.TestCase):
    def test_window_ranges(self):
        self.assertEqual(dm.window_ranges(30 * 300, 30.0, CFG)[:3], [(0, 180), (3600, 3780), (7200, 7380)])
        self.assertEqual(dm.window_ranges(100, 30.0, CFG), [])                       # shorter than one window
        self.assertEqual(len(dm.window_ranges(30 * 60 * 90, 30.0, CFG)), 45)          # a 90-minute match
        self.assertEqual([dm.stride_of(f, CFG) for f in (60.0, 30.0, 25.0, 15.0, 10.0)], [4, 2, 2, 1, 1])


class Candidates(unittest.TestCase):
    def setUp(self):
        self.objs = dm.candidates_of(*paths(), radius_px=7.2, cfg=CFG)
        self.by = {k: [o for o in self.objs if o["kind"] == k] for k in dm.KINDS}

    def test_missed_ball_from_a_hole_of_the_untiled_path(self):
        (o,) = self.by["missed_ball"]
        self.assertEqual((o["why"], o["first_frame"], o["last_frame"], o["frames_available"]), ("hole", 20, 27, 8))
        self.assertEqual(len(o["frames"]), CFG.frames_per_gap)
        for r in o["frames"]:                                  # the hint is on the ball's straight path
            self.assertTrue(r["hint"])
            self.assertAlmostEqual(r["center"][0], REAL(r["frame"])[0], delta=1.0)

    def test_look_alike_in_conflict_with_the_real_ball_is_still_mined(self):
        _, t, _ = paths()
        self.assertIn("CONFLICT", {f["reason"] for f in t["frames"]})           # the path itself hides it
        (o,) = self.by["tiled_only"]
        self.assertEqual((o["first_frame"], o["last_frame"], o["frames_available"]), (10, 50, 41))
        self.assertEqual(len(o["frames"]), CFG.frames_per_object)
        self.assertTrue(all(r["center"] == [float(CONE[0]), float(CONE[1])] and not r["hint"] for r in o["frames"]))

    def test_frames_the_tiles_add_to_the_real_ball(self):
        (o,) = self.by["tiled_extends"]
        self.assertEqual(o["frames_available"], 8)                               # frames 20-27
        self.assertTrue(all(20 <= r["frame"] <= 27 for r in o["frames"]))

    def test_long_gaps_are_not_missed_ball_candidates(self):
        u_doc = ball_doc([[] if 20 <= i <= 59 else [ball(*REAL(i))] for i in range(90)], ball_cfg=ON)
        u = ball_path_document(u_doc, PC, {}, sha256_doc(u_doc))
        objs = dm.candidates_of(u, u, u_doc, 7.2, CFG)                          # a 1.6 s gap > max_gap_s
        self.assertEqual([o for o in objs if o["kind"] == "missed_ball"], [])


class TrainOnly(unittest.TestCase):
    def test_only_train_matches_are_mined(self):
        dm._train_only("config/roboflow.yaml", "TRAIN_M002_EASTERN_SUBURBS")
        for bad in ("DEV_G1_SYDNEY_UNITED_APIA", "DEV_G2_NWS_SPIRIT_SD_RAIDERS", "SOME_EVAL_MATCH"):
            with self.subTest(bad), self.assertRaisesRegex(SystemExit, "TRAIN matches only"):
                dm._train_only("config/roboflow.yaml", bad)


class MineCommand(unittest.TestCase):
    def test_mine_resume_review_and_yield(self):
        import cv2
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            video = tmp / "m.avi"
            vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
            from tests.test_detection_camera import _texture
            ground = (_texture(H, W, seed=3) * 0.35).astype(np.uint8)   # dark texture: the camera needs features
            for i in range(60):
                img = ground.copy()
                for d in tiled(i):
                    x1, y1, x2, y2 = (int(v) for v in d.box)
                    img[y1:y2, x1:x2] = 255
                vw.write(img)
            vw.release()
            mcfg = tmp / "mine.yaml"                                  # one 2.4 s window, every frame
            mcfg.write_text((ROOT / "config" / "detector_mine.yaml").read_text()
                            .replace("window_s: 6.0", "window_s: 2.4").replace("window_every_s: 120.0",
                                                                               "window_every_s: 2.4")
                            .replace("analysis_fps: 15.0", "analysis_fps: 25.0"))
            argv = ["mine", "--video", str(video), "--match-id", "TRAIN_M002_EASTERN_SUBURBS", "--config",
                    TILING_CONFIG, "--dataset-config", "config/roboflow.yaml", "--model-record", "unused",
                    "--runtime", "torch", "--device", "cpu", "--out", str(tmp / "out"), "--mine-config", str(mcfg)]
            calls = []

            def stub(a, b, c):
                calls.append(1)
                return BlobDetector(max_width=1000), ["ball"]         # sees blobs on tiles only
            with mock.patch.object(fp, "_detector", stub):
                self.assertEqual(dm.main(argv), 0)
                self.assertEqual(dm.main(argv), 0)                    # resume: nothing left to detect
            self.assertEqual(len(calls), 1)
            m = "TRAIN_M002_EASTERN_SUBURBS"
            rep = json.loads((tmp / "out" / f"{m}.mine.json").read_text())
            self.assertEqual((rep["windows"], rep["frames_analysed"], rep["untiled_seen"]), (1, 60, 0))
            kinds = [o["kind"] for o in rep["objects"]]
            self.assertEqual(sorted(kinds), ["tiled_only", "tiled_only"])   # the ball and the look-alike: tiles only
            self.assertTrue(all(Path(p).is_file() for p in rep["contact_sheets"]["tiled_only"]))
            review = tmp / "out" / f"{m}.review.csv"
            lines = review.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            lines[1] = lines[1].rstrip(",") + ",match ball,"
            lines[2] = lines[2].rstrip(",") + ",equipment/cone,"
            review.write_text("\n".join(lines) + "\n")
            text = dm.yield_report(tmp / "out", dm.load_config(mcfg), fp.load_config())
            self.assertIn("| equipment/cone | 1 |", text)
            self.assertIn("THIN: under 40 frames", text)
            self.assertIn("Not reviewed yet: 0 of 2 objects.", text)
            review.write_text(review.read_text().replace("equipment/cone", "striker"))
            with self.assertRaises(SystemExit):
                dm.yield_report(tmp / "out", dm.load_config(mcfg), fp.load_config())
            with self.assertRaisesRegex(SystemExit, "TRAIN matches only"):
                dm.main([x if x != m else "DEV_G1_SYDNEY_UNITED_APIA" for x in argv])
            with self.assertRaisesRegex(SystemExit, "tiling config"):
                dm.main(argv[:6] + ["config/default.yaml"] + argv[7:-3] + [str(tmp / "out2"), "--mine-config", str(mcfg)])


if __name__ == "__main__":
    unittest.main()
