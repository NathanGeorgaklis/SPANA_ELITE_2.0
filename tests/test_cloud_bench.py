"""tools/cloud_bench.py: config fail-closed (what may leave the machine), upload refusal, decode hashes, projection."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from spana.dataset.config import LicensedMatch
from tools import cloud_bench as cb

REPO = Path(__file__).resolve().parents[1]
LIC = (LicensedMatch("DEV_A", "dev", "p", True), LicensedMatch("DEV_NOTP", "dev", "p", False),
       LicensedMatch("EVAL_X", "eval", "p", True), LicensedMatch("TRAIN_T", "train", "p", True))


def _raw():
    return {"schema_version": cb.SCHEMA_VERSION,
            "modal": {"app_name": "a", "volume_name": "v", "python_version": "3.13", "gpu": "L4", "max_gpus": 4,
                      "timeout_s": 60, "confirmed_matches": ["DEV_A"]},
            "footage": [{"name": "clip", "match_id": "DEV_A", "file": "clip.mp4", "sha256": "a" * 64}],
            "suites": {"s": [{"unit": "W", "footage": "clip", "start": 10, "stop": 20, "labels": None},
                             {"unit": "L", "footage": "clip", "start": 0, "stop": None, "labels": "data/labels/x/v1"}]},
            "baseline": {"window_identity": "a", "window_tracks": "b", "labelled_dir": "c", "labelled_tag": "base"},
            "decode_check_frames": 5}


class Config(unittest.TestCase):
    def test_repo_config_loads(self):
        cfg = cb.load_bench_config(REPO / "config" / "cloud_bench.yaml", REPO / "config" / "roboflow.yaml")
        self.assertTrue(cfg.footage and all(f.match_id in cfg.confirmed_matches for f in cfg.footage))
        self.assertIn("dev_standard", cfg.suites)

    def test_valid(self):
        cfg = cb.parse_config(_raw(), LIC)
        self.assertEqual([u.unit for u in cfg.suites["s"]], ["W", "L"])

    def test_only_licensed_dev_with_third_party_processing(self):
        for mid in ("EVAL_X", "TRAIN_T", "DEV_NOTP", "UNLISTED"):
            r = _raw()
            r["modal"]["confirmed_matches"] = [mid]
            r["footage"][0]["match_id"] = mid
            with self.subTest(match=mid), self.assertRaises(cb.BenchConfigError):
                cb.parse_config(r, LIC)

    def test_footage_must_belong_to_a_confirmed_match(self):
        r = _raw()
        r["footage"][0]["match_id"] = "DEV_NOTP"
        with self.assertRaises(cb.BenchConfigError):
            cb.parse_config(r, LIC)

    def test_fail_closed_shapes(self):
        bad = []
        r = _raw(); r["modal"]["extra"] = 1; bad.append(r)                                   # unknown key
        r = _raw(); del r["decode_check_frames"]; bad.append(r)                              # missing key
        r = _raw(); r["schema_version"] = "0.9"; bad.append(r)
        r = _raw(); r["footage"][0]["sha256"] = "abc"; bad.append(r)
        r = _raw(); r["footage"][0]["file"] = "../x/clip.mp4"; bad.append(r)
        r = _raw(); r["suites"]["s"][0]["footage"] = "other"; bad.append(r)                  # unlisted footage
        r = _raw(); r["suites"]["s"][0]["stop"] = 5; bad.append(r)                           # stop <= start
        r = _raw(); r["suites"]["s"][1]["labels"] = None; bad.append(r)                      # two windows
        r = _raw(); r["modal"]["max_gpus"] = 0; bad.append(r)
        for i, r in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(cb.BenchConfigError):
                cb.parse_config(r, LIC)


class Upload(unittest.TestCase):
    def test_by_content_only(self):
        cfg = cb.parse_config(_raw(), LIC)
        self.assertEqual(cb.footage_for_upload(cfg, Path("/x/clip.mp4"), "a" * 64).name, "clip")
        with self.assertRaises(cb.BenchConfigError):                      # unlisted content (e.g. EVAL, Game 3)
            cb.footage_for_upload(cfg, Path("/x/clip.mp4"), "b" * 64)
        with self.assertRaises(cb.BenchConfigError):                      # listed content under another name
            cb.footage_for_upload(cfg, Path("/x/other.mp4"), "a" * 64)


class Helpers(unittest.TestCase):
    def test_projection(self):
        p = cb.projection(det_ms=80.0, trk_ms=10.0, fps=30.0, footage_s=5400.0, max_gpus=4)
        self.assertEqual(p["1"], 243.0)                    # 90 ms x 162000 frames
        self.assertEqual(p["4"], 81.0)                     # (20 + 10) ms x 162000
        self.assertEqual(sorted(p), ["1", "2", "3", "4"])

    def test_dirty_code_detection(self):
        st = " M spana/cli.py\nM  tools/x.py\n M docs/STATUS.md\n M models/rfdetr_v1.json\n"
        self.assertEqual(cb.tracked_code_dirty(st), ["spana/cli.py", "tools/x.py", "models/rfdetr_v1.json"])
        self.assertEqual(cb.tracked_code_dirty(" M docs/STATUS.md\n"), [])

    def test_frame_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "v.avi")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
            rng = np.random.default_rng(1)
            for _ in range(8):
                w.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
            w.release()
            h = cb.frame_hashes(path, 2, 3)
            self.assertEqual(len(h), 3)
            self.assertEqual(h, cb.frame_hashes(path, 2, 3))
            self.assertEqual(h[1:], cb.frame_hashes(path, 3, 2))
            self.assertNotEqual(h[0], h[1])


if __name__ == "__main__":
    unittest.main()
