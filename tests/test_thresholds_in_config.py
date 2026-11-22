"""Values that were hard-coded in player tracking now come from config (schema 2.17.0), with the same defaults:
camera.cut_hist_hue_bins / cut_hist_sat_bins (cut-test histogram, was [32, 32]) and
tracking.confirm_min_observed_frames (was `observed_frames >= 2`)."""
from __future__ import annotations

import copy
import dataclasses
import unittest
from pathlib import Path

import cv2
import yaml

from spana.config import ConfigError, config_from_dict
from spana.tracking.camera import CameraMotionEstimator
from tests.synth import Player, cfg, run
from tests.test_detection_camera import _texture

ROOT = Path(__file__).resolve().parents[1]
RAW = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))


def with_keys(**sections) -> dict:
    d = copy.deepcopy(RAW)
    for sec, kv in sections.items():
        for k, v in kv.items():
            if v is None:
                d[sec].pop(k)
            else:
                d[sec][k] = v
    return d


class ConfigKeys(unittest.TestCase):
    def test_defaults_keep_the_old_values(self):
        c = cfg()
        self.assertEqual((c.camera.cut_hist_hue_bins, c.camera.cut_hist_sat_bins), (32, 32))
        self.assertEqual(c.tracking.confirm_min_observed_frames, 2)

    def test_missing_or_invalid_keys_are_refused(self):
        for sec, key in (("camera", "cut_hist_hue_bins"), ("camera", "cut_hist_sat_bins"),
                         ("tracking", "confirm_min_observed_frames")):
            for bad in (None, 0, 2.5, True):
                with self.subTest(key=key, value=bad), self.assertRaises(ConfigError):
                    config_from_dict(with_keys(**{sec: {key: bad}}))


class CutHistogramBins(unittest.TestCase):
    def test_estimator_uses_the_configured_bins(self):
        a = _texture(540, 960, seed=1)
        b = cv2.convertScaleAbs(a, alpha=0.9, beta=12)                 # same view, slightly different exposure
        dist = {}
        for bins in ((32, 32), (8, 8)):
            c = dataclasses.replace(cfg().camera, cut_hist_hue_bins=bins[0], cut_hist_sat_bins=bins[1])
            est = CameraMotionEstimator(c, 960, 540, 25.0)
            est.observe(a, [])
            dist[bins] = est.observe(b, []).hist_distance
        self.assertNotAlmostEqual(dist[(32, 32)], dist[(8, 8)], places=6)


class ConfirmMinObservedFrames(unittest.TestCase):
    def first_confirmed_frame(self, n: int) -> int:
        base = cfg()
        c = dataclasses.replace(base, tracking=dataclasses.replace(base.tracking, confirm_min_observed_frames=n))
        res = run([Player("A", (300, 500), (100, 0))], fps=25, duration_s=1, config=c)
        return next(i for i, ft in enumerate(res.frames) if ft.tracks and ft.tracks[0].confirmed)

    def test_frame_floor_is_read_from_config(self):
        # default: confirm_s (0.10 s = 2.5 frames at 25 fps) binds -> confirmed on frame 3 (4 observed frames)
        self.assertEqual(self.first_confirmed_frame(2), 3)
        self.assertEqual(self.first_confirmed_frame(4), 3)
        self.assertEqual(self.first_confirmed_frame(6), 5)            # the frame floor binds


if __name__ == "__main__":
    unittest.main()
