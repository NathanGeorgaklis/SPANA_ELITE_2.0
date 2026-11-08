"""Regression for the real DEV finding (FRESH_05): a camera estimate supported by few features followed the wrong
motion, was applied to every track as exact, and the Mahalanobis gate then rejected true continuations.
The estimator now reports its uncertainty from feature disagreement; tracks add it to their position variance."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from spana.config import load_config
from spana.records import CameraObservation, CameraStatus
from spana.tracking.camera import CameraMotionEstimator
from spana.tracking.kalman import BoxKalman
from tests.synth import Player, kit, run
from tests.test_detection_camera import _texture
from tests.metrics import evaluate
from tests.test_tracker import H, assert_safe

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")


class CameraDisagreement(unittest.TestCase):
    def test_consistent_motion_has_negligible_uncertainty(self):
        big = _texture(1200, 2200)
        est = CameraMotionEstimator(CFG.camera, 960, 540, 25.0)
        est.observe(big[100:640, 100:1060], [])
        obs = est.observe(big[106:646, 120:1080], [])
        self.assertEqual(obs.status, CameraStatus.OK)
        self.assertLess(obs.motion_std_px, 1.0)

    def test_two_conflicting_motions_report_large_uncertainty(self):
        big = _texture(1200, 2200)
        a = big[100:640, 100:1060].copy()
        b = a.copy()
        b[:, :560] = big[100:640, 120:680]          # most of the frame pans 20 px; the rest does not move
        est = CameraMotionEstimator(CFG.camera, 960, 540, 25.0)
        est.observe(a, [])
        obs = est.observe(b, [])
        self.assertEqual(obs.status, CameraStatus.OK)
        self.assertGreater(obs.motion_std_px, 5.0)

    def test_identity_observations_carry_no_uncertainty(self):
        self.assertEqual(CameraObservation.identity(CameraStatus.OK).motion_std_px, 0.0)


class KalmanCameraNoise(unittest.TestCase):
    def test_camera_std_adds_position_variance_only(self):
        a, b = BoxKalman(np.array([500., 400., 30., 80.]), CFG.kalman), BoxKalman(np.array([500., 400., 30., 80.]),
                                                                                CFG.kalman)
        A = np.array([[1., 0., -5.], [0., 1., 2.]])
        a.apply_camera(A)
        b.apply_camera(A, 4.0)
        np.testing.assert_array_equal(a.x, b.x)
        d = b.P - a.P
        self.assertAlmostEqual(d[0, 0], 16.0)
        self.assertAlmostEqual(d[1, 1], 16.0)
        d[0, 0] = d[1, 1] = 0.0
        self.assertFalse(np.any(d))


class WeakCameraTracking(unittest.TestCase):
    PAN = 0.75 * H                      # true camera pan per frame
    ERR = 0.30 * H                      # error of a weak estimate (inside the centre gate, outside the old chi2 gate)

    def _run(self, report_std: bool):
        def cam(i):
            if i == 0:
                return CameraObservation.identity(CameraStatus.FIRST)
            err = self.ERR if i % 5 == 0 else 0.0        # every 5th estimate is weak and wrong
            return CameraObservation(CameraStatus.OK, ((1, 0, -self.PAN + err), (0, 1, 0)), 30, 0.01,
                                     err if report_std else 0.0)
        ps = [Player("A", (300, 500), (0, 0), appearance=kit(0)),
              Player("B", (300 + 1.2 * H, 500), (0, 0), appearance=kit(1))]
        return run(ps, fps=30, duration_s=2.0, camera=cam, frame_w=40000,
                   world_to_image=lambda i, b: b - np.array([1, 0, 1, 0]) * self.PAN * i)

    def test_weak_estimate_widens_gate_and_keeps_identity(self):
        m = assert_safe(self, self._run(report_std=True))
        self.assertEqual(m.total_fragments, 0)

    def test_without_reported_uncertainty_the_same_scene_fragments(self):
        """Guard: proves the scenario exercises the gate (the pre-fix behaviour)."""
        self.assertGreater(evaluate(self._run(report_std=False)).total_fragments, 0)


if __name__ == "__main__":
    unittest.main()
