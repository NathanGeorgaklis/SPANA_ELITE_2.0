"""Local detector preprocess (records/identity_speed_2026-11-20): the per-plane in-place version must give the same
bits as the plain formula it replaced, on any frame size / resolution, deterministically, with no torch needed."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from spana.detection.rfdetr_local import MEAN, STD, preprocess


def reference(frame_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """The formula before 2026-11-20, kept as the fixture the fast path must reproduce exactly."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1)[None])


class Preprocess(unittest.TestCase):
    def test_bit_identical_to_reference(self):
        rng = np.random.default_rng(7)
        for (h, w), res in [((720, 1280), 1280), ((360, 640), 1280), ((1080, 1920), 576), ((37, 53), 32)]:
            with self.subTest(size=(h, w), res=res):
                img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                got, ref = preprocess(img, res), reference(img, res)
                self.assertEqual((got.dtype, got.shape), (ref.dtype, ref.shape))
                self.assertTrue(got.flags["C_CONTIGUOUS"])
                self.assertEqual(got.tobytes(), ref.tobytes())

    def test_extremes_and_determinism(self):
        for v in (0, 255):
            img = np.full((48, 64, 3), v, np.uint8)
            self.assertEqual(preprocess(img, 32).tobytes(), reference(img, 32).tobytes())
        img = np.random.default_rng(1).integers(0, 256, (90, 160, 3), dtype=np.uint8)
        self.assertEqual(preprocess(img, 64).tobytes(), preprocess(img, 64).tobytes())

    def test_input_not_modified(self):
        img = np.random.default_rng(2).integers(0, 256, (40, 60, 3), dtype=np.uint8)
        before = img.copy()
        preprocess(img, 32)
        np.testing.assert_array_equal(img, before)


if __name__ == "__main__":
    unittest.main()
