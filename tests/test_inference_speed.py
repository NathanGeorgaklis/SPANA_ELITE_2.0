"""Config `inference`: the local detector's GPU precision / torch.compile (records/identity_speed_2026-11-18).

Checks: the config fails closed; the CPU path ignores the GPU settings (fp32, eager: the same outputs as before);
on the Apple GPU, compile (fp32 and fp16) is deterministic (same outputs on the same input) and recorded in the model id.
Agreement with fp32 on real footage is a DEV measurement (tools/detector_speed_probe.py), not a unit test."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from spana.config import ConfigError, config_from_dict, load_config
from spana.detection.rfdetr_local import LocalModelRecord, make_runner, preprocess
from tests.synth import CONFIG_PATH

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "models" / "rfdetr_v1.json"
HAVE_WEIGHTS = (REPO / "models" / "rfdetr_v1" / "weights.pt").is_file()


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class Config(unittest.TestCase):
    def test_default_and_refusals(self):
        cfg = load_config(CONFIG_PATH)
        self.assertIn(cfg.inference.mps_precision, ("fp32", "fp16"))
        raw = yaml.safe_load(Path(CONFIG_PATH).read_text())
        raw["inference"]["mps_precision"] = "int8"
        with self.assertRaises(ConfigError):
            config_from_dict(raw)
        raw["inference"]["mps_precision"] = "fp16"
        del raw["inference"]["mps_compile"]
        with self.assertRaises(ConfigError):
            config_from_dict(raw)


def _frame():
    return (np.random.default_rng(1).random((360, 640, 3)) * 255).astype(np.uint8)


@unittest.skipUnless(HAVE_WEIGHTS and _torch() is not None, "detector v1 weights or torch not present")
class RealWeights(unittest.TestCase):
    def test_cpu_fallback_ignores_gpu_settings(self):
        rec = LocalModelRecord.load(REAL)
        x = preprocess(_frame(), rec.resolution)
        plain = make_runner(rec, "torch", "cpu")
        tuned = make_runner(rec, "torch", "cpu", "fp16", True)
        self.assertEqual((tuned.describe["precision"], tuned.describe["compiled"]), ("fp32", False))
        b1, l1 = plain(x)
        b2, l2 = tuned(x)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(l1, l2)

    @unittest.skipUnless(_torch() is not None and _torch().backends.mps.is_available(), "no Apple GPU")
    def test_gpu_compile_is_deterministic(self):
        rec = LocalModelRecord.load(REAL)
        x = preprocess(_frame(), rec.resolution)
        for precision in ("fp32", "fp16"):
            with self.subTest(precision=precision):
                run = make_runner(rec, "torch", "mps", precision, True)
                self.assertEqual((run.describe["precision"], run.describe["compiled"], run.describe["device"]),
                                 (precision, True, "mps"))
                b1, l1 = run(x)
                b2, l2 = run(x)
                np.testing.assert_array_equal(b1, b2)
                np.testing.assert_array_equal(l1, l2)
                self.assertTrue(np.isfinite(b1).all() and np.isfinite(l1).all())
                self.assertEqual(b1.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
