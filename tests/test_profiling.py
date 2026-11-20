"""Per-stage profiling (config `profiling`): off by default, and timing never changes what a pass computes."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from spana import cli
from spana.artifacts import read_json
from spana.config import ConfigError, config_from_dict, load_config
from spana.profiling import StageTimer
from tests.synth import CONFIG_PATH
from tests import test_cli_e2e as e2e


class Timer(unittest.TestCase):
    def test_disabled_is_a_no_op(self):
        t = StageTimer(False)
        with t.stage("x"):
            pass
        self.assertEqual(list(t.timed_iter(iter([1, 2, 3]), "decode", "seek")), [1, 2, 3])
        self.assertEqual(t.report("p")["stages"], [])

    def test_enabled_records_stages_and_first_step_apart(self):
        t = StageTimer(True)
        for _ in t.timed_iter(iter([1, 2, 3]), "decode", "seek"):
            with t.stage("work"):
                pass
            t.frames += 1
        r = t.report("p", realtime_fps=30.0)
        calls = {s["stage"]: s["calls"] for s in r["stages"]}
        self.assertEqual(calls, {"seek": 1, "decode": 3, "work": 3})   # decode: 2 frames + the final StopIteration
        self.assertEqual(r["frames"], 3)
        self.assertAlmostEqual(sum(s["share"] for s in r["stages"]), 1.0, places=3)
        self.assertIn("realtime_factor", r)


class Config(unittest.TestCase):
    def test_default_off_and_performance_target_in_config(self):
        cfg = load_config(CONFIG_PATH)
        self.assertFalse(cfg.profiling.enabled)
        self.assertGreater(cfg.performance.realtime_factor_target, 0)

    def test_fails_closed(self):
        raw = yaml.safe_load(Path(CONFIG_PATH).read_text())
        del raw["performance"]["realtime_factor_target"]
        with self.assertRaises(ConfigError):
            config_from_dict(raw)


class SameOutputs(unittest.TestCase):
    """detect + track with profiling on give the same frames as with it off, plus a profile per pass."""

    def setUp(self):
        e2e.EndToEnd.setUp(self)   # synthetic clip + fake pinned ONNX detector

    def _run(self, tag: str, enabled: bool):
        raw = yaml.safe_load(Path(self.config).read_text())
        raw["profiling"]["enabled"] = enabled
        cfgp = self.td / f"{tag}.yaml"
        cfgp.write_text(yaml.safe_dump(raw))
        d, t = self.td / f"{tag}.det.json", self.td / f"{tag}.trk.json"
        self.assertEqual(cli.main(["detect", "--video", str(self.video), "--config", str(cfgp), "--model-record",
                                   str(self.record), "--out", str(d)]), 0)
        self.assertEqual(cli.main(["track", "--video", str(self.video), "--config", str(cfgp), "--detections",
                                   str(d), "--out", str(t)]), 0)
        return d, t

    def test_profiling_changes_nothing(self):
        d0, t0 = self._run("off", False)
        d1, t1 = self._run("on", True)
        self.assertEqual(read_json(d0)["frames"], read_json(d1)["frames"])
        self.assertEqual(read_json(t0)["frames"], read_json(t1)["frames"])
        self.assertFalse(d0.with_suffix(".profile.json").exists())
        pd, pt = read_json(d1.with_suffix(".profile.json")), read_json(t1.with_suffix(".profile.json"))
        self.assertEqual((pd["pass"], pd["frames"]), ("detect", 20))
        self.assertEqual((pt["pass"], pt["frames"]), ("track", 20))
        self.assertTrue({"decode", "io.validate_write"} <= {s["stage"] for s in pd["stages"]})
        self.assertTrue({"decode", "camera_motion", "appearance", "tracking"} <= {s["stage"] for s in pt["stages"]})


if __name__ == "__main__":
    unittest.main()
