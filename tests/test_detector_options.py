"""tools/detector_options.py (DEV speed options) + config `performance.option_*` + non-square preprocess."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from spana.config import ConfigError, config_from_dict, load_config
from spana.detection.rfdetr_local import MEAN, STD, LocalModelRecord, preprocess
from tests.synth import CONFIG_PATH
from tools import detector_options as opt


def _summary(sw, unflagged, recall, gt=100, transfers=0, frags=0):
    return {"summary": {"id_switches": sw, "unflagged_transfers": unflagged, "detection_recall": recall,
                        "gt_boxes": gt, "identity_transfers": transfers, "fragments_total": frags,
                        "track_coverage": 0.9, "missed_players": 1}}


def _target(wrong, correct=0.8, followed=100.0):
    return {"clips": [{"pooled": {"wrong": wrong, "correct_time": correct, "followed_s": followed}}]}


class Config(unittest.TestCase):
    def test_defaults_and_fail_closed(self):
        perf = load_config(CONFIG_PATH).performance
        self.assertGreater(perf.option_realtime_factor_max, 0)
        raw = yaml.safe_load(Path(CONFIG_PATH).read_text())
        for key, bad in (("option_id_switches_max_rise", 1.5), ("option_wrong_frames_max_increase", -1),
                         ("option_recall_max_drop", -0.1)):
            r = yaml.safe_load(yaml.safe_dump(raw))
            r["performance"][key] = bad
            with self.subTest(key=key), self.assertRaises(ConfigError):
                config_from_dict(r)
        del raw["performance"]["option_realtime_factor_max"]
        with self.assertRaises(ConfigError):
            config_from_dict(raw)


class NonSquarePreprocess(unittest.TestCase):
    def test_shape_and_same_ops(self):
        img = np.random.default_rng(4).integers(0, 256, (72, 128, 3), dtype=np.uint8)
        got = preprocess(img, 128, 64)
        self.assertEqual(got.shape, (1, 3, 64, 128))
        rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (128, 64), interpolation=cv2.INTER_LINEAR)
        ref = np.ascontiguousarray((((rgb.astype(np.float32) / 255.0) - MEAN) / STD).transpose(2, 0, 1)[None])
        self.assertEqual(got.tobytes(), ref.tobytes())
        self.assertEqual(preprocess(img, 64, 64).tobytes(), preprocess(img, 64).tobytes())


class Variants(unittest.TestCase):
    def test_parse(self):
        v = opt.parse_variant("medium:1280x736")
        self.assertEqual((v.model, v.width, v.height, v.name), ("medium", 1280, 736, "medium:1280x736"))
        for bad in ("medium:1000x1000", "medium", "medium:960", "medium:0x960"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                opt.parse_variant(bad)


class Agreement(unittest.TestCase):
    def test_reproduced_by_height_and_class_group(self):
        ref = [[{"class": "player", "box": [0, 0, 10, 100], "score": 0.9},
                {"class": "player", "box": [50, 0, 60, 30], "score": 0.9},
                {"class": "player", "box": [80, 0, 90, 30], "score": 0.2}]]          # below score: not counted
        got = [[{"class": "referee", "box": [0, 0, 10, 98], "score": 0.8},           # another person class counts
                {"class": "ball", "box": [50, 0, 60, 30], "score": 0.9}]]            # other group: no match
        r = opt.reproduced(ref, got, ("player", "referee"), 0.5, 0.5)
        self.assertEqual(r["all"], {"ref_boxes": 2, "reproduced": 1, "share": 0.5})
        self.assertEqual(r["96-inf"]["reproduced"], 1)
        self.assertEqual(r["24-48"]["reproduced"], 0)

    def test_persons_found(self):
        frames = [{"gt": [{"class": "goalkeeper", "box": [0, 0, 10, 10]}, {"class": "ball", "box": [0, 0, 2, 2]}],
                   "pred": [{"class": "player", "box": [0, 0, 10, 10], "score": 0.6}]}]
        self.assertEqual(opt.persons_found(frames, ("player", "goalkeeper"), 0.5, 0.5),
                         {"labels": 1, "found": 1, "share": 1.0})


class Judge(unittest.TestCase):
    def setUp(self):
        self.perf = load_config(CONFIG_PATH).performance
        self.base = {"A": _summary(20, 0, 0.98), "B": _summary(20, 1, 0.96)}

    def test_equal_is_viable(self):
        j = opt.judge(self.base, self.base, _target(1), _target(1), self.perf)
        self.assertTrue(j["viable"])

    def test_each_limit_fails_closed(self):
        cases = {
            "wrong_frames_reviewer": (self.base, _target(2)),
            "unflagged_transfers": ({"A": _summary(20, 1, 0.98), "B": _summary(20, 1, 0.96)}, _target(1)),
            "id_switches": ({"A": _summary(22, 0, 0.98), "B": _summary(21, 1, 0.96)}, _target(1)),   # +7.5 %
            "detection_recall": ({"A": _summary(20, 0, 0.955), "B": _summary(20, 1, 0.96)}, _target(1)),  # -1.25 pt pooled
        }
        for name, (cand, tc) in cases.items():
            with self.subTest(check=name):
                j = opt.judge(self.base, cand, _target(1), tc, self.perf)
                self.assertFalse(j["checks"][name]["ok"])
                self.assertFalse(j["viable"])

    def test_within_limits(self):
        cand = {"A": _summary(21, 0, 0.975), "B": _summary(20, 1, 0.955)}            # +2.5 % switches, -0.5 pt
        self.assertTrue(opt.judge(self.base, cand, _target(1), _target(0), self.perf)["viable"])

    def test_clip_sets_must_match(self):
        with self.assertRaises(SystemExit):
            opt.judge(self.base, {"A": self.base["A"]}, _target(1), _target(1), self.perf)


class Record(unittest.TestCase):
    def test_dev_record_points_at_the_same_weights(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "m").mkdir()
            w, o = td / "m" / "w.pt", td / "m" / "x.onnx"
            w.write_bytes(b"weights")
            o.write_bytes(b"onnx")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            src = td / "m.json"
            src.write_text(json.dumps({
                "name": "n", "source": "s", "version": "v1", "licence": "l", "variant": "medium", "num_classes": 1,
                "resolution": 1280, "weights_path": "m/w.pt", "weights_sha256": sha(w), "onnx_path": "m/x.onnx",
                "onnx_sha256": sha(o), "output_classes": ["", "player"], "validation": "x"}))
            out = td / "runs" / "dev" / "r960.json"
            self.assertEqual(opt.main(["record", "--model-record", str(src), "--width", "960", "--height", "960",
                                       "--out", str(out)]), 0)
            rec = LocalModelRecord.load(out)
            self.assertEqual((rec.resolution, rec.weights_sha256), (960, sha(w)))
            self.assertEqual(rec.file("weights").resolve(), w.resolve())
            with self.assertRaises(SystemExit):                      # never overwritten
                opt.main(["record", "--model-record", str(src), "--width", "960", "--height", "960", "--out", str(out)])
            with self.assertRaises(SystemExit):                      # the product record is square only
                opt.main(["record", "--model-record", str(src), "--width", "1280", "--height", "736",
                          "--out", str(td / "x.json")])


if __name__ == "__main__":
    unittest.main()
