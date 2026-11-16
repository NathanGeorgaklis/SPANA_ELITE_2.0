"""Local RF-DETR backend: model record checks, pre/post-processing, ONNX runner end to end through `detect --backend
local` and `track`, refusals. One integration test runs the real pinned weights (torch vs ONNX) when present."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from spana import cli
from spana.artifacts import read_json, validate_detections, validate_tracks
from spana.config import load_config
from spana.detection.rfdetr_local import (LocalModelError, LocalModelRecord, LocalRFDETRDetector, decode, make_runner,
                                          preprocess)
from tests.synth import CONFIG_PATH
from tests.test_cli_e2e import write_video

REPO = Path(__file__).resolve().parents[1]
CFG = load_config(CONFIG_PATH)
CLASSES = ("player", "goalkeeper", "referee", "ball")
OUTPUTS = ["", "ball", "goalkeeper", "player", "referee"]


def logit(p):
    return float(np.log(p / (1 - p)))


def tiny_onnx(path: Path, res: int, boxes: np.ndarray, logits: np.ndarray) -> None:
    """A graph with the exported model's interface: input (1, 3, res, res) -> constant dets / labels."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, res, res])
    outs = [helper.make_tensor_value_info("dets", TensorProto.FLOAT, list(boxes.shape)),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, list(logits.shape))]
    nodes = [helper.make_node("Constant", [], ["dets"], value=helper.make_tensor("d", TensorProto.FLOAT, boxes.shape,
                                                                                   boxes.ravel().tolist())),
             helper.make_node("Constant", [], ["labels"], value=helper.make_tensor("l", TensorProto.FLOAT, logits.shape,
                                                                                     logits.ravel().tolist()))]
    m = helper.make_model(helper.make_graph(nodes, "tiny", [inp], outs), opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


def write_record(d: Path, onnx_path: Path, res: int, **over) -> Path:
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (d / "w.pt").write_bytes(b"not used by the onnx runtime")
    rec = {"name": "tiny", "source": "test", "version": "t1", "licence": "test", "variant": "medium", "num_classes": 4,
           "resolution": res, "weights_path": "w.pt", "weights_sha256": sha(d / "w.pt"),
           "onnx_path": onnx_path.name, "onnx_sha256": sha(onnx_path), "output_classes": OUTPUTS,
           "validation": "test", **over}
    p = d / "rec.json"
    p.write_text(json.dumps(rec))
    return p


# one player (output 3) twice as referee (output 4, weaker, same place), one ball (output 1), one weak query
BOXES = np.zeros((1, 6, 4), np.float32)
BOXES[0, 0] = BOXES[0, 1] = [0.5, 0.5, 0.0625, 0.28]          # 20 x 50 px centred in a 320 x 180 frame
BOXES[0, 2] = [0.2, 0.83, 0.02, 0.033]
BOXES[0, 3] = [0.9, 0.5, 0.05, 0.2]
LOGITS = np.full((1, 6, 5), -20.0, np.float32)
LOGITS[0, 0, 3] = logit(0.9)
LOGITS[0, 1, 4] = logit(0.3)
LOGITS[0, 2, 1] = logit(0.8)
LOGITS[0, 3, 3] = logit(0.05)                                   # below detection.score_low


class Units(unittest.TestCase):
    def test_preprocess(self):
        img = np.zeros((90, 160, 3), np.uint8)
        img[..., 2] = 255                                          # pure red (BGR)
        x = preprocess(img, 32)
        self.assertEqual(x.shape, (1, 3, 32, 32))
        np.testing.assert_allclose(x[0, :, 0, 0], ((np.array([1, 0, 0]) - [0.485, 0.456, 0.406])
                                                   / [0.229, 0.224, 0.225]), rtol=1e-5)

    def test_decode_maps_outputs_and_drops_unused(self):
        logits = np.full((2, 5), -20.0)
        logits[0, 0] = logit(0.99)                                 # output 0 is unused: never a detection
        logits[1, 2] = logit(0.7)                                  # goalkeeper
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.98, 0.5, 0.1, 0.4]])
        mapping = [-1, 3, 1, 0, 2]
        out = decode(boxes, logits, 100, 50, 0.1, mapping)
        self.assertEqual([(d.class_id, round(d.score, 3)) for d in out], [(1, 0.7)])
        self.assertEqual(out[0].box[2], 100.0)                     # clipped to the frame

    def test_record_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            tiny_onnx(d / "m.onnx", 32, BOXES, LOGITS)
            rec = LocalModelRecord.load(write_record(d, d / "m.onnx", 32))
            self.assertEqual(rec.file("onnx").resolve(), (d / "m.onnx").resolve())
            (d / "m.onnx").write_bytes(b"tampered")
            with self.assertRaisesRegex(LocalModelError, "sha256"):
                rec.file("onnx")
            with self.assertRaisesRegex(LocalModelError, "num_classes \\+ 1"):
                LocalModelRecord.load(write_record(d, d / "m.onnx", 32, output_classes=OUTPUTS[1:]))
            with self.assertRaisesRegex(LocalModelError, "each class once"):
                LocalModelRecord.load(write_record(d, d / "m.onnx", 32, output_classes=["", "ball", "ball", "player",
                                                                                         "referee"]))
            with self.assertRaisesRegex(LocalModelError, "device 'cpu' only"):
                make_runner(LocalModelRecord.load(write_record(d, d / "m.onnx", 32)), "onnx", "mps")


class CliLocalBackend(unittest.TestCase):
    def test_detect_then_track(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_video(d / "clip.avi", 6, 25.0)
            tiny_onnx(d / "m.onnx", 32, BOXES, LOGITS)
            rec = write_record(d, d / "m.onnx", 32)
            args = ["detect", "--backend", "local", "--video", str(d / "clip.avi"), "--config", str(CONFIG_PATH),
                    "--dataset-config", str(REPO / "config" / "roboflow.yaml"), "--model-record", str(rec),
                    "--runtime", "onnx", "--device", "cpu", "--out", str(d / "d.json")]
            self.assertEqual(cli.main(args), 0)
            ddoc = read_json(d / "d.json")
            validate_detections(ddoc)
            self.assertEqual(ddoc["class_names"], list(CLASSES))
            self.assertEqual(ddoc["model"]["backend"], "rfdetr_local")
            self.assertEqual(ddoc["model"]["confidence"], CFG.detection.score_low)
            for fr in ddoc["frames"]:                               # ball dropped, referee duplicate merged
                self.assertEqual([(r[5], r[4]) for r in fr["detections"]], [(0, 0.9)])
            self.assertEqual(cli.main(["track", "--video", str(d / "clip.avi"), "--config", str(CONFIG_PATH),
                                       "--detections", str(d / "d.json"), "--out", str(d / "t.json")]), 0)
            validate_tracks(read_json(d / "t.json"), ddoc)

    def test_missing_arguments_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(Exception, "needs --model-record, --dataset-config, --runtime and --device"):
                cli.main(["detect", "--backend", "local", "--video", "x.avi", "--config", str(CONFIG_PATH),
                          "--out", str(Path(td) / "d.json")])


REAL = REPO / "models" / "rfdetr_v1.json"


@unittest.skipUnless((REPO / "models" / "rfdetr_v1" / "weights.pt").is_file()
                     and (REPO / "models" / "rfdetr_v1" / "rfdetr-medium-1280.onnx").is_file(),
                     "pinned detector v1 weights / ONNX export not present (models/rfdetr_v1.json)")
class RealWeights(unittest.TestCase):
    def test_torch_and_onnx_agree(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed (requirements-local-detector.txt)")
        rec = LocalModelRecord.load(REAL)
        frame = (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)
        x = preprocess(frame, rec.resolution)
        b1, l1 = make_runner(rec, "onnx", "cpu")(x)
        b2, l2 = make_runner(rec, "torch", "cpu")(x)
        self.assertLess(float(np.abs(b1 - b2).max()), 1e-3)
        self.assertLess(float(np.abs(l1 - l2).max()), 1e-2)
        det = LocalRFDETRDetector(rec, make_runner(rec, "onnx", "cpu"), CLASSES, 0.5)
        self.assertEqual(det.class_names, CLASSES)


if __name__ == "__main__":
    unittest.main()
