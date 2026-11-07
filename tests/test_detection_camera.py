"""Detection pre/post-processing, pinned ONNX backend, and camera motion / cut estimation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spana.config import load_config
from spana.detection.onnx_backend import ModelRecord, ModelRecordError, OnnxPersonDetector
from spana.detection.postprocess import filter_detections, letterbox, nms
from spana.records import CameraStatus
from spana.tracking.camera import CameraMotionEstimator
from spana.video import sha256_file
from tests.synth import CONFIG_PATH

CFG = load_config(CONFIG_PATH)


def constant_onnx(path: Path, output: np.ndarray, input_size: int) -> None:
    """A model whose output is a fixed tensor: input * 0 summed, broadcast-added to the constant."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, input_size, input_size])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, list(output.shape))
    const = numpy_helper.from_array(output.astype(np.float32), "C")
    zero = numpy_helper.from_array(np.zeros(1, np.float32), "Z")
    nodes = [helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
             helper.make_node("Mul", ["s", "Z"], ["s0"]),
             helper.make_node("Add", ["C", "s0"], ["out"])]
    g = helper.make_graph(nodes, "const", [x], [y], initializer=[const, zero])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def write_record(td: Path, weights: Path, **over) -> Path:
    rec = {"name": "test-constant", "source": "unit test", "version": "0", "sha256": sha256_file(weights),
           "licence": "test-only", "weights_path": weights.name, "input_size": 64, "color_order": "RGB",
           "scale_to_unit": True, "pad_value": 114, "output_format": "cxcywh_class_scores",
           "output_layout": "channels_first", "person_class_id": 0}
    rec.update(over)
    p = td / "record.json"
    p.write_text(json.dumps(rec))
    return p


class PostProcess(unittest.TestCase):
    def test_letterbox_round_trip(self):
        img = np.zeros((360, 640, 3), np.uint8)
        _, lb = letterbox(img, 320, 114)
        src = np.array([[100.0, 50.0, 300.0, 350.0]])
        model = src * lb.scale + np.array([lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y])
        np.testing.assert_allclose(lb.to_source(model), src, atol=1e-6)

    def test_nms_deterministic(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
        self.assertEqual(nms(boxes, np.array([0.5, 0.5, 0.4]), 0.5), [0, 2])

    def test_filter_size_class_score(self):
        boxes = np.array([[0, 0, 10, 100], [0, 0, 10, 5], [200, 0, 210, 100], [400, 0, 410, 100]], float)
        out = filter_detections(boxes, np.array([0.9, 0.9, 0.05, 0.9]), np.array([0, 0, 0, 3]), frame_w=1920,
                                frame_h=1080, cfg=CFG.detection, keep_class=0)
        self.assertEqual([d.box for d in out], [(0.0, 0.0, 10.0, 100.0)])


class OnnxBackend(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        # Two candidates (channels-first: 4 coords + 2 classes). Candidate 0 = person in model px (cx=32, cy=32).
        out = np.zeros((1, 6, 2), np.float32)
        out[0, :, 0] = [32, 32, 8, 20, 0.9, 0.1]
        out[0, :, 1] = [10, 10, 4, 4, 0.1, 0.8]  # class 1, filtered out
        self.weights = self.td / "m.onnx"
        constant_onnx(self.weights, out, 64)

    def test_decodes_into_source_pixels(self):
        det = OnnxPersonDetector(ModelRecord.load(write_record(self.td, self.weights)), CFG.detection)
        frame = np.zeros((360, 640, 3), np.uint8)    # scale 0.1, pad_y 14: y = (22 - 14) / 0.1 = 80, (42 - 14) / 0.1 = 280
        ds = det.detect(frame)
        self.assertEqual(len(ds), 1)
        np.testing.assert_allclose(ds[0].box, [280.0, 80.0, 360.0, 280.0], atol=1e-4)
        self.assertAlmostEqual(ds[0].score, 0.9, places=5)

    def test_record_fails_closed(self):
        bad = [dict(sha256="0" * 64), dict(licence=" "), dict(output_layout="guess"), dict(output_format="x"),
               dict(weights_path="missing.onnx")]
        for over in bad:
            with self.assertRaises(ModelRecordError, msg=str(over)):
                ModelRecord.load(write_record(self.td, self.weights, **over))
        p = write_record(self.td, self.weights)
        d = json.loads(p.read_text()); del d["licence"]; p.write_text(json.dumps(d))
        with self.assertRaises(ModelRecordError):
            ModelRecord.load(p)


def _texture(h=540, w=960, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


class CameraMotion(unittest.TestCase):
    def test_translation_recovered_in_source_pixels(self):
        big = _texture(1200, 2200)
        a, b = big[100:640, 100:1060], big[100 + 6:640 + 6, 100 + 20:1060 + 20]   # camera moves right/down
        est = CameraMotionEstimator(CFG.camera, 960, 540, 25.0)
        self.assertEqual(est.observe(a, []).status, CameraStatus.FIRST)
        obs = est.observe(b, [])
        self.assertEqual(obs.status, CameraStatus.OK)
        A = obs.matrix()
        self.assertAlmostEqual(A[0, 2], -20.0, delta=1.5)
        self.assertAlmostEqual(A[1, 2], -6.0, delta=1.5)

    def test_moving_players_masked_out(self):
        big = _texture(1200, 2200)
        a, b = big[100:640, 100:1060].copy(), big[100:640, 100:1060].copy()   # static camera
        player = _texture(200, 80, seed=5)
        a[200:400, 300:380] = player
        b[200:400, 340:420] = player                        # the player moves 40 px; the background does not
        est = CameraMotionEstimator(CFG.camera, 960, 540, 25.0)
        est.observe(a, [(300, 200, 380, 400)])
        obs = est.observe(b, [(340, 200, 420, 400)])
        self.assertEqual(obs.status, CameraStatus.OK)
        self.assertLess(abs(obs.matrix()[0, 2]), 1.0)

    def test_cut_and_unknown(self):
        est = CameraMotionEstimator(CFG.camera, 960, 540, 25.0)
        est.observe(_texture(seed=1), [])
        green = np.zeros((540, 960, 3), np.uint8); green[:, :, 1] = 200
        self.assertEqual(est.observe(green, []).status, CameraStatus.CUT)
        self.assertEqual(est.observe(green.copy(), []).status, CameraStatus.UNKNOWN)   # textureless


if __name__ == "__main__":
    unittest.main()
