"""spana.identity.reid end to end on a synthetic clip: two players (red and blue shirts) tracked by the real tracker,
embedded by a tiny ONNX model (mean colour per channel) through the product's crop / record / document path, and the
`embed` CLI. The model record pins the weights by sha256 (wrong hash refused)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spana.artifacts import TRACKS_SCHEMA, read_json, tracklet_rows, tracks_frame, validate_track_embeddings
from spana.config import load_config
from spana.identity.reid import (ReidEmbedder, ReidModelError, ReidModelRecord, cosine, embeddings_document)
from spana.records import CameraObservation, CameraStatus, Detection
from spana.tracking.tracker import MultiPlayerTracker
from spana.video import VideoSource, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
N, FPS = 30, 25.0
RED, BLUE = (0, 0, 220), (220, 0, 0)         # BGR


def boxes(i):
    return [(40.0 + i, 40.0, 80.0 + i, 140.0), (200.0 - i, 40.0, 240.0 - i, 140.0)]


def write_video(path: Path) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 180))
    for i in range(N):
        img = np.full((180, 320, 3), 90, np.uint8)
        for (x1, y1, x2, y2), col in zip(boxes(i), (RED, BLUE)):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, -1)
        vw.write(img)
    vw.release()


def mean_colour_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["n", 3, 224, 112])
    y = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["n", 3])
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], [2, 3])
    g = helper.make_graph([helper.make_node("ReduceMean", ["pixel_values", "axes"], ["m"], keepdims=0),
                           helper.make_node("Identity", ["m"], ["embedding"])], "mean", [x], [y], [axes])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=8), str(path))


def write_record(d: Path, onnx_path: Path, sha: str = None) -> Path:
    rec = {"name": "test mean colour", "source": "tests", "version": "t1", "licence": "test", "onnx_path": onnx_path.name,
           "onnx_sha256": sha or sha256_file(onnx_path), "input_size_hw": [224, 112], "crop_pad": 0.05,
           "color_order": "rgb", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "embedding_dim": 3,
           "validation": "unit test"}
    p = d / "reid.json"
    p.write_text(json.dumps(rec))
    return p


def track(video: Path) -> dict:
    src = VideoSource(str(video))
    trk = MultiPlayerTracker(CFG, src.meta.width, src.meta.height)
    frames = []
    for i, t, _ in src.frames():
        dets = [Detection(b, 0.9, 0) for b in boxes(i)]
        cam = CameraObservation.identity(CameraStatus.FIRST if i == 0 else CameraStatus.OK, inliers=100)
        frames.append(tracks_frame(trk.update(i, t, dets, cam), CFG.output))
    return {"schema": TRACKS_SCHEMA[0], "schema_version": TRACKS_SCHEMA[1], "provenance": {},
            "video": src.meta.to_dict(), "detections_sha256": "0" * 64, "frames": frames,
            "tracklets": tracklet_rows(trk.summaries())}


class Embed(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "clip.avi"
        write_video(self.video)
        mean_colour_onnx(self.td / "m.onnx")
        self.record = write_record(self.td, self.td / "m.onnx")
        self.tracks = track(self.video)

    def test_signatures_separate_the_two_shirts(self):
        emb = ReidEmbedder(ReidModelRecord.load(self.record))
        doc = embeddings_document(VideoSource(str(self.video)), self.tracks, emb, CFG, ["test"])
        validate_track_embeddings(doc, self.tracks)
        sig = doc["tracks"]
        self.assertEqual(len(sig), 2)
        (a, b) = sorted(sig)
        for t in (a, b):
            self.assertEqual(len(sig[t]["first_frames"]), CFG.reid.samples)
            self.assertGreater(cosine(sig[t]["first"], sig[t]["last"]), 0.99)       # same shirt start and end
        self.assertLess(cosine(sig[a]["last"], sig[b]["first"]), cosine(sig[a]["last"], sig[a]["first"]))

    def test_wrong_weights_hash_is_refused(self):
        with self.assertRaises(ReidModelError):
            ReidModelRecord.load(write_record(self.td, self.td / "m.onnx", sha="f" * 64))

    def test_cli(self):
        from spana import cli
        trk = self.td / "trk.json"
        trk.write_text(json.dumps(self.tracks))
        out = self.td / "emb.json"
        self.assertEqual(cli.main(["embed", "--video", str(self.video), "--tracks", str(trk), "--config",
                                   str(ROOT / "config" / "default.yaml"), "--model-record", str(self.record),
                                   "--out", str(out)]), 0)
        validate_track_embeddings(read_json(out), self.tracks)


if __name__ == "__main__":
    unittest.main()
