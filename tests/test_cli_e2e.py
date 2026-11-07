"""End-to-end: video -> detect (pinned ONNX) -> detections artifact -> track -> tracks artifact + decision log."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spana import cli
from spana.artifacts import ArtifactError, read_json, validate_detections, validate_tracks
from spana.video import VideoError
from tests.synth import CONFIG_PATH
from tests.test_detection_camera import _texture, constant_onnx, write_record


def write_video(path: Path, n: int, fps: float, shift_px: int = 2, seed: int = 0) -> None:
    big = _texture(400, 900, seed=seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 180))
    for i in range(n):
        vw.write(big[50:230, 50 + i * shift_px:370 + i * shift_px])
    vw.release()


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "clip.avi"
        write_video(self.video, 20, 25.0)
        out = np.zeros((1, 6, 1), np.float32)
        out[0, :, 0] = [32, 32, 8, 20, 0.9, 0.1]
        constant_onnx(self.td / "m.onnx", out, 64)
        self.record = write_record(self.td, self.td / "m.onnx")

    def _detect(self, out: Path, **kw) -> None:
        args = ["detect", "--video", str(self.video), "--config", CONFIG_PATH, "--model-record", str(self.record),
                "--out", str(out)]
        for k, v in kw.items():
            args += [f"--{k.replace('_', '-')}", str(v)]
        self.assertEqual(cli.main(args), 0)

    def test_full_pipeline(self):
        dpath, tpath = self.td / "det.json", self.td / "trk.json"
        self._detect(dpath)
        ddoc = read_json(dpath)
        validate_detections(ddoc)
        self.assertEqual(len(ddoc["frames"]), 20)
        self.assertEqual(ddoc["model"]["sha256"], ddoc["provenance"]["inputs"]["weights"])
        self.assertEqual(cli.main(["track", "--video", str(self.video), "--config", CONFIG_PATH,
                                   "--detections", str(dpath), "--out", str(tpath)]), 0)
        tdoc = read_json(tpath)
        validate_tracks(tdoc)
        self.assertEqual(len(tdoc["frames"]), 20)
        self.assertTrue(tpath.with_suffix(".decisions.jsonl").stat().st_size > 0)
        states = {t["state"] for fr in tdoc["frames"] for t in fr["tracks"]}
        self.assertIn("CONFIRMED", states)
        # Overwrite is refused.
        with self.assertRaises(ArtifactError):
            self._detect(dpath)

    def test_inputs_fail_closed(self):
        with self.assertRaises(VideoError):
            self._detect(self.td / "a.json", video_sha256="0" * 64)
        dpath = self.td / "det.json"
        self._detect(dpath)
        other = self.td / "other.avi"
        write_video(other, 20, 25.0, seed=3)
        with self.assertRaises(ArtifactError):
            cli.main(["track", "--video", str(other), "--config", CONFIG_PATH, "--detections", str(dpath),
                      "--out", str(self.td / "t.json")])


if __name__ == "__main__":
    unittest.main()
