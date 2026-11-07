"""End-to-end: video -> detect (pinned ONNX) -> detections artifact -> track -> tracks artifact + decision log."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

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
        # The fake model answers every input (each tile too) with the same box: tiling off for plumbing tests.
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        raw["detection"]["tiling_enabled"] = False
        self.config = str(self.td / "cfg.yaml")
        Path(self.config).write_text(yaml.safe_dump(raw))

    def _detect(self, out: Path, **kw) -> None:
        args = ["detect", "--video", str(self.video), "--config", self.config, "--model-record", str(self.record),
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
        self.assertEqual(cli.main(["track", "--video", str(self.video), "--config", self.config,
                                   "--detections", str(dpath), "--out", str(tpath)]), 0)
        tdoc = read_json(tpath)
        validate_tracks(tdoc)
        self.assertEqual(len(tdoc["frames"]), 20)
        self.assertTrue(tpath.with_suffix(".decisions.jsonl").stat().st_size > 0)
        self.assertTrue(any(t["confirmed"] for fr in tdoc["frames"] for t in fr["tracks"]))
        self.assertEqual(len(tdoc["tracklets"]), 1)
        cams = [fr["camera"] for fr in tdoc["frames"][1:]]
        self.assertTrue(all(c["status"] == "OK" for c in cams), {c["status"] for c in cams})
        # The video pans 2 px/frame to the right -> scene content moves left by 2 px.
        self.assertTrue(all(abs(c["affine"][0][2] + 2.0) < 0.5 for c in cams), [c["affine"][0][2] for c in cams])
        # Overwrite is refused.
        with self.assertRaises(ArtifactError):
            self._detect(dpath)

    def test_config_mismatch_between_passes_refused(self):
        dpath = self.td / "det.json"
        self._detect(dpath)
        with open(self.config) as fh:
            raw = yaml.safe_load(fh)
        raw["detection"]["score_low"] = 0.2
        other_cfg = self.td / "cfg.yaml"
        other_cfg.write_text(yaml.safe_dump(raw))
        with self.assertRaises(ArtifactError):
            cli.main(["track", "--video", str(self.video), "--config", str(other_cfg), "--detections", str(dpath),
                      "--out", str(self.td / "t.json")])
        self.assertFalse((self.td / "t.json").exists())
        self.assertFalse((self.td / "t.decisions.jsonl").exists())

    def test_start_beyond_video_refused(self):
        with self.assertRaises(ArtifactError):
            self._detect(self.td / "x.json", start=500)

    def test_inputs_fail_closed(self):
        with self.assertRaises(VideoError):
            self._detect(self.td / "a.json", video_sha256="0" * 64)
        dpath = self.td / "det.json"
        self._detect(dpath)
        other = self.td / "other.avi"
        write_video(other, 20, 25.0, seed=3)
        with self.assertRaises(ArtifactError):
            cli.main(["track", "--video", str(other), "--config", self.config, "--detections", str(dpath),
                      "--out", str(self.td / "t.json")])


if __name__ == "__main__":
    unittest.main()
