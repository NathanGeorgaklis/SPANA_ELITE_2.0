"""Merging chunked detections artifacts of one video (long TRAIN windows are detected in chunks)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import DETECTIONS_SCHEMA, ArtifactError, detection_frame
from spana.config import load_config
from spana.records import Detection
from tools.merge_detections import merge

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")


def part(frames, stride=4, sha="a" * 64):
    return {"schema": DETECTIONS_SCHEMA[0], "schema_version": DETECTIONS_SCHEMA[1], "provenance": {"git": "x"},
            "video": {"sha256": sha}, "model": {"name": "m"}, "detection_config": {"k": 1}, "frame_stride": stride,
            "frames": [detection_frame(f, f / 60, [Detection((1.0, 1.0, 5.0, 9.0), 0.9, 0)], CFG.output)
                       for f in frames]}


class Merge(unittest.TestCase):
    def _write(self, docs):
        td = Path(tempfile.mkdtemp())
        paths = []
        for k, d in enumerate(docs):
            p = td / f"p{k}.json"
            p.write_text(json.dumps(d))
            paths.append(p)
        return paths

    def test_contiguous_parts_merge(self):
        out = merge(self._write([part(range(0, 16, 4)), part(range(16, 32, 4))]))
        self.assertEqual([f["frame"] for f in out["frames"]], list(range(0, 32, 4)))
        self.assertEqual(len(out["provenance"]["merged_from"]), 2)

    def test_refuses_gaps_and_mismatches(self):
        for bad in (part(range(20, 32, 4)), part(range(16, 32, 4), sha="b" * 64)):
            with self.assertRaises(ArtifactError):
                merge(self._write([part(range(0, 16, 4)), bad]))


if __name__ == "__main__":
    unittest.main()
