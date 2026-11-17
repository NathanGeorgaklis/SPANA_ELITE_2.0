"""tools/target_demo.py + tools/target_render.py: the synthetic demo scene is a valid tracks artifact, the real
follower produces the scripted behaviour, and the renderer draws frames (encoding needs ffmpeg)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from spana.config import load_config
from spana.identity.follower import target_document
from tools.target_demo import ACTIONS, H, N_FRAMES, SELECTION, W, scene, tracks_artifact
from tools.target_render import Renderer, encode

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")


class TargetDemoTest(unittest.TestCase):
    def setUp(self):
        self.tracks = tracks_artifact()
        self.doc = target_document(self.tracks, SELECTION, CFG, ["test"], ACTIONS)

    def test_scripted_behaviour(self):
        ans = {f["frame"]: (f["state"], f["reason"]) for f in self.doc["frames"]}
        self.assertEqual(ans[20], ("TARGET", "TRACK_OBSERVED"))
        self.assertEqual(ans[35], ("ABSTAIN", "TRACK_OCCLUDED"))
        self.assertEqual(ans[62], ("LOST", "TRACK_LOST"))
        self.assertEqual(ans[70], ("CANDIDATE", "RELINK_PREDECESSOR"))
        self.assertEqual(ans[95], ("TARGET", "TRACK_OBSERVED"))
        self.assertEqual(ans[115], ("ABSTAIN", "IDENTITY_UNCERTAIN"))
        self.assertEqual(ans[130], ("ABSTAIN", "AFTER_UNCERTAIN"))
        self.assertEqual(ans[185], ("LOST", "TRACK_ENDED"))
        self.assertEqual(ans[210], ("TARGET", "TRACK_OBSERVED"))
        self.assertEqual(len(self.doc["gaps"]), 4)

    def test_renderer_draws_and_encodes(self):
        r = Renderer(self.tracks, self.doc, W, H)
        for f in (0, 70, 115, 210):
            img = r.draw(scene(f), f, "caption")
            self.assertEqual(img.shape, (H, W, 3))
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo.mp4"
            encode((r.draw(scene(f), f) for f in range(0, N_FRAMES, 25)), str(out), 5.0, W, H)
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
