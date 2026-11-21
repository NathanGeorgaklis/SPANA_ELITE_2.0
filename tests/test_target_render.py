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
from tools.target_render import Renderer, encode, playback_fps

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


class BallOverlayTest(unittest.TestCase):
    """The optional ball overlay: a ring only where spana.ball claims a position, never for LOST / AMBIGUOUS."""

    def setUp(self):
        self.tracks = tracks_artifact()
        self.doc = target_document(self.tracks, SELECTION, CFG, ["test"], ACTIONS)

    @staticmethod
    def _fr(frame, state, center=None, diameter=None):
        return {"frame": frame, "state": state, "center": center, "diameter": diameter}

    def _yellow_near(self, img, c, r=14):
        x, y = c
        patch = img[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3)
        return bool(((patch[:, 2] > 200) & (patch[:, 1] > 180) & (patch[:, 0] < 60)).any())

    def test_ring_only_when_position_claimed(self):
        c = (W // 2, H // 2 + 60)
        ball = {"frames": [self._fr(20, "OBSERVED", list(c), 10.0), self._fr(21, "PREDICTED", list(c), 10.0),
                           self._fr(22, "LOST"), self._fr(23, "AMBIGUOUS")]}
        r = Renderer(self.tracks, self.doc, W, H, ball)
        self.assertTrue(self._yellow_near(r.draw(scene(20), 20), c))
        self.assertTrue(self._yellow_near(r.draw(scene(21), 21), c))
        self.assertFalse(self._yellow_near(r.draw(scene(22), 22), c))
        self.assertFalse(self._yellow_near(r.draw(scene(23), 23), c))
        self.assertFalse(self._yellow_near(r.draw(scene(24), 24), c))      # frame missing from the ball doc

    def test_without_ball_unchanged(self):
        a = Renderer(self.tracks, self.doc, W, H).draw(scene(70), 70)
        b = Renderer(self.tracks, self.doc, W, H, None).draw(scene(70), 70)
        self.assertTrue((a == b).all())



class PlaybackFpsTest(unittest.TestCase):
    def test_strided_frames_play_in_real_time(self):
        self.assertAlmostEqual(playback_fps([0, 3, 6, 9], 60.0), 20.0)
        self.assertAlmostEqual(playback_fps(list(range(10)), 25.0), 25.0)
        self.assertAlmostEqual(playback_fps([5], 30.0), 30.0)


if __name__ == "__main__":
    unittest.main()
