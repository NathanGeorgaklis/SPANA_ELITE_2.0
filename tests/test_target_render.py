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
import tools.target_render as target_render
from tools.target_render import OFFER_LABEL, TARGET_LABEL, Renderer, answer_label, encode

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


class PersistentLabelTest(unittest.TestCase):
    """The selected player keeps one label ("YOU") on every TARGET frame across track changes; the track id is only a
    secondary tag; CANDIDATE keeps the offer text; ABSTAIN / LOST label no player."""

    def setUp(self):
        self.tracks = tracks_artifact()
        self.doc = target_document(self.tracks, SELECTION, CFG, ["test"], ACTIONS)
        self.r = Renderer(self.tracks, self.doc, W, H)

    def drawn_text(self, f: int, answer=None) -> list:
        seen, orig = [], target_render._text
        target_render._text = lambda img, s, *a, **k: (seen.append(s), orig(img, s, *a, **k))
        try:
            if answer is not None:
                self.r.answers[f] = answer
            self.r.draw(scene(f), f)
        finally:
            target_render._text = orig
        return seen

    def test_label_persists_across_track_changes(self):
        target_tracks = set()
        for a in self.doc["frames"]:
            if a["state"] != "TARGET":
                continue
            target_tracks.add(a["track_id"])
            self.assertEqual(answer_label(a), (TARGET_LABEL, a["track_id"]))
            text = self.drawn_text(a["frame"])
            self.assertIn(TARGET_LABEL, text)
            self.assertFalse(any("following" in t and a["track_id"] in t for t in text))
        self.assertGreaterEqual(len(target_tracks), 3)          # relink + re-selection across a cut

    def test_candidate_keeps_offer_text(self):
        cand = [a for a in self.doc["frames"] if a["state"] == "CANDIDATE"]
        self.assertTrue(cand)
        for a in cand:
            self.assertEqual(answer_label(a), (OFFER_LABEL, None))
            text = self.drawn_text(a["frame"])
            self.assertIn(OFFER_LABEL, text)
            self.assertNotIn(TARGET_LABEL, text)

    def test_abstain_and_lost_label_no_player(self):
        quiet = [a for a in self.doc["frames"] if a["state"] in ("ABSTAIN", "LOST")]
        self.assertEqual({a["state"] for a in quiet}, {"ABSTAIN", "LOST"})
        for a in quiet:
            # fail closed even if an answer carried a box: no label, no box on any player
            forged = {**a, "box": self.tracks["frames"][a["frame"]]["tracks"][0]["box"]}
            self.assertIsNone(answer_label(a))
            self.assertIsNone(answer_label(forged))
            text = self.drawn_text(a["frame"], forged)
            self.assertNotIn(TARGET_LABEL, text)
            self.assertNotIn(OFFER_LABEL, text)
        self.assertIsNone(answer_label(None))


if __name__ == "__main__":
    unittest.main()
