"""DEV ball labels: record rules, ball pack, freeze (refusals + tamper check), and the page driven like a user."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from spana.ball_labels import BallLabelError, load_frozen, summary, validate_record
from tests.test_cli_e2e import write_video
from tests.test_labelling import _playwright
from tools.freeze_ball_labels import freeze
from tools.label_pack import make_pack

CLIP = {"name": "DEVCLIP", "sha256": "x" * 64, "fps": 25.0, "width": 320, "height": 180}


def record(frames, **kw):
    r = {"schema": "spana.ball_labels", "schema_version": "1.0.0", "split": "DEV", "clip": dict(CLIP),
         "frame_step": 5, "label_version": 1, "annotator": "QA", "frozen_utc": "2026-11-22T00:00:00Z",
         "frames": frames}
    r.update(kw)
    return r


def fr(frame, state, box=None, point=None, others=()):
    return {"frame": frame, "state": state, "box": box, "point": point, "other_balls": list(others)}


class RecordRules(unittest.TestCase):
    def test_valid_record(self):
        labels = validate_record(record([fr(0, "VISIBLE", [10, 10, 16, 16]), fr(5, "HIDDEN", point=[50, 60]),
                                         fr(10, "OUT", others=[[200, 20, 206, 26]]), fr(15, "UNSURE")]))
        self.assertEqual(labels[0].box, (10.0, 10.0, 16.0, 16.0))
        self.assertEqual(labels[5].point, (50.0, 60.0))
        self.assertEqual(summary(labels), {"frames_labelled": 4, "other_balls": 1,
                                           "states": {"VISIBLE": 1, "HIDDEN": 1, "OUT": 1, "UNSURE": 1}})

    def test_refusals(self):
        bad = [
            record([fr(0, "VISIBLE")]),                                   # seen without a box
            record([fr(0, "OUT", box=[1, 1, 5, 5])]),                     # box on a non-VISIBLE frame
            record([fr(0, "OUT", point=[3, 3])]),                         # point on a non-HIDDEN frame
            record([fr(0, "VISIBLE", [10, 10, 400, 16])]),                # box outside the image
            record([fr(0, "VISIBLE", [16, 10, 10, 16])]),                 # empty box
            record([fr(0, "SEEN")]),                                      # unknown state
            record([fr(0, "OUT"), fr(0, "UNSURE")]),                      # frame twice
            record([]),                                                   # nothing labelled
            record([fr(0, "OUT")], split="EVAL"),
            record([fr(0, "OUT")], annotator=" "),
            record([fr(0, "OUT")], schema_version="2.0.0"),
        ]
        for r in bad:
            with self.assertRaises(BallLabelError, msg=json.dumps(r["frames"])[:120]):
                validate_record(r)
        with self.assertRaises(BallLabelError):
            validate_record(record([fr(3, "OUT")]), allowed_frames=[0, 5])


class PackAndFreeze(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "DEVCLIP.avi"
        write_video(self.video, 20, 25.0)
        self.pack = self.td / "pack"
        self.m = make_pack(str(self.video), "DEV", 0.2, str(self.pack), kind="ball")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _write(self, rec, name="rec.json"):
        p = self.td / name
        p.write_text(json.dumps(rec))
        return p

    def test_ball_pack_has_ball_page(self):
        self.assertEqual(self.m["kind"], "ball")
        self.assertIn("SPANA_BALL_LABELLER", (self.pack / "index.html").read_text())
        with self.assertRaises(SystemExit):
            make_pack(str(self.video), "DEV", 0.2, str(self.td / "p2"), kind="goals")

    def test_freeze_and_load(self):
        rec = record([fr(0, "VISIBLE", [10, 10, 16, 16]), fr(5, "OUT")], clip=self.m["clip"])
        dest = freeze(str(self.pack), str(self._write(rec)), str(self.video), root=self.td / "repo")
        frozen = json.loads((dest / "FROZEN.json").read_text())
        self.assertEqual((frozen["frames_labelled"], frozen["states"]["VISIBLE"]), (2, 1))
        self.assertEqual(dest.parts[-3:], ("ball_labels", "DEVCLIP", "v1"))
        _, labels = load_frozen(dest)
        self.assertEqual(sorted(labels), [0, 5])
        with self.assertRaises(BallLabelError):                       # never overwrite a version
            freeze(str(self.pack), str(self._write(rec)), None, root=self.td / "repo")
        (dest / "ball_labels_frozen.json").write_text(json.dumps(rec) + " ")
        with self.assertRaises(BallLabelError):                       # edited after freezing
            load_frozen(dest)

    def test_freeze_refusals(self):
        ok = record([fr(0, "OUT")], clip=self.m["clip"])
        other_clip = record([fr(0, "OUT")], clip=dict(self.m["clip"], sha256="y" * 64))
        with self.assertRaises(BallLabelError):
            freeze(str(self.pack), str(self._write(other_clip)), None, root=self.td / "r1")
        with self.assertRaises(BallLabelError):                       # frame not in the pack
            freeze(str(self.pack), str(self._write(record([fr(1, "OUT")], clip=self.m["clip"]))), None,
                   root=self.td / "r2")
        wrong_video = self.td / "OTHER.avi"
        write_video(wrong_video, 12, 25.0)
        with self.assertRaises(BallLabelError):
            freeze(str(self.pack), str(self._write(ok)), str(wrong_video), root=self.td / "r3")


@unittest.skipIf(_playwright() is None or shutil.which("google-chrome") is None, "Playwright/Chrome not available")
class BrowserPage(unittest.TestCase):
    """Drives tools/ball_labeller.html like a user: box, other ball, hidden + point, out, unsure, copy, export."""

    def test_labelling_session(self):
        td = Path(tempfile.mkdtemp())
        video = td / "DEVCLIP.avi"
        write_video(video, 20, 25.0)
        pack = td / "pack"
        m = make_pack(str(video), "DEV", 0.2, str(pack), kind="ball")           # frames 0, 5, 10, 15
        with _playwright()() as p:
            browser = p.chromium.launch(executable_path=shutil.which("google-chrome"), headless=True)
            ctx = browser.new_context(accept_downloads=True, viewport={"width": 1400, "height": 800})
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto((pack / "index.html").resolve().as_uri())
            page.wait_for_function("window.SPANA_BALL_LABELLER !== undefined")
            box = page.locator("#cv").bounding_box()
            s = min(box["width"] / 320, box["height"] / 180)

            def at(x, y):
                return box["x"] + (box["width"] - 320 * s) / 2 + x * s, box["y"] + (box["height"] - 180 * s) / 2 + y * s

            def drag(a, b, shift=False):
                if shift:
                    page.keyboard.down("Shift")
                page.mouse.move(*at(*a)); page.mouse.down(); page.mouse.move(*at(*b), steps=4); page.mouse.up()
                if shift:
                    page.keyboard.up("Shift")

            drag((100, 50), (110, 60))                               # frame 0: match ball seen
            drag((200, 20), (208, 28), shift=True)                   # frame 0: a spare ball
            page.keyboard.press("d")
            page.keyboard.press("h")                                 # frame 5: hidden ...
            page.mouse.click(*at(150, 90))                           # ... best guess there
            page.keyboard.press("d")
            page.keyboard.press("o")                                 # frame 10: out (auto-advances)
            page.keyboard.press("c")                                 # frame 15: copy previous (out) ...
            page.keyboard.press("u")                                 # ... then changed to unsure
            page.fill("#annotator", "QA"); page.keyboard.press("Tab")
            st = page.evaluate("() => window.SPANA_BALL_LABELLER.state")
            self.assertEqual([st["answers"][str(f)]["state"] for f in (0, 5, 10, 15)],
                             ["VISIBLE", "HIDDEN", "OUT", "UNSURE"])
            with page.expect_download() as dl:
                page.click("#export")
            out = td / "export.json"
            dl.value.save_as(out)
            page.reload()
            page.wait_for_function("window.SPANA_BALL_LABELLER !== undefined")
            self.assertTrue(page.evaluate("() => window.SPANA_BALL_LABELLER.state.frozen"))   # survives reload
            browser.close()
        self.assertEqual(errors, [])
        rec = json.loads(out.read_text())
        labels = validate_record(rec, [f["index"] for f in m["frames"]])
        self.assertEqual(labels[0].state, "VISIBLE")
        self.assertTrue(abs(labels[0].box[0] - 100) < 2 and abs(labels[0].box[3] - 60) < 2)
        self.assertEqual(len(labels[0].other_balls), 1)
        self.assertTrue(abs(labels[5].point[0] - 150) < 2)
        dest = freeze(str(pack), str(out), str(video), root=td / "repo")
        self.assertTrue((dest / "FROZEN.json").is_file())
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
