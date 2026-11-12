"""DEV labelling tool: pack, browser page (headless Chrome via Playwright when available), MOT 1.1 parse, freeze."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spana.labels import LabelError, parse_mot
from tests.test_cli_e2e import write_video
from tools.freeze_labels import freeze
from tools.label_pack import make_pack


def _record(pack: Path, gt: str, frames, version=1, annotator="QA"):
    m = json.loads((pack / "manifest.json").read_text())
    return {"schema": "spana.labels_frozen", "split": "DEV", "clip": m["clip"], "frame_step": m["frame_step"],
            "label_version": version, "annotator": annotator, "frozen_utc": "2026-11-07T00:00:00Z",
            "frames_labelled": frames, "gt_sha256": hashlib.sha256(gt.encode()).hexdigest()}


class PackAndFreeze(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "DEVCLIP.avi"
        write_video(self.video, 20, 25.0)
        self.pack = self.td / "pack"
        self.m = make_pack(str(self.video), "DEV", 0.2, str(self.pack))

    def test_pack_contents(self):
        self.assertEqual(self.m["frame_step"], 5)
        self.assertEqual([f["index"] for f in self.m["frames"]], [0, 5, 10, 15])
        for f in self.m["frames"]:
            self.assertTrue((self.pack / f["file"]).is_file())
        self.assertTrue((self.pack / "index.html").is_file() and (self.pack / "manifest.js").is_file())

    def test_pack_refuses_eval_and_overwrite(self):
        with self.assertRaises(SystemExit):
            make_pack(str(self.video), "EVAL", 0.2, str(self.td / "e"))
        with self.assertRaises(SystemExit):
            make_pack(str(self.video), "DEV", 0.2, str(self.pack))

    def test_parse_mot_rules(self):
        ok = "1,1,10,10,20,40,1,1,1\n6,1,12,10,20,40,1,1,0.5\n6,0,100,0,50,50,0,1,1\n"
        gt = parse_mot(ok, 320, 180, [0, 5])
        self.assertEqual(gt.players[5]["1"], (12.0, 10.0, 32.0, 50.0))
        self.assertEqual(gt.occluded[5], {"1"})
        self.assertEqual(len(gt.ignore[5]), 1)
        bad = ["1,1,10,10,20,40,1,1\n", "2,1,10,10,20,40,1,1,1\n", "1,1,10,10,-2,40,1,1,1\n",
               "1,1,310,10,20,40,1,1,1\n", "1,1,1,1,5,5,1,1,1\n1,1,9,9,5,5,1,1,1\n", "1,0,1,1,5,5,1,1,1\n",
               "1,1,1,1,5,5,1,1,0.3\n"]
        for b in bad:
            with self.assertRaises(LabelError, msg=b):
                parse_mot(b, 320, 180, [0, 5])

    def test_freeze_happy_path_and_refusals(self):
        gt = "1,1,10,10,20,40,1,1,1\n6,1,12,10,20,40,1,1,0.5\n"
        g, r = self.td / "gt.txt", self.td / "rec.json"
        g.write_text(gt)
        r.write_text(json.dumps(_record(self.pack, gt, [0, 5])))
        root = self.td / "repo"
        dest = freeze(str(self.pack), str(g), str(r), str(self.video), root=root)
        fz = json.loads((dest / "FROZEN.json").read_text())
        self.assertEqual((fz["players"], fz["player_boxes"], fz["split"]), (1, 2, "DEV"))
        with self.assertRaises(LabelError):                       # never overwrite a version
            freeze(str(self.pack), str(g), str(r), None, root=root)
        g.write_text(gt + "11,1,12,10,20,40,1,1,1\n")              # edited after export
        with self.assertRaises(LabelError):
            freeze(str(self.pack), str(g), str(r), None, root=self.td / "r2")
        other = self.td / "other.avi"
        write_video(other, 20, 25.0, seed=4)
        g.write_text(gt)
        with self.assertRaises(LabelError):                       # wrong video
            freeze(str(self.pack), str(g), str(r), str(other), root=self.td / "r3")
        rec = _record(self.pack, gt, [0, 5]); rec["split"] = "EVAL"
        r.write_text(json.dumps(rec))
        with self.assertRaises(LabelError):
            freeze(str(self.pack), str(g), str(r), None, root=self.td / "r4")


def _playwright():
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError:
        return None


@unittest.skipIf(_playwright() is None or shutil.which("google-chrome") is None, "Playwright/Chrome not available")
class BrowserPage(unittest.TestCase):
    """Drives tools/labeller.html like a user: draw, ID, copy previous, move, occluded, done, export, reload."""

    def test_full_labelling_session(self):
        td = Path(tempfile.mkdtemp())
        video = td / "DEVCLIP.avi"
        write_video(video, 20, 25.0)
        pack = td / "pack"
        make_pack(str(video), "DEV", 0.2, str(pack))
        url = (pack / "index.html").resolve().as_uri()
        downloads = td / "dl"
        downloads.mkdir()
        with _playwright()() as p:
            browser = p.chromium.launch(executable_path=shutil.which("google-chrome"), headless=True)
            ctx = browser.new_context(accept_downloads=True, viewport={"width": 1400, "height": 800})
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url)
            page.wait_for_function("window.SPANA_LABELLER !== undefined")
            box = page.locator("#cv").bounding_box()
            to = page.evaluate("() => { const r = document.getElementById('cv').getBoundingClientRect();"
                               " return {w: r.width, h: r.height}; }")

            def img_to_screen(x, y):   # same fit rule as the page (320x180 clip)
                s = min(to["w"] / 320, to["h"] / 180)
                return box["x"] + (to["w"] - 320 * s) / 2 + x * s, box["y"] + (to["h"] - 180 * s) / 2 + y * s

            def drag(a, b):
                page.mouse.move(*img_to_screen(*a)); page.mouse.down()
                page.mouse.move(*img_to_screen(*b), steps=5); page.mouse.up()

            drag((20, 30), (60, 120))                          # player 1
            page.keyboard.press("n")
            drag((150, 40), (190, 130))                        # player 2
            st = page.evaluate("() => window.SPANA_LABELLER.state")
            self.assertEqual(sorted(b["id"] for b in st["boxes"]["0"]), [1, 2])
            page.fill("#annotator", "QA bot"); page.keyboard.press("Enter")
            page.locator("#annotator").dispatch_event("change")
            page.keyboard.press("Enter")                       # done (focus left the input)
            page.keyboard.press("d")                           # next frame
            page.keyboard.press("c")                           # copy previous
            drag((40, 75), (50, 75))                           # move player 1 by +10 px
            page.mouse.click(*img_to_screen(170, 85))          # select player 2
            page.keyboard.press("o")                           # occluded
            page.keyboard.press("Enter")                       # done
            st = page.evaluate("() => window.SPANA_LABELLER.state")
            f5 = {b["id"]: b for b in st["boxes"]["5"]}
            self.assertAlmostEqual(f5[1]["x1"], 30.0, delta=1.0)
            self.assertTrue(f5[2]["occluded"])
            self.assertEqual(sorted(int(k) for k in st["done"]), [0, 5])

            page.keyboard.press("d")                           # frame 10: a known player re-drawn via the list
            page.locator("#ids .id").nth(1).click()          # row for player 2
            drag((150, 40), (190, 130))
            drag((250, 40), (290, 130))                        # plain draw afterwards = a new player (3)
            st = page.evaluate("() => window.SPANA_LABELLER.state")
            self.assertEqual(sorted(b["id"] for b in st["boxes"]["10"]), [2, 3])
            page.keyboard.press("Delete")                      # remove player 3 again (selected)
            page.keyboard.press("a")
            page.on("dialog", lambda d: d.accept())
            got = []
            page.on("download", lambda d: got.append(d))
            page.click("#export")
            page.wait_for_timeout(1500)
            self.assertEqual(len(got), 2)
            for d in got:
                d.save_as(downloads / d.suggested_filename)
            st = page.evaluate("() => window.SPANA_LABELLER.state")
            self.assertTrue(st["frozen"])

            page.reload()                                       # autosave restores progress, still frozen
            page.wait_for_function("window.SPANA_LABELLER !== undefined")
            st2 = page.evaluate("() => window.SPANA_LABELLER.state")
            self.assertTrue(st2["frozen"])
            self.assertEqual(len(st2["boxes"]["5"]), 2)
            drag((200, 20), (240, 100))                         # frozen: drawing refused
            self.assertEqual(len(page.evaluate("() => window.SPANA_LABELLER.state")["boxes"]["0"]), 2)
            self.assertEqual(errors, [])
            browser.close()

        gt_file = next(downloads.glob("*_gt.txt"))
        rec_file = next(downloads.glob("*_labels_frozen.json"))
        gt = gt_file.read_text()
        self.assertEqual(len(gt.strip().splitlines()), 4)
        parsed = parse_mot(gt, 320, 180, [0, 5])
        self.assertEqual(sorted(parsed.players[5]), ["1", "2"])
        self.assertEqual(parsed.occluded.get(5), {"2"})
        self.assertAlmostEqual(parsed.players[5]["1"][0], 30.0, delta=1.0)
        rec = json.loads(rec_file.read_text())
        self.assertEqual((rec["annotator"], rec["frames_labelled"], rec["split"]), ("QA bot", [0, 5], "DEV"))
        dest = freeze(str(pack), str(gt_file), str(rec_file), str(video), root=td / "repo")
        self.assertEqual(json.loads((dest / "FROZEN.json").read_text())["player_boxes"], 4)


if __name__ == "__main__":
    unittest.main()


class BenchmarkTool(unittest.TestCase):
    """End to end: pack -> human labels -> freeze -> detect + track -> benchmark against frozen labels."""

    def test_benchmark_scores_against_frozen_labels(self):
        import subprocess
        import sys
        import yaml
        from tests.synth import CONFIG_PATH
        from tests.test_detection_camera import constant_onnx, write_record
        td = Path(tempfile.mkdtemp())
        video = td / "DEVCLIP.avi"
        write_video(video, 20, 25.0)
        pack = td / "pack"
        make_pack(str(video), "DEV", 0.2, str(pack))
        out = np.zeros((1, 6, 1), np.float32)
        out[0, :, 0] = [32, 32, 8, 20, 0.9, 0.1]          # one fixed person box: x 140-180, y 40-140 in source px
        constant_onnx(td / "m.onnx", out, 64)
        record = write_record(td, td / "m.onnx")
        raw = yaml.safe_load(open(CONFIG_PATH)); raw["detection"]["tiling_enabled"] = False
        cfg = td / "cfg.yaml"; cfg.write_text(yaml.safe_dump(raw))
        run = lambda *a: subprocess.run([sys.executable, "-m", *a], check=True, capture_output=True)  # noqa: E731
        run("spana.cli", "detect", "--video", str(video), "--config", str(cfg), "--model-record", str(record),
            "--out", str(td / "d.json"))
        run("spana.cli", "track", "--video", str(video), "--config", str(cfg), "--detections", str(td / "d.json"),
            "--out", str(td / "t.json"))
        for name, rows in (("good", "{f},1,140,40,40,100,1,1,1\n"), ("wrong", "{f},1,10,10,30,60,1,1,1\n")):
            gt = "".join(rows.format(f=f + 1) for f in (0, 5, 10, 15))
            g, r = td / f"{name}_gt.txt", td / f"{name}_rec.json"
            g.write_text(gt)
            rec = _record(pack, gt, [0, 5, 10, 15]); rec["label_version"] = 1 if name == "good" else 2
            r.write_text(json.dumps(rec))
            dest = freeze(str(pack), str(g), str(r), str(video), root=td / "repo")
            run("tools.benchmark", "--labels", str(dest), "--detections", str(td / "d.json"), "--tracks",
                str(td / "t.json"), "--decisions", str(td / "t.decisions.jsonl"), "--out", str(td / f"{name}.json"))
            s = json.loads((td / f"{name}.json").read_text())["summary"]
            if name == "good":
                self.assertEqual((s["detection_recall"], s["missed_players"], s["identity_transfers"]), (1.0, 0, 0))
            else:
                self.assertEqual((s["detection_recall"], s["missed_players"]), (0.0, 4))
