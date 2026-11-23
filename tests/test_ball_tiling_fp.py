"""tools/diagnostics/ball_tiling_fp.py: what the tiled detector mistook for the ball (analysis only).
A scripted scene through the real tracker and path: the real ball, then a steady false object in the overlap of the
two tiles (accepted by the path), and a brief blip (rejected)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spana.artifacts import sha256_doc
from spana.ball.path import ball_path_document
from spana.ball_labels import validate_record
from spana.detection.ball_view import tile_grid
from spana.reid_labels import ReidLabelError
from tests.synth import cfg
from tests.test_ball_path import FPS, H, SHA, W, ball_doc
from tests.test_ball_score import label_record
from tests.test_ball_tiling import SmallBallDetector
from tests.test_ball_tracker import ball
from tools.diagnostics import ball_tiling_fp as fp

ON = dataclasses.replace(cfg().ball, tile_size_frac=1.0)
PC = cfg().ball_path
CLIP = "SYNTH"
REAL = lambda i: (100 + 6 * i, 300)        # noqa: E731  the real ball's centre on frame i (visible on every frame)
FALSE_AT = (640, 500)                      # in the overlap of the two 720 px tiles of a 1280 px frame (560 <= x < 720)


def candidates(i):
    c = []
    if i < 20:
        c.append(ball(*REAL(i)))                           # detected for 0.76 s, then missed for the rest
    if 21 <= i <= 23:
        c.append(ball(1100, 200, score=0.4))               # a brief blip
    if i >= 25:
        c.append(ball(*FALSE_AT, score=0.6))               # a steady false object, 1.36 s
    return c


def write_run(tmp: Path, split="DEV", sha=SHA):
    frames = [candidates(i) for i in range(60)]
    bdoc = ball_doc(frames, ball_cfg=ON)
    bdoc["video"]["sha256"] = sha
    path = ball_path_document(bdoc, PC, {}, sha256_doc(bdoc))
    video = bdoc["video"]
    det = lambda fs: {"video": video, "frames": [{"frame": i, "detections": [  # noqa: E731
        [*d.box, d.score, 3, 0] for d in c]} for i, c in enumerate(fs)]}
    tiled = tmp / "tile"
    tiled.mkdir()
    for name, doc in ((f"{CLIP}.ball.json", bdoc), (f"{CLIP}.path.json", path), (f"{CLIP}.balldet.json", det(frames))):
        (tiled / name).write_text(json.dumps(doc))
    base = tmp / "base.balldet.json"                       # untiled: the false object only on frames 25-30
    base.write_text(json.dumps(det([[d for d in c if not (d.box[0] == FALSE_AT[0] - 5 and i > 30)]
                                    for i, c in enumerate(frames)])))
    script = {f: ({"state": "VISIBLE", "box": [REAL(f)[0] - 8, REAL(f)[1] - 8, REAL(f)[0] + 8, REAL(f)[1] + 8]}, None)
              for f in range(0, 60, 3)}
    rec = label_record(script, split=split)
    rec["clip"].update(name=CLIP, width=W, height=H, sha256=sha)
    labels = tmp / "labels" / "v1"
    labels.mkdir(parents=True)
    raw = json.dumps(rec).encode()
    (labels / "ball_labels_frozen.json").write_bytes(raw)
    (labels / "FROZEN.json").write_text(json.dumps({"record_sha256": hashlib.sha256(raw).hexdigest()}))
    return labels, tiled, base


def image(i):
    img = np.zeros((H, W, 3), np.uint8)
    for d in candidates(i):
        x1, y1, x2, y2 = (int(v) for v in d.box)
        img[y1:y2, x1:x2] = 255
    return img


class Analyse(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.labels, self.tiled, self.base = write_run(Path(self._tmp.name))
        self.report = fp.analyse(self.labels, self.tiled, self.base)

    def test_stretches_are_judged_against_the_labels(self):
        # The false object appears on frame 25; two frames later the lost real-ball hypothesis re-attaches to it
        # (far from its prediction), which the path splits into a stretch of its own: frames 27-59.
        got = {(s["class"], s["confident"]): s for s in self.report["stretches"]}
        self.assertEqual(set(got), {("REAL", True), ("FALSE", True), ("FALSE", False), ("UNLABELLED", False)})
        self.assertEqual(got[("FALSE", False)]["rejected_because"], ["too short"])        # the blip, frames 21-23
        self.assertEqual((got[("UNLABELLED", False)]["first_frame"], got[("UNLABELLED", False)]["last_frame"]),
                         (25, 26))                                                         # no labelled frame in it
        self.assertEqual((got[("FALSE", True)]["longest_run"], got[("FALSE", True)]["sightings"]), (33, 33))
        self.assertEqual(got[("FALSE", True)]["hypothesis"], got[("REAL", True)]["hypothesis"])
        self.assertEqual(got[("REAL", True)]["labelled_wrong"], 0)

    def test_accepted_false_detections(self):
        det = self.report["detections"]
        self.assertEqual([r["frame"] for r in det], list(range(27, 60)))            # every SEEN frame of the object
        self.assertEqual(sum(r["scored_wrong"] for r in det), 11)                  # labelled frames 27, 30, ... 57
        self.assertTrue(all(r["in_tile_overlap"] and (r["w_px"], r["h_px"], r["score"]) == (10.0, 10.0, 0.6)
                            for r in det))
        self.assertEqual(det[0]["box"], [FALSE_AT[0] - 5.0, FALSE_AT[1] - 5.0, FALSE_AT[0] + 5.0, FALSE_AT[1] + 5.0])
        self.assertEqual([r["in_untiled_baseline"]["present"] for r in det], [True] * 4 + [False] * 29)
        self.assertEqual(self.report["tile_grid"], tile_grid(W, H, ON))
        self.assertIsNone(self.report["source_checks"])

    def test_sources_from_a_detector_rerun(self):
        det = SmallBallDetector(max_width=1000)                                    # sees blobs on tiles only
        fp.add_sources(self.report, det, ["ball"], {i: image(i) for i in range(27, 60)})
        srcs = {r["source"]["source"] for r in self.report["detections"]}
        self.assertEqual(srcs, {"merged(tile_left+tile_right)"})                    # the overlap: both tiles see it
        self.assertEqual(self.report["source_checks"], {"detections": 33, "not_reproduced": 0, "merge_boosts": 0,
                                                        "seam_copies": 0})
        self.assertEqual(self.report["detections"][0]["source"]["pass_scores"], {"tile_left": 0.8, "tile_right": 0.8})

    def test_objects_sheets_categories_and_tables(self):
        dcfg = fp.load_config()
        (obj,) = fp.objects_of(self.report)
        self.assertEqual((obj["first"], obj["last"], len(obj["rows"]), obj["persistence"]["longest_run"]),
                         (27, 59, 33, 33))
        out = Path(self._tmp.name)
        sheets = fp.contact_sheets(self.report, {i: image(i) for i in range(25, 60)}, dcfg, out)
        self.assertEqual(len(sheets), 1)
        import cv2
        sheet = cv2.imread(sheets[0])
        self.assertEqual(sheet.shape[:2], (dcfg.cell_px + 22, dcfg.sheet_cols * dcfg.cell_px))   # 6 crops: one row
        csv_path = out / "c.csv"
        fp.write_categories_template(self.report, csv_path)
        with self.assertRaises(SystemExit):                                         # an empty category is refused
            fp.read_categories(csv_path, dcfg)
        csv_path.write_text(csv_path.read_text().replace(",33,,", ",33,boot/foot,"))
        cats = fp.read_categories(csv_path, dcfg)
        self.assertEqual(list(cats.values()), ["boot/foot"])
        text = fp.summarise(self.report, cats)
        for needle in ("Accepted false detections: 33 frames (11 of them", "| boot/foot | 1 | 33 |",
                       "| new with tiling | 29 |", "| also in the untiled baseline | 4 |",
                       "In the overlap of two tiles: 33 of 33", "too short: 1",
                       "| false objects the path ACCEPTED | 1 | 1.28 / 1.28 | 33 / 33 |",
                       "Stretches with no labelled frame (not judged): 1"):
            self.assertIn(needle, text)
        csv_path.write_text(csv_path.read_text().replace("boot/foot", "striker"))
        with self.assertRaises(SystemExit):                                         # generic categories only
            fp.read_categories(csv_path, dcfg)


class Sources(unittest.TestCase):
    def passes(self, squares, **kw):
        img = np.zeros((H, W, 3), np.uint8)
        for x, y in squares:
            img[y - 5:y + 5, x - 5:x + 5] = 255
        return fp.passes_of(SmallBallDetector(max_width=1000, **kw), ["ball"], ON, img)

    def box(self, x, y):
        return [x - 5.0, y - 5.0, x + 5.0, y + 5.0]

    def test_single_tile_full_frame_and_merged(self):
        final, passes = self.passes([(100, 400), (1200, 400)])
        self.assertEqual(fp.source_of(self.box(100, 400), final, passes, ON.nms_iou)["source"], "tile_left")
        self.assertEqual(fp.source_of(self.box(1200, 400), final, passes, ON.nms_iou)["source"], "tile_right")
        final, passes = self.passes([(100, 400)], full_frame_score=0.4)
        s = fp.source_of(self.box(100, 400), final, passes, ON.nms_iou)
        self.assertEqual((s["source"], s["pass_scores"], s["merge_boost"], s["reproduced"]),
                         ("merged(full+tile_left)", {"full": 0.4, "tile_left": 0.8}, False, True))

    def test_seam_copy_and_not_reproduced(self):
        final, passes = self.passes([(716, 400)])            # cut by the left tile's right edge (x = 720)
        s = fp.source_of(self.box(716, 400), final, passes, ON.nms_iou)
        self.assertEqual((s["source"], s["seam_copy"]), ("tile_right", True))
        s = fp.source_of(self.box(300, 300), final, passes, ON.nms_iou)             # nothing there in the re-run
        self.assertEqual((s["source"], s["reproduced"], s["rerun_iou"]), ("none", False, 0.0))

    def test_tile_names(self):
        self.assertEqual(fp.tile_names([(0, 0, 720), (560, 0, 720)]), ["tile_left", "tile_right"])
        self.assertEqual(fp.tile_names([(0, 0, 5), (3, 0, 5), (6, 0, 5)]), ["tile_0", "tile_1", "tile_2"])


class BlobDetector(SmallBallDetector):
    """SmallBallDetector on a decoded (lossy) video: bright pixels count as white."""

    def detect(self, img):
        return super().detect(np.where(img > 128, 255, 0).astype(np.uint8))


class InspectCommand(unittest.TestCase):
    def test_inspect_then_summarise_on_a_video(self):
        import cv2
        from unittest import mock
        from spana.video import sha256_file
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            video = tmp / "clip.avi"
            vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
            for i in range(60):
                vw.write(image(i))
            vw.release()
            labels, tiled, base = write_run(tmp, sha=sha256_file(video))
            out = tmp / "out"
            argv = ["inspect", "--labels", str(labels), "--video", str(video), "--tiled-dir", str(tiled),
                    "--baseline-balldet", str(base), "--config",
                    "records/ball_tiling_v1_2026-11-23/tiling_on.config.yaml", "--dataset-config", "unused",
                    "--model-record", "unused", "--runtime", "torch", "--device", "cpu", "--out", str(out)]
            with mock.patch.object(fp, "_detector", lambda a, b, c: (BlobDetector(max_width=1000), ["ball"])):
                self.assertEqual(fp.main(argv), 0)
                with self.assertRaises(Exception):
                    fp.main(argv)                                                  # never overwrites its outputs
            report = json.loads((out / f"{CLIP}.fp.json").read_text())
            self.assertEqual(report["source_checks"]["detections"], 33)
            self.assertEqual(report["source_checks"]["not_reproduced"], 0)
            self.assertEqual({r["source"]["source"] for r in report["detections"]}, {"merged(tile_left+tile_right)"})
            self.assertEqual(len(report["contact_sheets"]), 1)
            self.assertTrue(Path(report["contact_sheets"][0]).is_file())
            cats = out / f"{CLIP}.categories.csv"
            cats.write_text(cats.read_text().replace(",33,,", ",33,unclear,"))
            self.assertEqual(fp.main(["summarise", "--out", str(out)]), 0)
            with mock.patch.object(fp, "_detector", lambda a, b, c: (BlobDetector(), ["ball"])), \
                    self.assertRaisesRegex(SystemExit, "not the config"):          # the untiled config is refused
                fp.main(argv[:10] + ["config/default.yaml"] + argv[11:-1] + [str(tmp / "out2")])


class Guards(unittest.TestCase):
    def test_dev_only_and_same_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels, tiled, base = write_run(Path(tmp))
            other = json.loads(base.read_text())
            other["video"] = {**other["video"], "sha256": "b" * 64}
            (Path(tmp) / "other.json").write_text(json.dumps(other))
            with self.assertRaises(SystemExit):
                fp.analyse(labels, tiled, Path(tmp) / "other.json")
        with tempfile.TemporaryDirectory() as tmp:
            labels, tiled, base = write_run(Path(tmp), split="EVAL")
            with self.assertRaises(Exception) as e:                               # EVAL labels never load
                fp.analyse(labels, tiled, base)
            self.assertNotIsInstance(e.exception, (KeyError, AttributeError))

    def test_helpers(self):
        self.assertEqual(fp.longest_run([1, 2, 3, 7, 8, 20]), 3)
        self.assertEqual(fp.longest_run([]), 0)
        self.assertEqual(len(fp.sample(list(range(35)), 6)), 6)
        self.assertEqual(fp.sample([1, 2], 6), [1, 2])
        self.assertIn("unclear", fp.load_config().categories)


if __name__ == "__main__":
    unittest.main()
