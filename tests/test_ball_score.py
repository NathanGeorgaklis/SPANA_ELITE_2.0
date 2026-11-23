"""tools/ball_score.py: spana.ball against frozen DEV ball labels, by centre distance, report only."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import validate_ball
from spana.ball_labels import load_frozen, validate_record
from spana.config import ConfigError
from spana.reid_labels import ReidLabelError
from tools.ball_score import (DEFAULT_CONFIG, BallScoreConfig, _centre, _ranks, load_config, main, pooled,
                              score_clip)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(DEFAULT_CONFIG)
W, H, SHA = 1920, 1080, "a" * 64
R = CFG.hit_radius_frac_height * H                      # 10.8 px


def sq(cx, cy, half=12.0):
    return [cx - half, cy - half, cx + half, cy + half]


# frame -> (label entry, ball answer (state, centre or None))
SCRIPT = {
    0: ({"state": "VISIBLE", "box": sq(100, 100)}, ("OBSERVED", (105, 100))),            # HIT_OBSERVED, 5 px
    3: ({"state": "VISIBLE", "box": sq(200, 200)}, ("OBSERVED", (230, 200))),            # WRONG_POSITION
    6: ({"state": "VISIBLE", "box": sq(300, 300), "other_balls": [sq(500, 500, 8)]},
        ("OBSERVED", (502, 500))),                                                       # WRONG_OTHER_BALL
    9: ({"state": "VISIBLE", "box": sq(600, 300)}, ("PREDICTED", (600 + R, 300))),        # HIT_PREDICTED (on radius)
    12: ({"state": "VISIBLE", "box": sq(700, 300)}, ("LOST", None)),                     # MISS
    15: ({"state": "HIDDEN", "point": [400, 400]}, ("OBSERVED", (400, 400))),            # WRONG_CLAIMED_SEEN
    18: ({"state": "HIDDEN", "point": [400, 400]}, ("PREDICTED", (403, 404))),           # PREDICTED_HIDDEN, 5 px
    21: ({"state": "HIDDEN"}, ("LOST", None)),                                           # ABSTAIN_OK
    24: ({"state": "OUT"}, ("PREDICTED", (50, 50))),                                     # WRONG_PHANTOM
    27: ({"state": "OUT"}, ("LOST", None)),                                              # ABSTAIN_OK
    30: ({"state": "UNSURE"}, ("OBSERVED", (900, 900))),                                 # not scored
    33: ({"state": "VISIBLE", "box": sq(800, 300)}, ("AMBIGUOUS", None)),                # MISS
}
EXPECTED = {0: "HIT_OBSERVED", 3: "WRONG_POSITION", 6: "WRONG_OTHER_BALL", 9: "HIT_PREDICTED", 12: "MISS",
            15: "WRONG_CLAIMED_SEEN", 18: "PREDICTED_HIDDEN", 21: "ABSTAIN_OK", 24: "WRONG_PHANTOM",
            27: "ABSTAIN_OK", 33: "MISS"}


def label_record(script, sha=SHA, split="DEV") -> dict:
    frames = [{"frame": f, "state": e["state"], "box": e.get("box"), "point": e.get("point"),
               "other_balls": e.get("other_balls", [])} for f, (e, _) in sorted(script.items())]
    return {"schema": "spana.ball_labels", "schema_version": "1.0.0", "split": split,
            "clip": {"name": "SYNTH", "sha256": sha, "fps": 30, "width": W, "height": H}, "frame_step": 3,
            "label_version": 1, "annotator": "test", "frozen_utc": "2026-11-22T00:00:00Z", "frames": frames}


_REASON = {"OBSERVED": "CONTINUING", "PREDICTED": "SHORT_GAP", "LOST": "NO_CONFIRMED_BALL",
           "AMBIGUOUS": "RIVAL_CANDIDATES"}


def ball_doc(answers: dict, n: int, sha=SHA, w=W, h=H) -> dict:
    """A valid spana.ball document: `answers` frame -> (state, centre); other frames LOST."""
    frames = []
    for f in range(n):
        state, c = answers.get(f, ("LOST", None))
        on = state in ("OBSERVED", "PREDICTED")
        frames.append({"frame": f, "t": round(f / 30, 6), "camera_status": "FIRST" if f == 0 else "OK",
                       "state": state, "reason": _REASON[state], "hypothesis": 1 if on else None,
                       "center": [float(c[0]), float(c[1])] if on else None, "diameter": 15.0 if on else None,
                       "confidence": 0.8 if on else None, "contested": False,
                       "candidates": int(state == "OBSERVED"), "confirmed_live": 1})
    obs = [f for f in range(n) if answers.get(f, ("LOST",))[0] == "OBSERVED"] or [0]
    doc = {"schema": "spana.ball", "schema_version": "1.0.0", "provenance": {}, "detections_sha256": "0" * 64,
           "video": {"path": "synthetic", "sha256": sha, "fps": 30.0, "width": w, "height": h,
                     "frame_count_reported": n, "timestamp_source": "synthetic"},
           "ball_config": {}, "frames": frames,
           "hypotheses": [{"hypothesis": 1, "first_frame": 0, "last_observed_frame": obs[-1],
                           "observed_frames": len(obs), "confirmed_frame": 0, "end_frame": None,
                           "end_reason": None}]}
    validate_ball(doc)
    return doc


def scripted():
    rec = label_record(SCRIPT)
    return rec, validate_record(rec), ball_doc({f: a for f, (_, a) in SCRIPT.items()}, 34)


class Outcomes(unittest.TestCase):
    def test_every_cell_of_the_table(self):
        rec, labels, ball = scripted()
        out = score_clip(rec, labels, ball, CFG)
        self.assertEqual({r["frame"]: r["outcome"] for r in out["frames"]}, EXPECTED)
        self.assertEqual(out["unsure_frames"], 1)
        self.assertEqual({r["frame"] for r in out["wrong_frames"]}, {3, 6, 15, 24})

    def test_summary(self):
        s = score_clip(*scripted(), CFG)["summary"]
        self.assertEqual(s["frames_scored"], 11)
        self.assertEqual(s["labels"], {"VISIBLE": 6, "HIDDEN": 3, "OUT": 2})
        self.assertAlmostEqual(s["coverage_observed"], round(1 / 6, 5))
        self.assertAlmostEqual(s["coverage_answered"], round(2 / 6, 5))
        self.assertEqual((s["wrong_observed"]["answers"], s["wrong_observed"]["wrong"]), (4, 3))
        self.assertEqual((s["wrong_predicted"]["answers"], s["wrong_predicted"]["wrong"]), (3, 1))
        self.assertEqual((s["wrong_answered"]["answers"], s["wrong_answered"]["wrong"]), (7, 4))
        self.assertGreater(s["wrong_answered"]["upper_bound"], 4 / 7)
        self.assertEqual(s["hit_distance_observed"]["px"]["median"], 5.0)
        self.assertEqual(s["hit_distance_observed"]["frac_height"]["median"], round(5 / H, 5))
        self.assertEqual(s["hidden_predicted_distance"]["px"]["max"], 5.0)

    def test_near_the_ball_and_a_spare_ball_is_a_hit(self):
        script = {0: ({"state": "VISIBLE", "box": sq(100, 100), "other_balls": [sq(106, 100, 8)]},
                      ("OBSERVED", (103, 100)))}
        rec = label_record(script)
        out = score_clip(rec, validate_record(rec), ball_doc({0: script[0][1]}, 1), CFG)
        self.assertEqual(out["frames"][0]["outcome"], "HIT_OBSERVED")

    def test_percentiles_are_nearest_rank(self):
        self.assertEqual(_ranks([4.0, 1.0, 3.0, 2.0], 2), {"median": 2.0, "p90": 4.0, "max": 4.0})
        self.assertEqual(_ranks([float(k) for k in range(1, 21)], 2), {"median": 10.0, "p90": 18.0, "max": 20.0})

    def test_radius_comes_from_config(self):
        tight = BallScoreConfig(CFG.schema_version, 4.0 / H, CFG.bound_confidence)      # 4 px: frame 0 misses
        out = score_clip(*scripted(), tight)
        self.assertEqual(out["frames"][0]["outcome"], "WRONG_POSITION")


class Refusals(unittest.TestCase):
    def test_other_video_or_size(self):
        rec, labels, ball = scripted()
        for b in (ball_doc({}, 34, sha="b" * 64), ball_doc({}, 34, w=1280, h=720)):
            with self.assertRaises(SystemExit):
                score_clip(rec, labels, b, CFG)

    def test_ball_document_without_video_fields(self):
        rec, labels, ball = scripted()
        for k in ("sha256", "width", "height"):
            b = copy.deepcopy(ball)
            del b["video"][k]
            with self.subTest(k), self.assertRaises(SystemExit):
                score_clip(rec, labels, b, CFG)

    def test_same_video_twice_is_refused_when_pooled(self):
        c = score_clip(*scripted(), CFG)
        with self.assertRaises(SystemExit):
            pooled([c, c], CFG)

    def test_uncovered_labelled_frame(self):
        rec, labels, _ = scripted()
        with self.assertRaises(SystemExit):
            score_clip(rec, labels, ball_doc({}, 30), CFG)            # frame 33 labelled, not in the document

    def test_eval_labels_refused(self):
        rec, labels, ball = scripted()
        rec = {**copy.deepcopy(rec), "split": "EVAL"}
        with self.assertRaises(ReidLabelError):
            score_clip(rec, labels, ball, CFG)

    def test_bad_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text in ('schema_version: "1.0.0"\nhit_radius_frac_height: 0\nbound_confidence: 0.95\n',
                         'schema_version: "1.0.0"\nhit_radius_frac_height: 0.01\n',
                         'schema_version: "9.9.9"\nhit_radius_frac_height: 0.01\nbound_confidence: 0.95\n'):
                p = Path(tmp) / "c.yaml"
                p.write_text(text)
                with self.subTest(text), self.assertRaises(ConfigError):
                    load_config(p)


class Pooling(unittest.TestCase):
    def test_two_clips_of_different_heights(self):
        a = score_clip(*scripted(), CFG)                                   # 1080p: frame 0 hit at 5 px
        script = {0: ({"state": "VISIBLE", "box": sq(100, 100)}, ("OBSERVED", (102, 100)))}
        rec = label_record(script, sha="c" * 64)
        rec["clip"].update(width=1280, height=720)
        b = score_clip(rec, validate_record(rec), ball_doc({0: script[0][1]}, 1, sha="c" * 64, w=1280, h=720), CFG)
        p = pooled([a, b], CFG)
        self.assertEqual(p["frames_scored"], a["summary"]["frames_scored"] + 1)
        self.assertEqual(p["outcomes"]["HIT_OBSERVED"], 2)
        d = p["hit_distance_observed"]
        self.assertNotIn("px", d)                                         # px of two resolutions do not pool
        self.assertEqual(d["frac_height"]["max"], round(5 / 1080, 5))
        self.assertIn("px", pooled([a], CFG)["hit_distance_observed"])


class FrozenDevLabels(unittest.TestCase):
    """The real frozen DEV labels load and score: a perfect tracker (OBSERVED on every VISIBLE centre, LOST
    elsewhere) gets coverage 1 and no wrong answer on each clip. Uses no inference output."""

    def test_perfect_tracker_on_each_clip(self):
        dirs = sorted((ROOT / "data" / "ball_labels").glob("*/v1"))
        self.assertTrue(dirs)
        for d in dirs:
            rec, labels = load_frozen(d)
            c = rec["clip"]
            answers = {f: ("OBSERVED", _centre(lab.box)) for f, lab in labels.items() if lab.state == "VISIBLE"}
            ball = ball_doc(answers, max(labels) + 1, sha=c["sha256"], w=c["width"], h=c["height"])
            s = score_clip(rec, labels, ball, CFG)["summary"]
            with self.subTest(clip=c["name"]):
                self.assertEqual(s["coverage_observed"], 1.0)
                self.assertEqual(s["wrong_answered"]["wrong"], 0)
                self.assertEqual(s["frames_scored"], sum(lab.state != "UNSURE" for lab in labels.values()))


class Cli(unittest.TestCase):
    def test_end_to_end_and_never_overwrites(self):
        rec, _, ball = scripted()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "SYNTH" / "v1"
            d.mkdir(parents=True)
            raw = json.dumps(rec).encode()
            (d / "ball_labels_frozen.json").write_bytes(raw)
            (d / "FROZEN.json").write_text(json.dumps({"record_sha256": hashlib.sha256(raw).hexdigest()}))
            b, out = Path(tmp) / "b.json", Path(tmp) / "report.json"
            b.write_text(json.dumps(ball))
            argv = ["--clip", str(d), str(b), "--out", str(out)]
            self.assertEqual(main(argv), 0)
            rep = json.loads(out.read_text())
            self.assertEqual(rep["pooled"]["wrong_answered"]["wrong"], 4)
            self.assertEqual(rep["config"]["hit_radius_frac_height"], CFG.hit_radius_frac_height)
            with self.assertRaises(Exception):
                main(argv)


if __name__ == "__main__":
    unittest.main()
