"""tools/ball_score.py: every outcome category, the centre-distance rule, the detector-layer split, refusals."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from spana.ball_labels import validate_record
from tools.ball_score import DEFAULT_SCORE_CONFIG, load_score_config, on_box, run, score_clip

CFG = load_score_config(str(DEFAULT_SCORE_CONFIG))
CLIP = {"name": "DEVCLIP", "sha256": "a" * 64, "fps": 25.0, "width": 320, "height": 180}
BALL = [100, 100, 110, 110]          # 10 px ball, centre (105, 105): "on" within 10 px


def labels(entries):
    rec = {"schema": "spana.ball_labels", "schema_version": "1.0.0", "split": "DEV", "clip": CLIP, "frame_step": 1,
           "label_version": 1, "annotator": "QA", "frozen_utc": "2026-11-22T00:00:00Z", "frames": entries}
    return rec, validate_record(rec)


def lab(f, state, box=None, point=None, others=()):
    return {"frame": f, "state": state, "box": box, "point": point, "other_balls": list(others)}


def ans(f, state, center=None, reason="X"):
    return {"frame": f, "state": state, "reason": reason, "center": center}


class Categories(unittest.TestCase):
    def test_every_category(self):
        _, L = labels([
            lab(0, "VISIBLE", BALL), lab(1, "VISIBLE", BALL), lab(2, "VISIBLE", BALL, others=[[200, 50, 210, 60]]),
            lab(3, "VISIBLE", BALL), lab(4, "VISIBLE", BALL), lab(5, "HIDDEN"), lab(6, "HIDDEN"), lab(7, "OUT"),
            lab(8, "OUT"), lab(9, "UNSURE"), lab(10, "VISIBLE", BALL)])
        ball = {"frames": [
            ans(0, "OBSERVED", [107, 104]),            # on the ball
            ans(1, "PREDICTED", [112, 108]),           # on (7.6 px)
            ans(2, "OBSERVED", [205, 55]),             # on the other ball
            ans(3, "OBSERVED", [140, 105]),            # 35 px away
            ans(4, "LOST", None, "NO_CONFIRMED_BALL"),
            ans(5, "OBSERVED", [10, 10]), ans(6, "PREDICTED", [10, 10]),
            ans(7, "OBSERVED", [50, 50]), ans(8, "AMBIGUOUS"),
            ans(9, "OBSERVED", [1, 1])]}               # frame 10 not analysed
        r = score_clip(L, ball, CFG)
        c = r["counts"]
        self.assertEqual((c["correct_observed"], c["correct_predicted"]), (1, 1))
        self.assertEqual((c["wrong_other_ball"], c["wrong_position"], c["false_ball"]), (1, 1, 1))
        self.assertEqual((c["missed"], c["observed_while_hidden"], c["predicted_while_hidden"]), (1, 1, 1))
        self.assertEqual((c["quiet_while_out"], c["unscored_unsure"], c["not_analysed"]), (1, 1, 1))
        self.assertEqual((r["claims"], r["wrong"]), (5, 3))         # hidden-frame claims are not in the wrong rate
        self.assertEqual(r["coverage"], round(2 / 5, 4))
        self.assertEqual(r["missed_by_state_reason"], {"LOST/NO_CONFIRMED_BALL": 1})
        self.assertGreater(r["wrong_bound"], r["wrong_rate"])
        self.assertNotIn("detector_recall", r)

    def test_centre_rule_uses_short_side_and_floor(self):
        loose = [100, 100, 130, 110]                   # loose box: short side 10 -> radius 10, not 30
        self.assertTrue(on_box((124, 105), loose, CFG))          # 9 px from centre (115, 105)
        self.assertFalse(on_box((126, 105), loose, CFG))
        tiny = [100, 100, 102, 102]                    # 2 px -> floor 6 px
        self.assertTrue(on_box((106, 101), tiny, CFG))

    def test_detector_layer_split(self):
        _, L = labels([lab(0, "VISIBLE", BALL), lab(1, "VISIBLE", BALL)])
        ball = {"frames": [ans(0, "LOST", None, "NO_CONFIRMED_BALL"), ans(1, "AMBIGUOUS", None, "RIVAL_CANDIDATES")]}
        view = {"model": {"ball_view": True}, "frames": [
            {"frame": 0, "detections": [[101, 101, 109, 109, 0.9, 3, 0]]},          # on the ball, unused
            {"frame": 1, "detections": [[250, 10, 258, 18, 0.9, 3, 0]]}]}           # elsewhere only
        r = score_clip(L, ball, CFG, view)
        self.assertEqual((r["counts"]["missed_candidate_present"], r["counts"]["missed_no_candidate"]), (1, 1))
        self.assertEqual(r["detector_recall"], 0.5)


class Refusals(unittest.TestCase):
    def test_wrong_clip_refused(self):
        td = Path(tempfile.mkdtemp())
        try:
            from tools.freeze_ball_labels import freeze
            pack = td / "pack"
            pack.mkdir()
            (pack / "manifest.json").write_text(json.dumps({"split": "DEV", "clip": CLIP,
                                                            "frames": [{"index": 0}]}))
            rec, _ = labels([lab(0, "OUT")])
            (td / "rec.json").write_text(json.dumps(rec))
            dest = freeze(str(pack), str(td / "rec.json"), None, root=td / "repo")
            ball = {"schema": "spana.ball", "schema_version": "1.0.0", "video": {"sha256": "b" * 64}}
            (td / "ball.json").write_text(json.dumps(ball))
            from unittest import mock
            with mock.patch("tools.ball_score.validate_ball"), self.assertRaises(SystemExit) as e:
                run([[str(dest), str(td / "ball.json")]], str(DEFAULT_SCORE_CONFIG))
            self.assertIn("not from the labelled clip", str(e.exception))
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
