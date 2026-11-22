"""Possession / touches v1 (spana/possession/touches.py, spana.touches 1.0.0) on scripted scenes: who is on the ball
per frame, fail closed, and the possessions built from it."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import ArtifactError, sha256_doc, validate_touches
from spana.cli import main as cli_main
from spana.config import ConfigError, config_from_dict
from spana.possession.touches import possessions, touches_document
from tests.synth import cfg
from tests.test_thresholds_in_config import with_keys
from tests.touches_synth import (SCENES, ball_doc, ball_behind_player, camera_cut, crossing_uncertain, dribble,
                                 fifty_fifty, lofted_over_player, occluded_dribble, tracks_doc)
from tools.touches_scenarios import score_scene

PC = cfg().possession


def run(scene):
    s = scene()
    tdoc, bdoc = tracks_doc(s), ball_doc(s)
    doc, log = touches_document(tdoc, bdoc, PC, {"note": "test"}, sha256_doc(tdoc), sha256_doc(bdoc))
    validate_touches(doc, tdoc, bdoc)
    return s, tdoc, bdoc, doc, log


def answers(doc, frames):
    return {f["frame"]: (f["state"], f["reason"], f["track_id"]) for f in doc["frames"] if f["frame"] in frames}


class ScenarioTable(unittest.TestCase):
    def test_every_scene_as_scripted_and_never_wrong(self):
        for make in SCENES:
            r = score_scene(make(), cfg())
            with self.subTest(scene=r["scene"]):
                self.assertEqual(r["wrong"], 0, r["wrong_frames"])
                self.assertEqual(r["mismatch"], 0, r["mismatch_frames"])
                if r["possessions_ok"] is not None:
                    self.assertTrue(r["possessions_ok"], (r["possessions_found"], r["possessions_scripted"]))


class Reasons(unittest.TestCase):
    def test_dribble_is_control_after_confirmation(self):
        _, _, _, doc, _ = run(dribble)
        self.assertEqual(answers(doc, {0, 1, 49}), {0: ("UNKNOWN", "TRACK_NOT_CERTAIN", None),
                                                    1: ("CONTROL", "NEAREST_FEET", "T1"),
                                                    49: ("CONTROL", "NEAREST_FEET", "T1")})

    def test_ball_in_the_air_over_a_player_is_never_control(self):
        _, _, _, doc, _ = run(lofted_over_player)
        near = {f: a for f, a in answers(doc, set(range(40))).items() if a[0] == "UNKNOWN"}
        self.assertEqual(sorted(near), [19, 20, 21])
        self.assertTrue(all(a[1] == "BALL_AIRBORNE_SUSPECT" for a in near.values()))

    def test_ball_far_behind_a_player_is_never_control(self):
        _, _, _, doc, _ = run(ball_behind_player)
        a = answers(doc, set(range(30)))
        self.assertEqual({v[1] for f, v in a.items() if 5 <= f <= 22}, {"BALL_SCALE_MISMATCH"})
        self.assertNotIn("CONTROL", {v[0] for v in a.values()})

    def test_fifty_fifty_is_contested_with_both_candidates(self):
        _, _, _, doc, _ = run(fifty_fifty)
        fr = doc["frames"][20]
        self.assertEqual((fr["state"], fr["reason"], fr["track_id"]), ("CONTESTED", "RIVAL_WITHIN_MARGIN", None))
        self.assertEqual([c["track_id"] for c in fr["candidates"]], ["T2", "T1"])

    def test_uncertain_identity_near_the_ball_is_unknown(self):
        _, _, _, doc, _ = run(crossing_uncertain)
        self.assertEqual({v[1] for v in answers(doc, set(range(24, 37))).values()}, {"TRACK_NOT_CERTAIN"})

    def test_hidden_ball_is_unknown_and_short_gaps_bridge(self):
        _, _, _, doc, _ = run(occluded_dribble)
        self.assertEqual({v[1] for v in answers(doc, set(range(15, 20))).values()}, {"BALL_NOT_OBSERVED"})
        self.assertEqual(doc["possessions"][0], {"track_id": "T1", "first_frame": 1, "last_frame": 29,
                                                 "frames_control": 24, "frames_gap": 5})

    def test_camera_cut_is_a_discontinuity(self):
        _, _, _, doc, log = run(camera_cut)
        self.assertEqual(answers(doc, {25})[25], ("UNKNOWN", "CAMERA_DISCONTINUITY", None))
        self.assertEqual(doc["frames"][25]["candidates"], [])
        self.assertNotIn(25, {r["frame"] for r in log})

    def test_contested_ball_answer_is_unknown(self):
        s = dribble()
        ball = s.ball
        s.ball = lambda f: ball(f)[:3] + (f == 10,)
        tdoc, bdoc = tracks_doc(s), ball_doc(s)
        doc, _ = touches_document(tdoc, bdoc, PC, {}, sha256_doc(tdoc), sha256_doc(bdoc))
        self.assertEqual(answers(doc, {10})[10], ("UNKNOWN", "BALL_CONTESTED", None))
        self.assertEqual(len(doc["possessions"]), 1)                    # one frame: bridged


class Possessions(unittest.TestCase):
    @staticmethod
    def fr(f, state, tid=None, reason="X"):
        return {"frame": f, "t": f / 25.0, "state": state, "reason": reason, "track_id": tid}

    def test_breaks(self):
        c = lambda f, tid="T1": self.fr(f, "CONTROL", tid)          # noqa: E731
        cases = {
            "camera": [c(0), self.fr(1, "UNKNOWN", reason="CAMERA_DISCONTINUITY"), c(2)],
            "contested": [c(0), self.fr(1, "CONTESTED"), c(2)],
            "other track": [c(0), c(1, "T2"), c(2)],
        }
        for name, frames in cases.items():
            with self.subTest(name):
                self.assertGreaterEqual(len(possessions(frames, PC)), 2)

    def test_bridge_limit(self):
        n = round(PC.bridge_s * 25)                                    # CONTROL frames exactly bridge_s apart
        gap = [self.fr(k, "FREE", reason="NO_TRACK_IN_REACH") for k in range(1, n)]
        ok = [self.fr(0, "CONTROL", "T1")] + gap + [self.fr(n, "CONTROL", "T1")]
        self.assertEqual(len(possessions(ok, PC)), 1)
        far = ok[:-1] + [self.fr(n, "FREE", reason="NO_TRACK_IN_REACH"), self.fr(n + 1, "CONTROL", "T1")]
        self.assertEqual(len(possessions(far, PC)), 2)


class Validation(unittest.TestCase):
    def setUp(self):
        _, self.tdoc, self.bdoc, self.doc, _ = run(dribble)

    def refused(self, doc, *src):
        with self.assertRaises(ArtifactError):
            validate_touches(doc, *src)

    def test_tampering_is_refused(self):
        d = copy.deepcopy(self.doc)
        d["frames"][10]["track_id"] = "T2"                             # CONTROL handed to someone else
        self.refused(d)
        d = copy.deepcopy(self.doc)
        d["frames"][10]["candidates"][0]["usable"] = False             # CONTROL on an unusable track
        self.refused(d)
        d = copy.deepcopy(self.doc)
        d["possessions"][0]["last_frame"] = 40
        self.refused(d)
        d = copy.deepcopy(self.doc)
        d["frames"][10]["candidates"][0]["distance_h"] = 0.2           # plausible, but not what the inputs give
        validate_touches(d)
        self.refused(d, self.tdoc, self.bdoc)

    def test_inputs_must_match(self):
        self.refused(self.doc, self.tdoc, ball_doc(lofted_over_player()))   # other ball artifact
        self.refused(self.doc, self.tdoc, None)
        other = ball_doc(lofted_over_player())                            # 40 frames vs 50
        with self.assertRaises(ValueError):
            touches_document(self.tdoc, other, PC, {}, "", "")
        moved = copy.deepcopy(self.bdoc)
        moved["video"]["sha256"] = "2" * 64
        with self.assertRaises(ValueError):
            touches_document(self.tdoc, moved, PC, {}, "", "")


class Config(unittest.TestCase):
    def test_bad_possession_config_refused(self):
        for bad in ({"diameter_to_height_min": 0.3}, {"bridge_s": -0.1}, {"reach_h": 0}, {"reach_h": None}):
            with self.subTest(bad), self.assertRaises(ConfigError):
                config_from_dict(with_keys(possession=bad))


class Cli(unittest.TestCase):
    def test_touches_command(self):
        s = occluded_dribble()
        with tempfile.TemporaryDirectory() as tmp:
            t, b, out = Path(tmp) / "x.tracks.json", Path(tmp) / "x.ball.json", Path(tmp) / "x.touches.json"
            t.write_text(json.dumps(tracks_doc(s)))
            b.write_text(json.dumps(ball_doc(s)))
            argv = ["touches", "--tracks", str(t), "--ball", str(b), "--config", "config/default.yaml",
                    "--out", str(out)]
            self.assertEqual(cli_main(argv), 0)
            doc = json.loads(out.read_text())
            validate_touches(doc, json.loads(t.read_text()), json.loads(b.read_text()))
            self.assertEqual(len(doc["possessions"]), 2)
            self.assertTrue(out.with_suffix(".decisions.jsonl").read_text().strip())
            with self.assertRaises(Exception):
                cli_main(argv)                                         # never overwrites


if __name__ == "__main__":
    unittest.main()
