"""Ball layer v1: ball view, ball tracker (synthetic sequences with ground truth), spana.ball artifact, CLI.
No network, no real footage: generic scenes only."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from spana import cli
from spana.artifacts import ArtifactError, read_json, sha256_doc, validate_ball, validate_detections
from spana.ball.tracker import BallTracker
from spana.config import ConfigError, config_from_dict, load_config
from spana.detection.ball_view import BallView, BallViewError
from spana.records import BallReason, BallState, CameraObservation, CameraStatus, Detection
from tests.synth import CONFIG_PATH
from tests.test_cli_e2e import write_video
from tests.test_person_view import NAMES, FakeClient, FakeDetector
from tests.test_roboflow_prep import raw as raw_dataset_cfg

CFG = load_config(CONFIG_PATH)
B = CFG.ball
FPS = 25.0
W, H = 1280, 720


def ball(x, y, d=10.0, score=0.8):
    return Detection((x - d / 2, y - d / 2, x + d / 2, y + d / 2), score, 3)


def cam(status=CameraStatus.OK, dx=0.0, dy=0.0):
    return CameraObservation(status, ((1.0, 0.0, dx), (0.0, 1.0, dy)), 100, 0.0)


def run(frames, cfg=B, cams=None):
    """frames: list of candidate lists, one per frame at FPS. Returns the BallFrame answers."""
    trk = BallTracker(cfg, W, H)
    out = []
    for i, cands in enumerate(frames):
        c = cams[i] if cams else (cam(CameraStatus.FIRST) if i == 0 else cam())
        out.append(trk.update(i, i / FPS, cands, c))
    return out, trk


def run_with_players(frames, players, cfg=B):
    """Like run(), with the same player boxes on every frame (None = no player knowledge)."""
    trk = BallTracker(cfg, W, H)
    return [trk.update(i, i / FPS, cands, cam(CameraStatus.FIRST) if i == 0 else cam(), players)
            for i, cands in enumerate(frames)], trk


def states(out):
    return [f.state for f in out]


class BallTrackerTests(unittest.TestCase):
    def test_moving_ball_confirms_then_observed(self):
        out, _ = run([[ball(100 + 8 * i, 300)] for i in range(10)])
        k = B.confirm_min_obs - 1
        self.assertTrue(all(f.state is BallState.LOST for f in out[:k]))       # not yet confirmed: no answer
        self.assertTrue(all(f.state is BallState.OBSERVED for f in out[k:]))
        self.assertEqual(len({f.hypothesis for f in out[k:]}), 1)
        self.assertAlmostEqual(out[-1].center[0], 100 + 8 * 9)

    def test_lone_blip_never_answered(self):
        out, trk = run([[ball(400, 400)]] + [[] for _ in range(20)])
        self.assertTrue(all(f.state is BallState.LOST for f in out))
        self.assertEqual(trk.summaries()[0].end_reason, "UNCONFIRMED_EXPIRED")

    def test_short_gap_predicted_then_lost(self):
        n_seen = int(B.seen_share_window_s * FPS) + 1              # followed for longer than the seen-share window
        # (the tracker compares gap > predict_max_s without a tolerance: n_seen keeps the original float rounding)
        seen = [[ball(100 + 8 * i, 300)] for i in range(n_seen)]
        out, _ = run(seen + [[] for _ in range(20)])
        n_pred = int(B.predict_max_s * FPS + 1e-9)
        gap = out[n_seen:]
        self.assertTrue(all(f.state is BallState.PREDICTED for f in gap[:n_pred]))
        self.assertAlmostEqual(gap[0].center[0], 100 + 8 * n_seen, delta=1.0)   # extrapolated with its velocity
        self.assertTrue(gap[0].confidence > gap[n_pred - 1].confidence)
        self.assertEqual(gap[n_pred].state, BallState.LOST)
        self.assertEqual(gap[n_pred].reason, BallReason.GAP_TOO_LONG)

    def test_flickering_false_object_is_never_the_answer(self):
        # seen 3 frames in a row (confirmed), then only every 4th frame: kept alive, but sparse -> never answered
        cands = [[ball(600, 400)] if i < 3 or i % 4 == 0 else [] for i in range(60)]
        out, trk = run(cands)
        late = out[int(B.seen_share_window_s * FPS):]
        self.assertTrue(all(f.state is BallState.LOST for f in late))
        self.assertIn(BallReason.SPARSE_OBSERVATIONS, {f.reason for f in late})
        self.assertNotIn(BallState.OBSERVED, {f.state for f in late})

    def test_flickering_object_does_not_take_the_answer_from_nothing_but_ball_still_wins(self):
        # the real ball moves and is seen every frame; a sparse false object never takes over and never answers
        cands = [[ball(100 + 6 * i, 300)] + ([ball(900, 500)] if i < 3 or i % 4 == 0 else []) for i in range(60)]
        out, _ = run(cands)
        ans = [f for f in out if f.state in (BallState.OBSERVED, BallState.PREDICTED)]
        self.assertTrue(ans)
        self.assertTrue(all(abs(f.center[1] - 300) < 1 for f in ans))        # only ever the real ball

    def test_young_ball_hidden_goes_sparse_before_predict_max(self):
        seen = [[ball(100 + 8 * i, 300)] for i in range(6)]          # followed 0.24 s, then hidden
        out, _ = run(seen + [[] for _ in range(20)])
        gap = out[6:]
        self.assertEqual(gap[0].state, BallState.PREDICTED)
        first_lost = next(f for f in gap if f.state is BallState.LOST)
        self.assertEqual(first_lost.reason, BallReason.SPARSE_OBSERVATIONS)
        self.assertLess(first_lost.frame_index - 6, int(B.predict_max_s * FPS))

    def test_seen_share_comes_from_config(self):
        cands = [[ball(600, 400)] if i < 3 or i % 4 == 0 else [] for i in range(60)]
        loose = dataclasses.replace(B, min_seen_share=0.2)
        out, _ = run(cands, cfg=loose)
        self.assertIn(BallState.OBSERVED, {f.state for f in out[30:]})

    # ---- player-aware re-acquisition (patterns of the DEV errors, records/ball_player_aware_2026-11-23)
    P1 = (480.0, 140.0, 540.0, 310.0)          # the player with the ball at his feet (feet y = 310)
    P2 = (270.0, 150.0, 330.0, 330.0)          # another player ~200 px away

    def at_feet(self, i):
        return ball(500 + 2 * i, 300)           # inside P1's box, in the feet zone

    def test_jump_to_a_steady_detection_on_another_players_body(self):
        # ball at P1's feet, unseen 4 frames, then a steady detection on P2's lower back (mid-body)
        n = 30
        frames = [[self.at_feet(i)] for i in range(n)] + [[] for _ in range(4)] + \
                 [[ball(300, 235, score=0.35)] for _ in range(6)]
        blind, _ = run_with_players(frames, None)
        self.assertEqual(blind[-1].state, BallState.OBSERVED)                    # the bug: it takes the jersey
        self.assertAlmostEqual(blind[-1].center[0], 300, delta=1)
        aware, trk = run_with_players(frames, [self.P1, self.P2])
        self.assertFalse([f for f in aware[n:] if f.state is BallState.OBSERVED])
        self.assertTrue(all(abs(f.center[0] - 300) > 100 for f in aware[n:] if f.center))
        self.assertEqual(len(trk.summaries()), 1)                                # the jersey never became a ball

    def test_body_detection_never_starts_a_ball(self):
        glove = [[ball(300, 200, score=0.5)] for _ in range(40)]                 # steady, on P2's upper body
        self.assertIn(BallState.OBSERVED, set(states(run_with_players(glove, None)[0])))
        out, trk = run_with_players(glove, [self.P2])
        self.assertEqual(set(states(out)), {BallState.LOST})
        self.assertEqual(trk.summaries(), [])
        self.assertTrue(all(d["reason"] == "ON_BODY" for f in out for d in f.decisions if d["kind"] == "blocked"))

    def test_single_far_detection_after_a_gap_is_not_the_ball(self):
        # ball at P1's feet, unseen 3 frames, one detection 250 px away in the open (not on a body), then nothing
        n = 30
        frames = [[self.at_feet(i)] for i in range(n)] + [[] for _ in range(3)] + [[ball(810, 290, score=0.42)]] + \
                 [[] for _ in range(3)]
        blind, _ = run_with_players(frames, None)
        self.assertEqual(blind[n + 3].state, BallState.OBSERVED)                 # the bug
        aware, _ = run_with_players(frames, [self.P1])
        self.assertNotEqual(aware[n + 3].state, BallState.OBSERVED)
        self.assertTrue(all(abs(f.center[0] - 810) > 100 for f in aware[n:] if f.center))

    def test_ball_back_near_its_prediction_rejoins_at_once(self):
        n = 30
        frames = [[self.at_feet(i)] for i in range(n)] + [[] for _ in range(3)] + \
                 [[self.at_feet(i)] for i in range(n + 3, n + 8)]
        out, _ = run_with_players(frames, [self.P1])
        self.assertTrue(all(f.state is BallState.OBSERVED for f in out[n + 3:]))
        self.assertEqual(len({f.hypothesis for f in out if f.hypothesis}), 1)

    def test_ball_followed_across_a_body_stays_observed(self):
        # a ball in the air passes in front of P2's body, detected on every frame
        frames = [[ball(150 + 10 * i, 240)] for i in range(40)]                  # x 150 -> 540 crosses P2 (270-330)
        out, _ = run_with_players(frames, [self.P2])
        k = B.confirm_min_obs - 1
        self.assertTrue(all(f.state is BallState.OBSERVED for f in out[k:]))
        self.assertEqual(len({f.hypothesis for f in out[k:]}), 1)

    def test_ball_reappearing_far_away_is_confirmed_as_a_new_ball(self):
        # a real pass missed for a few frames: seen steadily far from the prediction -> answered again, new hypothesis
        n = 30
        frames = [[self.at_feet(i)] for i in range(n)] + [[] for _ in range(4)] + \
                 [[ball(800 + 6 * i, 320)] for i in range(40)]
        out, _ = run_with_players(frames, [self.P1])
        late = [f for f in out[-10:]]
        self.assertTrue(all(f.state is BallState.OBSERVED and f.center[0] > 700 for f in late))
        self.assertEqual(len({f.hypothesis for f in out if f.hypothesis}), 2)

    def test_feet_zone_and_rejoin_distance_come_from_config(self):
        glove = [[ball(300, 200, score=0.5)] for _ in range(40)]
        out, _ = run_with_players(glove, [self.P2], cfg=dataclasses.replace(B, feet_zone_h=0.99))
        self.assertIn(BallState.OBSERVED, set(states(out)))                      # everything is "feet": not blocked
        n = 30
        far = [[self.at_feet(i)] for i in range(n)] + [[] for _ in range(3)] + [[ball(810, 290)]] + [[]]
        out, _ = run_with_players(far, [self.P1], cfg=dataclasses.replace(B, reacquire_max_d=1000.0))
        self.assertEqual(out[n + 3].state, BallState.OBSERVED)

    def test_hidden_ball_reacquired_on_same_hypothesis(self):
        seen = [[ball(100 + 8 * i, 300)] for i in range(6)]
        back = [[ball(100 + 8 * i, 300)] for i in range(10, 14)]
        out, _ = run(seen + [[] for _ in range(4)] + back)
        self.assertEqual(out[-1].state, BallState.OBSERVED)
        self.assertEqual(out[-1].hypothesis, out[5].hypothesis)

    def test_kick_changes_velocity_within_speed_bound(self):
        a = [[ball(300 + 4 * i, 300)] for i in range(6)]                       # slow dribble
        b = [[ball(320 + 40 * j, 300 - 10 * j)] for j in range(1, 8)]          # struck: 10x faster, new direction
        out, _ = run(a + b)
        self.assertTrue(all(f.state is BallState.OBSERVED for f in out[B.confirm_min_obs - 1:]))
        self.assertEqual(len({f.hypothesis for f in out[B.confirm_min_obs - 1:]}), 1)

    def test_camera_pan_compensated(self):
        # The ball is still on the pitch while the camera pans 20 px / frame: it moves in the image only by the pan.
        frames = [[ball(600 - 20 * i, 300)] for i in range(12)]
        cams = [cam(CameraStatus.FIRST)] + [cam(dx=-20.0) for _ in range(11)]
        frames[8] = []                                                          # hidden one frame mid-pan
        out, _ = run(frames, cams=cams)
        self.assertEqual(out[8].state, BallState.PREDICTED)
        self.assertAlmostEqual(out[8].center[0], 600 - 20 * 8, delta=1.0)
        self.assertEqual(out[-1].hypothesis, out[5].hypothesis)

    def test_camera_cut_drops_everything(self):
        frames = [[ball(100 + 8 * i, 300)] for i in range(12)]
        cams = [cam(CameraStatus.FIRST)] + [cam() for _ in range(5)] + [cam(CameraStatus.CUT)] + \
            [cam() for _ in range(5)]
        out, trk = run(frames, cams=cams)
        self.assertEqual(out[6].state, BallState.LOST)
        self.assertEqual(out[6].reason, BallReason.CAMERA_DISCONTINUITY)
        self.assertNotEqual(out[-1].hypothesis, out[5].hypothesis)               # re-confirmed from scratch
        self.assertEqual(trk.summaries()[0].end_reason, "CAMERA_DISCONTINUITY")

    def test_static_false_positive_does_not_steal_the_ball(self):
        # A confirmed static blob (pitch spot) next to the moving ball: the ball, once selected, keeps the answer;
        # it is marked contested while both are observed.
        frames = [[ball(100 + 8 * i, 300)] for i in range(4)] + \
            [[ball(100 + 8 * i, 300), ball(900, 500, score=0.9)] for i in range(4, 12)]
        out, _ = run(frames)
        sel = out[3].hypothesis
        self.assertTrue(all(f.hypothesis == sel and f.state is BallState.OBSERVED for f in out[3:]))
        self.assertTrue(out[-1].contested)

    def test_two_balls_from_the_start_ambiguous(self):
        frames = [[ball(100 + 8 * i, 300), ball(900 - 8 * i, 500)] for i in range(10)]
        out, _ = run(frames)
        self.assertTrue(all(f.state in (BallState.LOST, BallState.AMBIGUOUS) for f in out))
        self.assertEqual(out[-1].reason, BallReason.RIVAL_CANDIDATES)

    def test_no_silent_switch_while_selected_hidden(self):
        # The ball disappears; another confirmed candidate is seen while the ball is still within predict_max_s.
        a = [[ball(100 + 8 * i, 300)] for i in range(6)]
        other = [[ball(1000, 600)] for _ in range(6)]
        out, _ = run(a + other)
        k = B.confirm_min_obs - 1
        hidden = out[6 + k:6 + int(B.predict_max_s * FPS)]
        self.assertTrue(hidden and all(f.state is BallState.AMBIGUOUS for f in hidden))
        self.assertTrue(all(f.reason is BallReason.SELECTED_HIDDEN_RIVAL for f in hidden))

    def test_prediction_leaving_picture_is_lost(self):
        a = [[ball(W - 60 + 20 * i, 300)] for i in range(3)]                    # heading out at the right edge
        out, _ = run(a + [[] for _ in range(5)])
        self.assertIn(BallReason.PREDICTED_OUTSIDE, [f.reason for f in out])

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        frames = [[ball(*rng.uniform(50, 700, 2)) for _ in range(rng.integers(0, 4))] for _ in range(40)]
        a, _ = run(frames)
        b, _ = run(frames)
        self.assertEqual([dataclasses.astuple(f) for f in a], [dataclasses.astuple(f) for f in b])


class BallConfigTests(unittest.TestCase):
    def test_predict_must_not_exceed_lost(self):
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        raw["ball"]["predict_max_s"] = raw["ball"]["lost_max_s"] + 0.1
        with self.assertRaisesRegex(ConfigError, "predict_max_s"):
            config_from_dict(raw)


class BallViewTests(unittest.TestCase):
    frame = np.zeros((180, 320, 3), np.uint8)

    def test_keeps_ball_class_only_and_merges(self):
        dets = [Detection((100.0, 50.0, 120.0, 100.0), 0.9, 0),       # player
                Detection((200.0, 90.0, 206.0, 96.0), 0.95, 3),       # ball
                Detection((201.0, 90.0, 207.0, 96.0), 0.60, 3),       # same ball again
                Detection((40.0, 40.0, 44.0, 44.0), 0.10, 3)]         # below ball.score_min
        out = BallView(FakeDetector(dets), ("ball",), B).detect(self.frame)
        self.assertEqual([(d.class_id, d.score) for d in out], [(3, 0.95)])

    def test_refusals(self):
        with self.assertRaisesRegex(BallViewError, "score_min"):
            BallView(FakeDetector([], confidence=0.9), ("ball",), B)
        with self.assertRaisesRegex(BallViewError, "not all in"):
            BallView(FakeDetector([]), ("puck",), B)


class BallCli(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "clip.avi"
        write_video(self.video, 8, 25.0)
        d = raw_dataset_cfg()
        d["licensed_matches"] = [{"match_id": "M_DEV", "split": "dev", "permission": "test",
                                  "third_party_processing_confirmed": True}]
        d["preflight"]["split_matches"] = ["train:M_X", "valid:M_Y"]
        self.dcfg = self.td / "roboflow.yaml"
        self.dcfg.write_text(yaml.safe_dump(d))
        FakeClient.calls = []

    def detect(self, *extra):
        with mock.patch("spana.detection.roboflow_backend.HttpInferenceClient", FakeClient), \
                mock.patch("spana.secrets.load_secret", return_value="k"):
            return cli.main(["detect", "--backend", "roboflow", "--video", str(self.video), "--config",
                             str(CONFIG_PATH), "--dataset-config", str(self.dcfg), "--match-id", "M_DEV",
                             "--out", str(self.td / "d.json"), *extra])

    def test_detect_ball_out_then_ball(self):
        bd, bo = self.td / "bd.json", self.td / "ball.json"
        self.assertEqual(self.detect("--ball-out", str(bd)), 0)
        self.assertEqual(len(FakeClient.calls), 8)                              # one request per frame, shared
        ddoc, bdoc = read_json(self.td / "d.json"), read_json(bd)
        validate_detections(bdoc)
        self.assertEqual(bdoc["model"]["ball_view"], {"ball_classes": ["ball"]})
        self.assertTrue(all([r[5] for r in f["detections"]] == [NAMES.index("ball")] for f in bdoc["frames"]))
        self.assertTrue(all(NAMES.index("ball") not in [r[5] for r in f["detections"]] for f in ddoc["frames"]))
        self.assertEqual(cli.main(["ball", "--video", str(self.video), "--config", str(CONFIG_PATH),
                                   "--detections", str(bd), "--out", str(bo)]), 0)
        doc = read_json(bo)
        validate_ball(doc, bdoc)
        self.assertEqual(doc["frames"][-1]["state"], "OBSERVED")              # a still ball, confirmed
        self.assertTrue(bo.with_suffix(".decisions.jsonl").is_file())
        with self.assertRaisesRegex(ArtifactError, "ball-view"):               # player tracking refuses it
            cli.main(["track", "--video", str(self.video), "--config", str(CONFIG_PATH),
                      "--detections", str(bd), "--out", str(self.td / "t.json")])
        with self.assertRaisesRegex(ArtifactError, "not a ball-view"):         # ball refuses person detections
            cli.main(["ball", "--video", str(self.video), "--config", str(CONFIG_PATH),
                      "--detections", str(self.td / "d.json"), "--out", str(self.td / "b2.json")])

    def test_ball_artifact_validation(self):
        bd, bo = self.td / "bd.json", self.td / "ball.json"
        self.detect("--ball-out", str(bd))
        cli.main(["ball", "--video", str(self.video), "--config", str(CONFIG_PATH), "--detections", str(bd),
                  "--out", str(bo)])
        doc = read_json(bo)
        bad = {**doc, "frames": [dict(f) for f in doc["frames"]]}
        bad["frames"][0] = {**bad["frames"][0], "state": "OBSERVED", "reason": "SINGLE_CONFIRMED"}
        with self.assertRaises(ArtifactError):                                  # an answer with no position
            validate_ball(bad)
        bad["frames"][0] = {**doc["frames"][0], "reason": "SHORT_GAP"}
        with self.assertRaises(ArtifactError):                                  # reason not allowed with LOST
            validate_ball(bad)

        self.assertEqual((doc["schema_version"], doc["tracks_sha256"]), ("1.2.0", None))
        old = {k: v for k, v in doc.items() if k != "tracks_sha256"}
        sparse = {**doc, "frames": [dict(f) for f in doc["frames"]]}
        sparse["frames"][0] = {**sparse["frames"][0], "state": "LOST", "reason": "SPARSE_OBSERVATIONS"}
        validate_ball(sparse)                                                   # 1.1.0 on: allowed
        with self.assertRaises(ArtifactError):                                  # 1.0.0 had no such reason
            validate_ball({**old, "frames": sparse["frames"], "schema_version": "1.0.0"})
        validate_ball({**old, "schema_version": "1.1.0"})
        with self.assertRaises(ArtifactError):                                  # 1.2.0 needs tracks_sha256
            validate_ball(old)
        self.assertNotIn("SPARSE_OBSERVATIONS", {f["reason"] for f in doc["frames"]})   # a still ball, always seen
        validate_ball({**old, "schema_version": "1.0.0"})                       # 1.0.0 documents still validate

    def test_ball_with_player_tracks(self):
        bd, tr, bo = self.td / "bd.json", self.td / "t.json", self.td / "ball.json"
        self.detect("--ball-out", str(bd))
        self.assertEqual(cli.main(["track", "--video", str(self.video), "--config", str(CONFIG_PATH),
                                   "--detections", str(self.td / "d.json"), "--out", str(tr)]), 0)
        argv = ["ball", "--video", str(self.video), "--config", str(CONFIG_PATH), "--detections", str(bd)]
        self.assertEqual(cli.main(argv + ["--tracks", str(tr), "--out", str(bo)]), 0)
        doc, tdoc = read_json(bo), read_json(tr)
        validate_ball(doc, read_json(bd))
        self.assertEqual(doc["tracks_sha256"], sha256_doc(tdoc))
        self.assertEqual(doc["provenance"]["inputs"]["tracks"], sha256_doc(tdoc))
        short = {**tdoc, "frames": tdoc["frames"][:-1],                         # tracks not covering every frame
                 "tracklets": tdoc["tracklets"]}
        (self.td / "short.json").write_text(json.dumps(short))
        with self.assertRaises(ArtifactError):
            cli.main(argv + ["--tracks", str(self.td / "short.json"), "--out", str(self.td / "b3.json")])
        other = {**tdoc, "video": {**tdoc["video"], "sha256": "f" * 64}}
        (self.td / "other.json").write_text(json.dumps(other))
        with self.assertRaisesRegex(ArtifactError, "different video"):
            cli.main(argv + ["--tracks", str(self.td / "other.json"), "--out", str(self.td / "b4.json")])

    def test_ball_reuses_detections_unless_the_view_changed(self):
        bd = self.td / "bd.json"
        self.detect("--ball-out", str(bd))
        raw = yaml.safe_load(Path(CONFIG_PATH).read_text())
        for key, value, ok in (("min_seen_share", 0.3, True), ("predict_max_s", 0.3, True),
                               ("score_min", 0.45, False), ("nms_iou", 0.5, False)):
            with self.subTest(key=key):
                r = {**raw, "ball": {**raw["ball"], key: value}}
                cfgp = self.td / f"c_{key}.yaml"
                cfgp.write_text(yaml.safe_dump(r))
                argv = ["ball", "--video", str(self.video), "--config", str(cfgp), "--detections", str(bd),
                        "--out", str(self.td / f"b_{key}.json")]
                if ok:                                                          # tracker settings: same detections
                    self.assertEqual(cli.main(argv), 0)
                else:                                                           # the view changed: refused
                    with self.assertRaisesRegex(ArtifactError, "different ball view"):
                        cli.main(argv)


if __name__ == "__main__":
    unittest.main()
