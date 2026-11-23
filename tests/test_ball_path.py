"""Ball path v1 (spana/ball/path.py, spana.ball_path 1.0.0): the ball decided offline over all sightings.
Every scene runs the real ball tracker on scripted candidates and builds the spana.ball 1.3.0 document the way
`spana.cli ball` does. The error patterns are those found on DEV (records/ball_*_2026-11-23)."""
from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from spana import cli
from spana.artifacts import (BALL_SCHEMA, ArtifactError, ball_frame, ball_hypothesis_rows, sha256_doc, validate_ball,
                             validate_ball_path)
from spana.ball.path import as_ball_answers, ball_path, ball_path_document, summary
from spana.ball.tracker import BallTracker
from spana.ball_labels import validate_record
from spana.records import CameraStatus
from tests.synth import CONFIG_PATH, cfg
from tests.test_ball_score import label_record
from tests.test_ball_tracker import ball, cam
from tools.ball_score import load_config as load_score_config
from tools.ball_score import score_clip

CFG = cfg()
PC = CFG.ball_path
FPS, W, H = 25.0, 1280, 720
SHA = "a" * 64


def ball_doc(frames, cams=None, players=None, fps=FPS, ball_cfg=None) -> dict:
    """spana.ball 1.3.0 of scripted candidate lists (one list per frame), built like `spana.cli ball`."""
    bc = ball_cfg or CFG.ball
    trk = BallTracker(bc, W, H)
    rows = []
    for i, cands in enumerate(frames):
        c = cams[i] if cams else (cam(CameraStatus.FIRST) if i == 0 else cam())
        rows.append(ball_frame(trk.update(i, i / fps, cands, c, players), CFG.output))
    doc = {"schema": BALL_SCHEMA[0], "schema_version": BALL_SCHEMA[1], "provenance": {},
           "video": {"path": "synthetic", "sha256": SHA, "fps": fps, "width": W, "height": H,
                     "frame_count_reported": len(frames), "timestamp_source": "synthetic"},
           "detections_sha256": "0" * 64, "tracks_sha256": None, "ball_config": dataclasses.asdict(bc),
           "frames": rows, "hypotheses": ball_hypothesis_rows(trk.summaries())}
    validate_ball(doc)
    return doc


def path_of(frames, pc=PC, **kw):
    bdoc = ball_doc(frames, **kw)
    doc = ball_path_document(bdoc, pc, {}, sha256_doc(bdoc))
    validate_ball_path(doc, bdoc)
    return doc, bdoc


def states(doc):
    return [(f["state"], f["reason"]) for f in doc["frames"]]


def moving(i, x0=100, y=300, step=6):
    return ball(x0 + step * i, y)


SEEN = ("SEEN", "CONFIDENT_STRETCH")


class ConfidentStretch(unittest.TestCase):
    def test_steady_ball_is_seen_on_every_frame_with_its_position(self):
        doc, bdoc = path_of([[moving(i)] for i in range(60)])
        self.assertEqual(set(states(doc)), {SEEN})                     # also the frames before online confirmation
        self.assertEqual(bdoc["frames"][0]["state"], "LOST")
        self.assertEqual([f["center"] for f in doc["frames"]], [[100.0 + 6 * i, 300.0] for i in range(60)])
        self.assertEqual((len(doc["stretches"]), doc["stretches"][0]["confident"], doc["gaps"]), (1, True, []))

    def test_short_false_objects_are_never_the_ball(self):
        cases = {"phantom, 2 sightings": [1, 1], "phantom answered on 2 frames (4 sightings)": [1] * 4,
                 "glove / jersey, 5 frames": [1] * 5, "boot, 6 of 8 frames": [1, 1, 0, 1, 1, 0, 1, 1]}
        for name, pattern in cases.items():
            frames = [[] for _ in range(10)] + [[ball(600, 400)] if on else [] for on in pattern] + \
                     [[] for _ in range(30)]
            with self.subTest(name):
                doc, bdoc = path_of(frames)
                self.assertEqual(set(states(doc)), {("UNKNOWN", "NO_BALL")})
                self.assertFalse(any(s["confident"] for s in doc["stretches"]))
                self.assertTrue(doc["stretches"])                                   # recorded, not hidden
        for n, answered in ((4, 2), (5, 3)):                                        # pass 1 alone answers them
            _, bdoc = path_of([[] for _ in range(10)] + [[ball(600, 400)]] * n + [[] for _ in range(30)])
            self.assertEqual(sum(f["state"] == "OBSERVED" for f in bdoc["frames"]), answered)

    def test_long_flicker_is_never_the_ball(self):
        # confirmed by 3 frames in a row, then seen every 4th frame for 3 s: one long stretch at ~25 %
        doc, _ = path_of([[ball(600, 400)] if i < 3 or i % 4 == 0 else [] for i in range(75)])
        self.assertNotIn(SEEN, set(states(doc)))
        longest = max(doc["stretches"], key=lambda s: s["duration_s"])
        self.assertGreater(longest["duration_s"], 2.5)
        self.assertLess(longest["density"], PC.min_density)

    def test_false_object_does_not_block_the_real_ball(self):
        # a false object is selected first; the real ball appears while it is still followed
        frames = [[ball(900, 500)] if i < 8 else [] for i in range(60)]
        for i in range(4, 60):
            frames[i] = frames[i] + [moving(i)]
        doc, bdoc = path_of(frames)
        self.assertNotIn("OBSERVED", {f["state"] for f in bdoc["frames"][8:14]})    # pass 1: ambiguous / waiting
        self.assertEqual(set(states(doc)[4:]), {SEEN})
        self.assertTrue(all(f["center"][1] == 300.0 for f in doc["frames"][4:]))

    def test_thresholds_come_from_config(self):
        five = [[] for _ in range(10)] + [[ball(600, 400)]] * 5 + [[] for _ in range(30)]
        short, _ = path_of(five, pc=dataclasses.replace(PC, min_stretch_s=0.1))
        self.assertIn(SEEN, set(states(short)))
        flicker = [[ball(600, 400)] if i < 3 or i % 4 == 0 else [] for i in range(75)]
        loose, _ = path_of(flicker, pc=dataclasses.replace(PC, min_density=0.2))
        self.assertIn(SEEN, set(states(loose)))


class HolesAndGaps(unittest.TestCase):
    def setUp(self):
        self.pre = [[] for _ in range(5)]

    def test_short_hole_stays_one_stretch_without_a_position(self):
        frames = [[moving(i)] if not 30 <= i < 35 else [] for i in range(60)]      # 0.2 s unseen
        doc, _ = path_of(frames)
        self.assertEqual({s for s in states(doc)[30:35]}, {("UNKNOWN", "HOLE")})
        self.assertTrue(all(f["center"] is None and f["stretch"] for f in doc["frames"][30:35]))
        self.assertEqual((len([s for s in doc["stretches"] if s["confident"]]), doc["gaps"]), (1, []))

    def test_long_hole_is_a_gap_recorded_from_both_ends(self):
        frames = self.pre + [[moving(i)] if not 30 <= i < 45 else [] for i in range(75)]   # 0.6 s unseen
        doc, _ = path_of(frames)
        st = states(doc)
        self.assertEqual(set(st[:5]), {("UNKNOWN", "NO_BALL")})
        self.assertEqual(set(st[35:50]), {("UNKNOWN", "GAP")})
        self.assertEqual(set(st[5:35]) | set(st[50:]), {SEEN})
        (g,) = doc["gaps"]
        self.assertEqual((g["from"]["frame"], g["to"]["frame"], g["frames"], g["kind"]), (34, 50, 15, "OPEN"))
        self.assertEqual(g["from"]["center"], doc["frames"][34]["center"])
        self.assertAlmostEqual(g["duration_s"], 16 / FPS)
        self.assertLess(g["distance_d"], g["allowed_d"])
        wide, _ = path_of(frames, pc=dataclasses.replace(PC, max_hole_s=1.0))
        self.assertEqual(wide["gaps"], [])                                         # one stretch: a hole, not a gap

    def test_implausible_gap_is_flagged(self):
        # with the default speed bound (150 d/s) almost any gap is reachable; a slow bound shows the rule itself
        slow = dataclasses.replace(CFG.ball, max_speed_d_per_s=20.0)                # allowed after 0.64 s: 15.8 d
        first = [[ball(100 + 2 * i, 300)] for i in range(30)]
        for x, kind in ((250, "OPEN"), (1250, "IMPLAUSIBLE")):                     # 9 d / 109 d from frame 29
            frames = first + [[] for _ in range(15)] + [[ball(x, 300)] for _ in range(30)]
            doc, _ = path_of(frames, ball_cfg=slow)
            with self.subTest(x=x):
                self.assertEqual([g["kind"] for g in doc["gaps"]], [kind])
                self.assertEqual({s for s in states(doc)[45:]}, {SEEN})             # both stretches stay answers
        doc, _ = path_of(first + [[] for _ in range(15)] + [[ball(1250, 300)] for _ in range(30)])
        self.assertEqual([g["kind"] for g in doc["gaps"]], ["OPEN"])               # default bound: reachable

    def test_false_tail_after_a_hole_is_not_the_ball(self):
        # real ball 40 frames; unseen 5; then the tracker rejoins a different object 150 px off the path for 5 frames
        frames = [[moving(i)] for i in range(40)] + [[] for _ in range(5)] + \
                 [[ball(100 + 6 * 45, 450)] for _ in range(5)] + [[] for _ in range(25)]
        doc, bdoc = path_of(frames)
        self.assertEqual(len({s["hypothesis"] for f in bdoc["frames"] for s in f["sightings"]}), 1)   # one hypothesis
        self.assertGreater(bdoc["frames"][45]["sightings"][0]["rejoin_d"], CFG.ball.reacquire_max_d)
        self.assertEqual(set(states(doc)[:40]), {SEEN})
        self.assertNotIn(SEEN, set(states(doc)[40:]))                              # the tail is never an answer
        self.assertEqual([s["confident"] for s in doc["stretches"]], [True, False])
        loose = dataclasses.replace(CFG.ball, reacquire_max_d=1000.0)             # without the split: the tail is SEEN
        tail, _ = path_of(frames, ball_cfg=loose)
        self.assertIn(SEEN, set(states(tail)[45:50]))

    def test_kicked_ball_that_goes_on_is_kept_with_a_gap_between(self):
        # real ball 40 frames; unseen 3; found 150 px off its path (a kick) and followed for another second
        frames = [[moving(i)] for i in range(40)] + [[], [], []] + [[ball(100 + 6 * 43 + 4 * k, 450 - 2 * k)]
                                                                     for k in range(27)]
        doc, _ = path_of(frames)
        self.assertEqual(set(states(doc)[:40]) | set(states(doc)[43:]), {SEEN})
        self.assertEqual([s["confident"] for s in doc["stretches"]], [True, True])
        (g,) = doc["gaps"]
        self.assertEqual((g["from"]["frame"], g["to"]["frame"], g["frames"], g["kind"]), (39, 43, 3, "OPEN"))
        self.assertEqual({s for s in states(doc)[40:43]}, {("UNKNOWN", "GAP")})

    def test_camera_cut_inside_a_gap(self):
        n = 75
        cams = [cam(CameraStatus.FIRST)] + [cam() for _ in range(n - 1)]
        cams[38] = cam(CameraStatus.CUT)
        frames = [[moving(i)] if not 30 <= i < 45 else [] for i in range(n)]
        doc, _ = path_of(frames, cams=cams)
        (g,) = doc["gaps"]
        self.assertEqual((g["kind"], g["distance_d"], g["allowed_d"]), ("CAMERA_BREAK", None, None))

    def test_unknown_camera_inside_a_gap_is_a_camera_break(self):
        n = 75
        cams = [cam(CameraStatus.FIRST)] + [cam() for _ in range(n - 1)]
        cams[38] = cam(CameraStatus.UNKNOWN)
        doc, _ = path_of([[moving(i)] if not 30 <= i < 45 else [] for i in range(n)], cams=cams)
        self.assertEqual([g["kind"] for g in doc["gaps"]], ["CAMERA_BREAK"])

    def test_cut_while_the_ball_is_seen_gives_adjacent_segments(self):
        n = 60
        cams = [cam(CameraStatus.FIRST)] + [cam() for _ in range(n - 1)]
        cams[30] = cam(CameraStatus.CUT)
        doc, _ = path_of([[moving(i)] for i in range(n)], cams=cams)
        self.assertEqual(set(states(doc)), {SEEN})
        (g,) = doc["gaps"]
        self.assertEqual((g["from"]["frame"], g["to"]["frame"], g["frames"], g["kind"]), (29, 30, 0, "CAMERA_BREAK"))

    def test_trailing_frames_after_the_last_stretch_are_no_ball(self):
        doc, _ = path_of([[moving(i)] for i in range(30)] + [[] for _ in range(20)])
        self.assertEqual(set(states(doc)[30:]), {("UNKNOWN", "NO_BALL")})
        self.assertEqual(doc["gaps"], [])

    def test_known_limit_a_near_rejoin_onto_another_object_is_accepted(self):
        # DESIGN "risks added after the review", item 1: unseen 3 frames, then something within 3 diameters of the
        # predicted position that then goes its own way. Nothing here can tell it from the ball: it stays SEEN.
        # Pinned so that a later change to this behaviour is visible.
        frames = [[moving(i)] for i in range(40)] + [[], [], []] + \
                 [[ball(100 + 6 * 43 + 5, 300 + 12 + 5 * k)] for k in range(20)]
        doc, bdoc = path_of(frames)
        self.assertLessEqual(bdoc["frames"][43]["sightings"][0]["rejoin_d"], CFG.ball.reacquire_max_d)
        self.assertEqual(set(states(doc)[43:]), {SEEN})
        self.assertEqual(len([s for s in doc["stretches"] if s["confident"]]), 1)

    def test_camera_pan_is_compensated_in_the_plausibility_check(self):
        # a ball at rest on the pitch while the camera pans 80 px per frame during the gap (1200 px in the picture)
        n, pan = 75, 80.0
        cams = [cam(CameraStatus.FIRST)] + [cam(dx=pan) if 30 <= i < 45 else cam() for i in range(1, n)]
        x = lambda i: 40.0 + pan * min(max(i - 29, 0), 15)        # noqa: E731
        frames = [[ball(x(i), 300)] if not 30 <= i < 45 else [] for i in range(n)]
        doc, _ = path_of(frames, cams=cams)
        (g,) = doc["gaps"]
        self.assertEqual(g["kind"], "OPEN")
        self.assertLess(g["distance_d"], 1.0)                                       # carried: it did not move


class Boundaries(unittest.TestCase):
    """Exactly at each setting (30 fps, where 0.4 s and 0.5 s are whole frames): >= keeps, > splits."""

    @staticmethod
    def frames(sighted, n, rejoin=None):
        return [{"frame": i, "t": round(i / 30.0, 6), "camera_status": "OK",
                 "sightings": [{"hypothesis": "B000001", "rejoin_d": (rejoin or {}).get(i)}] if i in sighted else []}
                for i in range(n)]

    def rows(self, sighted, n, rejoin=None):
        from spana.ball.path import stretches
        return stretches(self.frames(set(sighted), n, rejoin), PC, CFG.ball.reacquire_max_d)

    def test_duration(self):
        self.assertEqual([r["confident"] for r in self.rows(range(16), 16)], [True])      # 15 / 30 = 0.5 s exactly
        self.assertEqual([r["confident"] for r in self.rows(range(15), 15)], [False])     # 0.4667 s

    def test_density(self):
        half = [0, 1, 3, 5, 7, 9, 11, 15]                                              # 8 of 16 frames
        self.assertEqual([(r["density"], r["confident"]) for r in self.rows(half, 16)], [(0.5, True)])
        self.assertEqual([r["confident"] for r in self.rows([0, 3, 5, 7, 9, 11, 15], 16)], [False])   # 7 of 16

    def test_hole(self):
        self.assertEqual(len(self.rows(list(range(5)) + [16, 17], 18)), 1)               # 12 frames = 0.4 s: one
        self.assertEqual(len(self.rows(list(range(5)) + [17, 18], 19)), 2)               # 13 frames: split

    def test_far_rejoin(self):
        at = CFG.ball.reacquire_max_d
        self.assertEqual(len(self.rows([0, 1, 2, 5, 6], 7, {5: at})), 1)                 # exactly the bound: near
        self.assertEqual(len(self.rows([0, 1, 2, 5, 6], 7, {5: at + 0.001})), 2)

    def test_unknown_camera_breaks_a_stretch_like_a_cut(self):
        from spana.ball.path import stretches
        for status in ("CUT", "UNKNOWN"):
            fr = self.frames(set(range(40)), 40)
            fr[20]["camera_status"] = status
            self.assertEqual(len(stretches(fr, PC, CFG.ball.reacquire_max_d)), 2, status)


class Safeguards(unittest.TestCase):
    def test_a_stretch_never_spans_a_camera_break(self):
        # the tracker ends every hypothesis at a cut; the path must not rely on that
        from spana.ball.path import stretches
        frames = [{"frame": i, "t": i / FPS, "camera_status": "CUT" if i == 20 else "OK",
                   "sightings": [{"hypothesis": "B000001", "rejoin_d": None}]} for i in range(40)]
        rows = stretches(frames, PC, CFG.ball.reacquire_max_d)
        self.assertEqual([(r["first_frame"], r["last_frame"]) for r in rows], [(0, 19), (20, 39)])
        frames[20]["camera_status"] = "OK"
        self.assertEqual(len(stretches(frames, PC, CFG.ball.reacquire_max_d)), 1)

    def test_no_frame_but_seen_ever_carries_a_position(self):
        scenes = ([[moving(i)] if not 30 <= i < 45 else [] for i in range(75)],
                  [[moving(i)] if i % 3 else [] for i in range(60)],
                  [[moving(i), ball(1100 - 3 * i, 600)] for i in range(40)])
        for frames in scenes:
            doc, _ = path_of(frames)
            for f in doc["frames"]:
                self.assertEqual(f["state"] == "SEEN", f["center"] is not None)
                self.assertEqual(f["state"] == "SEEN", f["score"] is not None)


class OneBall(unittest.TestCase):
    def test_two_confident_stretches_at_once_are_a_conflict(self):
        frames = [[moving(i)] if i < 50 else [] for i in range(80)]                 # A: frames 0-49
        for i in range(30, 80):
            frames[i] = frames[i] + [ball(1100 - 3 * i, 600)]                       # B: frames 30-79
        doc, _ = path_of(frames)
        st = states(doc)
        self.assertEqual(set(st[:30]) | set(st[50:]), {SEEN})
        self.assertEqual(set(st[30:50]), {("UNKNOWN", "CONFLICT")})
        self.assertTrue(all(f["center"] is None for f in doc["frames"][30:50]))
        self.assertEqual([g["kind"] for g in doc["gaps"]], ["CONFLICT"])
        self.assertEqual((doc["gaps"][0]["from"]["frame"], doc["gaps"][0]["to"]["frame"]), (29, 50))
        self.frames = frames

    def test_gap_kind_precedence(self):
        self.test_two_confident_stretches_at_once_are_a_conflict()
        slow = dataclasses.replace(CFG.ball, max_speed_d_per_s=20.0)                # the 29 -> 50 gap is also too far
        doc, _ = path_of(self.frames, ball_cfg=slow)
        (g,) = doc["gaps"]
        self.assertEqual(g["kind"], "CONFLICT")                                     # CONFLICT before IMPLAUSIBLE
        self.assertGreater(g["distance_d"], g["allowed_d"])
        # A on frames 0-59, B on 20-79, a cut at 40: conflict on 20-59, with the cut inside it
        frames = [[moving(i)] if i < 60 else [] for i in range(80)]
        for i in range(20, 80):
            frames[i] = frames[i] + [ball(1100 - 3 * i, 600)]
        cams = [cam(CameraStatus.FIRST)] + [cam() for _ in range(79)]
        cams[40] = cam(CameraStatus.CUT)
        doc, _ = path_of(frames, cams=cams)
        (g,) = doc["gaps"]
        self.assertEqual((g["from"]["frame"], g["to"]["frame"]), (19, 60))
        self.assertEqual(set(states(doc)[20:60]), {("UNKNOWN", "CONFLICT")})
        self.assertEqual((g["kind"], g["distance_d"]), ("CAMERA_BREAK", None))      # CAMERA_BREAK before CONFLICT


class Validation(unittest.TestCase):
    def setUp(self):
        self.doc, self.bdoc = path_of([[] for _ in range(5)] + [[moving(i)] if not 30 <= i < 45 else []
                                                                for i in range(75)])

    def refused(self, doc, *src):
        with self.assertRaises(ArtifactError):
            validate_ball_path(doc, *src)

    def test_deterministic_and_recomputed(self):
        again = ball_path_document(self.bdoc, PC, {}, sha256_doc(self.bdoc))
        self.assertEqual(again, self.doc)
        moved = copy.deepcopy(self.doc)
        moved["frames"][10]["center"][0] += 3.0                                     # plausible, but not the input
        validate_ball_path(moved)
        self.refused(moved, self.bdoc)
        self.refused(self.doc, ball_doc([[moving(i)] for i in range(60)]))          # another ball document

    def test_no_position_without_a_sighting(self):
        d = copy.deepcopy(self.doc)
        gap = next(f for f in d["frames"] if f["reason"] == "GAP")
        gap.update(center=[300.0, 300.0], diameter=10.0, score=0.5)                 # a guessed position
        self.refused(d)

    def test_stretch_and_gap_rules(self):
        d = copy.deepcopy(self.doc)
        d["stretches"][0].update(duration_s=0.2)                                    # too short to be confident
        self.refused(d)
        d = copy.deepcopy(self.doc)
        d["gaps"][0]["from"]["center"] = [0.0, 0.0]                                 # not the SEEN frame's position
        self.refused(d)
        d = copy.deepcopy(self.doc)
        d["gaps"][0].update(distance_d=500.0)                                       # OPEN beyond the bound
        self.refused(d)

    def test_alone_the_validator_still_enforces_the_central_rule(self):
        frames = [[moving(i)] if i < 50 else [] for i in range(80)]
        for i in range(30, 80):
            frames[i] = frames[i] + [ball(1100 - 3 * i, 600)]
        doc, _ = path_of(frames)                                                    # conflict on frames 30-49
        s1 = doc["frames"][0]["stretch"]
        d = copy.deepcopy(doc)
        d["frames"][35].update(state="SEEN", reason="CONFIDENT_STRETCH", stretch=s1, center=[1.0, 2.0],
                               diameter=10.0, score=0.5)                            # a conflict frame made SEEN
        self.refused(d)
        d = copy.deepcopy(doc)
        d["frames"][60].update(stretch=s1)                                          # SEEN outside its stretch's span
        self.refused(d)
        d = copy.deepcopy(doc)
        d["frames"][35].update(reason="GAP")                                        # two stretches cover it: CONFLICT
        self.refused(d)

    def test_alone_the_validator_checks_gaps_against_the_frames(self):
        base = self.doc
        end = lambda f: {k: f[k] for k in ("frame", "t", "center", "diameter", "stretch")}    # noqa: E731
        d = copy.deepcopy(base)
        a, b = d["frames"][10], d["frames"][12]                                     # two SEEN frames of one stretch
        d["gaps"].insert(0, {"from": end(a), "to": end(b), "frames": 1, "duration_s": round(b["t"] - a["t"], 6),
                             "kind": "OPEN", "distance_d": 0.0, "allowed_d": 9.0})
        self.refused(d)                                                             # a SEEN frame inside a "gap"
        for change in ({"frames": 3}, {"duration_s": 9.0}, {"allowed_d": None}, {"kind": "CONFLICT"},
                       {"kind": "CAMERA_BREAK"}, {"kind": "IMPLAUSIBLE"}):
            d = copy.deepcopy(base)
            d["gaps"][0].update(change)
            with self.subTest(change):
                self.refused(d)
        d = copy.deepcopy(base)
        d["gaps"][0]["from"]["frame"] = [34]                                        # refused, not a crash
        self.refused(d)

    def test_path_must_name_the_ball_documents_video(self):
        d = copy.deepcopy(self.doc)
        d["video"] = {**d["video"], "sha256": "b" * 64}
        validate_ball_path(d)
        self.refused(d, self.bdoc)

    def test_ball_document_without_the_needed_settings_is_refused(self):
        b = copy.deepcopy(self.bdoc)
        del b["ball_config"]["reacquire_max_d"]
        with self.assertRaises(ValueError):
            ball_path(b, PC)

    def test_older_ball_documents_are_refused(self):
        old = {**self.bdoc, "schema_version": "1.2.0",
               "frames": [{k: v for k, v in f.items() if k not in ("sightings", "camera_affine")}
                          for f in self.bdoc["frames"]]}
        validate_ball(old)
        with self.assertRaises(ValueError):
            ball_path(old, PC)

    def test_bad_config_refused(self):
        raw = __import__("yaml").safe_load(Path(CONFIG_PATH).read_text())
        from spana.config import ConfigError, config_from_dict
        for bad in ({"min_density": 1.5}, {"max_hole_s": 0}, {"min_stretch_s": -1}):
            with self.subTest(bad), self.assertRaises(ConfigError):
                config_from_dict({**raw, "ball_path": {**raw["ball_path"], **bad}})


class Scoring(unittest.TestCase):
    def test_scored_like_a_ball_document_and_phantoms_do_not_count(self):
        # real ball frames 0-39 (labels every 3rd frame), ball out of the picture after; a 2-frame phantom at 60-61
        frames = [[moving(i)] for i in range(40)] + [[] for _ in range(20)] + [[ball(800, 500)]] * 2 + \
                 [[] for _ in range(10)]
        doc, bdoc = path_of(frames)
        script = {}
        for f in range(0, 72, 3):
            c = (100 + 6 * f, 300)
            script[f] = ({"state": "VISIBLE", "box": [c[0] - 8, c[1] - 8, c[0] + 8, c[1] + 8]} if f < 40
                         else {"state": "OUT"}, None)
        script[60] = ({"state": "OUT"}, None)
        rec = label_record(script)
        rec["clip"].update(width=W, height=H)
        sc = load_score_config()
        path_score = score_clip(rec, validate_record(rec), as_ball_answers(doc), sc)["summary"]
        raw_score = score_clip(rec, validate_record(rec), bdoc, sc)["summary"]
        self.assertEqual(path_score["wrong_answered"]["wrong"], 0)
        self.assertEqual(path_score["coverage_observed"], 1.0)
        self.assertLess(raw_score["coverage_observed"], 1.0)                        # pass 1 waits to confirm
        self.assertEqual(summary(doc)["seen"], 40)


class Cli(unittest.TestCase):
    def test_ball_path_command(self):
        bdoc = ball_doc([[moving(i)] for i in range(40)])
        with tempfile.TemporaryDirectory() as tmp:
            b, out = Path(tmp) / "b.json", Path(tmp) / "p.json"
            b.write_text(json.dumps(bdoc))
            argv = ["ball-path", "--ball", str(b), "--config", str(CONFIG_PATH), "--out", str(out)]
            self.assertEqual(cli.main(argv), 0)
            doc = json.loads(out.read_text())
            validate_ball_path(doc, json.loads(b.read_text()))
            self.assertEqual(doc["ball_sha256"], sha256_doc(bdoc))
            with self.assertRaises(Exception):
                cli.main(argv)                                                      # never overwrites

    def test_ball_score_scores_a_path_only_after_recomputing_it(self):
        from tools.ball_score import load_answers
        bdoc = ball_doc([[moving(i)] for i in range(40)])
        doc = ball_path_document(bdoc, PC, {}, sha256_doc(bdoc))
        with tempfile.TemporaryDirectory() as tmp:
            write = lambda name, d: (Path(tmp) / name).write_text(json.dumps(d)) and str(Path(tmp) / name)  # noqa
            b, p = write("b.json", bdoc), write("p.json", doc)
            kind, answers = load_answers(p, b)
            self.assertEqual(kind, "spana.ball_path")
            self.assertEqual({f["state"] for f in answers["frames"]}, {"OBSERVED"})
            self.assertEqual(load_answers(b)[0], "spana.ball")
            with self.assertRaisesRegex(SystemExit, "give its ball document"):      # a path alone is not scored
                load_answers(p)
            with self.assertRaises(SystemExit):                                     # a ball has no source document
                load_answers(b, b)
            moved = copy.deepcopy(doc)
            moved["frames"][10]["center"][0] += 3.0
            with self.assertRaises(ArtifactError):                                  # not what the ball document gives
                load_answers(write("moved.json", moved), b)
            loose_pc = dataclasses.replace(PC, min_stretch_s=0.1)
            loose = ball_path_document(bdoc, loose_pc, {}, sha256_doc(bdoc))
            with self.assertRaisesRegex(SystemExit, "recorded"):                    # made with other settings
                load_answers(write("loose.json", loose), b)


if __name__ == "__main__":
    unittest.main()
