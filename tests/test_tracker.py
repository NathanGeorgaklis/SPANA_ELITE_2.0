"""Multi-player tracker: unit and synthetic integration tests for the general failure classes."""
from __future__ import annotations

import unittest

import numpy as np

from spana.records import CameraObservation, CameraStatus, Detection, EndReason, PositionSource, TrackState
from spana.tracking.tracker import MultiPlayerTracker
from tests.synth import Player, cfg, run


def _states(res, gid):
    return [o for ft in res.frames for o in ft.tracks
            if o.detection_index is not None and res.gt_of_detection[ft.frame_index][o.detection_index] == gid]


class SinglePlayer(unittest.TestCase):
    def test_constant_motion_one_confirmed_track(self):
        res = run([Player("A", (300, 500), (200, 0))], fps=25, duration_s=3)
        self.assertEqual(res.gid_tracks["A"], [res.gid_tracks["A"][0]])
        self.assertEqual(res.frames[-1].tracks[0].state, TrackState.CONFIRMED)
        self.assertEqual(res.false_continuity, {})

    def test_confirmation_time_is_fps_independent(self):
        c = cfg()
        for fps in (25.0, 30.0, 50.0, 60.0):
            res = run([Player("A", (300, 500), (100, 0))], fps=fps, duration_s=1)
            first = next(ft.time_s for ft in res.frames if ft.tracks and ft.tracks[0].state == TrackState.CONFIRMED)
            self.assertGreaterEqual(first + 1e-9, c.tracking.confirm_s, fps)
            self.assertLessEqual(first, c.tracking.confirm_s + 1.0 / fps + 1e-9, fps)

    def test_short_gap_bridged_and_recorded(self):
        c = cfg()
        gap = c.tracking.rematch_max_s * 0.75
        p = Player("A", (300, 500), (150, 0), visible=lambda t: not (1.0 <= t < 1.0 + gap))
        res = run([p], fps=30, duration_s=2)
        self.assertEqual(len(res.gid_tracks["A"]), 1)
        hidden = [ft.tracks[0] for ft in res.frames if 1.0 <= ft.time_s < 1.0 + gap]
        self.assertTrue(hidden)
        self.assertTrue(all(o.source == PositionSource.PREDICTED and o.state == TrackState.LOST for o in hidden))
        last = res.frames[-1].tracks[0]
        self.assertEqual(last.gaps_bridged, 1)
        self.assertGreater(last.last_gap_s, 0)

    def test_gap_longer_than_rematch_window_ends_continuity(self):
        c = cfg()
        gap = c.tracking.rematch_max_s * 1.5
        p = Player("A", (300, 500), (150, 0), visible=lambda t: not (1.0 <= t < 1.0 + gap))
        res = run([p], fps=30, duration_s=2)
        self.assertEqual(len(res.gid_tracks["A"]), 2)   # fragmentation is the safe outcome
        self.assertEqual(res.false_continuity, {})

    def test_long_absence_held_then_removed(self):
        c = cfg()
        p = Player("A", (300, 500), (0, 0), visible=lambda t: t < 1.0)
        res = run([p], fps=30, duration_s=1.0 + c.tracking.lost_max_s + 0.5)
        sources = {ft.tracks[0].source for ft in res.frames
                   if ft.tracks and ft.time_s > 1.0 + c.tracking.predict_max_s + 0.05}
        self.assertEqual(sources, {PositionSource.HELD})
        ended = [r for ft in res.frames for _, r in ft.ended]
        self.assertEqual(ended, [EndReason.LOST_EXPIRED])
        self.assertEqual(res.frames[-1].tracks, [])

    def test_reappearance_far_away_is_not_the_same_track(self):
        # Detection jumps by 3 body heights after 0.1 s: beyond any plausible speed -> new track, never continuity.
        p1 = Player("A", (300, 500), (0, 0), visible=lambda t: t < 1.0)
        p2 = Player("B", (300 + 240, 500), (0, 0), visible=lambda t: t >= 1.1)
        res = run([p1, p2], fps=30, duration_s=2)
        self.assertEqual(res.false_continuity, {})
        self.assertNotEqual(res.gid_tracks["A"][0], res.gid_tracks["B"][0])


class LostTrackSafety(unittest.TestCase):
    """Regressions for QA review H1-H3: a LOST track must never be inherited by another player."""
    H = 80.0

    def test_new_player_near_lost_track(self):
        for fps in (10, 25, 120):
            for d in (0.3, 0.5, 1.0, 1.5, 1.9):
                for appear in (0.05, 0.15, 0.6):
                    b = Player("B", (800, 500), (0, 0), visible=lambda t: t < 1.0)
                    a = Player("A", (800 + d * self.H, 500), (0, 0), visible=lambda t, s=1.0 + appear: t >= s)
                    res = run([a, b], fps=fps, duration_s=2.0)
                    if appear * 1.0 > cfg().tracking.rematch_max_s or d > 1.0:
                        self.assertEqual(res.false_continuity, {}, (fps, d, appear))

    def test_new_player_on_extrapolated_path(self):
        b = Player("B", (400, 500), (300, 0), visible=lambda t: t < 1.0)
        a = Player("A", (400 + 300 * 1.3, 500), (0, 0), visible=lambda t: t >= 1.3)
        self.assertEqual(run([a, b], fps=25, duration_s=2).false_continuity, {})

    def test_player_walks_through_occluded_players_position(self):
        for fps in (10, 25, 120):
            b = Player("B", (800, 500), (0, 0), visible=lambda t: not (1.0 <= t < 2.5))
            a = Player("A", (500, 560), (200, 0), visible=lambda t: t >= 1.2)
            self.assertEqual(run([a, b], fps=fps, duration_s=3).false_continuity, {}, fps)

    def test_lost_track_cannot_steal_continuing_players_detection(self):
        for off in (0.3, 0.4, 0.45, 0.5):
            class Jump(Player):
                def box_at(self, t, o=off):
                    b = super().box_at(t)
                    return b + (np.array([1, 0, 1, 0]) * o * 80 if t >= 1.0 else 0)
            a = Jump("A", (800, 500), (0, 0))
            b = Player("B", (800 + 1.2 * 80, 500), (0, 0), visible=lambda t: t < 0.95)
            self.assertEqual(run([a, b], fps=25, duration_s=1.6).false_continuity, {}, off)

    def test_rematch_refused_when_another_detection_is_in_the_gate(self):
        c = cfg()
        gap = c.tracking.rematch_max_s * 0.5
        a = Player("A", (800, 500), (0, 0), visible=lambda t: not (1.0 <= t < 1.0 + gap))
        n = Player("N", (800 + 0.25 * 80, 500), (0, 0), visible=lambda t: t >= 1.0 + gap)
        res = run([a, n], fps=30, duration_s=1.5)
        self.assertEqual(res.false_continuity, {})
        refused = [d for ft in res.frames for d in ft.decisions if d["kind"] == "rematch_refused"]
        self.assertTrue(refused)


class MultiPlayer(unittest.TestCase):
    def test_crossing_players_no_false_continuity(self):
        for fps in (25, 50):
            a = Player("A", (400, 500), (300, 0))
            b = Player("B", (1000, 510), (-300, 0))
            res = run([a, b], fps=fps, duration_s=2)
            self.assertEqual(res.false_continuity, {}, fps)
            peak = max(o.max_overlap_iou for ft in res.frames for o in ft.tracks)
            self.assertGreater(peak, 0.5, "physical overlap at the crossing must be reported")

    def test_equidistant_detection_is_contested(self):
        # Two stationary confirmed tracks; one detection appears exactly between them -> ambiguous association.
        def vis_a(t): return t < 1.0
        a = Player("A", (500, 500), (0, 0), visible=vis_a)
        b = Player("B", (500 + 16, 500), (0, 0), visible=vis_a)
        def extra(i, t):
            return [Detection((492.0, 460.0, 524.0, 540.0), 0.9, 0)] if t >= 1.0 else []
        res = run([a, b], fps=30, duration_s=1.2, extra=extra)
        first = next(ft for ft in res.frames if ft.time_s >= 1.0)
        obs = [o for o in first.tracks if o.detection_index is not None]
        self.assertEqual(len(obs), 1)
        self.assertTrue(obs[0].contested)

    def test_crossing_with_occlusion_no_false_continuity(self):
        # B is hidden behind A around the crossing point (merged detection -> only A is detected).
        a = Player("A", (400, 500), (300, 0))
        b = Player("B", (1000, 505), (-300, 0), visible=lambda t: not (0.85 <= t < 1.15))
        res = run([a, b], fps=30, duration_s=2)
        self.assertEqual(res.false_continuity, {})

    def test_low_score_detection_continues_but_does_not_spawn(self):
        c = cfg()
        weak = (c.detection.score_low + c.detection.score_high) / 2
        a = Player("A", (300, 500), (100, 0), score=lambda t: 0.9 if t < 1.0 else weak)
        b = Player("B", (900, 500), (0, 0), score=lambda t: weak)
        res = run([a, b], fps=30, duration_s=2)
        self.assertEqual(len(res.gid_tracks["A"]), 1)
        self.assertNotIn("B", res.gid_tracks)

    def test_duplicate_box_does_not_spawn_track(self):
        def extra(i, t):
            b = Player("A", (300, 500), (0, 0)).box_at(t) + np.array([1, 1, 1, 1])
            return [Detection(tuple(float(v) for v in b), 0.8, 0)]
        res = run([Player("A", (300, 500), (0, 0))], fps=30, duration_s=1, extra=extra)
        live = {o.track_id for ft in res.frames for o in ft.tracks}
        self.assertEqual(len(live), 1)

    def test_scale_invariance(self):
        # Same scene at two zoom levels (all sizes and speeds scaled): identical decisions, frame by frame.
        logs = []
        for s in (0.5, 2.0):
            a = Player("A", (400 * s, 500 * s), (300 * s, 0), height=80 * s, visible=lambda t: not .9 < t < 1.0)
            b = Player("B", (1000 * s, 510 * s), (-300 * s, 0), height=80 * s)
            res = run([a, b], fps=30, duration_s=2, frame_w=int(1920 * s), frame_h=int(1080 * s))
            logs.append([[(o.track_id, o.state, o.source, o.detection_index, o.contested) for o in ft.tracks]
                         for ft in res.frames])
        self.assertEqual(logs[0], logs[1])

    def test_deterministic(self):
        def once():
            a = Player("A", (400, 500), (300, 0)); b = Player("B", (1000, 510), (-300, 0))
            c = Player("C", (700, 700), (0, -50), visible=lambda t: not .5 < t < .6)
            return [[(o.track_id, o.box, o.source) for o in ft.tracks] for ft in run([a, b, c], fps=30,
                                                                                     duration_s=2).frames]
        self.assertEqual(once(), once())


class Camera(unittest.TestCase):
    PAN_PX_PER_FRAME = 60.0  # 0.75 body heights per frame: far beyond the uncompensated gate

    def _pan(self, i, box):
        return box - np.array([1, 0, 1, 0]) * self.PAN_PX_PER_FRAME * i

    def test_compensated_pan_keeps_track(self):
        def cam(i):
            if i == 0:
                return CameraObservation.identity(CameraStatus.FIRST)
            return CameraObservation(CameraStatus.OK, ((1, 0, -self.PAN_PX_PER_FRAME), (0, 1, 0)), 100, 0.01)
        res = run([Player("A", (300, 500), (0, 0))], fps=30, duration_s=0.5, camera=cam,
                  world_to_image=self._pan, frame_w=40000)
        self.assertEqual(len(res.gid_tracks["A"]), 1)

    def test_unknown_camera_motion_breaks_continuity(self):
        def cam(i):
            return CameraObservation.identity(CameraStatus.FIRST if i == 0 else CameraStatus.UNKNOWN)
        a = Player("A", (300, 500), (0, 0))
        b = Player("B", (300 + 70, 500), (0, 0))
        res = run([a, b], fps=30, duration_s=0.5, camera=cam, world_to_image=self._pan, frame_w=40000)
        self.assertEqual(res.false_continuity, {})
        ended = {r for ft in res.frames for _, r in ft.ended}
        self.assertEqual(ended, {EndReason.CAMERA_MOTION_UNKNOWN})

    def test_cut_ends_all_tracks_and_starts_new_epoch(self):
        def cam(i):
            if i == 0:
                return CameraObservation.identity(CameraStatus.FIRST)
            return CameraObservation.identity(CameraStatus.CUT if i == 30 else CameraStatus.OK, inliers=100)
        res = run([Player("A", (300, 500), (0, 0)), Player("B", (900, 500), (0, 0))], fps=30, duration_s=2,
                  camera=cam)
        cut = res.frames[30]
        self.assertEqual(sorted(r for _, r in cut.ended), [EndReason.CAMERA_CUT] * 2)
        before = {o.track_id for o in res.frames[29].tracks}
        after = {o.track_id for o in res.frames[31].tracks}
        self.assertFalse(before & after)
        self.assertTrue(all(o.epoch == 1 for o in res.frames[31].tracks))


class Lifecycle(unittest.TestCase):
    def test_first_status_only_on_first_frame(self):
        t = MultiPlayerTracker(cfg(), 1920, 1080)
        with self.assertRaises(ValueError):
            t.update(0, 0.0, [], CameraObservation.identity(CameraStatus.OK))
        t.update(0, 0.0, [], CameraObservation.identity(CameraStatus.FIRST))
        with self.assertRaises(ValueError):
            t.update(1, 0.04, [], CameraObservation.identity(CameraStatus.FIRST))

    def test_frames_must_increase(self):
        t = MultiPlayerTracker(cfg(), 1920, 1080)
        t.update(5, 0.2, [], CameraObservation.identity(CameraStatus.FIRST))
        with self.assertRaises(ValueError):
            t.update(5, 0.2, [], CameraObservation.identity(CameraStatus.OK))

    def test_track_leaving_image_is_removed(self):
        p = Player("A", (1800, 500), (600, 0), visible=lambda t: t < 0.5)
        res = run([p], fps=30, duration_s=1.5)
        ended = [r for ft in res.frames for _, r in ft.ended]
        self.assertEqual(ended, [EndReason.LEFT_FRAME])

    def test_single_spurious_detection_never_confirms(self):
        def extra(i, t):
            return [Detection((100, 100, 140, 200), 0.9, 0)] if i == 10 else []
        res = run([], fps=30, duration_s=1, extra=extra)
        states = {o.state for ft in res.frames for o in ft.tracks}
        self.assertEqual(states, {TrackState.TENTATIVE})
        ended = [r for ft in res.frames for _, r in ft.ended]
        self.assertEqual(ended, [EndReason.TENTATIVE_EXPIRED])

    def test_decision_log_records_every_match(self):
        res = run([Player("A", (300, 500), (50, 0))], fps=30, duration_s=0.5)
        for ft in res.frames[1:]:
            matches = [d for d in ft.decisions if d["kind"] == "match"]
            self.assertEqual(len(matches), 1)
            self.assertIn("cost", matches[0])


if __name__ == "__main__":
    unittest.main()
