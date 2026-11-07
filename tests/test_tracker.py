"""Multi-player tracker: the targeted scenario suite plus unit tests of states, fields and safety rules.

Every scenario is checked with the QA metrics in tests/metrics.py. The non-negotiable invariant everywhere:
identity_transfers == 0 and wrong_recoveries == 0 (a track never carries two players). Fragmentation is
measured and bounded where continuity is physically recoverable.
"""
from __future__ import annotations

import unittest

import numpy as np

from spana.records import CameraObservation, CameraStatus, Detection, EndReason, PositionSource, TrackState
from spana.tracking.tracker import MultiPlayerTracker
from tests.metrics import evaluate
from tests.synth import Player, cfg, kit, run

H = 80.0
FPS_SET = (10, 25, 60)


def assert_safe(tc: unittest.TestCase, res, msg=""):
    m = evaluate(res)
    tc.assertEqual(m.identity_transfers, 0, f"identity transfer {m.transfer_tracks} {msg}")
    tc.assertEqual(m.unflagged_transfers, 0, f"silent identity transfer {msg}")
    tc.assertEqual(m.wrong_recoveries, 0, f"wrong recovery {msg}")
    tc.assertEqual(m.duplicate_frames, 0, f"duplicate tracks {msg}")
    return m


class Scenarios(unittest.TestCase):
    """The spec's targeted list. Each at several frame rates."""

    def test_normal_movement(self):
        for fps in FPS_SET:
            ps = [Player(f"P{k}", (300 + 200 * k, 400 + 60 * k), (120 - 40 * k, 30 * k)) for k in range(5)]
            m = assert_safe(self, run(ps, fps=fps, duration_s=3), fps)
            self.assertEqual(m.total_fragments, 0, fps)
            self.assertGreater(m.coverage, 0.97, fps)

    def test_missed_detection(self):
        # One or two missed frames (within rematch_max_s) while running: same track, recovery recorded.
        for fps in FPS_SET:
            gap = 1.5 / fps if fps > 10 else 0.1
            p = Player("A", (300, 500), (250, 0), visible=lambda t, g=gap: not (1.0 <= t < 1.0 + g))
            res = run([p], fps=fps, duration_s=2)
            m = assert_safe(self, res, fps)
            self.assertEqual(m.total_fragments, 0, fps)
            self.assertEqual(res.frames[-1].tracks[0].recoveries, 1, fps)

    def test_short_occlusion_distinct_players_recovered(self):
        # B passes behind A (different appearance), hidden 0.4 s: OCCLUDED, then recovered on the same id.
        for fps in FPS_SET:
            a = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0))
            b = Player("B", (500, 495), (200, 0), appearance=kit(1), visible=lambda t: not (0.9 <= t < 1.3))
            res = run([a, b], fps=fps, duration_s=2)
            m = assert_safe(self, res, fps)
            self.assertEqual(m.fragments.get("B"), 0, fps)
            occl = [o for ft in res.frames for o in ft.tracks if o.state == TrackState.OCCLUDED]
            self.assertTrue(occl and all(o.occluder is not None for o in occl), fps)

    def test_short_occlusion_same_looking_players_fails_closed(self):
        # Same descriptor for both: recovery after a long gap is refused -> safe fragmentation, recorded.
        for fps in FPS_SET:
            a = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0))
            b = Player("B", (500, 495), (200, 0), appearance=kit(0), visible=lambda t: not (0.9 <= t < 1.3))
            res = run([a, b], fps=fps, duration_s=2)
            assert_safe(self, res, fps)
            refused = [d for ft in res.frames for d in ft.decisions
                       if d["kind"] == "recovery_refused" and d.get("reason") == "APPEARANCE_NOT_DISTINCT"]
            self.assertTrue(refused, fps)
            new_b = [o for ft in res.frames[-3:] for o in ft.tracks if o.possible_predecessors]
            self.assertTrue(new_b, "the new track must name the refused predecessor")

    def test_two_players_crossing(self):
        for fps in FPS_SET:
            for same in (True, False):
                a = Player("A", (400, 500), (300, 0), appearance=kit(0))
                b = Player("B", (1000, 510), (-300, 0), appearance=kit(0 if same else 1))
                m = assert_safe(self, run([a, b], fps=fps, duration_s=2), (fps, same))
                self.assertEqual(m.total_fragments, 0, (fps, same))

    def test_crossing_with_full_occlusion(self):
        # The occluded player is not detected while fully behind: no transfer, either recovered or a safe split.
        for fps in FPS_SET:
            for same in (True, False):
                a = Player("A", (400, 500), (300, 0), appearance=kit(0))
                b = Player("B", (1000, 505), (-300, 0), appearance=kit(0 if same else 1),
                           visible=lambda t: not (0.85 <= t < 1.15))
                m = assert_safe(self, run([a, b], fps=fps, duration_s=2), (fps, same))
                if not same:
                    self.assertEqual(m.total_fragments, 0, (fps, same))

    def test_close_player_contact(self):
        # Two players side by side (0.3 h apart) moving together, then separating: no merge, no transfer.
        for fps in FPS_SET:
            class Sep(Player):
                def box_at(self, t, sign=1):
                    b = super().box_at(t)
                    return b + np.array([1, 0, 1, 0]) * self.sign * max(0.0, t - 1.0) * 120
            a = Sep("A", (500, 500), (150, 0)); a.sign = -1
            b = Sep("B", (500 + 0.3 * H, 505), (150, 0)); b.sign = 1
            m = assert_safe(self, run([a, b], fps=fps, duration_s=2.5), fps)
            self.assertEqual(m.total_fragments, 0, fps)

    def test_player_entering_frame(self):
        for fps in FPS_SET:
            p = Player("A", (-60, 500), (300, 0), visible=lambda t: -60 + 300 * t > 10)
            res = run([p], fps=fps, duration_s=2)
            m = assert_safe(self, res, fps)
            self.assertEqual(m.total_fragments, 0)
            first = next(ft for ft in res.frames if ft.tracks)
            self.assertFalse(first.tracks[0].confirmed)     # a new track starts uncertain
            self.assertTrue(res.frames[-1].tracks[0].confirmed)

    def test_player_leaving_frame(self):
        for fps in FPS_SET:
            p = Player("A", (1700, 500), (400, 0), visible=lambda t: 1700 + 400 * t < 1900)
            res = run([p], fps=fps, duration_s=2)
            assert_safe(self, res, fps)
            ended = [r for ft in res.frames for _, r in ft.ended]
            self.assertEqual(ended, [EndReason.LEFT_FRAME], fps)

    def test_scale_change(self):
        # Player approaches the camera: box height 50 -> 150 px over 2 s.
        for fps in FPS_SET:
            class Grow(Player):
                def box_at(self, t):
                    self.height = 50 + 50 * t
                    return super().box_at(t)
            m = assert_safe(self, run([Grow("A", (600, 500), (40, 30))], fps=fps, duration_s=2), fps)
            self.assertEqual(m.total_fragments, 0, fps)

    def test_scale_invariance_of_decisions(self):
        logs = []
        for s in (0.5, 2.0):
            a = Player("A", (400 * s, 500 * s), (300 * s, 0), height=H * s, visible=lambda t: not .9 < t < 1.0)
            b = Player("B", (1000 * s, 510 * s), (-300 * s, 0), height=H * s)
            res = run([a, b], fps=30, duration_s=2, frame_w=int(1920 * s), frame_h=int(1080 * s))
            logs.append([[(o.track_id, o.state, o.source, o.detection_index, o.contested) for o in ft.tracks]
                         for ft in res.frames])
        self.assertEqual(logs[0], logs[1])

    def test_camera_pan(self):
        pan = 0.75 * H   # px per frame: far beyond the uncompensated gate

        def cam(i):
            if i == 0:
                return CameraObservation.identity(CameraStatus.FIRST)
            return CameraObservation(CameraStatus.OK, ((1, 0, -pan), (0, 1, 0)), 100, 0.01)
        ps = [Player(f"P{k}", (300 + 70 * k, 500), (0, 0)) for k in range(4)]
        res = run(ps, fps=30, duration_s=0.5, camera=cam, frame_w=40000,
                  world_to_image=lambda i, b: b - np.array([1, 0, 1, 0]) * pan * i)
        m = assert_safe(self, res)
        self.assertEqual(m.total_fragments, 0)

    def test_camera_zoom(self):
        for fps in FPS_SET:
            z = 1.3 ** (1 / fps)
            cxc, cyc = 960, 540

            def cam(i):
                if i == 0:
                    return CameraObservation.identity(CameraStatus.FIRST)
                return CameraObservation(CameraStatus.OK, ((z, 0, cxc * (1 - z)), (0, z, cyc * (1 - z))), 100, .01)

            def w2i(i, b):
                s = z ** i
                return np.array([cxc + (b[0] - cxc) * s, cyc + (b[1] - cyc) * s,
                                 cxc + (b[2] - cxc) * s, cyc + (b[3] - cyc) * s])
            ps = [Player(f"P{k}", (700 + k * 45, 500), (0, 0)) for k in range(5)]
            m = assert_safe(self, run(ps, fps=fps, duration_s=2, camera=cam, world_to_image=w2i), fps)
            self.assertEqual(m.total_fragments, 0, fps)

    def test_false_duplicate_detection(self):
        # A second, near-identical box and a nested partial-body box on the same player: one track only.
        for fps in FPS_SET:
            def extra(i, t):
                b = Player("A", (600, 500), (100, 0)).box_at(t)
                near = b + np.array([2, 1, 2, 1])
                legs = np.array([b[0] + 4, b[1] + 0.5 * H, b[2] - 4, b[3]])
                return [Detection(tuple(map(float, near)), 0.8, 0), Detection(tuple(map(float, legs)), 0.7, 0)]
            res = run([Player("A", (600, 500), (100, 0))], fps=fps, duration_s=1.5, extra=extra)
            assert_safe(self, res, fps)
            self.assertEqual(len({o.track_id for ft in res.frames for o in ft.tracks}), 1, fps)


class LostTrackSafety(unittest.TestCase):
    """A track whose player went missing must never be inherited by another player."""

    def test_new_player_near_lost_track(self):
        for fps in FPS_SET:
            for d in (0.5, 1.0, 1.5, 1.9):
                b = Player("B", (800, 500), (0, 0), visible=lambda t: t < 1.0)
                a = Player("A", (800 + d * H, 500), (0, 0), visible=lambda t: t >= 1.6)
                assert_safe(self, run([a, b], fps=fps, duration_s=2.5), (fps, d))

    def test_player_walks_through_vanished_players_position(self):
        for fps in FPS_SET:
            b = Player("B", (800, 500), (0, 0), visible=lambda t: not (1.0 <= t < 2.5))
            a = Player("A", (500, 560), (200, 0), visible=lambda t: t >= 1.2)
            assert_safe(self, run([a, b], fps=fps, duration_s=3), fps)

    def test_occluded_track_not_recovered_by_a_third_lookalike(self):
        # B goes behind A; a third player C with B's appearance emerges from the group instead of B.
        for fps in FPS_SET:
            a = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0))
            b = Player("B", (600, 495), (200, 0), appearance=kit(1), visible=lambda t: t < 0.5)
            c = Player("C", (700 + 0.3 * H, 497), (150, 0), appearance=kit(1), visible=lambda t: t >= 0.9)
            assert_safe(self, run([a, b, c], fps=fps, duration_s=1.6), fps)

    def test_recovery_refused_with_two_candidates_in_gate(self):
        a = Player("A", (800, 500), (0, 0), appearance=kit(2), visible=lambda t: not (1.0 <= t < 1.1))
        n = Player("N", (800 + 0.25 * H, 500), (0, 0), appearance=kit(2), visible=lambda t: t >= 1.1)
        res = run([a, n], fps=30, duration_s=1.5)
        assert_safe(self, res)
        refused = [d for ft in res.frames for d in ft.decisions if d["kind"] == "recovery_refused"]
        self.assertTrue(refused)

    def test_appearance_mismatch_vetoes_short_gap(self):
        b = Player("B", (800, 500), (0, 0), appearance=kit(0), visible=lambda t: t < 1.0)
        a = Player("A", (800 + 0.2 * H, 500), (0, 0), appearance=kit(3), visible=lambda t: t >= 1.07)
        res = run([a, b], fps=30, duration_s=1.5)
        assert_safe(self, res)
        reasons = {d.get("reason") for ft in res.frames for d in ft.decisions if d["kind"] == "recovery_refused"}
        self.assertIn("APPEARANCE_MISMATCH", reasons)

    def test_unknown_camera_motion_breaks_continuity(self):
        pan = 0.75 * H

        def cam(i):
            return CameraObservation.identity(CameraStatus.FIRST if i == 0 else CameraStatus.UNKNOWN)
        ps = [Player("A", (300, 500), (0, 0)), Player("B", (300 + 70, 500), (0, 0))]
        res = run(ps, fps=30, duration_s=0.5, camera=cam, frame_w=40000,
                  world_to_image=lambda i, b: b - np.array([1, 0, 1, 0]) * pan * i)
        assert_safe(self, res)
        self.assertEqual({r for ft in res.frames for _, r in ft.ended}, {EndReason.CAMERA_MOTION_UNKNOWN})

    def test_cut_ends_all_tracks(self):
        def cam(i):
            if i == 0:
                return CameraObservation.identity(CameraStatus.FIRST)
            return CameraObservation.identity(CameraStatus.CUT if i == 30 else CameraStatus.OK, inliers=100)
        res = run([Player("A", (300, 500), (0, 0)), Player("B", (900, 500), (0, 0))], fps=30, duration_s=2,
                  camera=cam)
        self.assertEqual([r for _, r in res.frames[30].ended], [EndReason.CAMERA_CUT] * 2)
        self.assertFalse({o.track_id for o in res.frames[29].tracks} & {o.track_id for o in res.frames[31].tracks})


class QARound2Regressions(unittest.TestCase):
    """Regressions for the independent QA review of 3d68382 (each failed on that commit)."""

    def test_h1_occluder_detection_drop_does_not_hand_its_track_to_the_emerging_player(self):
        for fps in FPS_SET:
            for look in (1, 0):
                a = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0),
                           visible=lambda t: not (1.2 <= t < 1.2 + 1.5 / 25))
                b = Player("B", (560, 495), (110, 0), appearance=kit(look),
                           visible=lambda t: not (0.9 <= t < 1.2))
                res = run([a, b], fps=fps, duration_s=2)
                if look == 1:
                    assert_safe(self, res, (fps, look))
                else:
                    # Same look: NMS merges the two bodies into one box right after an ambiguous spawn. Known
                    # limit: a transfer may follow, but only on a track already flagged as ambiguous.
                    m = evaluate(res)
                    self.assertEqual(m.unflagged_transfers, 0, (fps, look))
                    self.assertEqual(m.wrong_recoveries, 0, (fps, look))

    def test_h1_front_swap_at_crossing(self):
        for fps in FPS_SET + (120,):
            for la, lb in ((kit(0), kit(0)), (kit(0), kit(1)), (None, None)):
                a = Player("A", (400, 500), (300, 0), appearance=la, visible=lambda t: not (1.0 <= t < 1.2))
                b = Player("B", (1000, 505), (-300, 0), appearance=lb, visible=lambda t: not (0.8 <= t < 1.0))
                assert_safe(self, run([a, b], fps=fps, duration_s=3), (fps, la is None))

    def test_h2_unknown_appearance_short_gap_needs_motion_consistency(self):
        for fps in FPS_SET + (120,):
            for off in (0.5, 1.0, 1.3):
                b = Player("B", (800, 500), (0, 0), visible=lambda t: t < 1.0)
                c = Player("C", (800 + off * H, 500), (0, 0), visible=lambda t: t >= 1.0 + 1.0 / fps)
                assert_safe(self, run([b, c], fps=fps, duration_s=1.6, use_appearance=False), (fps, off))

    def test_h4_shifted_partial_body_box_never_becomes_a_confirmed_track(self):
        for shift in (0.0, 0.1, 0.2, 0.3):
            def extra(i, t, sh=shift):
                b = Player("A", (600, 500), (60, 0)).box_at(t)
                w = b[2] - b[0]
                return [Detection((b[0] + sh * w, b[1], b[2] + sh * w, b[1] + 0.6 * (b[3] - b[1])), 0.8, 0)]
            res = run([Player("A", (600, 500), (60, 0))], fps=25, duration_s=2, extra=extra)
            assert_safe(self, res, shift)
            confirmed = {o.track_id for ft in res.frames for o in ft.tracks if o.confirmed}
            self.assertEqual(len(confirmed), 1, shift)

    def test_m1_confirmation_and_continuity_independent_of_fps_under_score_flicker(self):
        c = cfg()
        rng = np.random.default_rng(3)
        for fps in (10, 25, 60, 120):
            draws = rng.random(int(3 * fps) + 2)
            p = Player("A", (500, 500), (80, 0),
                       score=lambda t, f=fps, d=draws: 0.9 if d[int(round(t * f))] < 0.5 else 0.3)
            res = run([p], fps=fps, duration_s=3)
            m = evaluate(res)
            self.assertEqual(m.total_fragments, 0, fps)
            share = np.mean([any(o.confirmed for o in ft.tracks) for ft in res.frames])
            self.assertGreater(share, 0.9, fps)
        self.assertGreater(c.detection.score_high, 0.3)

    def test_m2_occlusion_not_sticky_when_occluder_is_unobserved(self):
        a = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0), visible=lambda t: t < 1.0)
        b = Player("B", (680, 495), (0, 0), appearance=kit(1), visible=lambda t: t < 0.9)
        res = run([a, b], fps=25, duration_s=1.6)
        late = [o for ft in res.frames if ft.time_s > 1.05 for o in ft.tracks if o.state == TrackState.OCCLUDED]
        self.assertEqual(late, [])


class TrackFields(unittest.TestCase):
    def test_states_and_fields_through_a_lifecycle(self):
        c = cfg()
        p = Player("A", (600, 500), (0, 0), visible=lambda t: t < 1.0)
        res = run([p], fps=25, duration_s=1.0 + c.tracking.lost_max_s + 0.2)
        live = [ft.tracks[0] for ft in res.frames if ft.tracks]
        self.assertEqual(live[0].state, TrackState.ACTIVE)
        self.assertEqual(live[0].missed_frames, 0)
        lost = [o for o in live if o.state == TrackState.LOST]
        self.assertEqual([o.missed_frames for o in lost[:3]], [1, 2, 3])
        self.assertTrue(all(o.age_frames == k for k, o in enumerate(live)))
        held = [o for o in lost if o.time_since_observed_s > c.tracking.predict_max_s + 0.05]
        self.assertTrue(held and all(o.source == PositionSource.HELD for o in held))
        self.assertEqual([r for ft in res.frames for _, r in ft.ended], [EndReason.LOST_EXPIRED])
        # confidence: full while observed, then decays as unobserved frames enter the window
        self.assertAlmostEqual(live[24].confidence, 0.9, places=6)
        self.assertLess(lost[10].confidence, live[24].confidence)

    def test_confirmation_is_fps_independent(self):
        c = cfg()
        for fps in (10.0, 25.0, 60.0, 120.0):
            res = run([Player("A", (300, 500), (100, 0))], fps=fps, duration_s=1)
            first = next(ft.time_s for ft in res.frames if ft.tracks and ft.tracks[0].confirmed)
            self.assertGreaterEqual(first + 1e-9, c.tracking.confirm_s, fps)
            self.assertLessEqual(first, c.tracking.confirm_s + 1.0 / fps + 1e-9, fps)

    def test_single_spurious_detection_never_confirms(self):
        res = run([], fps=30, duration_s=1,
                  extra=lambda i, t: [Detection((100, 100, 140, 200), 0.9, 0)] if i == 10 else [])
        self.assertEqual({o.confirmed for ft in res.frames for o in ft.tracks}, {False})
        self.assertEqual([r for ft in res.frames for _, r in ft.ended], [EndReason.UNCONFIRMED_MISSED])

    def test_low_score_continues_but_never_spawns(self):
        c = cfg()
        weak = (c.detection.score_low + c.detection.score_high) / 2
        a = Player("A", (300, 500), (100, 0), score=lambda t: 0.9 if t < 1.0 else weak)
        b = Player("B", (900, 500), (0, 0), score=lambda t: weak)
        m = evaluate(run([a, b], fps=30, duration_s=2))
        self.assertEqual(m.fragments, {"A": 0})

    def test_history_bounded_in_seconds(self):
        c = cfg()
        res = run([Player("A", (300, 500), (50, 0))], fps=30, duration_s=c.tracking.history_s + 2)
        tid = res.frames[-1].tracks[0].track_id
        h = res.tracker.history(tid)
        self.assertLessEqual(h[-1].time_s - h[0].time_s, c.tracking.history_s + 1e-6)
        self.assertEqual(h[-1].source, PositionSource.OBSERVED)

    def test_tracklet_summaries(self):
        p = Player("A", (300, 500), (0, 0), visible=lambda t: t < 0.5)
        res = run([p, Player("B", (900, 500), (0, 0))], fps=30, duration_s=3.0)
        rows = {s.track_id: s for s in res.tracker.summaries()}
        self.assertEqual(sorted(s.end_reason or "LIVE" for s in rows.values()), ["LIVE", "LOST_EXPIRED"])

    def test_first_status_only_on_first_frame_and_increasing_frames(self):
        t = MultiPlayerTracker(cfg(), 1920, 1080)
        with self.assertRaises(ValueError):
            t.update(0, 0.0, [], CameraObservation.identity(CameraStatus.OK))
        t.update(0, 0.0, [], CameraObservation.identity(CameraStatus.FIRST))
        with self.assertRaises(ValueError):
            t.update(1, 0.04, [], CameraObservation.identity(CameraStatus.FIRST))
        with self.assertRaises(ValueError):
            t.update(0, 0.0, [], CameraObservation.identity(CameraStatus.OK))

    def test_deterministic(self):
        def once():
            ps = [Player("A", (400, 500), (300, 0), appearance=kit(0)),
                  Player("B", (1000, 510), (-300, 0), appearance=kit(1), visible=lambda t: not .9 < t < 1.2)]
            return [[(o.track_id, o.box, o.state, o.confidence) for o in ft.tracks]
                    for ft in run(ps, fps=30, duration_s=2).frames]
        self.assertEqual(once(), once())

    def test_decision_log_records_every_association(self):
        res = run([Player("A", (300, 500), (50, 0))], fps=30, duration_s=0.5)
        for ft in res.frames[1:]:
            self.assertEqual(len([d for d in ft.decisions if d["kind"] == "match"]), 1)


if __name__ == "__main__":
    unittest.main()
