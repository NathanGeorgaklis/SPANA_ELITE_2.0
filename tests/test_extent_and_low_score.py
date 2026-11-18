"""Extent-change link (S3c, tracking.extent_change_link) and low-score short-gap recovery (S4,
tracking.low_score_recovery). Real DEV finding (records/gate_break_diagnosis_2026-11-18): a confirmed player's box
shrank behind an occluder or grew back as he re-emerged, failing the height or centre gate, and a short gap closed on
a low-score box was refused; each spawned a new track. Both rules are exclusive and confirmed-only."""
from __future__ import annotations

import unittest

from tests.test_appearance_doubt_link import RAW, ids_of
import copy

from spana.config import config_from_dict
from spana.records import CameraObservation, CameraStatus, Detection
from spana.tracking.tracker import MultiPlayerTracker

FPS = 30.0
X0, VX, TOP, H, W = 100.0, 3.0, 400.0, 150.0, 60.0


def cfg(**tracking):
    raw = copy.deepcopy(RAW)
    raw["tracking"].update(tracking)
    return config_from_dict(raw)


def run(config, players, frames):
    """players: callables frame -> (box, score) or None."""
    trk = MultiPlayerTracker(config, 1920, 1080)
    out = []
    for f in range(frames):
        dets = [Detection(tuple(float(v) for v in r[0]), r[1], 0) for r in (p(f) for p in players) if r is not None]
        cam = CameraObservation.identity(CameraStatus.FIRST if f == 0 else CameraStatus.OK, inliers=100)
        out.append(trk.update(f, f / FPS, dets, cam))
    return out


def body(f, x0=X0, keep="top", visible=1.0, score=0.9):
    x = x0 + VX * f
    if keep == "top":
        return (x, TOP, x + W, TOP + H * visible), score
    return (x, TOP + H * (1 - visible), x + W, TOP + H), score


HIDE, BACK = 20, 30          # frames [HIDE, BACK) the player's lower body is hidden (box keeps its top edge)


def hidden_legs(f, x0=X0, keep="top"):
    return body(f, x0, keep, visible=0.6 if HIDE <= f < BACK else 1.0)


def kinds(out):
    return [d["kind"] for ft in out for d in ft.decisions]


def tracks(out):
    return set.union(*ids_of(out))


class ExtentLink(unittest.TestCase):
    def test_off_breaks_on_shrink_and_on_regrowth(self):
        out = run(cfg(), [hidden_legs], 45)
        self.assertGreaterEqual(len(tracks(out)), 2)

    def test_on_keeps_one_track_through_shrink_and_regrowth(self):
        out = run(cfg(extent_change_link=True), [hidden_legs], 45)
        self.assertEqual(len(tracks(out)), 1)
        self.assertIn("extent_link", kinds(out))
        self.assertTrue(all(o.identity.value == "CERTAIN" for ft in out for o in ft.tracks))

    def test_bottom_edge_kept_also_links(self):
        out = run(cfg(extent_change_link=True), [lambda f: hidden_legs(f, keep="bottom")], 45)
        self.assertEqual(len(tracks(out)), 1)

    def test_no_edge_kept_is_refused(self):
        def centred(f):          # shrinks around its centre: neither edge kept -> not a partly hidden body
            x = X0 + VX * f
            m = 0.2 * H if HIDE <= f < BACK else 0.0
            return (x, TOP + m, x + W, TOP + H - m), 0.9
        out = run(cfg(extent_change_link=True), [centred], 45)
        self.assertNotIn("extent_link", kinds(out))

    def test_two_candidate_boxes_are_refused(self):
        # Upper and lower part boxes both keep an edge and nest in the prediction: ambiguous, no link.
        def lower(f):
            return body(f, keep="bottom", visible=0.6) if HIDE <= f < BACK else None
        out = run(cfg(extent_change_link=True), [hidden_legs, lower], 45)
        self.assertNotIn("extent_link", [d["kind"] for ft in out[HIDE:HIDE + 1] for d in ft.decisions])

    def test_box_another_track_could_take_is_refused(self):
        # Another player stands just behind (box 60 px higher, same column) from frame 5: the shrunk box lies in his
        # box and keeps his bottom edge, so it could be his. No link (fail closed).
        def behind(f):
            if f < 5:
                return None
            x = X0 + VX * f
            return (x, TOP - 60, x + W, TOP - 60 + H), 0.9
        out = run(cfg(extent_change_link=True), [hidden_legs, behind], 45)
        self.assertGreaterEqual(len(set.union(*ids_of(out[:HIDE]))), 2)          # he is tracked
        self.assertNotIn("extent_link", [d["kind"] for ft in out[HIDE:HIDE + 1] for d in ft.decisions])

    def test_unconfirmed_tracks_never_extent_link(self):
        # The box shrinks on the frame right after the player appears (track still unconfirmed).
        def young(f):
            if f < 10:
                return None
            return body(f, visible=0.6 if f >= 11 else 1.0)
        out = run(cfg(extent_change_link=True, confirm_s=0.5, unconfirmed_max_s=1.0), [young], 20)
        self.assertNotIn("extent_link", kinds(out))


class ExtentLinkReviewCases(unittest.TestCase):
    def test_one_track_after_the_link_too(self):
        # After a size jump the next frames must still match: the linked box replaces the Kalman extent instead of
        # being averaged with the old one (found while reviewing: the averaged box kept neither edge).
        out = run(cfg(extent_change_link=True), [lambda f: body(f, keep="bottom", visible=0.5 if f < HIDE else 1.0)], 40)
        self.assertEqual(len(tracks(out)), 1)

    """Cases added after the code review of eecbc72."""

    def test_link_across_a_gap_is_recorded_as_a_recovery(self):
        def missed_then_shrunk(f):          # not detected for 3 frames, then back with the lower body hidden
            if HIDE <= f < HIDE + 3:
                return None
            return body(f, visible=0.6 if HIDE + 3 <= f < BACK else 1.0)
        out = run(cfg(extent_change_link=True), [missed_then_shrunk], 45)
        rec = [d for ft in out for d in ft.decisions if d["kind"] == "recovery" and d.get("rule") == "EXTENT_CHANGE"]
        self.assertEqual(len(rec), 1)
        self.assertEqual(len(tracks(out)), 1)
        o = out[HIDE + 3].tracks[0]
        self.assertEqual(o.recoveries, 1)
        self.assertGreater(o.last_gap_s, 0.0)

    def test_link_across_a_gap_must_fit_own_motion(self):
        # Tracked as a half-height box (upper body hidden), missed 3 frames, back at full height with the same bottom
        # edge: the extent rule fits (ratio 2.0, nested, bottom kept) but the centre moved half a predicted height,
        # beyond the acceleration bound for the gap. Without the gap the same change links (control).
        def doubled(f, gap=True):
            if gap and HIDE <= f < HIDE + 3:
                return None
            return body(f, keep="bottom", visible=0.5 if f < HIDE + (3 if gap else 0) else 1.0)
        out = run(cfg(extent_change_link=True), [doubled], 45)
        self.assertNotIn("S3c", [d.get("stage") for d in out[HIDE + 3].decisions])
        ctrl = run(cfg(extent_change_link=True), [lambda f: doubled(f, gap=False)], 45)
        self.assertIn("extent_link", kinds(ctrl))

    def test_appearance_veto_blocks_the_link(self):
        import numpy as np
        a, b = np.full(16, 1 / 16), np.zeros(16)
        b[0] = 1.0
        trk = MultiPlayerTracker(cfg(extent_change_link=True), 1920, 1080)
        seen = []
        for f in range(40):
            box, sc = hidden_legs(f)
            cam = CameraObservation.identity(CameraStatus.FIRST if f == 0 else CameraStatus.OK, inliers=100)
            seen.append(trk.update(f, f / FPS, [Detection(box, sc, 0)], cam, [a if f < HIDE else b]))
        self.assertNotIn("extent_link", kinds(seen))
        self.assertIn("extent_link", kinds(run(cfg(extent_change_link=True), [hidden_legs], 40)))   # control

    def test_refusals_hold_on_every_hidden_frame(self):
        def lower(f):
            return body(f, keep="bottom", visible=0.6) if HIDE <= f < BACK else None
        out = run(cfg(extent_change_link=True), [hidden_legs, lower], 45)
        self.assertNotIn("extent_link", kinds(out[HIDE:BACK]))


GAP0, GAP1 = 20, 23            # frames the player is not detected (0.1 s < rematch_max_s 0.2 s)


def weak_return(f, score_back=0.3):
    if GAP0 <= f < GAP1:
        return None
    return body(f, score=score_back if GAP1 <= f < GAP1 + 3 else 0.9)


class LowScoreRecovery(unittest.TestCase):
    def test_off_refuses_and_breaks(self):
        out = run(cfg(), [weak_return], 40)
        self.assertGreaterEqual(len(tracks(out)), 2)

    def test_on_recovers_on_the_low_score_box(self):
        out = run(cfg(low_score_recovery=True), [weak_return], 40)
        self.assertEqual(len(tracks(out)), 1)
        at = [n for n, ft in enumerate(out) for d in ft.decisions if d["kind"] == "recovery"]
        self.assertEqual(at, [GAP1])                      # recovered on the low-score box, no high-score box present

    def test_gap_longer_than_rematch_is_not_recovered_on_low_score(self):
        def long_gap(f):
            if 20 <= f < 30:       # 0.33 s > rematch_max_s
                return None
            return body(f, score=0.3 if 30 <= f < 33 else 0.9)
        out = run(cfg(low_score_recovery=True), [long_gap], 45)
        self.assertGreaterEqual(len(tracks(out)), 2)

    def test_unconfirmed_tracks_never_recover_on_low_score(self):
        def young(f):
            if f < 10 or 11 <= f < 13:
                return None
            return body(f, score=0.3 if f >= 13 else 0.9)
        out = run(cfg(low_score_recovery=True, confirm_s=0.5, unconfirmed_max_s=1.0), [young], 20)
        self.assertNotIn("recovery", kinds(out))


if __name__ == "__main__":
    unittest.main()
