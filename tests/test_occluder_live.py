"""Tracker invariant: an OCCLUDED track's occluder is a track that is live on the same frame.

Real DEV finding (10-minute window of the full DEV match G2, frame 174397): a hidden confirmed track was given as
occluder an unconfirmed track that was matched on that frame but ended there (UNCONFIRMED_TIMEOUT), so the emitted
frame named an occluder that was no longer live and the tracks artifact failed validation."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from spana.config import load_config
from spana.geometry import box_to_cxcywh
from spana.records import (CameraObservation, CameraStatus, Detection, FrameTracks, TrackletSummary,
                           TrackState)
from spana.tracking.kalman import BoxKalman
from spana.tracking.tracker import MultiPlayerTracker, _Track

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")


def _track(tid, box, t_first, t_last, confirmed, observed_frames):
    return _Track(tid, BoxKalman(box_to_cxcywh(np.asarray(box, float)), CFG.kalman), 0, 0, t_first, t_last, 0,
                  t_last, confirmed=confirmed, observed_frames=observed_frames)


def _tracker(*tracks):
    trk = MultiPlayerTracker(CFG, 1280, 720)
    trk._tracks = list(tracks)
    for t in tracks:
        trk._summaries[t.track_id] = TrackletSummary(t.track_id, 0, 0, 99, 99, t.observed_frames, t.confirmed, None,
                                                     0, ())
    return trk


class OccluderLiveTest(unittest.TestCase):
    def test_occluder_ending_on_the_same_frame_is_not_used(self):
        now = 5.0
        hidden = _track("T1", (100, 100, 140, 200), 0.0, now - 0.1, True, 30)        # confirmed, unseen now
        # matched this frame, never confirmed, older than unconfirmed_max_s: ends with UNCONFIRMED_TIMEOUT
        expiring = _track("T2", (95, 95, 145, 205), now - CFG.tracking.unconfirmed_max_s - 0.5, now - 0.1, False, 1)
        trk = _tracker(hidden, expiring)
        dets = [Detection((95.0, 95.0, 145.0, 205.0), 0.9, 0)]             # covers the hidden track's box
        out = FrameTracks(100, now, CameraObservation.identity(CameraStatus.OK))
        matched = {1: (0, False)}
        trk._lifecycle(matched, dets, 100, now, out)
        self.assertIn(("T2", "UNCONFIRMED_TIMEOUT"), [(t, r.value) for t, r in out.ended])
        live = {t.track_id for t in trk._tracks}
        self.assertEqual(live, {"T1"})
        self.assertTrue(hidden.occluder is None or hidden.occluder in live)
        self.assertEqual(hidden.state, TrackState.LOST)                     # nothing live covers it

    def test_surviving_occluder_is_kept(self):
        now = 5.0
        hidden = _track("T1", (100, 100, 140, 200), 0.0, now - 0.1, True, 30)
        cover = _track("T3", (95, 95, 145, 205), 0.0, now - 0.1, True, 30)          # confirmed, survives
        trk = _tracker(hidden, cover)
        dets = [Detection((95.0, 95.0, 145.0, 205.0), 0.9, 0)]
        out = FrameTracks(100, now, CameraObservation.identity(CameraStatus.OK))
        trk._lifecycle({1: (0, False)}, dets, 100, now, out)
        self.assertEqual((hidden.state, hidden.occluder), (TrackState.OCCLUDED, "T3"))


if __name__ == "__main__":
    unittest.main()
