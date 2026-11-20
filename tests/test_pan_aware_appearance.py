"""appearance.pan_aware (records/pan_aware_appearance_2026-11-20; root cause records/veto_root_cause_2026-11-20).

FRESH_08 player 10: the stored appearance was learned only from motion-blurred crops during a fast pan (torso band half
background); the first sharp crop of the same player then scored below min_similarity and the colour veto split the
track. With pan_aware, a crop taken while the camera moves > pan_blur_px per frame never overwrites a mostly-sharp
stored appearance, and an appearance learned mostly from such crops vetoes only below weak_min_similarity. A clearly
different player is still vetoed. Static world, the camera pans; descriptors are synthetic histograms."""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from spana.appearance import similarity
from spana.records import CameraObservation, CameraStatus, Detection
from spana.tracking.tracker import MultiPlayerTracker
from tests.synth import cfg, kit

SHIRT = kit(0)
BLURRED = 0.4 * kit(0) + 0.6 * kit(5)          # shirt smeared with background
OTHER = kit(9)                                 # another player
FPS = 30.0


def config(on: bool):
    c = cfg()
    return dataclasses.replace(c, appearance=dataclasses.replace(c.appearance, pan_aware=on))


def run(script, on: bool):
    """script: per frame (camera dx px, [(world x, descriptor)]) - one detection per entry, 40 x 100 px boxes at
    world x (image x = world x + accumulated camera dx), y 500."""
    trk = MultiPlayerTracker(config(on), 1920, 1080)
    frames, shift = [], 0.0
    script = [(0.0, [])] + list(script)            # frame 0 (camera FIRST) has no motion measurement: keep it empty
    for i, (dx, people) in enumerate(script):
        shift += dx if i else 0.0
        dets = [Detection((x + shift, 500.0, x + shift + 40.0, 600.0), 0.9, 0) for x, _ in people]
        cam = CameraObservation.identity(CameraStatus.FIRST) if i == 0 else \
            CameraObservation(CameraStatus.OK, ((1.0, 0.0, dx), (0.0, 1.0, 0.0)), 100, None)
        frames.append(trk.update(i, i / FPS, dets, cam, [d for _, d in people]))
    return trk, frames


def ids(frames):
    return sorted({t.track_id for ft in frames for t in ft.tracks})


class PanAware(unittest.TestCase):
    def test_setup(self):
        self.assertLess(similarity(SHIRT, BLURRED), 0.70)        # vetoed today
        self.assertGreater(similarity(SHIRT, BLURRED), 0.50)     # above the weak bar
        self.assertLess(similarity(OTHER, BLURRED), 0.50)        # another player stays vetoed

    def test_a_pan_learned_appearance_does_not_split_its_own_player(self):
        script = [(10.0, [(800.0, BLURRED)])] * 30 + [(0.0, [(800.0, SHIRT)])] * 10
        _, off = run(script, False)
        self.assertEqual(ids(off), ["T000001", "T000002"])       # the veto split the player (as on DEV)
        trk, on = run(script, True)
        self.assertEqual(ids(on), ["T000001"])

    def test_another_player_is_still_vetoed(self):
        script = [(10.0, [(800.0, BLURRED)])] * 30 + [(0.0, [(800.0, OTHER)])] * 10
        for on in (False, True):
            _, frames = run(script, on)
            self.assertEqual(ids(frames), ["T000001", "T000002"], on)    # never continued onto the other player

    def test_a_pan_starting_neither_splits_nor_overwrites_a_sharp_appearance(self):
        script = [(0.0, [(800.0, SHIRT)])] * 30 + [(10.0, [(800.0, BLURRED)])] * 30
        _, off = run(script, False)
        self.assertEqual(ids(off), ["T000001", "T000002"])       # today: the first blurred crop is vetoed
        trk, on = run(script, True)
        self.assertEqual(ids(on), ["T000001"])                   # blur may explain 0.68: not vetoed
        self.assertAlmostEqual(similarity(trk._tracks[0].appearance, SHIRT), 1.0, places=6)   # kept sharp
        self.assertEqual(trk._tracks[0].appearance_blur, 0.0)

    def test_another_player_during_a_pan_is_still_vetoed(self):
        script = [(0.0, [(800.0, SHIRT)])] * 30 + [(10.0, [(800.0, OTHER)])] * 10
        for on in (False, True):
            _, frames = run(script, on)
            self.assertEqual(ids(frames), ["T000001", "T000002"], on)

    def test_on_by_default(self):
        self.assertTrue(cfg().appearance.pan_aware)          # met the agreed tracking bar on DEV


if __name__ == "__main__":
    unittest.main()
