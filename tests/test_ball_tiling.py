"""Ball-only tiled detector pass (spana/detection/ball_view.py): the full frame plus overlapping frame-height tiles,
merged by the ball view's NMS. A stand-in detector reports a ball only where the picture it is given shows the ball
large enough, which is what tiling is for."""
from __future__ import annotations

import argparse
import dataclasses
import unittest

import numpy as np

from spana.artifacts import ArtifactError
from spana.cli import _ball_view
from spana.config import ConfigError, config_from_dict
from spana.detection.ball_view import VIEW_KEYS, BallView, BallViewError, tile_grid
from spana.records import Detection
from tests.synth import cfg
from tests.test_thresholds_in_config import with_keys

B = cfg().ball
ON = dataclasses.replace(B, tile_size_frac=1.0)
W, H = 1920, 1080
NAMES = ("player", "goalkeeper", "referee", "ball")
BALL, PLAYER = 3, 0


def frame_with(*balls, size=12):
    """A black frame with one white square per ball (x, y = top-left corner)."""
    img = np.zeros((H, W, 3), np.uint8)
    for x, y in balls:
        img[y:y + size, x:x + size] = 255
    return img


class SmallBallDetector:
    """Sees a white blob as a ball only when the picture is narrower than `max_width` (a tile). Boxes are in the
    coordinates of the picture it was given, like a real detector. Records the widths it was called with."""

    def __init__(self, max_width=1200, full_frame_score=None, confidence=0.1, extra=()):
        self.max_width, self.full_frame_score, self.conf, self.extra = max_width, full_frame_score, confidence, extra
        self.calls = []

    class_names = NAMES

    @property
    def model_id(self):
        return {"backend": "fake", "confidence": self.conf}

    def detect(self, img):
        self.calls.append(img.shape[1])
        tile = img.shape[1] <= self.max_width
        score = 0.8 if tile else self.full_frame_score
        out = [Detection(tuple(float(v) for v in b), s, c) for b, s, c in self.extra if tile]
        if score is None:
            return out
        mask = img[:, :, 0] == 255
        seen = np.zeros(mask.shape, bool)
        for y, x in zip(*np.nonzero(mask)):                       # one box per connected white square
            if seen[y, x]:
                continue
            x2 = x
            while x2 + 1 < mask.shape[1] and mask[y, x2 + 1]:
                x2 += 1
            y2 = y
            while y2 + 1 < mask.shape[0] and mask[y2 + 1, x]:
                y2 += 1
            seen[y:y2 + 1, x:x2 + 1] = True
            out.append(Detection((float(x), float(y), float(x2 + 1), float(y2 + 1)), score, BALL))
        return out


def view(det, cfg_ball=ON):
    return BallView(det, ["ball"], cfg_ball, tile_detector=det)


class TileGrid(unittest.TestCase):
    def test_two_frame_height_tiles_on_a_16_9_frame(self):
        self.assertEqual(tile_grid(1920, 1080, ON), [(0, 0, 1080), (840, 0, 1080)])
        self.assertEqual(tile_grid(1280, 720, ON), [(0, 0, 720), (560, 0, 720)])

    def test_no_tiling_when_off_or_when_one_tile_is_the_frame(self):
        self.assertEqual(tile_grid(1920, 1080, B), [])
        self.assertEqual(B.tile_size_frac, 0.0)                                 # off by default
        self.assertEqual(tile_grid(1080, 1080, ON), [])

    def test_tiles_cover_the_frame_with_the_minimum_overlap(self):
        for frac in (1.0, 0.75, 0.5):
            c = dataclasses.replace(ON, tile_size_frac=frac)
            tiles = tile_grid(W, H, c)
            side = tiles[0][2]
            xs = sorted({t[0] for t in tiles})
            ys = sorted({t[1] for t in tiles})
            self.assertEqual((xs[0], xs[-1] + side, ys[0], ys[-1] + side), (0, W, 0, H), frac)
            for o in (xs, ys):
                self.assertTrue(all(b - a <= side - c.tile_overlap_frac * side + 1 for a, b in zip(o, o[1:])), frac)


class TiledBallView(unittest.TestCase):
    def test_small_ball_found_only_with_tiles_and_in_frame_coordinates(self):
        img = frame_with((1500, 400))                             # in the right tile only (x >= 1080)
        self.assertEqual(view(SmallBallDetector(), B).detect(img), [])
        det = SmallBallDetector()
        (d,) = view(det).detect(img)
        self.assertEqual(d.box, (1500.0, 400.0, 1512.0, 412.0))
        self.assertEqual((d.score, d.class_id), (0.8, BALL))
        self.assertEqual(det.calls, [W, 1080, 1080])              # the full frame, then two tiles

    def test_ball_in_the_overlap_of_two_tiles_is_one_ball(self):
        (d,) = view(SmallBallDetector()).detect(frame_with((950, 500)))      # 840 <= x < 1080: both tiles
        self.assertEqual(d.box, (950.0, 500.0, 962.0, 512.0))

    def test_ball_cut_by_an_internal_tile_edge_is_taken_from_the_other_tile(self):
        # the left tile ends at x = 1080: it sees only 6 px of this ball; the right tile sees all 12
        (d,) = view(SmallBallDetector()).detect(frame_with((1074, 500)))
        self.assertEqual(d.box, (1074.0, 500.0, 1086.0, 512.0))
        (d,) = view(SmallBallDetector()).detect(frame_with((836, 500)))      # cut by the right tile's left edge
        self.assertEqual(d.box, (836.0, 500.0, 848.0, 512.0))

    def test_ball_at_a_frame_edge_is_kept(self):
        got = view(SmallBallDetector()).detect(frame_with((0, 500), (W - 12, 500)))
        self.assertEqual(sorted(d.box[0] for d in got), [0.0, float(W - 12)])

    def test_full_frame_and_tile_detections_of_one_ball_merge(self):
        det = SmallBallDetector(full_frame_score=0.4)             # seen weakly on the full frame, well on a tile
        (d,) = view(det).detect(frame_with((300, 300)))
        self.assertEqual(d.score, 0.8)
        (d,) = view(SmallBallDetector(full_frame_score=0.4), B).detect(frame_with((300, 300)))   # tiling off
        self.assertEqual(d.score, 0.4)

    def test_two_balls_stay_two(self):
        got = view(SmallBallDetector()).detect(frame_with((300, 300), (1500, 700)))
        self.assertEqual(len(got), 2)

    def test_only_the_ball_class_comes_from_tiles(self):
        det = SmallBallDetector(extra=(((100, 100, 160, 300), 0.9, PLAYER),))
        got = view(det).detect(frame_with((300, 300)))
        self.assertEqual([d.class_id for d in got], [BALL])

    def test_weak_tile_balls_are_dropped_like_weak_full_frame_ones(self):
        det = SmallBallDetector(extra=(((500, 500, 512, 512), ON.score_min - 0.05, BALL),))
        self.assertEqual(view(det).detect(frame_with()), [])


class Wiring(unittest.TestCase):
    def test_tiling_needs_a_tile_detector(self):
        det = SmallBallDetector()
        with self.assertRaises(BallViewError):
            BallView(det, ["ball"], ON)                                         # e.g. hosted inference
        BallView(det, ["ball"], B)                                              # tiling off: fine

    def test_cli_refuses_tiling_without_a_local_detector(self):
        dcfg = argparse.Namespace(detector=argparse.Namespace(ball_classes=["ball"]))
        a = argparse.Namespace(ball_out="x.json")
        c = dataclasses.replace(cfg(), ball=ON)
        with self.assertRaisesRegex(ArtifactError, "local backend"):
            _ball_view(a, SmallBallDetector(), dcfg, c)
        self.assertIsNotNone(_ball_view(a, SmallBallDetector(), dcfg, c, tile_detector=SmallBallDetector()))
        self.assertIsNone(_ball_view(argparse.Namespace(ball_out=None), SmallBallDetector(), dcfg, c))

    def test_tile_settings_are_part_of_the_ball_view(self):
        for k in ("tile_size_frac", "tile_overlap_frac", "tile_edge_margin_frac"):
            self.assertIn(k, VIEW_KEYS)

    def test_bad_tile_config_refused(self):
        for bad in ({"tile_size_frac": 1.5}, {"tile_size_frac": -0.1}, {"tile_overlap_frac": 1.0},
                    {"tile_edge_margin_frac": 0.2}):                            # margin > overlap / 2
            with self.subTest(bad), self.assertRaises(ConfigError):
                config_from_dict(with_keys(ball=bad))


if __name__ == "__main__":
    unittest.main()
