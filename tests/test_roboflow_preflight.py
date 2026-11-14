"""Preflight before a dataset version: label, class, box, split and ball-size checks. No network (fake project)."""
from __future__ import annotations

import copy
import unittest

from spana.config import ConfigError
from spana.dataset.config import PolicyError
from tests.test_roboflow_prep import img, make_cfg
from tools import roboflow_preflight as pf


def entry(i, mid="M1", split="train", boxes=None, **kw):
    e = {"id": f"id{i}", "name": f"{mid}_f{i}.jpg", "match_id": mid, "split": split, "record_split": split,
         "split_tags": [split], "width": 1280, "height": 720, "annotated": True, "error": None,
         "boxes": [{"class": c, "box": [100.0 + 40 * k, 100.0, 100.0 + 40 * k + s, 100.0 + s]}
                   for k, (c, s) in enumerate(boxes if boxes is not None else
                                              [("player", 30.0), ("goalkeeper", 30.0), ("referee", 30.0),
                                               ("ball", 12.0)])]}
    e.update(kw)
    return e


def good():
    return [entry(1), entry(2, boxes=[("ball", 20.0)]), entry(3, "M2", "valid")]


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def fails(self, entries, pattern, cfg=None):
        with self.assertRaisesRegex(PolicyError, pattern):
            pf.check_labels(cfg or self.cfg, entries)

    def test_pass_counts_and_ball_statistics(self):
        out = pf.check_labels(self.cfg, good())
        self.assertEqual(out["frames"], {"train": 2, "valid": 1})
        self.assertEqual(out["boxes"]["train"], {"player": 1, "goalkeeper": 1, "referee": 1, "ball": 2})
        ball = out["ball"]["all"]
        self.assertEqual(ball["boxes"], 3)
        self.assertEqual(ball["share_below_small_px"], round(2 / 3, 4))      # 12, 12 < 16 <= 20
        self.assertEqual(ball["buckets_px"], {"0-8": 0, "8-12": 0, "12-16": 2, "16-24": 1, ">=24": 0})
        self.assertEqual(ball["short_side_version_px"]["q50"], 12.0)        # 1280x720 in 1280x1280: scale 1

    def test_each_check_fails_closed(self):
        e = good()
        self.fails([dict(e[0], annotated=False, boxes=[])] + e[1:], "without a readable saved annotation")
        self.fails([dict(e[0], boxes=None, error="unexpected annotation shape")] + e[1:], "readable saved annotation")
        self.fails([dict(e[0], record_split="valid")] + e[1:], "differs between the listing")
        self.fails([dict(e[0], split_tags=["valid"])] + e[1:], "split:<name> tag")
        self.fails(e + [entry(4, boxes=[("staff", 30.0)])], "class outside")
        self.fails(e + [entry(4, boxes=[("player", 0.0)])], "empty boxes or boxes outside")
        out = dict(e[0], boxes=[{"class": "player", "box": [1270.0, 10.0, 1300.0, 60.0]}])
        self.fails(e + [out], "outside the frame")
        self.fails(e + [entry(5, "M2", "train")], "holds")                      # valid match also in train
        self.fails(e[:2], "preflight.split_matches expects")                    # valid match missing
        self.fails(e + [entry(6, "M_OTHER", "valid")], "holds")
        self.fails([entry(1, boxes=[("player", 30.0)]), e[2]], "no 'ball' box in the train split")

    def test_border_rounding_is_tolerated(self):
        e = good()
        e.append(entry(7, boxes=[]))
        e[-1]["boxes"] = [{"class": "referee", "box": [-1e-9, 77.8, 26.7, 145.0]}]
        pf.check_labels(self.cfg, e)

    def test_ball_too_small_at_version_size(self):
        cfg = make_cfg(lambda d: (d["version"].update(resize_width=320, resize_height=320),
                                  d["detector"].update(input_resolution=320)))
        self.fails(good(), "below preflight.ball_min_version_px", cfg)          # 12 px * 0.25 = 3 px < 4
        cfg = make_cfg(lambda d: d["version"].update(resize_format="Stretch to"))
        self.fails(good(), "does not keep the aspect ratio", cfg)

    def test_size_buckets(self):
        edges = (8.0, 12.0)
        self.assertEqual([pf.size_bucket(v, edges) for v in (0, 7.9, 8, 11.9, 12, 99)],
                         ["0-8", "0-8", "8-12", "8-12", ">=12", ">=12"])
        self.assertEqual(pf.bucket_names(edges), ["0-8", "8-12", ">=12"])

    def test_read_labels_reads_only_version_frames(self):
        class Project:
            calls = []

            def get_image(self, image_id):
                self.calls.append(image_id)
                return {"id": image_id, "split": "train",
                        "annotation": {"boxes": [{"label": "ball", "x": 10, "y": 10, "width": 4, "height": 6}]}}
        rev = ("reviewed",)
        images = [img(1, tags=rev, in_dataset=True), img(2, tags=rev, in_dataset=False),
                  img(3, tags=rev + (self.cfg.prelabel.blur.excluded_tag,), in_dataset=True)]
        p = Project()
        out = pf.read_labels(self.cfg, p, images)
        self.assertEqual(p.calls, ["id001"])
        self.assertEqual(out[0]["boxes"], [{"class": "ball", "box": [8.0, 7.0, 12.0, 13.0]}])
        self.assertEqual(out[0]["split_tags"], ["train"])

    def test_config_fails_closed(self):
        for mutate in (lambda d: d["preflight"].update(split_matches=["test:M1"]),
                       lambda d: d["preflight"].update(split_matches=["train:M1", "valid:M1"]),
                       lambda d: d["preflight"].update(ball_size_buckets_px=[12.0, 8.0]),
                       lambda d: d["preflight"].update(ball_quantile=1.5),
                       lambda d: d["detector"].update(input_resolution=640),          # would shrink the ball
                       lambda d: d["detector"].update(online_augmentations=["hue"]),
                       lambda d: d["detector"].update(model_id="spana-elite"),
                       lambda d: d["preflight"].pop("ball_min_version_px")):
            with self.assertRaises(ConfigError):
                make_cfg(mutate)


if __name__ == "__main__":
    unittest.main()
