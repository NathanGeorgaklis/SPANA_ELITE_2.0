"""Review-priority flags (mocked Roboflow client, synthetic frames, no network): one test per rule, plus the tool's
write discipline (tags only, licence / review gate, batches, resume)."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from spana.config import ConfigError
from spana.dataset import review_flags as rf
from spana.dataset.guard import GuardedProject
from spana.dataset.prefilter import grass_mask
from spana.dataset.rf_project import RoboflowApiError
from tests.test_roboflow_prep import FakeProject, img, make_cfg
from tools import roboflow_review_flags as tool

H, W = 180, 320


def cfg_with(**flags):
    def mutate(d):
        d["prelabel"]["grass"]["enabled"] = True
        d["review_flags"]["min_interval_s"] = 0.0
        for k, v in flags.items():
            if isinstance(v, dict):
                d["review_flags"][k].update(v)
            else:
                d["review_flags"][k] = v
    return make_cfg(mutate)


def pitch():
    """Grey stand on top, grass below row 60 (the grass boundary is the far touchline)."""
    f = np.full((H, W, 3), (120, 120, 120), np.uint8)
    f[60:, :] = (40, 150, 40)
    return f


def box(cls, x1, y1, x2, y2):
    return {"class": cls, "box": [float(x1), float(y1), float(x2), float(y2)]}


def ann(*boxes):
    return {"boxes": [{"label": b["class"], "x": (b["box"][0] + b["box"][2]) / 2, "y": (b["box"][1] + b["box"][3]) / 2,
                       "width": b["box"][2] - b["box"][0], "height": b["box"][3] - b["box"][1]} for b in boxes]}


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_with()
        self.frame = pitch()
        self.mask = grass_mask(self.frame, self.cfg.prelabel.grass)

    def test_annotation_boxes_are_read_as_corners_and_bad_shapes_fail_closed(self):
        got = rf.boxes_from_annotation({"boxes": [{"label": "ball", "x": 10, "y": 20, "width": 4, "height": 6}]})
        self.assertEqual(got, [{"class": "ball", "box": [8.0, 17.0, 12.0, 23.0]}])
        self.assertEqual(rf.boxes_from_annotation(None), [])
        for bad in ({"boxes": "x"}, {"boxes": [{"label": "ball"}]}, [1]):
            with self.assertRaises(ValueError):
                rf.boxes_from_annotation(bad)

    def test_ball_tiny(self):
        limit = self.cfg.review_flags.ball_tiny.min_size_frac                      # 0.008 x 180 px = 1.44 px
        tiny, ok = box("ball", 100, 100, 101, 101.2), box("ball", 50, 100, 54, 104)
        self.assertEqual(len(rf.ball_tiny([tiny, ok], H, limit)), 1)
        self.assertIn("longer side 1.2 px", rf.ball_tiny([tiny, ok], H, limit)[0])
        self.assertEqual(rf.ball_tiny([ok], H, limit), [])

    def test_ball_offpitch_uses_the_box_centre(self):
        in_stand, on_grass, straddling = box("ball", 100, 20, 106, 26), box("ball", 100, 100, 106, 106), \
            box("ball", 200, 40, 210, 50)                                          # bottom near the grass, centre off
        got = rf.ball_offpitch([in_stand, on_grass, straddling], self.mask)
        self.assertEqual(len(got), 2)
        self.assertTrue(all("outside grass mask" in g for g in got))

    def test_player_edge_flags_feet_on_the_grass_boundary_only(self):
        r = cfg_with(player_edge={"band_frac": 0.05}).review_flags.player_edge      # 9 px band
        margin = self.cfg.prelabel.grass.frame_edge_margin_frac
        touchline = box("player", 100, 30, 110, 63)        # feet 3 px inside the far boundary: possible sideline staff
        midfield = box("player", 150, 80, 160, 120)
        off = box("referee", 200, 20, 210, 50)             # feet in the stand: outside the mask
        cut = box("player", 300, 150, 320, 180)            # cut by the frame's bottom border: foot not observed
        left = box("player", 0, 100, 8, 140)               # at the left frame border, on grass: not a boundary
        got = rf.player_edge([touchline, midfield, off, cut, left], self.mask, r.band_frac, margin)
        self.assertEqual(len(got), 2)
        self.assertIn("inside the grass boundary", got[0])
        self.assertIn("outside the grass mask", got[1])

    def test_dense_counts_heavily_overlapping_person_pairs(self):
        a, b, c = box("player", 10, 10, 30, 60), box("player", 16, 10, 36, 60), box("goalkeeper", 20, 12, 40, 62)
        far = box("player", 200, 10, 220, 60)
        self.assertEqual(rf.dense([a, far], 0.3, 1), [])
        self.assertEqual(len(rf.dense([a, b, far], 0.3, 1)), 1)
        self.assertEqual(rf.dense([a, b, far], 0.3, 2), [])                         # one pair, two needed
        self.assertIn("3 overlapping", rf.dense([a, b, c, far], 0.3, 2)[0])

    def test_frame_flags_splits_ball_and_person_rules(self):
        boxes = [box("ball", 100, 20, 106, 26), box("player", 150, 80, 160, 120)]
        self.assertEqual(sorted(rf.frame_flags(self.cfg, self.frame, boxes, self.mask)), ["ball_offpitch"])
        self.assertEqual(rf.frame_flags(self.cfg, self.frame, [boxes[1]], self.mask), {})


def scene(seed=3, h=480, w=900):
    """Textured static scene (random blocks): gives ORB background features."""
    rng = np.random.default_rng(seed)
    s = cv2.resize(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(s, (3, 3), 0)


def view(sc, x0, y0=100, h=240, w=400, balls=()):
    """Camera window on the scene; balls given in SCENE coordinates are drawn and returned in frame coordinates."""
    sc = sc.copy()
    boxes = []
    for (bx, by) in balls:
        cv2.circle(sc, (bx, by), 4, (255, 255, 255), -1)
        boxes.append(box("ball", bx - x0 - 5, by - y0 - 5, bx - x0 + 5, by - y0 + 5))
    return np.ascontiguousarray(sc[y0:y0 + h, x0:x0 + w]), boxes


class BallStaticTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_with()
        self.s = self.cfg.review_flags.ball_static
        self.scene = scene()

    def _pairs(self, a, b):
        (fa, ba), (fb, bb) = a, b
        t = rf.camera_transform(rf.background_features(fa, ba, self.s), rf.background_features(fb, bb, self.s),
                                fa.shape[0], self.s)
        return rf.ball_static(ba, bb, t, fa.shape[0], self.s), t

    def test_ball_fixed_in_the_scene_while_the_camera_pans_is_flagged(self):
        pairs, t = self._pairs(view(self.scene, 100, balls=[(300, 220)]), view(self.scene, 140, balls=[(300, 220)]))
        self.assertIsNotNone(t)
        self.assertEqual(len(pairs), 1)
        _, _, resid, shift = pairs[0]
        self.assertLess(resid, 1.7)                       # 0.007 x 240 px
        self.assertAlmostEqual(shift, 40.0, delta=1.5)    # the pan

    def test_ball_that_moved_in_the_scene_is_not_flagged(self):
        pairs, t = self._pairs(view(self.scene, 100, balls=[(300, 220)]), view(self.scene, 140, balls=[(330, 240)]))
        self.assertIsNotNone(t)
        self.assertEqual(pairs, [])

    def test_still_camera_is_not_flagged(self):
        """With no camera motion a dead ball in play looks the same as a spare ball: the rule needs a pan."""
        pairs, _ = self._pairs(view(self.scene, 100, balls=[(300, 220)]), view(self.scene, 100, balls=[(300, 220)]))
        self.assertEqual(pairs, [])

    def test_no_reliable_camera_estimate_means_no_flag(self):
        flat = np.full((240, 400, 3), 90, np.uint8)
        b = [box("ball", 100, 100, 110, 110)]
        pairs, t = self._pairs((flat, b), (flat.copy(), b))
        self.assertIsNone(t)
        self.assertEqual(pairs, [])


class ToolTests(unittest.TestCase):
    """compute_flags is read-only; apply_tags only ever ADDS flag tags, through the guard."""

    def setUp(self):
        self.cfg = cfg_with(write_batch_size=2)
        self.frame = pitch()
        self.offpitch = ann(box("ball", 100, 20, 106, 26), box("player", 150, 80, 160, 120))
        self.clean = ann(box("player", 150, 80, 160, 120))

    def _project(self, images, annotations):
        proj = FakeProject(images)
        for i, a in zip(images, annotations):
            proj.annotation[i.id] = a
        return proj

    def _compute(self, proj, images, **kw):
        return tool.compute_flags(self.cfg, proj, images, lambda i: self.frame, **kw)

    def test_compute_reads_current_annotations_and_writes_nothing(self):
        images = [img(1, tags=("prelabel:v2",)), img(2, tags=("prelabel:v2",)), img(3),
                  img(4, tags=("prelabel:v2", "reviewed")), img(5, tags=("prelabel:v2", "excluded:blur")),
                  img(6, mid="M_OTHER", tags=("prelabel:v2",))]
        proj = self._project(images, [self.offpitch, self.clean, None, self.offpitch, self.offpitch, self.offpitch])
        entries, skipped = self._compute(proj, images)
        self.assertEqual([(e["name"], sorted(e["flags"]), e["tags"]) for e in entries],
                         [("M1_f0000001.jpg", ["ball_offpitch"], ["review:ball-offpitch"]), ("M1_f0000002.jpg", [], [])])
        self.assertEqual(sorted(s["reason"].split(":")[0] for s in skipped),
                         ["already reviewed (never touched)", "excluded", "licence gate", "not pre-labelled"])
        self.assertEqual({e[0] for e in proj.events}, {"get_image"})                # no write of any kind
        self.assertEqual(self.cfg.to_dict(), cfg_with(write_batch_size=2).to_dict())

    def test_unreadable_annotation_fails_that_frame_only(self):
        images = [img(1, tags=("prelabel:v2",)), img(2, tags=("prelabel:v2",))]
        proj = self._project(images, [{"boxes": [{"label": "ball"}]}, self.offpitch])
        entries, _ = self._compute(proj, images)
        self.assertEqual([e["status"] for e in entries], ["failed", "ok"])
        self.assertEqual(entries[0]["tags"], [])

    def test_apply_only_adds_tags_in_batches_and_never_touches_annotations(self):
        images = [img(i, tags=("prelabel:v2",)) for i in range(1, 6)]
        proj = self._project(images, [self.offpitch] * 4 + [self.clean])
        proj.tags[images[1].id].add("review:ball-offpitch")                          # tagged by an earlier run
        entries, _ = self._compute(proj, images)
        before = dict(proj.annotation)
        proj.events.clear()
        saves = []
        tool.apply_tags(self.cfg, GuardedProject(self.cfg, proj, images), entries, save=lambda: saves.append(1))
        self.assertEqual([e["written"] for e in entries], ["added", "already_tagged", "added", "added", None])
        self.assertEqual(proj.tag_calls, [(i.id, ("review:ball-offpitch",), ()) for i in (images[0], images[2], images[3])])
        self.assertEqual(proj.ann_calls, [])
        self.assertEqual(proj.annotation, before)
        self.assertEqual(len(saves), 2)                                              # after 2 writes, and at the end
        self.assertTrue(all(c[2] == () for c in proj.tag_calls))                     # nothing is ever removed

    def test_reviewed_or_delisted_after_listing_is_not_written(self):
        images = [img(1, tags=("prelabel:v2",)), img(2, tags=("prelabel:v2",))]
        proj = self._project(images, [self.offpitch, self.offpitch])
        entries, _ = self._compute(proj, images)
        proj.tags[images[0].id].add("reviewed")                                      # a human finished it meanwhile
        cfg = make_cfg(lambda d: (d["prelabel"]["grass"].update(enabled=True),
                                  d.update(licensed_matches=[m for m in d["licensed_matches"] if m["match_id"] != "M1"])))
        tool.apply_tags(self.cfg, GuardedProject(self.cfg, proj, images), entries, save=lambda: None)
        self.assertTrue(entries[0]["written"].startswith("skipped"))
        self.assertEqual(entries[1]["written"], "added")
        entries[1]["written"] = None
        proj.tag_calls.clear()
        tool.apply_tags(cfg, GuardedProject(cfg, proj, images), entries, save=lambda: None)   # M1 off the allow-list
        self.assertEqual(proj.tag_calls, [])
        self.assertTrue(entries[1]["written"].startswith("skipped: changed since listing: licence gate"))

    def test_write_failure_stops_the_run_and_resume_continues_where_it_left_off(self):
        images = [img(i, tags=("prelabel:v2",)) for i in range(1, 5)]
        proj = self._project(images, [self.offpitch] * 4)
        entries, skipped = self._compute(proj, images)
        real, calls = proj.update_tags, []

        def flaky(image_id, add=(), remove=()):
            calls.append(image_id)
            if len(calls) == 3:
                raise RoboflowApiError("HTTP 429: Too Many Requests")
            return real(image_id, add=add, remove=remove)
        proj.update_tags = flaky
        saves = []
        with self.assertRaises(RoboflowApiError):
            tool.apply_tags(self.cfg, GuardedProject(self.cfg, proj, images), entries, save=lambda: saves.append(1))
        self.assertEqual([(e["written"] or "")[:6] for e in entries], ["added", "added", "failed", ""])
        self.assertTrue(saves)                                                       # the log was saved before stopping
        doc = tool.build_doc(self.cfg, entries, skipped, apply=True, match=None)
        tool.check_resume(self.cfg, doc)                                             # same settings: accepted
        proj.update_tags = real
        proj.tag_calls.clear()
        tool.apply_tags(self.cfg, GuardedProject(self.cfg, proj, images), doc["entries"], save=lambda: None)
        self.assertEqual([c[0] for c in proj.tag_calls], [images[2].id, images[3].id])   # only what was missing
        self.assertEqual([e["written"] for e in entries], ["added"] * 4)
        other = cfg_with(ball_tiny={"min_size_frac": 0.02})
        with self.assertRaises(Exception):
            tool.check_resume(other, doc)                                            # settings changed: recompute

    def test_summary_counts_per_match_and_orders_flagged_frames_first(self):
        images = [img(1, tags=("prelabel:v2",)), img(2, tags=("prelabel:v2",)), img(3, tags=("prelabel:v2",)),
                  img(4, mid="M2", tags=("prelabel:v2",))]
        both = ann(box("ball", 100, 20, 106, 26), box("player", 10, 80, 30, 130), box("player", 14, 80, 34, 130),
                   box("player", 18, 80, 38, 130))
        proj = self._project(images, [self.clean, self.offpitch, both, self.offpitch])
        entries, skipped = self._compute(proj, images)
        s = tool.summarize(self.cfg, entries, skipped)
        self.assertEqual((s["per_match"]["M1"]["ball_offpitch"], s["per_match"]["M1"]["dense"],
                          s["per_match"]["M1"]["flagged"], s["per_match"]["M2"]["ball_offpitch"]), (2, 1, 2, 1))
        self.assertEqual([r["name"] for r in s["review_order"]["M1"]],
                         ["M1_f0000003.jpg", "M1_f0000002.jpg", "M1_f0000001.jpg"])
        text = tool.summary_text(tool.build_doc(self.cfg, entries, skipped, apply=False, match=None))
        self.assertIn("review:ball-offpitch", text)
        self.assertIn("DRY RUN", text)


class ConfigTests(unittest.TestCase):
    def test_flag_tags_must_be_distinct_and_never_match_or_split_or_reviewed(self):
        for bad in ({"ball_tiny": {"tag": "review:dense"}}, {"dense": {"tag": "reviewed"}},
                    {"dense": {"tag": "match:M1"}}, {"dense": {"tag": "prelabel:v2"}}, {"dense": {"min_pairs": 0}},
                    {"write_batch_size": 0}):
            with self.assertRaises(ConfigError):
                cfg_with(**bad)

    def test_guard_refuses_a_flag_write_on_a_reviewed_frame(self):
        cfg = cfg_with()
        images = [img(1, tags=("prelabel:v2", "reviewed"))]
        proj = FakeProject(images)
        g = GuardedProject(cfg, proj, images)
        g.refresh(images[0].id)
        with self.assertRaises(PermissionError):
            g.update_tags(images[0].id, add=["review:dense"])
        self.assertEqual(proj.tag_calls, [])


if __name__ == "__main__":
    unittest.main()
