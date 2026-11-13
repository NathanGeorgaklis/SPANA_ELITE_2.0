"""Pre-label v2 (mocked Roboflow client, synthetic frames, no network): write guard, run control, blur gate, grass
filter, ball tiling / merge, per-class thresholds, migration backup ordering."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

import cv2
import numpy as np

from spana.dataset.config import PolicyError
from spana.dataset.guard import GuardedProject
from spana.dataset.prefilter import blur_score, grass_mask, merge_boxes, on_grass, tiles
from tests.test_roboflow_prep import FakeProject, img, make_cfg, p, resp, sharp
from tools import roboflow_prelabel as pl

WRITES = ("update_tags", "save_annotation")


def writes(proj):
    return [e for e in proj.events if e[0] in WRITES]


def cfg_with(**prelabel):
    def mutate(d):
        for k, v in prelabel.items():
            if isinstance(v, dict):
                d["prelabel"][k].update(v)
            else:
                d["prelabel"][k] = v
    return make_cfg(mutate)


def run(cfg, proj, images, infer=None, frame=None, **kw):
    infer = infer or mock.Mock(return_value=resp([p(10, 10, 4, 10, 0.9, "person")]))
    frame = sharp() if frame is None else frame
    args = dict(limit=20, match=None, confirm_full_run=False, load=lambda i: frame, render_dir=None)
    args.update(kw)
    return pl.run_prelabel(cfg, proj, infer, images, **args)


class WriteGuardTests(unittest.TestCase):
    """Allow-list (and reviewed tag) enforced on every write path, re-read immediately before writing."""

    def setUp(self):
        self.cfg = make_cfg()

    def _guard(self, images):
        proj = FakeProject(images)
        return proj, GuardedProject(self.cfg, proj, images)

    def test_every_write_path_checks_the_allow_list(self):
        bad = [img(1, mid="M_OTHER"), img(2, mid="M_EVAL"), img(3, mid="M_UNCONF"), img(4, tags=("reviewed",)),
               img(5, mid=None)]
        proj, g = self._guard(bad)
        for i in bad:
            g.refresh(i.id)
            calls = [lambda: g.update_tags(i.id, add=["x"]),
                     lambda: g.update_tags(i.id, remove=["prelabel:v1"]),
                     lambda: g.save_annotation(i.id, name="a.txt", content="", labelmap={}, add_to_dataset=False),
                     lambda: g.save_annotation(i.id, name="a.txt", content="", labelmap={}, add_to_dataset=False,
                                               overwrite=True)]
            for c in calls:
                with self.assertRaises(PolicyError):
                    c()
        self.assertEqual(writes(proj), [])
        self.assertEqual(g.writes, [])

    def test_unknown_or_unrefreshed_image_and_forbidden_tags_refused(self):
        proj, g = self._guard([img(1)])
        with self.assertRaisesRegex(PolicyError, "not in this run's listing"):
            g.update_tags("nope", add=["x"])
        with self.assertRaisesRegex(PolicyError, "re-read"):
            g.update_tags("id001", add=["x"])
        g.refresh("id001")
        for add, rem in ((["reviewed"], []), (["match:M2"], []), ([], ["split:train"])):
            with self.assertRaises(PolicyError):
                g.update_tags("id001", add=add, remove=rem)
        self.assertEqual(writes(proj), [])
        g.update_tags("id001", add=["prelabel:v2"])
        self.assertEqual(writes(proj), [("update_tags", "id001", ("prelabel:v2",), ())])

    def test_frame_reviewed_after_listing_is_never_touched(self):
        images = [img(1), img(2)]
        proj = FakeProject(images)
        proj.tags["id002"].add("reviewed")              # a human reviewed it after the listing was taken
        doc = run(self.cfg, proj, images)
        st = {e["id"]: e["status"] for e in doc["entries"]}
        self.assertEqual(st, {"id001": "labelled", "id002": "skipped"})
        self.assertEqual({e[1] for e in writes(proj)}, {"id001"})

    def test_allow_list_change_after_listing_blocks_writes(self):
        images = [img(1)]
        proj = FakeProject(images)
        proj.images = [img(1, mid="M_OTHER")]           # server now says another (unlicensed) match
        proj.tags["id001"] = set(proj.images[0].tags)
        doc = run(self.cfg, proj, images)
        self.assertEqual(doc["entries"][0]["status"], "skipped")
        self.assertIn("licence gate", doc["entries"][0]["reason"])
        self.assertEqual(writes(proj), [])

    def test_run_never_modifies_the_allow_list(self):
        before = copy.deepcopy(self.cfg.licensed_matches)
        run(self.cfg, FakeProject([img(1)]), [img(1)])
        self.assertEqual(self.cfg.licensed_matches, before)

    def test_diagnostic_is_read_only_and_licence_gated(self):
        images = [img(1), img(2, mid="M_OTHER")]
        proj = FakeProject(images)
        infer = mock.Mock(return_value=resp([p(10, 10, 2, 2, 0.4, "sports ball"), p(20, 10, 4, 10, 0.9, "person")]))
        doc = pl.diagnose_ball(self.cfg, lambda jpg, conf: infer(jpg), images, ["id001", "id002"],
                               lambda i: sharp())
        self.assertEqual(doc["frames"][0]["full"]["top"][0]["score"], 0.4)    # raw score, below the class threshold
        self.assertIn("licence gate", doc["frames"][1]["status"])
        self.assertEqual(proj.events, [])


class RunControlTests(unittest.TestCase):
    """Stops at prelabel.sample_size; going beyond needs confirmation AND a sample made with these exact settings."""

    def setUp(self):
        self.cfg = make_cfg()
        self.images = [img(i) for i in range(60)]

    def _sample(self, cfg=None, **kw):
        cfg = cfg or self.cfg
        return run(cfg, FakeProject(self.images), self.images, limit=cfg.prelabel.sample_size, **kw)

    def test_stops_after_sample_size(self):
        proj = FakeProject(self.images)
        doc = run(self.cfg, proj, self.images, limit=self.cfg.prelabel.sample_size)
        n = self.cfg.prelabel.sample_size
        self.assertEqual(doc["summary"]["selected"], n)
        self.assertEqual(len({e[1] for e in writes(proj)}), n)
        for kw in (dict(limit=n + 1), dict(limit=None), dict(limit=n + 1, confirm_full_run=True),
                   dict(limit=None, confirm_full_run=True)):
            proj = FakeProject(self.images)
            with self.assertRaises(PolicyError):
                run(self.cfg, proj, self.images, **kw)
            self.assertEqual(proj.events, [])           # refused before any read or write

    def test_approval_must_match_current_settings(self):
        sample = self._sample()
        doc = run(self.cfg, FakeProject(self.images), self.images, limit=None, confirm_full_run=True,
                  approved_sample=sample)
        self.assertEqual(doc["summary"]["selected"], 60 - 0)
        other = cfg_with(class_thresholds=["player:0.50", "ball:0.60"])
        with self.assertRaisesRegex(PolicyError, "sha256"):
            run(other, FakeProject(self.images), self.images, limit=None, confirm_full_run=True, approved_sample=sample)
        v1_like = dict(sample, schema_version="1.0.0")
        with self.assertRaisesRegex(PolicyError, "this version"):
            run(self.cfg, FakeProject(self.images), self.images, limit=None, confirm_full_run=True,
                approved_sample=v1_like)
        dry = self._sample(dry_run=True)
        with self.assertRaisesRegex(PolicyError, "dry run"):
            run(self.cfg, FakeProject(self.images), self.images, limit=None, confirm_full_run=True,
                approved_sample=dry)

    def test_dry_run_writes_nothing(self):
        proj = FakeProject(self.images)
        doc = run(self.cfg, proj, self.images, dry_run=True)
        self.assertEqual(writes(proj), [])
        self.assertGreater(doc["writes"], 0)            # the writes it would have made are counted


class BlurGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        rng = np.random.default_rng(3)
        self.sharp = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
        self.blurred = cv2.GaussianBlur(self.sharp, (0, 0), 6)

    def test_score_orders_sharp_over_blurred_and_is_resolution_independent(self):
        w = self.cfg.prelabel.blur.analysis_width_px
        s, b = blur_score(self.sharp, w), blur_score(self.blurred, w)
        self.assertGreater(s, self.cfg.prelabel.blur.laplacian_var_min)
        self.assertLess(b, self.cfg.prelabel.blur.laplacian_var_min)
        big = cv2.resize(self.blurred, (640, 360))
        self.assertLess(blur_score(big, w), self.cfg.prelabel.blur.laplacian_var_min)

    def test_blurred_frame_is_tagged_and_skipped(self):
        images = [img(1)]
        proj = FakeProject(images)
        infer = mock.Mock()
        doc = run(self.cfg, proj, images, infer=infer, frame=self.blurred)
        e = doc["entries"][0]
        self.assertEqual((e["status"], e["blurred"]), ("excluded_blur", True))
        infer.assert_not_called()
        self.assertEqual(writes(proj), [("update_tags", "id001", ("excluded:blur",), ())])
        self.assertIn("excluded (blur)", pl.eligibility(self.cfg, img(1, tags=("excluded:blur",))))

    def test_blurred_v1_frame_in_migration_is_cleared_after_backup(self):
        images = [img(1, tags=("prelabel:v1",), annotated=True, classes=("player",))]
        proj = FakeProject(images)
        backups = []
        doc = run(self.cfg, proj, images, frame=self.blurred, migrate=True,
                  write_backup=lambda d: backups.append(d) or {"path": "b.json"})
        self.assertEqual(doc["entries"][0]["status"], "excluded_blur")
        self.assertEqual(proj.annotation["id001"], {"boxes": []})
        self.assertEqual(proj.tags["id001"], {"match:M1", "split:train", "excluded:blur"})
        self.assertEqual(backups[0]["frames"][0]["annotation"], {"boxes": [{"label": "player"}]})


class GrassFilterTests(unittest.TestCase):
    """Synthetic frame: grass in the lower 2/3, grey stand on top; players are non-green blobs on the grass."""

    def setUp(self):
        self.cfg = cfg_with(grass={"enabled": True})
        h, w = 180, 320
        f = np.full((h, w, 3), (120, 120, 120), np.uint8)                       # stand (grey)
        f[60:, :] = (40, 150, 40)                                                 # pitch (BGR green)
        f[90:140, 100:120] = (30, 30, 200)                                        # front player's kit (red)
        f[40:60, 200:212] = (200, 200, 200)                                       # spectator in the stand
        self.frame = f
        self.on_pitch = {"class": "player", "class_id": 0, "box": [40.0, 100.0, 60.0, 150.0], "score": 0.9}
        self.occluded = {"class": "player", "class_id": 0, "box": [102.0, 80.0, 118.0, 120.0], "score": 0.8}
        self.spectator = {"class": "player", "class_id": 0, "box": [200.0, 35.0, 212.0, 59.0], "score": 0.7}
        self.edge = {"class": "player", "class_id": 0, "box": [250.0, 30.0, 262.0, 61.0], "score": 0.6}

    def test_off_pitch_dropped_with_reason_on_pitch_kept(self):
        kept, dropped = pl.grass_filter(self.cfg, self.frame, [self.on_pitch, self.occluded, self.spectator, self.edge])
        self.assertEqual(kept, [self.on_pitch, self.occluded, self.edge])   # occluded: feet on another kit
        self.assertEqual([d["box"] for d in dropped], [self.spectator["box"]])
        self.assertIn("outside grass mask", dropped[0]["reason"])

    def test_without_hull_the_occluded_player_would_be_lost(self):
        g = cfg_with(grass={"enabled": True, "convex_hull": False, "close_frac": 0.0}).prelabel.grass
        self.assertFalse(on_grass(grass_mask(self.frame, g), self.occluded["box"]))
        self.assertTrue(on_grass(grass_mask(self.frame, self.cfg.prelabel.grass), self.occluded["box"]))

    def test_filter_is_logged_per_frame_and_counts_players(self):
        images = [img(1)]
        frame = self.frame
        cfg = cfg_with(grass={"enabled": True}, blur={"laplacian_var_min": 0.0})   # flat synthetic frame
        infer = mock.Mock(return_value=resp([p(50, 125, 20, 50, 0.9, "person"), p(206, 47, 12, 24, 0.7, "person")],
                                            w=320, h=180))
        doc = run(cfg, FakeProject(images), images, infer=infer, frame=frame)
        e = doc["entries"][0]
        self.assertEqual((e["players_kept"], e["players_dropped"]), (1, 1))
        self.assertEqual(doc["summary"]["per_match"]["M1"]["players_dropped_off_pitch"], 1)

    def test_disabled_filter_keeps_everything(self):
        cfg = make_cfg()
        kept, dropped = pl.grass_filter(cfg, self.frame, [self.spectator])
        self.assertEqual((len(kept), dropped), (1, []))


class BallTilingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_with(ball_tiling={"enabled": True, "tile_px": 64, "overlap_frac": 0.25, "merge_iou": 0.3})
        self.frame = sharp(120, 200)

    def _infer_for(self, balls, players=()):
        """Fake model: reports every ball fully inside the request window (each tile and the full frame)."""
        frame_wh = (self.frame.shape[1], self.frame.shape[0])
        windows = iter([(0, 0, *frame_wh)] + tiles(*frame_wh, 64, 0.25))

        def infer(jpg):
            x1, y1, x2, y2 = next(windows)
            preds = [p(bx - x1, by - y1, 6, 6, s, "sports ball") for bx, by, s in balls
                     if x1 + 3 <= bx <= x2 - 3 and y1 + 3 <= by <= y2 - 3]
            if (x1, y1) == (0, 0) and (x2, y2) == frame_wh:
                preds += [p(px, py, 10, 30, s, "person") for px, py, s in players]
            return resp(preds, w=x2 - x1, h=y2 - y1)
        return infer

    def test_tiles_cover_frame_deterministically(self):
        t = tiles(200, 120, 64, 0.25)
        self.assertEqual(t, tiles(200, 120, 64, 0.25))
        cover = np.zeros((120, 200), bool)
        for x1, y1, x2, y2 in t:
            cover[y1:y2, x1:x2] = True
            self.assertEqual((x2 - x1, y2 - y1), (64, 64))
        self.assertTrue(cover.all())
        self.assertEqual(tiles(50, 40, 64, 0.25), [(0, 0, 50, 40)])

    def test_one_ball_seen_in_many_tiles_gives_one_box(self):
        boxes, info = pl.detect(self.cfg, self.frame, self._infer_for([(55, 55, 0.9)]))
        balls = [b for b in boxes if b["class"] == "ball"]
        self.assertEqual(len(balls), 1)
        self.assertGreater(info["ball_merged"], 0)
        self.assertEqual(info["tiles"], len(tiles(200, 120, 64, 0.25)))

    def test_two_distinct_balls_stay_two(self):
        boxes, _ = pl.detect(self.cfg, self.frame, self._infer_for([(20, 20, 0.9), (170, 100, 0.8)]))
        self.assertEqual(len([b for b in boxes if b["class"] == "ball"]), 2)

    def test_merge_keeps_highest_score_and_handles_cut_balls(self):
        whole = {"box": [10.0, 10.0, 16.0, 16.0], "score": 0.7}
        cut = {"box": [10.0, 10.0, 13.0, 16.0], "score": 0.9}            # same ball cut by a tile edge
        far = {"box": [80.0, 80.0, 86.0, 86.0], "score": 0.6}
        kept, merged = merge_boxes([whole, cut, far], 0.3)
        self.assertEqual((kept, merged), ([cut, far], 1))

    def test_tiling_never_changes_player_boxes(self):
        infer_args = dict(balls=[(55, 55, 0.9)], players=[(100, 60, 0.9)])
        on, _ = pl.detect(self.cfg, self.frame, self._infer_for(**infer_args))
        off, _ = pl.detect(make_cfg(), self.frame, self._infer_for(**infer_args))
        self.assertEqual([b for b in on if b["class"] == "player"], [b for b in off if b["class"] == "player"])

    def test_oversized_ball_box_dropped_and_logged(self):
        cfg = cfg_with(ball_max_size_frac=0.1)
        boxes = [{"class": "ball", "class_id": 3, "box": [0.0, 0.0, 20.0, 20.0], "score": 0.9},
                 {"class": "ball", "class_id": 3, "box": [50.0, 50.0, 56.0, 56.0], "score": 0.9},
                 {"class": "player", "class_id": 0, "box": [0.0, 0.0, 30.0, 90.0], "score": 0.9}]
        kept, dropped = pl.ball_size_filter(cfg, 120, boxes)
        self.assertEqual([b["box"] for b in kept], [[50.0, 50.0, 56.0, 56.0], [0.0, 0.0, 30.0, 90.0]])
        self.assertIn("ball_max_size_frac", dropped[0]["reason"])


class ThresholdIndependenceTests(unittest.TestCase):
    PREDS = [p(10, 10, 4, 10, 0.55, "person"), p(30, 10, 4, 10, 0.95, "person"),
             p(40, 30, 2, 2, 0.55, "sports ball"), p(50, 30, 2, 2, 0.95, "sports ball")]

    def _classes(self, cfg):
        boxes, _, _ = pl.map_predictions(cfg, resp(self.PREDS), 64, 36)
        return ([b["score"] for b in boxes if b["class"] == "player"], [b["score"] for b in boxes if b["class"] == "ball"])

    def test_ball_and_player_thresholds_are_independent(self):
        base = self._classes(cfg_with(class_thresholds=["player:0.50", "ball:0.50"]))
        strict_ball = self._classes(cfg_with(class_thresholds=["player:0.50", "ball:0.90"]))
        strict_player = self._classes(cfg_with(class_thresholds=["player:0.90", "ball:0.50"]))
        self.assertEqual(base, ([0.95, 0.55], [0.95, 0.55]))
        self.assertEqual(strict_ball, (base[0], [0.95]))          # players unchanged
        self.assertEqual(strict_player, ([0.95], base[1]))        # balls unchanged


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.images = [img(1, tags=("prelabel:v1",), annotated=True, classes=("player", "player")),
                       img(2, tags=("prelabel:v1", "reviewed"), annotated=True, classes=("player",)),
                       img(3)]

    def test_backup_written_before_any_change_and_reviewed_untouched(self):
        proj = FakeProject(self.images)
        order = []

        def backup(doc):
            order.append(("backup", len(writes(proj))))
            self.backup = doc
            return {"path": "b.json"}
        doc = run(self.cfg, proj, self.images, migrate=True, write_backup=backup)
        self.assertEqual(order, [("backup", 0)])                 # no write had happened yet
        self.assertEqual([f["id"] for f in self.backup["frames"]], ["id001"])
        self.assertEqual(self.backup["frames"][0]["annotation"], {"boxes": [{"label": "player"}, {"label": "player"}]})
        self.assertEqual([e["id"] for e in doc["entries"]], ["id001"])
        self.assertNotIn("id002", {e[1] for e in writes(proj)})  # reviewed: never touched
        self.assertNotIn("id003", {e[1] for e in writes(proj)})  # not a v1 frame
        # overwrite, then (after read-back) swap tags in one call
        w = writes(proj)
        self.assertEqual(w[0], ("save_annotation", "id001", True))
        self.assertEqual(w[-1], ("update_tags", "id001", ("prelabel:v2",), ("prelabel:v1",)))
        self.assertEqual(proj.tags["id001"], {"match:M1", "split:train", "prelabel:v2"})

    def test_backup_failure_means_no_writes(self):
        proj = FakeProject(self.images)

        def boom(doc):
            raise OSError("disk full")
        with self.assertRaises(OSError):
            run(self.cfg, proj, self.images, migrate=True, write_backup=boom)
        self.assertEqual(writes(proj), [])
        with self.assertRaisesRegex(PolicyError, "backup"):
            run(self.cfg, FakeProject(self.images), self.images, migrate=True)

    def test_read_back_mismatch_keeps_old_tag(self):
        proj = FakeProject(self.images)
        proj.annotate_result = "already_annotated"              # server silently kept the old annotation
        doc = run(self.cfg, proj, self.images, migrate=True, write_backup=lambda d: {"path": "b.json"},
                  infer=mock.Mock(return_value=resp([p(10, 10, 4, 10, 0.9, "person")])))
        self.assertEqual(doc["entries"][0]["status"], "failed")
        self.assertIn("old pre-label kept", doc["entries"][0]["reason"])
        self.assertIn("prelabel:v1", proj.tags["id001"])
        self.assertNotIn("prelabel:v2", proj.tags["id001"])

    def test_same_classes_but_old_boxes_on_server_is_a_mismatch(self):
        images = [img(1, tags=("prelabel:v1",), annotated=True)]
        proj = FakeProject(images)
        proj.annotation["id001"] = {"boxes": [{"label": "player", "x": 50, "y": 20, "width": 4, "height": 10}]}
        real_save = proj.save_annotation

        def save_but_keep(image_id, **kw):          # server says "saved" but keeps the old boxes
            real_save(image_id, **kw)
            proj.annotation[image_id] = {"boxes": [{"label": "player", "x": 50, "y": 20, "width": 4, "height": 10}]}
            return "saved"
        proj.save_annotation = save_but_keep
        doc = run(self.cfg, proj, images, migrate=True, write_backup=lambda d: {"path": "b.json"})
        self.assertEqual(doc["entries"][0]["status"], "failed")
        self.assertIn("read-back", doc["entries"][0]["reason"])
        self.assertIn("prelabel:v1", proj.tags["id001"])


    def test_v1_frames_need_migrate(self):
        doc = run(self.cfg, FakeProject(self.images), self.images)
        reasons = {s["id"]: s["reason"] for s in doc["not_selected_skips"]}
        self.assertIn("--migrate", reasons["id001"])
        self.assertIn("reviewed", reasons["id002"])


class ReviewFixTests(unittest.TestCase):
    """Regressions from the code review of the first v2 commit."""

    def setUp(self):
        self.cfg = make_cfg()

    def test_annotated_outside_dataset_is_seen_on_refresh(self):
        images = [img(1)]                           # search listing: not annotated (search omits it)
        proj = FakeProject(images)
        proj.annotation["id001"] = {"boxes": [{"label": "referee"}]}    # human labels on the server
        doc = run(self.cfg, proj, images)
        self.assertEqual(doc["entries"][0]["status"], "skipped")
        self.assertIn("already annotated", doc["entries"][0]["reason"])
        self.assertEqual(writes(proj), [])

    def test_unexpected_error_fails_one_frame_and_run_continues(self):
        images = [img(1), img(2)]

        def load(i):
            if i.id == "id001":
                raise cv2.error("decoder exploded")
            return sharp()
        doc = run(self.cfg, FakeProject(images), images, load=load)
        self.assertEqual([e["status"] for e in doc["entries"]], ["failed", "labelled"])

    def test_render_failure_does_not_fail_a_written_frame(self):
        images = [img(1)]
        with mock.patch.object(pl, "render", side_effect=RuntimeError("no display")):
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as d:
                doc = run(self.cfg, FakeProject(images), images, render_dir=Path(d))
        self.assertEqual(doc["entries"][0]["status"], "labelled")
        self.assertIn("no display", doc["entries"][0]["render_error"])

    def test_reviewed_during_inference_blocks_the_write(self):
        images = [img(1)]
        proj = FakeProject(images)

        def infer(jpg):
            proj.tags["id001"].add("reviewed")      # a reviewer tags it while the model runs
            return resp([p(10, 10, 4, 10, 0.9, "person")])
        doc = run(self.cfg, proj, images, infer=infer)
        self.assertEqual(doc["entries"][0]["status"], "skipped")
        self.assertEqual(writes(proj), [])

    def test_oversize_full_frame_ball_cannot_swallow_a_tile_ball(self):
        cfg = cfg_with(ball_tiling={"enabled": True, "tile_px": 64, "overlap_frac": 0.25, "merge_iou": 0.3},
                       ball_max_size_frac=0.1)
        frame = sharp(120, 200)
        windows = iter([(0, 0, 200, 120)] + tiles(200, 120, 64, 0.25))

        def infer(jpg):
            x1, y1, x2, y2 = next(windows)
            if (x2 - x1, y2 - y1) == (200, 120):     # full frame: one huge high-score "ball" (an overlay)
                return resp([p(55, 55, 40, 40, 0.99, "sports ball")], w=200, h=120)
            preds = [p(55 - x1, 55 - y1, 6, 6, 0.8, "sports ball")] if x1 + 3 <= 55 <= x2 - 3 and y1 + 3 <= 55 <= y2 - 3 else []
            return resp(preds, w=x2 - x1, h=y2 - y1)
        boxes, info = pl.detect(cfg, frame, infer)
        balls = [b for b in boxes if b["class"] == "ball"]
        self.assertEqual(len(balls), 1)
        self.assertAlmostEqual(balls[0]["box"][2] - balls[0]["box"][0], 6.0)
        self.assertEqual(len(info["too_big"]), 1)


if __name__ == "__main__":
    unittest.main()
