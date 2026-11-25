"""tools/dataset_v3.py: review file -> frozen review -> preflight -> staged dataset version (no upload, no training).
Synthetic review files and generated images only; the one real input is the frozen labels of dataset version 2
(read-only, no pixels)."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from spana.dataset import detector_review as rv
from spana.dataset.config import load_dataset_config
from tools import dataset_v3 as ds
from tools import detector_mine as dm

ROOT = Path(__file__).resolve().parents[1]
DCFG = load_dataset_config(ROOT / "config" / "roboflow.yaml")
SPLIT_OF = DCFG.preflight.expected()
TRAIN_M = next(m for m, s in SPLIT_OF.items() if s == "train")       # match ids come from the config, not the test
VALID_M = next(m for m, s in SPLIT_OF.items() if s == "valid")
DEV_M = next(m.match_id for m in DCFG.licensed_matches if m.split == "dev")
BALL = DCFG.prelabel.ball_class
REAL = yaml.safe_load((ROOT / "config" / "dataset_v3.yaml").read_text())
W, H = 320, 180
CONE = "equipment/cone"
MIN = 2                                                              # min_per_category of the test mine config


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jpg(seed: int, ball_box=None) -> bytes:
    img = np.random.default_rng(seed).integers(20, 90, (H, W, 3), dtype=np.uint8)
    if ball_box:
        x1, y1, x2, y2 = (int(v) for v in ball_box)
        img[y1:y2, x1:x2] = 255
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def base_frame(n: int, match: str, split: str, balls) -> dict:
    boxes = [{"class": "player", "box": [10.0, 40.0, 30.0, 100.0]}, {"class": "player", "box": [150.0, 30.0, 170.0, 90.0]}]
    return {"annotated": True, "boxes": boxes + [{"class": BALL, "box": [float(v) for v in b]} for b in balls],
            "error": None, "height": H, "id": f"img{n:03d}", "match_id": match, "name": f"{match}_f{n:07d}.jpg",
            "record_split": split, "split": split, "split_tags": [SPLIT_OF[match]], "width": W}


BASE_BALL, SPARE_BOX = [100, 60, 112, 72], [290, 165, 310, 180]
BASE = [base_frame(1, TRAIN_M, "train", [BASE_BALL]),
        base_frame(2, TRAIN_M, "train", []),                             # no ball: no scale copy
        base_frame(3, VALID_M, "valid", [BASE_BALL, SPARE_BOX]),          # the two-ball frame: second box is a spare
        base_frame(4, VALID_M, "test", [BASE_BALL])]                      # platform test frame: never copied


class Tmp:
    """A throw-away repository root: base labels, image cache, mined frames, configs."""

    def __init__(self, t: str, min_per_category: int = MIN, **cfg_changes):
        self.root = Path(t)
        labels = json.dumps({"classes": list(DCFG.classes), "created_utc": "x", "frames": BASE,
                             "kind": "spana.roboflow_labels", "project": "p"}).encode()
        self.write("labels.json", labels)
        self.base_sha = sha(labels)
        for f in BASE:
            self.write(f"{DCFG.paths.image_cache_dir}/{f['id']}.jpg", jpg(int(f["id"][3:]), BASE_BALL))
        self.dev_jpg = jpg(900)
        self.write("dev.json", json.dumps({"frames": [{"name": f"{DEV_M}_c00_f0000001.jpg",
                                                       "image_sha256": sha(self.dev_jpg)}]}).encode())
        cfg = {**copy.deepcopy(REAL), "base_labels": "labels.json", "base_labels_sha256": self.base_sha,
               "dev_labels": ["dev.json"], **cfg_changes}
        self.write("v3.yaml", yaml.safe_dump(cfg).encode())
        mine = yaml.safe_load((ROOT / "config" / "detector_mine.yaml").read_text())
        self.write("mine.yaml", yaml.safe_dump({**mine, "min_per_category": min_per_category}).encode())
        self.ctx = ds.load_context(self.root / "v3.yaml", ROOT / "config" / "roboflow.yaml", self.root / "mine.yaml",
                                   ROOT / "config" / "ball_fp_diag.yaml", self.root)
        self.cfg = self.ctx.cfg
        self.seed = 100

    def write(self, rel: str, data: bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def frame(self, match: str, n: int, **fields) -> dict:
        """A mined frame on disk and its review entry."""
        self.seed += 1
        data = jpg(self.seed, fields.get("match_ball_box"))
        name = f"{match}_mined_f{n:07d}.jpg"
        self.write(f"{self.cfg.frames_dir}/{name}", data)
        return {"name": name, "sha256": sha(data), "object_box": None, "match_ball_box": None,
                "match_ball_absent": False, **fields}

    def review(self, candidates, existing="default", version=1) -> dict:
        if existing == "default":
            existing = [{"image_id": "img003", "box": SPARE_BOX, "verdict": rv.SPARE_BALL},
                        {"image_id": "img003", "box": BASE_BALL, "verdict": rv.MATCH_BALL}]
        return {"schema": rv.SCHEMA[0], "schema_version": rv.SCHEMA[1], "review_version": version,
                "reviewer": "test", "reviewed_utc": "2026-01-01T00:00:00+00:00", "base_labels_sha256": self.base_sha,
                "candidates": candidates, "existing_labels": existing}

    def build(self, review: dict) -> Path:
        return ds.build(self.write(f"review_v{review['review_version']}.json", json.dumps(review).encode()), self.ctx)

    def staged(self) -> bool:
        return (self.root / self.cfg.stage_dir).exists()


def cand(cid: str, match: str, verdict: str, frames, category=None, split="TRAIN") -> dict:
    return {"id": cid, "match_id": match, "split": split, "kind": "tiled_only", "verdict": verdict,
            "category": category, "frames": frames}


def good(t: Tmp) -> list:
    """A review that passes: every verdict, a frame shared with an UNSURE candidate, one ball seen by two candidates."""
    ball_a, cone, spare = [40, 50, 50, 60], [200, 100, 212, 112], [5, 150, 17, 162]
    shared = t.frame(TRAIN_M, 14, match_ball_box=ball_a)
    twice = t.frame(TRAIN_M, 15, match_ball_box=ball_a)
    return [
        cand("a1", TRAIN_M, rv.MATCH_BALL, [t.frame(TRAIN_M, 10, match_ball_box=ball_a), shared, twice]),
        cand("a2", TRAIN_M, rv.MATCH_BALL, [{**twice, "match_ball_box": [41, 50, 51, 60]}]),   # the same ball again
        cand("b1", TRAIN_M, rv.NOT_BALL, [t.frame(TRAIN_M, 20, object_box=cone, match_ball_absent=True),
                                          t.frame(TRAIN_M, 21, object_box=cone, match_ball_box=ball_a)], CONE),
        cand("c1", VALID_M, rv.SPARE_BALL, [t.frame(VALID_M, 30, object_box=spare, match_ball_box=ball_a),
                                            t.frame(VALID_M, 31, object_box=spare, match_ball_absent=True)]),
        cand("u1", TRAIN_M, rv.UNSURE, [{"name": shared["name"], "sha256": None}, {"name": f"{TRAIN_M}_gone.jpg"}]),
        cand("s1", TRAIN_M, rv.SKIP, [{"name": f"{TRAIN_M}_never_exported.jpg"}]),
    ]


class HappyPath(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.t = Tmp(self._t.name)
        self.dest = self.t.build(self.t.review(good(self.t)))
        self.m = json.loads((self.dest / "stage.json").read_text())
        self.frames = {f["name"]: f for f in self.m["frames"]}

    def test_review_is_frozen_with_sha256_and_provenance(self):
        d = self.t.root / self.t.cfg.review_dir / "v1"
        frozen = json.loads((d / "FROZEN.json").read_text())
        self.assertEqual(frozen["record_sha256"], sha((d / "review_frozen.json").read_bytes()))
        self.assertEqual(frozen["record_sha256"], self.m["review"]["record_sha256"])
        self.assertEqual((frozen["reviewer"], frozen["split"], frozen["base_labels_sha256"]),
                         ("test", "TRAIN", self.t.base_sha))
        self.assertEqual(frozen["candidates"], {"MATCH_BALL": 2, "NOT_BALL": 1, "SPARE_BALL": 1, "UNSURE": 1, "SKIP": 1})

    def test_only_reviewed_frames_are_staged_and_nothing_is_uploaded(self):
        self.assertFalse(self.m["uploaded"])
        self.assertEqual(len(self.frames), 6)                          # 2 ball + 2 look-alike + 2 spare
        self.assertEqual(self.m["frames_left_out"], [f"{TRAIN_M}_mined_f0000014.jpg"])   # shared with UNSURE
        self.assertEqual(self.m["counts"]["candidates_never_used"], {"UNSURE": 1, "SKIP": 1})
        self.assertEqual(self.m["counts"]["mined_frames"], {"train": 4, "valid": 2})
        for f in self.frames.values():
            self.assertEqual(sha((self.dest / f["file"]).read_bytes()), f["sha256"])
            self.assertEqual(f["split"], SPLIT_OF[f["match_id"]])
            self.assertIn(f"match:{f['match_id']}", f["tags"])
            self.assertIn(self.t.cfg.tags.persons_pending, f["tags"])
            self.assertNotIn(DCFG.review.reviewed_tag, f["tags"])       # the version gate keeps them out for now
            self.assertLessEqual(len(f["boxes"]), 1)
        staged = sorted(p.name for p in (self.dest / "images").rglob("*.jpg") if "__" not in p.name)
        self.assertEqual(staged, sorted(self.frames))

    def test_match_ball_is_boxed_spare_balls_and_look_alikes_are_not(self):
        tags = self.t.cfg.tags
        ball = self.frames[f"{TRAIN_M}_mined_f0000010.jpg"]
        self.assertEqual(ball["boxes"], [{"class": BALL, "box": [40.0, 50.0, 50.0, 60.0]}])
        self.assertIn(tags.small_ball, ball["tags"])
        self.assertEqual(len(self.frames[f"{TRAIN_M}_mined_f0000015.jpg"]["boxes"]), 1)   # two candidates, one ball
        neg = self.frames[f"{TRAIN_M}_mined_f0000020.jpg"]
        self.assertEqual((neg["boxes"], neg["categories"]), ([], [CONE]))
        self.assertIn(tags.negative_prefix + CONE, neg["tags"])
        with_ball = self.frames[f"{TRAIN_M}_mined_f0000021.jpg"]        # the look-alike is unboxed, the real ball is boxed
        self.assertEqual([b["box"] for b in with_ball["boxes"]], [[40.0, 50.0, 50.0, 60.0]])
        for n, boxes in ((30, 1), (31, 0)):
            spare = self.frames[f"{VALID_M}_mined_f{n:07d}.jpg"]
            self.assertIn(tags.spare_ball, spare["tags"])
            self.assertEqual(len(spare["boxes"]), boxes)                # never a box on the spare ball itself
        self.assertEqual(self.m["counts"]["lookalike_frames_per_category"], {CONE: 2})

    def test_known_spare_ball_box_is_removed_by_its_stored_id(self):
        name = BASE[2]["name"]
        self.assertEqual(self.m["project_changes"], [
            {"op": "remove_box", "image_id": "img003", "name": name, "class": BALL, "box": [float(v) for v in SPARE_BOX]},
            {"op": "add_tag", "image_id": "img003", "name": name, "tag": self.t.cfg.tags.spare_ball}])
        copies = [c for c in self.m["scale_copies"] if c["source_image_id"] == "img003"]
        self.assertTrue(copies)
        for c in copies:                                               # copies follow the corrected labels
            self.assertEqual(sum(b["class"] == BALL for b in c["boxes"]), 1)
            self.assertIn(self.t.cfg.tags.spare_ball, c["tags"])
            self.assertEqual(c["split"], "valid")

    def test_scale_copies(self):
        sc, copies = self.t.cfg.scale_copies, self.m["scale_copies"]
        self.assertEqual({c["source_image_id"] for c in copies}, {"img001", "img003"})   # with a ball, train / valid
        f = sc.shrink_factors[0]
        for c in copies:
            img = cv2.imread(str(self.dest / c["file"]))
            self.assertEqual(img.shape[:2], (c["height"], c["width"]))
            self.assertEqual(sha((self.dest / c["file"]).read_bytes()), c["sha256"])
            src = next(b for b in BASE if b["id"] == c["source_image_id"])
            self.assertEqual(c["split"], src["split"])                 # a copy never changes split
            self.assertIn(self.t.cfg.tags.source_prefix + src["id"], c["tags"])
            (ball,) = [b["box"] for b in c["boxes"] if b["class"] == BALL]
            cx, cy = int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2)
            self.assertGreater(int(img[cy, cx].min()), 200)            # the box is still on the (white) ball
            side = ball[2] - ball[0]
            if c["kind"] == "shrink":
                self.assertEqual((c["width"], c["height"]), (W, H))
                self.assertAlmostEqual(side, 12 * round(W * f) / W, places=1)
                self.assertEqual(len(c["boxes"]), 3)
                self.assertEqual(int(img[0, 0].max()), sc.pad_value)
            else:
                self.assertEqual(c["width"], round(H * sc.tile_frac_height * sc.tile_enlarge[0]))
                self.assertAlmostEqual(side, 12 * sc.tile_enlarge[0], delta=0.2)
        kinds = [c["kind"] for c in copies]
        self.assertEqual(kinds.count("shrink"), 2)
        self.assertGreaterEqual(kinds.count("tile"), 2)

    def test_staged_version_and_frozen_review_are_never_overwritten(self):
        path = self.t.root / "review_v1.json"
        other = self.t.write("other.json", json.dumps({**json.loads(path.read_text()), "reviewer": "other"}).encode())
        with self.assertRaises(ds.Refused) as e:                       # same version, other content
            ds.build(other, self.t.ctx)
        self.assertEqual(e.exception.codes, {"FROZEN_EXISTS"})
        with self.assertRaises(ds.Refused) as e:                       # the same bytes: same freeze, but already staged
            ds.build(path, self.t.ctx)
        self.assertEqual(e.exception.codes, {"STAGE_EXISTS"})
        frozen = self.t.root / self.t.cfg.review_dir / "v1" / "review_frozen.json"
        frozen.write_bytes(frozen.read_bytes() + b" ")
        with self.assertRaises(ds.Refused) as e:
            ds.load_frozen(frozen.parent)
        self.assertEqual(e.exception.codes, {"FROZEN_CHANGED"})


class Refusals(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.t = Tmp(self._t.name)

    def refused(self, review: dict, *codes: str, t: Tmp = None) -> ds.Refused:
        t = t or self.t
        with self.assertRaises(ds.Refused) as e:
            t.build(review)
        self.assertEqual(e.exception.codes, set(codes), e.exception.lines())
        self.assertFalse(t.staged())                                   # a refusal stages nothing
        return e.exception

    def one(self, verdict=rv.MATCH_BALL, match=TRAIN_M, category=None, split="TRAIN", n=2, **fields) -> list:
        fields = fields or {"match_ball_box": [40, 50, 50, 60]}
        return [cand("x", match, verdict, [self.t.frame(match, 50 + i, **fields) for i in range(n)], category, split)]

    def test_match_not_a_train_match_on_the_allow_list(self):
        self.refused(self.t.review(self.one(match="UNLISTED_MATCH")), "NOT_ALLOW_LISTED")
        e = self.refused(self.t.review(self.one(match=DEV_M), version=2), "NOT_ALLOW_LISTED")   # a DEV match as TRAIN
        self.assertIn("registered for split 'dev'", str(e))

    def test_dev_and_eval_frames(self):
        for v, split in enumerate(("DEV", "EVAL"), 1):
            self.refused(self.t.review(self.one(split=split), version=v), "DEV_OR_EVAL")
        dev_copy = {"name": f"{TRAIN_M}_looks_like_train.jpg", "sha256": sha(self.t.dev_jpg),
                    "match_ball_box": [40, 50, 50, 60]}                # a DEV picture under a TRAIN name
        self.t.write(f"{self.t.cfg.frames_dir}/{dev_copy['name']}", self.t.dev_jpg)
        self.refused(self.t.review([cand("x", TRAIN_M, rv.MATCH_BALL, [dev_copy])], version=3), "DEV_OR_EVAL")
        dev_name = {"name": f"{DEV_M}_c00_f0000001.jpg"}               # even on a candidate that is never used
        self.refused(self.t.review(self.one() + [cand("u", TRAIN_M, rv.SKIP, [dev_name])], version=4), "DEV_OR_EVAL")

    def test_boxed_spare_ball(self):
        box = [40, 50, 50, 60]
        self.refused(self.t.review(self.one(rv.SPARE_BALL, object_box=box, match_ball_box=box)), "BOXED_SPARE_BALL")
        self.refused(self.t.review(self.one(rv.NOT_BALL, category=CONE, object_box=box, match_ball_box=box),
                                   version=2), "BOXED_LOOKALIKE")

    def test_two_ball_boxes_on_one_frame(self):
        f = self.t.frame(TRAIN_M, 60, match_ball_box=[40, 50, 50, 60])
        e = self.refused(self.t.review([cand("x", TRAIN_M, rv.MATCH_BALL, [f]),
                                        cand("y", TRAIN_M, rv.MATCH_BALL, [{**f, "match_ball_box": [200, 50, 210, 60]}])]),
                         "TWO_BALL_BOXES")
        self.assertIn(f["name"], str(e))

    def test_base_labels_keep_two_balls_until_the_spare_is_removed(self):
        e = self.refused(self.t.review(self.one(), existing=[]), "TWO_BALL_BOXES")
        self.assertIn(BASE[2]["name"], str(e))
        self.refused(self.t.review(self.one(), existing=[{"image_id": "img003", "box": [1, 2, 3, 4],
                                                          "verdict": rv.SPARE_BALL}], version=2),
                     "EXISTING_LABEL", "TWO_BALL_BOXES")

    def test_missing_frames_and_hashes(self):
        c = self.one()
        (self.t.root / self.t.cfg.frames_dir / c[0]["frames"][0]["name"]).unlink()
        self.refused(self.t.review(c), "MISSING_FRAME")
        c = self.one()
        c[0]["frames"][1]["sha256"] = None
        self.refused(self.t.review(c, version=2), "MISSING_HASH")
        c = self.one()
        self.t.write(f"{self.t.cfg.frames_dir}/{c[0]['frames'][0]['name']}", jpg(7))
        self.refused(self.t.review(c, version=3), "HASH_MISMATCH")
        (self.t.root / DCFG.paths.image_cache_dir / "img001.jpg").unlink()       # a base frame for the scale copies
        self.refused(self.t.review(self.one(), version=4), "MISSING_FRAME")

    def test_category_below_min_per_category_reports_the_shortfall(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tmp(d, min_per_category=dm.load_config().min_per_category)        # the real threshold
            need = t.ctx.min_per_category
            c = [cand("x", TRAIN_M, rv.NOT_BALL, [t.frame(TRAIN_M, 70 + i, object_box=[1, 1, 9, 9],
                                                          match_ball_absent=True) for i in range(3)], CONE)]
            e = self.refused(t.review(c), "CATEGORY_SHORTFALL", t=t)
            self.assertIn(f"{CONE!r}: 3 frames, {need - 3} short of min_per_category {need}", str(e))
        self.refused(self.t.review(self.one(rv.NOT_BALL, category=CONE, n=MIN - 1, object_box=[1, 1, 9, 9],
                                            match_ball_absent=True)), "CATEGORY_SHORTFALL")

    def test_ball_must_be_boxed_or_stated_absent(self):
        self.refused(self.t.review(self.one(match_ball_box=None)), "MATCH_BALL_NO_BOX")
        self.refused(self.t.review(self.one(rv.NOT_BALL, category=CONE, object_box=[1, 1, 9, 9]), version=2),
                     "BALL_UNACCOUNTED")
        self.refused(self.t.review(self.one(rv.SPARE_BALL, object_box=[1, 1, 9, 9]), version=3), "BALL_UNACCOUNTED")

    def test_other_refusals(self):
        neg = dict(object_box=[1, 1, 9, 9], match_ball_absent=True)
        for v, cat in enumerate(self.t.cfg.not_lookalike_categories + ("striker",), 1):
            self.refused(self.t.review(self.one(rv.NOT_BALL, category=cat, **neg), version=v), "CATEGORY")
        self.refused(self.t.review(self.one(match_ball_box=[300, 170, 330, 190]), version=10), "BOX")
        clash = {**self.t.frame(TRAIN_M, 80, match_ball_box=[40, 50, 50, 60]), "name": BASE[0]["name"]}
        self.refused(self.t.review([cand("x", TRAIN_M, rv.MATCH_BALL, [clash])], version=11), "FRAME_NAME",
                     "MISSING_FRAME")
        other = {**self.t.review(self.one(), version=12), "base_labels_sha256": "0" * 64}
        self.refused(other, "BASE_LABELS")
        self.refused(self.t.review([cand("u", TRAIN_M, rv.UNSURE, [{"name": f"{TRAIN_M}_a.jpg"}])], version=13), "EMPTY")

    def test_malformed_review_is_refused_and_not_frozen(self):
        bad = self.t.review(self.one())
        bad["candidates"][0]["verdict"] = "PROBABLY_BALL"
        self.refused(bad, "SCHEMA")
        self.assertFalse((self.t.root / self.t.cfg.review_dir).exists())
        for change in ({"schema_version": "9.9.9"}, {"review_version": 0}, {"existing_labels": None},
                       {"existing_labels": [{"image_id": "img003", "box": SPARE_BOX, "verdict": rv.UNSURE}]}):
            self.assertTrue(rv.validate_review({**self.t.review(self.one()), **change}), change)
        self.assertEqual(rv.validate_review(self.t.review(good(self.t))), [])


class Determinism(unittest.TestCase):
    def test_same_input_gives_byte_identical_output(self):
        trees = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                t = Tmp(d)
                dest = t.build(t.review(good(t)))
                trees.append({str(p.relative_to(dest)): p.read_bytes() for p in sorted(dest.rglob("*")) if p.is_file()})
                frozen = t.root / t.cfg.review_dir / "v1" / "review_frozen.json"
                trees[-1]["review_frozen.json"] = frozen.read_bytes()
        self.assertGreater(len(trees[0]), 10)
        self.assertEqual(sorted(trees[0]), sorted(trees[1]))
        self.assertTrue(all(trees[0][k] == trees[1][k] for k in trees[0]))

    def test_check_writes_nothing_and_the_command_line_reports_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tmp(d)
            path = t.write("r.json", json.dumps(t.review(good(t))).encode())
            before = sorted(p for p in t.root.rglob("*"))
            argv = ["--review", str(path), "--config", str(t.root / "v3.yaml"), "--mine-config",
                    str(t.root / "mine.yaml"), "--root", str(t.root)]
            self.assertEqual(ds.main(["check"] + argv), 0)
            self.assertEqual(sorted(p for p in t.root.rglob("*")), before)
            self.assertEqual(ds.main(["build"] + argv), 0)
            self.assertEqual(ds.main(["build"] + argv), 3)             # already staged


class RealConfigAndLabels(unittest.TestCase):
    """The shipped config against the real frozen labels of version 2 (no pixels: scale copies switched off)."""

    def test_config_points_at_the_frozen_labels(self):
        cfg = ds.load_config()
        self.assertEqual(sha((ROOT / cfg.base_labels).read_bytes()), cfg.base_labels_sha256)
        self.assertEqual(json.loads((ROOT / "records/roboflow_version/version_2.json").read_text())["labels_sha256"],
                         cfg.base_labels_sha256)
        self.assertTrue(all((ROOT / p).is_file() for p in cfg.dev_labels))
        self.assertEqual(cfg.tags.spare_ball, "has-spare-ball")       # DECISIONS 2026-11-24

    def test_version_2_needs_exactly_its_one_spare_ball_removed(self):
        cfg = ds.load_config()
        labels = json.loads((ROOT / cfg.base_labels).read_text())
        two = [f for f in labels["frames"] if sum(b["class"] == BALL for b in f["boxes"]) > 1]
        self.assertEqual(len(two), 1)
        balls = [b["box"] for b in two[0]["boxes"] if b["class"] == BALL]
        with tempfile.TemporaryDirectory() as d:
            t = Tmp(d, base_labels=cfg.base_labels, base_labels_sha256=cfg.base_labels_sha256,
                    dev_labels=list(cfg.dev_labels), scale_copies={**REAL["scale_copies"], "enabled": False})
            t.ctx = ds.load_context(t.root / "v3.yaml", ROOT / "config" / "roboflow.yaml", t.root / "mine.yaml",
                                    ROOT / "config" / "ball_fp_diag.yaml", t.root)
            for rel in (cfg.base_labels, *cfg.dev_labels):
                t.write(rel, (ROOT / rel).read_bytes())
            t.base_sha = cfg.base_labels_sha256
            c = [cand("x", TRAIN_M, rv.MATCH_BALL, [t.frame(TRAIN_M, 1, match_ball_box=[40, 50, 50, 60])])]
            with self.assertRaises(ds.Refused) as e:
                ds.preflight(t.review(c, existing=[]), t.ctx)
            self.assertEqual(e.exception.codes, {"TWO_BALL_BOXES"})
            self.assertIn(two[0]["name"], str(e.exception))
            plan = ds.preflight(t.review(c, existing=[{"image_id": two[0]["id"], "box": balls[1],
                                                       "verdict": rv.SPARE_BALL}]), t.ctx)
            self.assertEqual([x["op"] for x in plan["changes"]], ["remove_box", "add_tag"])
            self.assertEqual((plan["changes"][0]["image_id"], plan["base_frames"]), (two[0]["id"], len(labels["frames"])))


if __name__ == "__main__":
    unittest.main()
