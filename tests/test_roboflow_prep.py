"""Dataset preparation for detector v1: licence gate on every step, match-level split, pre-label class mapping and
per-class thresholds, version gate, read-only audit, pHash, REST client. No network: fake project / mocked session."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from spana.config import ConfigError
from spana.dataset.config import PolicyError, dataset_config_from_dict, match_of
from spana.dataset.phash import hamming, near_duplicate_groups, phash
from spana.dataset.prep import SkipFrame, evenly_spaced
from spana.dataset.rf_project import ProjectImage, RoboflowApiError, RoboflowProject, annotation_classes
from tools import roboflow_audit, roboflow_prelabel, roboflow_progress, roboflow_split, roboflow_version

REPO = Path(__file__).resolve().parent.parent
CFG_PATH = REPO / "config" / "roboflow.yaml"
SECRET = "rf_TEST_SECRET_never_leaks_456"


def raw():
    return yaml.safe_load(CFG_PATH.read_text())


def lm(mid, split="train", confirmed=True):
    return {"match_id": mid, "split": split, "permission": "test", "third_party_processing_confirmed": confirmed}


def make_cfg(mutate=None):
    d = raw()
    d["licensed_matches"] = [lm("M1"), lm("M2"), lm("M_UNCONF", confirmed=False), lm("M_EVAL", "eval")]
    if mutate:
        mutate(d)
    return dataset_config_from_dict(d)


def img(i, mid="M1", split="train", tags=(), in_dataset=False, annotated=False, classes=(), fi=None, sha="a" * 64):
    t = (f"match:{mid}", f"split:train") if mid else ()
    return ProjectImage(id=f"id{i:03d}", name=f"{mid}_f{i:07d}.jpg", split=split, in_dataset=in_dataset,
                        tags=tuple(sorted(set(t) | set(tags))),
                        metadata={"match_id": mid, "frame_index": i if fi is None else fi, "image_sha256": sha},
                        width=64, height=36, url=f"https://img.invalid/{i}.jpg", classes=tuple(classes),
                        annotated=annotated)


class FakeProject:
    def __init__(self, images=()):
        self.images = list(images)
        self.tag_calls, self.ann_calls, self.gen_calls = [], [], []
        self.annotate_result = "saved"

    def list_images(self, page_size):
        return list(self.images)

    def update_tags(self, image_id, add=(), remove=()):
        self.tag_calls.append((image_id, tuple(add), tuple(remove)))

    def save_annotation(self, image_id, *, name, content, labelmap, add_to_dataset):
        self.ann_calls.append((image_id, name, content, labelmap, add_to_dataset))
        return self.annotate_result

    def generate_version(self, settings):
        self.gen_calls.append(settings)
        return {"version": 1}


class ConfigTests(unittest.TestCase):
    def test_default_sections_load(self):
        c = make_cfg()
        self.assertEqual(c.prelabel_class_map(), {"person": "player", "sports ball": "ball"})
        thr = c.prelabel_thresholds()
        self.assertEqual(set(thr), {"player", "ball"})
        self.assertGreater(thr["ball"], thr["player"])          # a missed ball beats a wrong ball box
        self.assertNotIn("goalkeeper", c.prelabel_class_map().values())
        self.assertNotIn("referee", c.prelabel_class_map().values())
        self.assertEqual(c.version.augmentations, ())

    def test_fail_closed(self):
        bad = [
            lambda d: d["prelabel"].update(class_thresholds=["player:0.5"]),                 # ball has no threshold
            lambda d: d["prelabel"].update(class_thresholds=["player:0.5", "ball:0.9", "referee:0.5"]),
            lambda d: d["prelabel"].update(class_thresholds=["player:0.5", "ball:1.5"]),
            lambda d: d["prelabel"].update(request_confidence=0.6),                         # above player threshold
            lambda d: d["prelabel"].update(class_map=["person:player", "person:ball"]),
            lambda d: d["prelabel"].update(class_map=["person:spectator"]),
            lambda d: d["prelabel"].update(model_id="rfdetr-base"),
            lambda d: d["version"].update(augmentations=["hue"]),
            lambda d: d["version"].update(augmentations=["brightness"]),
            lambda d: d["version"].update(resize_format="Squash"),
            lambda d: d["splits"].update(assignable=["train", "valid", "test"]),
            lambda d: d["splits"].update(move_method="delete"),
            lambda d: d["splits"].update(probe_max=0),
            lambda d: d["review"].update(reviewed_tag="prelabel:v1"),
            lambda d: d["audit"].update(phash_hamming_max=65),
        ]
        for m in bad:
            with self.assertRaises(ConfigError):
                make_cfg(m)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def test_allow_list_enforced(self):
        self.assertIsNotNone(match_of(self.cfg, img(1).tags, img(1).metadata)[0])
        cases = {
            "not on the licensed_matches allow-list": img(1, mid="M_OTHER"),
            "no match": img(1, mid=None),
            "third-party": img(1, mid="M_UNCONF"),
            "EVAL": img(1, mid="M_EVAL"),
        }
        for want, i in cases.items():
            entry, reason = match_of(self.cfg, i.tags, i.metadata)
            self.assertIsNone(entry)
            self.assertIn(want.split()[0], reason)
        i = img(1, tags=("split:eval",))
        self.assertIn("eval", match_of(self.cfg, i.tags, i.metadata)[1])
        i = img(1, tags=("match:M2",))
        self.assertIn("several match tags", match_of(self.cfg, i.tags, i.metadata)[1])
        i = img(1)
        self.assertIn("disagrees", match_of(self.cfg, i.tags, dict(i.metadata, match_id="M2"))[1])


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.images = ([img(i, "M1") for i in range(5)] + [img(10 + i, "M2") for i in range(4)]
                       + [img(20, "M_OTHER"), img(21, mid=None)])

    def test_match_level_assignment(self):
        plan = roboflow_split.plan_split(self.cfg, self.images, {"M2": "valid"}, exclude_ids=["id001"])
        self.assertEqual(plan["match_splits_after"], {"M1": "train", "M2": "valid"})
        per_match = {}
        for r in plan["rows"]:
            per_match.setdefault(r["match_id"], set()).add(r["after"])
        self.assertTrue(all(len(s) == 1 for s in per_match.values()))      # no match in two splits
        self.assertEqual(plan["n_moves"], 4)
        self.assertEqual(plan["n_exclude"], 1)
        self.assertEqual({s["reason"].split()[0] for s in plan["skipped"]}, {"match", "no"})   # gate failures listed

    def test_refusals(self):
        with self.assertRaisesRegex(PolicyError, "EVAL"):
            roboflow_split.plan_split(self.cfg, self.images + [img(30, "M_EVAL")], {"M_EVAL": "valid"})
        tagged = self.images + [img(31, "M2", tags=("split:eval",))]
        with self.assertRaisesRegex(PolicyError, "split:eval"):
            roboflow_split.plan_split(self.cfg, tagged, {"M2": "valid"})
        with self.assertRaisesRegex(PolicyError, "allow-list"):
            roboflow_split.plan_split(self.cfg, self.images, {"M_OTHER": "valid"})
        with self.assertRaisesRegex(PolicyError, "assignable"):
            roboflow_split.plan_split(self.cfg, self.images, {"M2": "test"})
        with self.assertRaisesRegex(PolicyError, "no frames"):
            roboflow_split.plan_split(self.cfg, self.images, {"M_UNCONF": "valid"})
        with self.assertRaisesRegex(PolicyError, "empty"):
            roboflow_split.plan_split(self.cfg, self.images, {"M2": "valid"},
                                      exclude_ids=[f"id{10 + i:03d}" for i in range(4)])
        with self.assertRaisesRegex(PolicyError, "twice"):
            roboflow_split.parse_assign(["valid=M2", "train=M2"])

    def test_apply_refuses_moves_without_changing_anything(self):
        proj = FakeProject(self.images)
        plan = roboflow_split.plan_split(self.cfg, self.images, {"M2": "valid"}, exclude_ids=["id001"])
        with self.assertRaisesRegex(PolicyError, "Nothing was changed"):
            roboflow_split.apply_plan(self.cfg, proj, plan, exclusions_only=False)
        self.assertEqual(proj.tag_calls, [])
        log = roboflow_split.apply_plan(self.cfg, proj, plan, exclusions_only=True)
        self.assertEqual(proj.tag_calls, [("id001", ("excluded:near-duplicate",), ())])
        self.assertEqual(len(log), 1)


class FakeUploader:
    """Mimics Roboflow re-upload of identical bytes: same id back; `honour_split` decides whether the split changes."""

    def __init__(self, project, honour_split=True, new_id=False):
        self.project, self.honour, self.new_id, self.calls = project, honour_split, new_id, []

    def single_upload(self, *, image_path, split, batch_name, tag_names, metadata, num_retry_uploads):
        self.calls.append((Path(image_path).name, split, tuple(tag_names), metadata))
        name = Path(image_path).name
        idx = next(k for k, i in enumerate(self.project.images) if i.name == name)
        old = self.project.images[idx]
        if self.new_id:
            return {"image": {"id": "NEW", "duplicate": False}}
        if self.honour:
            self.project.images[idx] = ProjectImage(**{**old.__dict__, "split": split})
        return {"image": {"id": old.id, "duplicate": True}}


class MoveTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.cache = Path(self.td.name)
        self.images = [img(i, "M1") for i in range(3)] + [img(10 + i, "M2") for i in range(2)]
        for i in self.images:                                   # verified cache: bytes whose sha is recorded
            (self.cache / f"{i.id}.jpg").write_bytes(b"jpeg-bytes")
        import hashlib
        sha = hashlib.sha256(b"jpeg-bytes").hexdigest()
        self.images = [ProjectImage(**{**i.__dict__, "metadata": dict(i.metadata, image_sha256=sha)})
                       for i in self.images]

    def tearDown(self):
        self.td.cleanup()

    def _plan(self, cfg):
        return roboflow_split.plan_split(cfg, self.images, {"M1": "valid"})

    def test_probe_moves_one_frame_and_verifies(self):
        cfg = make_cfg()
        proj = FakeProject(self.images)
        up = FakeUploader(proj)
        log = roboflow_split.apply_plan(cfg, proj, self._plan(cfg), exclusions_only=False, uploader=up,
                                        images=self.images, cache_dir=self.cache, probe=1)
        self.assertEqual(len(up.calls), 1)
        name, split, tags, meta = up.calls[0]
        self.assertEqual((name, split), ("M1_f0000000.jpg", "valid"))
        self.assertEqual(tags, self.images[0].tags)                  # same tags and metadata as the original
        self.assertEqual(meta, self.images[0].metadata)
        self.assertEqual(log[0]["split_after"], "valid")
        self.assertEqual(log[0]["action"], "probe_move")
        with self.assertRaisesRegex(PolicyError, "probe_max"):
            roboflow_split.apply_plan(cfg, proj, self._plan(cfg), exclusions_only=False, uploader=up,
                                      images=self.images, cache_dir=self.cache, probe=2)

    def test_split_not_honoured_or_new_id_fails_closed(self):
        cfg = make_cfg()
        proj = FakeProject(self.images)
        with self.assertRaisesRegex(PolicyError, "did not move"):
            roboflow_split.apply_plan(cfg, proj, self._plan(cfg), exclusions_only=False,
                                      uploader=FakeUploader(proj, honour_split=False), images=self.images,
                                      cache_dir=self.cache, probe=1)
        with self.assertRaisesRegex(PolicyError, "returned id"):
            roboflow_split.apply_plan(cfg, proj, self._plan(cfg), exclusions_only=False,
                                      uploader=FakeUploader(proj, new_id=True), images=self.images,
                                      cache_dir=self.cache, probe=1)

    def test_full_apply_needs_reupload_method_and_keeps_matches_whole(self):
        cfg = make_cfg()
        proj = FakeProject(self.images)
        with self.assertRaisesRegex(PolicyError, "move_method: none"):
            roboflow_split.apply_plan(cfg, proj, self._plan(cfg), exclusions_only=False, uploader=FakeUploader(proj),
                                      images=self.images, cache_dir=self.cache)
        cfg = make_cfg(lambda d: d["splits"].update(move_method="reupload"))
        log = roboflow_split.apply_plan(cfg, proj, self._plan(cfg), exclusions_only=False,
                                        uploader=FakeUploader(proj), images=self.images, cache_dir=self.cache)
        self.assertEqual(len(log), 3)
        splits = {}
        for i in proj.images:
            splits.setdefault(i.metadata["match_id"], set()).add(i.split)
        self.assertEqual(splits, {"M1": {"valid"}, "M2": {"train"}})

    def test_unverifiable_bytes_refused(self):
        cfg = make_cfg()
        (self.cache / f"{self.images[0].id}.jpg").write_bytes(b"tampered")
        proj = FakeProject(self.images)
        proj.fetch_image = mock.Mock(side_effect=RoboflowApiError("image bytes do not match the sha256"))
        up = FakeUploader(proj)
        with self.assertRaisesRegex(PolicyError, "cannot verify"):
            roboflow_split.apply_plan(cfg, proj, self._plan(cfg), exclusions_only=False, uploader=up,
                                      images=self.images, cache_dir=self.cache, probe=1)
        self.assertEqual(up.calls, [])


def resp(preds, w=64, h=36):
    return {"image": {"width": w, "height": h}, "predictions": preds}


def p(x, y, w, h, c, label):
    return {"x": x, "y": y, "width": w, "height": h, "confidence": c, "class": label}


class PrelabelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.thr = self.cfg.prelabel_thresholds()

    def test_class_mapping_and_per_class_thresholds(self):
        preds = [p(10, 10, 4, 10, self.thr["player"] + 0.01, "person"),
                 p(20, 10, 4, 10, self.thr["player"] - 0.01, "person"),       # below player threshold
                 p(30, 30, 2, 2, self.thr["ball"] + 0.01, "sports ball"),
                 p(40, 30, 2, 2, self.thr["ball"] - 0.01, "sports ball"),      # above player thr, below ball thr
                 p(50, 10, 8, 4, 0.99, "car")]                                 # unmapped
        boxes, dropped, below = roboflow_prelabel.map_predictions(self.cfg, resp(preds), 64, 36)
        self.assertEqual([b["class"] for b in boxes], ["player", "ball"])
        self.assertEqual(dict(dropped), {"car": 1})
        self.assertEqual(dict(below), {"player": 1, "ball": 1})
        self.assertEqual(boxes[0]["class_id"], self.cfg.classes.index("player"))
        self.assertEqual(boxes[1]["class_id"], self.cfg.classes.index("ball"))
        self.assertEqual(boxes[0]["box"], [8.0, 5.0, 12.0, 15.0])
        txt, labelmap = roboflow_prelabel.yolo_annotation(self.cfg, boxes, 64, 36)
        self.assertEqual(labelmap, {str(i): c for i, c in enumerate(self.cfg.classes)})
        first = txt.splitlines()[0].split()
        self.assertEqual(first[0], str(self.cfg.classes.index("player")))
        np.testing.assert_allclose([float(v) for v in first[1:]], [10 / 64, 10 / 36, 4 / 64, 10 / 36], atol=1e-6)
        with self.assertRaises(Exception):
            roboflow_prelabel.map_predictions(self.cfg, resp(preds, w=128), 64, 36)

    def _run(self, images, **kw):
        proj = FakeProject(images)
        frame = np.zeros((36, 64, 3), np.uint8)
        infer = mock.Mock(return_value=resp([p(10, 10, 4, 10, 0.9, "person"), p(30, 30, 2, 2, 0.95, "sports ball")]))
        args = dict(limit=20, match=None, confirm_full_run=False, load=lambda i: frame, render_dir=None)
        args.update(kw)
        doc = roboflow_prelabel.run_prelabel(self.cfg, proj, infer, images, **args)
        return doc, proj, infer

    def test_gate_skips_and_logging(self):
        images = [img(1), img(2, mid="M_OTHER"), img(3, tags=("excluded:near-duplicate",)),
                  img(4, annotated=True), img(5, tags=("prelabel:v1",)), img(6, mid="M_EVAL"), img(7)]
        doc, proj, infer = self._run(images)
        self.assertEqual([e["id"] for e in doc["entries"]], ["id001", "id007"])
        self.assertEqual(infer.call_count, 2)
        reasons = {s["id"]: s["reason"] for s in doc["not_selected_skips"]}
        self.assertIn("allow-list", reasons["id002"])
        self.assertIn("excluded", reasons["id003"])
        self.assertIn("already annotated", reasons["id004"])
        self.assertIn("already pre-labelled", reasons["id005"])
        self.assertIn("EVAL", reasons["id006"])
        for e in doc["entries"]:
            self.assertEqual(e["status"], "labelled")
            self.assertEqual(e["boxes_per_class"], {"player": 1, "ball": 1})
        self.assertEqual(doc["summary"]["boxes_per_class"], {"ball": 2, "player": 2})
        # tag before annotation, never overwrite, not added to the Dataset before review
        self.assertEqual([c[0] for c in proj.tag_calls], ["id001", "id007"])
        self.assertTrue(all(c[1] == ("prelabel:v1",) for c in proj.tag_calls))
        self.assertTrue(all(c[4] is False for c in proj.ann_calls))
        self.assertTrue(all(set(c[3].values()) == set(self.cfg.classes) for c in proj.ann_calls))

    def test_sample_limits_and_full_run_confirmation(self):
        images = [img(i) for i in range(60)]
        doc, proj, _ = self._run(images)
        self.assertEqual(doc["summary"]["selected"], self.cfg.prelabel.sample_size)
        idx = [e["frame_index"] for e in doc["entries"]]
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 59)
        self.assertEqual(idx, sorted(idx))
        with self.assertRaisesRegex(PolicyError, "sample_size"):
            self._run(images, limit=self.cfg.prelabel.sample_size + 1)
        with self.assertRaisesRegex(PolicyError, "confirm-full-run"):
            self._run(images, limit=None)
        doc, _, _ = self._run(images, limit=None, confirm_full_run=True)
        self.assertEqual(doc["summary"]["selected"], 60)

    def test_skipped_frames_and_server_conflict(self):
        def load(i):
            if i.id == "id001":
                raise SkipFrame("image bytes do not match the sha256 recorded at upload")
            return np.zeros((36, 64, 3), np.uint8)
        doc, proj, infer = self._run([img(1), img(2)], load=load)
        st = {e["id"]: (e["status"], e["reason"]) for e in doc["entries"]}
        self.assertEqual(st["id001"][0], "skipped")
        self.assertIn("sha256", st["id001"][1])
        self.assertEqual(st["id002"][0], "labelled")
        proj = FakeProject()
        proj.annotate_result = "already_annotated"
        doc = roboflow_prelabel.run_prelabel(self.cfg, proj, mock.Mock(return_value=resp([p(10, 10, 4, 10, 0.9, "person")])),
                                             [img(1)], limit=1, match=None, confirm_full_run=False,
                                             load=lambda i: np.zeros((36, 64, 3), np.uint8), render_dir=None)
        self.assertEqual(doc["entries"][0]["status"], "skipped")

    def test_match_filter(self):
        doc, _, _ = self._run([img(1, "M1"), img(2, "M2")], match="M2")
        self.assertEqual([e["match_id"] for e in doc["entries"]], ["M2"])


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def ready_images(self):
        rev = ("reviewed", "prelabel:v1")
        return ([img(i, "M1", "train", rev, in_dataset=True, annotated=True) for i in range(3)]
                + [img(10 + i, "M2", "valid", rev, in_dataset=True, annotated=True) for i in range(2)])

    def test_ready_and_settings(self):
        out = roboflow_version.check_ready(self.cfg, self.ready_images())
        self.assertEqual(out["splits"], {"train": 3, "valid": 2})
        s = roboflow_version.version_settings(self.cfg)
        self.assertEqual(s["preprocessing"]["resize"], {"width": self.cfg.version.resize_width,
                                                        "height": self.cfg.version.resize_height,
                                                        "format": self.cfg.version.resize_format})
        self.assertEqual(s["preprocessing"]["filter-tags"], {self.cfg.splits.excluded_tag: False})
        colour = {"hue", "saturation", "brightness", "exposure", "rgrayscale", "camera-gain", "noise", "blur"}
        self.assertFalse(colour & set(s["augmentation"]))
        self.assertNotIn("grayscale", s["preprocessing"])
        self.assertNotIn("contrast", s["preprocessing"])

    def test_refusals(self):
        imgs = self.ready_images()
        unrev = copy.copy(imgs)
        unrev[0] = img(0, "M1", "train", ("prelabel:v1",), in_dataset=True, annotated=True)
        with self.assertRaisesRegex(PolicyError, "not reviewed"):
            roboflow_version.check_ready(self.cfg, unrev)
        unrev2 = imgs + [img(50, "M1", "train", ("prelabel:v1",), in_dataset=False)]   # not even in the Dataset
        with self.assertRaisesRegex(PolicyError, "not reviewed"):
            roboflow_version.check_ready(self.cfg, unrev2)
        no_valid = [i for i in imgs if i.split != "valid"]
        with self.assertRaisesRegex(PolicyError, "valid split of the version would be empty"):
            roboflow_version.check_ready(self.cfg, no_valid)
        mixed = imgs + [img(60, "M2", "train", ("reviewed",), in_dataset=True)]
        with self.assertRaisesRegex(PolicyError, "several splits"):
            roboflow_version.check_ready(self.cfg, mixed)
        gate = imgs + [img(70, "M_OTHER", "train", ("reviewed",), in_dataset=True)]
        with self.assertRaisesRegex(PolicyError, "licence gate"):
            roboflow_version.check_ready(self.cfg, gate)


class AuditProgressTests(unittest.TestCase):
    def test_audit_is_read_only_and_counts(self):
        cfg = make_cfg()
        rng = np.random.default_rng(1)
        base = rng.integers(0, 255, (36, 64, 3), dtype=np.uint8)
        other = rng.integers(0, 255, (36, 64, 3), dtype=np.uint8)
        frames = {"id001": base, "id002": base.copy(), "id003": other}
        images = [img(1, "M1"), img(2, "M1"), img(3, "M2", split="valid"), img(4, "M2"), img(5, mid=None)]

        def load(i):
            if i.id not in frames:
                raise SkipFrame("no image_sha256 recorded at upload")
            return frames[i.id]
        doc = roboflow_audit.audit(cfg, images, load=load)
        self.assertEqual(doc["matches"]["M1"]["n"], 2)
        self.assertEqual(doc["matches"]["M2"]["splits"], {"train": 1, "valid": 1})
        self.assertEqual(doc["matches_in_several_splits"], ["M2"])
        self.assertEqual([u["id"] for u in doc["untagged"]], ["id005"])
        groups = doc["near_duplicates"]["groups"]
        self.assertEqual([(g["keep"], g["exclude"]) for g in groups], [("id001", ["id002"])])
        self.assertEqual(len(doc["near_duplicates"]["unverified"]), 2)
        self.assertIn("matches in several splits: ['M2']", roboflow_audit.summary_text(doc))
        proj = FakeProject(images)            # the audit function takes no project: nothing can be written
        self.assertEqual((proj.tag_calls, proj.ann_calls, proj.gen_calls), ([], [], []))

    def test_progress(self):
        cfg = make_cfg()
        images = [img(1, tags=("prelabel:v1", "reviewed"), annotated=True, classes=("player", "player", "ball")),
                  img(2, tags=("prelabel:v1",), annotated=True, classes=("player",)),
                  img(3, tags=("excluded:near-duplicate",)), img(4, "M2", "valid")]
        doc = roboflow_progress.progress(cfg, images)
        m1 = doc["matches"]["M1"]
        self.assertEqual((m1["frames"], m1["prelabelled"], m1["reviewed"], m1["prelabelled_not_reviewed"],
                          m1["excluded"]), (3, 2, 1, 1, 1))
        self.assertEqual(m1["boxes_annotated"], {"player": 3, "goalkeeper": 0, "referee": 0, "ball": 1})
        self.assertEqual(m1["boxes_reviewed"], {"player": 2, "goalkeeper": 0, "referee": 0, "ball": 1})
        self.assertIn("M1", roboflow_progress.table(doc))


class PhashTests(unittest.TestCase):
    def test_hash_and_groups(self):
        rng = np.random.default_rng(0)
        a = cv2.resize(rng.integers(0, 255, (9, 16, 3), dtype=np.uint8), (320, 180), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(rng.integers(0, 255, (9, 16, 3), dtype=np.uint8), (320, 180), interpolation=cv2.INTER_LINEAR)
        noisy = np.clip(a.astype(int) + rng.integers(-3, 4, a.shape), 0, 255).astype(np.uint8)
        ha, hb, hn = phash(a), phash(b), phash(noisy)
        self.assertLessEqual(hamming(ha, hn), 4)
        self.assertGreater(hamming(ha, hb), 10)
        items = [("x3", (2,), hb), ("x2", (1,), hn), ("x1", (0,), ha)]
        g1 = near_duplicate_groups(items, 4)
        g2 = near_duplicate_groups(list(reversed(items)), 4)
        self.assertEqual(g1, g2)
        self.assertEqual([(g["keep"], g["exclude"]) for g in g1], [("x1", ["x2"])])
        self.assertEqual(evenly_spaced(list(range(10)), 3), [0, 4, 9])   # round-half-to-even, deterministic


class ClientTests(unittest.TestCase):
    def _proj(self, session):
        return RoboflowProject(api_url="https://api.invalid", workspace="ws", project="proj", api_key=SECRET,
                               timeout_s=5, session=session)

    def test_list_images_paging_and_parsing(self):
        pages = [{"results": [{"id": "b", "name": "b.jpg", "tags": ["match:M1"], "projectData": {"proj": {"split": "valid",
                  "inDataset": True}}, "annotations": {"proj": {"boxes": [{"label": "player"}, {"label": "ball"}]}}}],
                  "total": 2, "continuationToken": "t1"},
                 {"results": [{"id": "a", "name": "a.jpg", "tags": [], "split": "train"}], "total": 2}]
        session = mock.Mock()
        session.request.side_effect = [mock.Mock(status_code=200, json=mock.Mock(return_value=pg)) for pg in pages]
        out = self._proj(session).list_images(250)
        self.assertEqual([i.id for i in out], ["a", "b"])
        self.assertEqual((out[1].split, out[1].in_dataset, out[1].classes, out[1].annotated),
                         ("valid", True, ("player", "ball"), True))
        self.assertEqual((out[0].split, out[0].annotated), ("train", False))
        second_body = session.request.call_args_list[1][1]["json"]
        self.assertEqual(second_body["continuationToken"], "t1")
        self.assertEqual(second_body["query"], "project:proj")

    def test_annotation_shapes(self):
        self.assertEqual(annotation_classes(None), (False, ()))
        self.assertEqual(annotation_classes([{"class": "ball"}]), (True, ("ball",)))
        self.assertEqual(annotation_classes({"boxes": [{"label": "player"}]}), (True, ("player",)))

    def test_save_never_overwrites_and_errors_are_redacted(self):
        session = mock.Mock()
        session.request.return_value = mock.Mock(status_code=409, text='{"error":{"message":"Image was already annotated"}}')
        pr = self._proj(session)
        self.assertEqual(pr.save_annotation("i1", name="a.txt", content="0 0.5 0.5 0.1 0.1\n", labelmap={"0": "player"},
                                            add_to_dataset=False), "already_annotated")
        args, kw = session.request.call_args
        self.assertEqual(args, ("POST", "https://api.invalid/dataset/proj/annotate/i1"))
        self.assertNotIn("overwrite", kw["params"])
        self.assertEqual(kw["params"]["addToDataset"], "false")
        self.assertEqual(json.loads(kw["data"])["labelmap"], {"0": "player"})
        session.request.return_value = mock.Mock(status_code=500, text=f"boom {SECRET}")
        with self.assertRaises(RoboflowApiError) as cm:
            pr.update_tags("i1", add=["x"])
        self.assertNotIn(SECRET, str(cm.exception))
        session.request.return_value = mock.Mock(status_code=200, content=b"bytes")
        with self.assertRaisesRegex(RoboflowApiError, "sha256"):
            pr.fetch_image("https://img.invalid/1.jpg", "0" * 64)


if __name__ == "__main__":
    unittest.main()
