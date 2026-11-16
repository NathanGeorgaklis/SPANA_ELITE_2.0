"""DEV benchmark freeze + score (tools/dev_bench_score.py). No network: fake project, synthetic masks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spana.dataset import det_metrics as dm
from spana.dataset.config import PolicyError
from spana.dataset.rf_project import ProjectImage
from tests.test_roboflow_prep import lm, make_cfg
from tools import dev_bench_score as ds


def cfg():
    return make_cfg(lambda d: d["licensed_matches"].extend([lm("M_DEV", "dev"), lm("M_DEV2", "dev")]))


def bench_img(i, mid="M_DEV", reviewed=True, in_dataset=True, extra=()):
    tags = {f"match:{mid}", "split:dev", "bench:v1", "prelabel:detector-v1", *extra}
    if reviewed:
        tags.add("reviewed")
    return ProjectImage(id=f"b{i:03d}", name=f"{mid}_c00_f{i:07d}.jpg", split="valid", in_dataset=in_dataset,
                        tags=tuple(sorted(tags)), metadata={"match_id": mid, "frame_index": i, "image_sha256": "a" * 64},
                        width=64, height=36, url=None)


class FakeBench:
    def __init__(self, images, boxes=None):
        self.images, self.boxes = list(images), boxes or {}

    def list_images(self, page_size, attempts=1, wait_s=0.0):
        return list(self.images)

    def get_image(self, image_id):
        i = next(x for x in self.images if x.id == image_id)
        bx = self.boxes.get(image_id, [{"label": "player", "x": 10, "y": 10, "width": 4, "height": 8}])
        return {"id": image_id, "tags": list(i.tags), "annotation": {"boxes": bx}}


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_freeze_reviewed_frames_only_and_never_twice(self):
        c = cfg()
        proj = FakeBench([bench_img(1), bench_img(2, "M_DEV2"), bench_img(3, reviewed=False, in_dataset=False)])
        out = ds.freeze(c, proj, expect=2, root=self.root)
        frozen = json.loads((out / "FROZEN.json").read_text())
        self.assertEqual(frozen["n_frames"], 2)
        self.assertEqual(frozen["frames_per_match"], {"M_DEV": 1, "M_DEV2": 1})
        self.assertEqual(frozen["bench_frames_not_reviewed"], ["M_DEV_c00_f0000003.jpg"])
        labels, _ = ds.load_frozen(c, self.root)
        self.assertEqual(labels["frames"][0]["boxes"], [{"class": "player", "box": [8.0, 6.0, 12.0, 14.0]}])
        with self.assertRaisesRegex(PolicyError, "never re-frozen"):
            ds.freeze(c, proj, expect=2, root=self.root)

    def test_freeze_refusals(self):
        c = cfg()
        with self.assertRaisesRegex(PolicyError, "--expect 3"):
            ds.freeze(c, FakeBench([bench_img(1)]), expect=3, root=self.root)
        with self.assertRaisesRegex(PolicyError, "not in the project's Dataset"):
            ds.freeze(c, FakeBench([bench_img(1, in_dataset=False)]), expect=1, root=self.root)
        with self.assertRaisesRegex(PolicyError, "disagree with the allow-list split 'train'"):
            ds.freeze(c, FakeBench([bench_img(1, mid="M1")]), expect=1, root=self.root)   # a TRAIN match
        with self.assertRaisesRegex(PolicyError, "not configured classes"):
            ds.freeze(c, FakeBench([bench_img(1)], {"b001": [{"label": "coach", "x": 1, "y": 1, "width": 1,
                                                              "height": 1}]}), expect=1, root=self.root)
        self.assertFalse(ds.labels_dir(c, self.root).exists())             # nothing written by a refused freeze

    def test_tampered_labels_refused(self):
        c = cfg()
        out = ds.freeze(c, FakeBench([bench_img(1)]), expect=1, root=self.root)
        doc = json.loads((out / "labels.json").read_text())
        doc["frames"][0]["boxes"] = []
        (out / "labels.json").write_text(json.dumps(doc))
        with self.assertRaisesRegex(PolicyError, "does not match its FROZEN.json"):
            ds.load_frozen(c, self.root)


class OffPitchTests(unittest.TestCase):
    def test_background_predictions_split_by_grass_mask(self):
        c = cfg()
        mask = np.zeros((36, 64), bool)
        mask[10:30, :] = True                                              # grass band rows 10..29
        f = {"id": "f", "match_id": "M_DEV",
             "gt": [{"class": "player", "box": [0, 10, 4, 20]}],
             "pred": [{"class": "player", "box": [0, 10, 4, 20], "score": 0.9},       # true positive
                      {"class": "player", "box": [20, 0, 24, 8], "score": 0.9},       # feet above the grass: off
                      {"class": "player", "box": [30, 10, 34, 20], "score": 0.9},     # on the grass
                      {"class": "ball", "box": [40, 30, 42, 36], "score": 0.9},       # cut by the bottom border
                      {"class": "referee", "box": [50, 0, 54, 8], "score": 0.1}]}     # below detector.confidence
        out = ds.off_pitch(c, [f], {"f": mask}, {"f": 36})
        self.assertEqual(out["player"], {"false_positives": 2, "off_pitch": 1, "on_pitch": 1, "foot_unobserved": 0})
        self.assertEqual(out["ball"], {"false_positives": 1, "off_pitch": 0, "on_pitch": 0, "foot_unobserved": 1})
        self.assertEqual(out["referee"]["false_positives"], 0)
        op = dm.at_confidence([f], c.detector.confidence)
        bg = dm.confusion(op, c.classes, c.detector_eval.match_iou)[dm.BACKGROUND]
        self.assertEqual({k: v["false_positives"] for k, v in out.items()}, {k: bg[k] for k in c.classes})

    def test_scores_split_per_match(self):
        c = cfg()
        fr = lambda i, m: {"id": i, "match_id": m, "gt": [{"class": "player", "box": [0, 10, 4, 20]}],
                           "pred": [{"class": "player", "box": [0, 10, 4, 20], "score": 0.9}]}
        frames = [fr("a", "M_DEV"), fr("b", "M_DEV2"), {**fr("c", "M_DEV2"), "pred": []}]
        masks = {k: np.ones((36, 64), bool) for k in "abc"}
        res = ds.score_frames(c, frames, masks, {k: 36 for k in "abc"})
        self.assertEqual(res["overall"]["per_class"]["player"]["recall"], round(2 / 3, 4))
        self.assertEqual(res["per_match"]["M_DEV"]["per_class"]["player"]["recall"], 1.0)
        self.assertEqual(res["per_match"]["M_DEV2"]["per_class"]["player"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
