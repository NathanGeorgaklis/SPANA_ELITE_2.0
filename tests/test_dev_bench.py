"""DEV generalisation benchmark tool (tools/dev_bench.py): selection rules, licence gate, separation from training,
upload + pre-label through the write guard. No network: uploader, project and detector are fakes."""
from __future__ import annotations

import json
import tempfile

import cv2
import numpy as np
import unittest
from datetime import datetime, timezone
from pathlib import Path

from spana.config import ConfigError
from spana.dataset.config import PolicyError
from spana.dataset.rf_project import ProjectImage
from spana.records import Detection
from tests.test_roboflow_prep import FakeProject, img, lm, make_cfg
from tools import dev_bench, roboflow_version
from tools.dev_bench import Candidate, run_bench, select_bench

def write_video(path: Path, seconds: float, seed: int, fps: float = 25.0) -> None:
    """A new random scene every second (so sampled frames are not near-duplicates)."""
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 90))
    scene = None
    for i in range(int(fps * seconds)):
        if i % int(fps) == 0:
            scene = cv2.resize(rng.integers(0, 255, (9, 16, 3), dtype=np.uint8), (160, 90),
                               interpolation=cv2.INTER_NEAREST)
        vw.write(scene)
    vw.release()


NOW = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)


def bench_cfg(**bench):
    def mutate(d):
        d["licensed_matches"] += [lm("M_DEV", "dev"), lm("M_DEV_UNCONF", "dev", confirmed=False)]
        d["detector"].update(model_id="proj/2")
        d["dev_bench"].update(min_interval_s=0.0, **bench)
    return make_cfg(mutate)


def cand(clip, fi, blur=500.0, h=None):
    if h is None:   # an independent random 64-bit hash: about 32 bits from any other
        h = int(np.random.default_rng(1000 * clip + fi + 7).integers(0, 2 ** 63)) | (1 << 63)
    return Candidate(clip, fi, fi / 25.0, ("interval",), f"M_DEV_c{clip:02d}_f{fi:07d}.jpg", "b" * 64, blur, h)


class SelectionTests(unittest.TestCase):
    def test_blur_and_near_duplicates_dropped_then_evenly_spread_over_the_match(self):
        cfg = bench_cfg(frames_per_match=4)
        cs = [cand(0, 0), cand(0, 50, blur=1.0), cand(0, 100, h=0), cand(0, 150, h=1),   # 150 is 1 bit from 100
              cand(1, 0), cand(1, 50), cand(1, 100), cand(1, 150)]
        chosen, excluded = select_bench(cfg, list(reversed(cs)))          # input order does not matter
        self.assertEqual(excluded, {"0:50": "blur", "0:150": "near-duplicate of 0:100"})
        self.assertEqual([c.key for c in chosen], [(0, 0), (1, 0), (1, 50), (1, 150)])
        self.assertEqual(chosen[0].key, (0, 0))                          # first and last frame of the match kept
        self.assertEqual(chosen[-1].key, (1, 150))

    def test_too_few_usable_frames_refused(self):
        with self.assertRaisesRegex(PolicyError, "only 1 usable frames"):
            select_bench(bench_cfg(frames_per_match=2), [cand(0, 0), cand(0, 10, blur=0.0)])


class SeparationTests(unittest.TestCase):
    def test_bench_project_must_differ_from_training_project(self):
        with self.assertRaisesRegex(ConfigError, "dev_bench.project must differ"):
            make_cfg(lambda d: d["dev_bench"].update(project=d["roboflow"]["project"]))

    def test_bench_tag_must_be_unique(self):
        with self.assertRaisesRegex(ConfigError, "must all differ"):
            make_cfg(lambda d: d["dev_bench"].update(bench_tag=d["review"]["reviewed_tag"]))

    def test_training_version_refuses_bench_frames(self):
        cfg = bench_cfg()
        images = [img(1, tags=("reviewed",), in_dataset=True),
                  img(2, mid="M2", tags=("reviewed", cfg.dev_bench.bench_tag), in_dataset=True)]
        with self.assertRaisesRegex(PolicyError, "DEV benchmark tag"):
            roboflow_version.check_ready(cfg, images)


class FakeUploader:
    def __init__(self, project):
        self.project, self.calls = project, []

    def single_upload(self, *, image_path, split, batch_name, tag_names, metadata, num_retry_uploads):
        self.calls.append({"name": Path(image_path).name, "split": split, "batch": batch_name, "tags": tag_names})
        uid = f"u{len(self.calls):03d}"
        self.project.add(uid, Path(image_path).name, tag_names, metadata)
        return {"image": {"id": uid, "duplicate": False}}


class BenchProject(FakeProject):
    def add(self, uid, name, tags, metadata):
        self.images.append(ProjectImage(id=uid, name=name, split="valid", in_dataset=False, tags=tuple(tags),
                                        metadata=dict(metadata), width=160, height=90, url=None))
        self.tags[uid] = set(tags)
        self.annotation[uid] = None

    def update_tags(self, image_id, add=(), remove=()):
        self.tags[image_id] = (self.tags[image_id] | set(add)) - set(remove)

    def save_annotation(self, image_id, *, name, content, labelmap, add_to_dataset, overwrite=False):
        self.ann_calls.append((image_id, add_to_dataset))
        boxes = []
        for line in content.strip().splitlines():
            k, cx, cy, w, h = line.split()
            boxes.append({"label": labelmap[k], "x": float(cx) * 160, "y": float(cy) * 90,
                          "width": float(w) * 160, "height": float(h) * 90})
        self.annotation[image_id] = {"boxes": boxes}
        return "saved"


class FakeDetector:
    class_names = ("player", "goalkeeper", "referee", "ball")

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection((10.0, 20.0, 30.0, 60.0), 0.9, 0), Detection((80.0, 40.0, 84.0, 44.0), 0.7, 3)]


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.videos = [str(self.root / "clip_a.avi"), str(self.root / "clip_b.avi")]
        write_video(Path(self.videos[0]), 10.0, seed=1)
        write_video(Path(self.videos[1]), 8.0, seed=2)
        self.cfg = make_cfg(lambda d: (d["licensed_matches"].append(lm("M_DEV", "dev")),
                                       d["detector"].update(model_id="proj/2"),
                                       d["prelabel"]["blur"].update(laplacian_var_min=0.0),
                                       d["dev_bench"].update(frames_per_match=4, min_interval_s=0.0)))

    def tearDown(self):
        self.tmp.cleanup()

    def run_real(self, cfg=None, match_id="M_DEV"):
        proj = BenchProject()
        up, det = FakeUploader(proj), FakeDetector()
        out = run_bench(cfg or self.cfg, match_id=match_id, videos=self.videos, dry_run=False, root=self.root,
                        uploader=up, project=proj, detector=det, api_key="k", now=NOW, sleep=lambda s: None)
        return out, proj, up, det

    def test_refusals_before_any_video_is_read(self):
        for mid, msg in (("M_EVAL", "registered for split 'eval'"), ("M1", "registered for split 'train'"),
                         ("GAME_3_NOT_LISTED", "not in licensed_matches")):
            with self.assertRaisesRegex(PolicyError, msg):
                run_bench(self.cfg, match_id=mid, videos=["/nonexistent.mp4"], dry_run=True, root=self.root)
        cfg = make_cfg(lambda d: d["licensed_matches"].append(lm("M_DEV", "dev", confirmed=False)))
        with self.assertRaisesRegex(PolicyError, "third-party"):
            run_bench(cfg, match_id="M_DEV", videos=["/nonexistent.mp4"], dry_run=False, root=self.root,
                      uploader=object(), project=object(), detector=object())

    def test_dry_run_uploads_nothing(self):
        out = run_bench(self.cfg, match_id="M_DEV", videos=self.videos, dry_run=True, root=self.root, now=NOW)
        doc = json.loads(out.read_text())
        self.assertTrue(out.is_relative_to(self.root / self.cfg.paths.dev_bench_frames_dir))
        self.assertEqual(doc["summary"]["status"], {"not_uploaded": 4})
        self.assertEqual({f["clip_index"] for f in doc["frames"]}, {0, 1})   # spread over both clips of the match

    def test_upload_tags_and_prelabel_through_guard(self):
        out, proj, up, det = self.run_real()
        doc = json.loads(out.read_text())
        b = self.cfg.dev_bench
        self.assertEqual(out.parent, self.root / self.cfg.paths.dev_bench_dir)
        self.assertEqual(doc["project"], b.project)
        self.assertNotEqual(doc["project"], self.cfg.roboflow.project)
        self.assertEqual(doc["summary"]["status"], {"uploaded": 4})
        self.assertEqual(doc["summary"]["prelabel_boxes"], {"ball": 4, "player": 4})
        self.assertEqual(doc["detector"], {"model_id": "proj/2", "confidence": self.cfg.detector.confidence,
                                           "iou": self.cfg.detector.iou})
        for c in up.calls:
            self.assertEqual(sorted(c["tags"]), sorted(["match:M_DEV", "split:dev", b.bench_tag]))
            self.assertEqual(c["split"], b.roboflow_split)
        for uid, tags in proj.tags.items():
            self.assertIn(b.prelabel_tag, tags)
            self.assertEqual(len(proj.annotation[uid]["boxes"]), 2)
        self.assertTrue(all(a is b.add_to_dataset for _, a in proj.ann_calls))
        with self.assertRaisesRegex(PolicyError, "already has benchmark manifest"):
            self.run_real()                                                # the sample is never redrawn

    def test_reviewed_frame_is_never_written(self):
        class Reviewed(BenchProject):
            def add(self, uid, name, tags, metadata):
                super().add(uid, name, list(tags) + ["reviewed"], metadata)
        proj = Reviewed()
        out = run_bench(self.cfg, match_id="M_DEV", videos=self.videos, dry_run=False, root=self.root,
                        uploader=FakeUploader(proj), project=proj, detector=FakeDetector(), api_key="k", now=NOW)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["summary"]["n_errors"], 4)
        self.assertIn("reviewed", doc["frames"][0]["error"])                 # first refusal stops the run
        self.assertEqual([f["status"] for f in doc["frames"]], ["uploaded"] + ["not_uploaded"] * 3)
        self.assertEqual(proj.ann_calls, [])


if __name__ == "__main__":
    unittest.main()
