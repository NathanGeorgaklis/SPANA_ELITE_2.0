"""Tracker bench v1 (bench/): frozen detection cache, candidate adapters behind spana.tracking.protocol.Tracker, and
the scorers. Synthetic clip only (no footage). Deep-EIoU / GTA-Link run only when their pinned checkouts exist
(`python -m bench.external fetch`); Roboflow `trackers` only when installed at the pinned version."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from bench import adapters
from bench.cache import DetectionCache, build_cache
from bench.config import REPO, load_bench
from bench.external import checkout_dir, head
from bench.metrics import target_stats, trackeval_data
from bench.run import run_online
from spana.artifacts import DETECTIONS_SCHEMA, validate_detections
from spana.config import load_config
from spana.labels import GroundTruth
from spana.tracking.protocol import FrameInput, TrackOutput, check_outputs
from spana.video import VideoSource

FPS, W, H, N = 30.0, 640, 360, 45
COLOURS = [(30, 30, 220), (220, 60, 30), (40, 200, 40)]          # three "kits" (BGR)


def boxes_at(i: int):
    """Three players walking; players 0 and 1 cross mid-clip."""
    return [(60 + 6.0 * i, 120.0, 100 + 6.0 * i, 220.0),
            (400 - 6.0 * i, 130.0, 440 - 6.0 * i, 230.0),
            (300.0, 40.0 + 2.0 * i, 336.0, 130.0 + 2.0 * i)]


def write_clip(path: Path) -> None:
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (5, 5), 0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(N):
        img = bg.copy()
        for (x1, y1, x2, y2), c in zip(boxes_at(i), COLOURS):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, -1)
        vw.write(img)
    vw.release()


def write_detections(path: Path, video: Path) -> dict:
    meta = VideoSource(video).meta
    frames = []
    for i in range(N):
        rows = [[*b, 0.9, 0, 0] for b in boxes_at(i)]
        rows.append([580.0, 300.0, 600.0, 340.0, 0.3, 0, 0])             # a low-score box every frame
        frames.append({"frame": i, "t": round(i / FPS, 6), "detections": rows})
    doc = {"schema": DETECTIONS_SCHEMA[0], "schema_version": DETECTIONS_SCHEMA[1], "provenance": {"git": "test"},
           "video": meta.to_dict(), "model": {"name": "synthetic"}, "class_names": ["person"],
           "detection_config": dataclasses.asdict(load_config(REPO / "config" / "default.yaml").detection),
           "frame_stride": 1, "frames": frames}
    validate_detections(doc)
    path.write_text(json.dumps(doc))
    return doc


class FakeEmbedder:
    """Mean crop colour as an 8-d unit vector: deterministic and identity-bearing on the synthetic clip."""
    rec = SimpleNamespace(crop_pad=0.0, input_size_hw=(32, 16), embedding_dim=8)

    def model_id(self):
        return {"name": "fake", "version": "test", "sha256": "0" * 64}

    def __call__(self, crops):
        v = np.array([[*c.reshape(-1, 3).mean(0) / 255.0, 0.1, 0.1, 0.1, 0.1, 0.1] for c in crops])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class Recorder:
    """Wraps a tracker and hashes exactly what it is given."""
    def __init__(self, inner):
        self.inner, self.name, self.h = inner, inner.name, hashlib.sha256()

    def update(self, detections, frame: FrameInput):
        for d in detections:
            self.h.update(np.array([*d.box, d.score], float).tobytes())
        if frame.embeddings is not None:
            self.h.update(np.ascontiguousarray(frame.embeddings, np.float32).tobytes())
        self.h.update(np.array([frame.index, frame.t]).tobytes())
        return self.inner.update(detections, frame)


def trackers_ready() -> bool:
    if importlib.util.find_spec("trackers") is None:
        return False
    from importlib.metadata import version
    return version("trackers") == "2.7.0"


def external_ready(bench, name) -> bool:
    c = next(c for c in bench.candidates if c.name == name)
    d = checkout_dir(bench.external_dir, c)
    # the checkout alone is not enough: its bench-only imports (bench/requirements-bench.txt) live in the bench venv
    deps = {"deep_eiou": ("lap", "cython_bbox"), "gta_link": ("sklearn", "torch", "seaborn")}.get(name, ())
    return (d / ".git").exists() and head(d) == c.implementation["commit"] and \
        all(importlib.util.find_spec(m) is not None for m in deps)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = Path(tempfile.mkdtemp())
        cls.video = cls.td / "clip.avi"
        write_clip(cls.video)
        cls.det_path = cls.td / "det.json"
        cls.doc = write_detections(cls.det_path, cls.video)
        sha = cls.doc["video"]["sha256"]
        build_cache(cls.td / "cache", "SYN", cls.det_path, sha, cls.video, FakeEmbedder())
        cls.cache = DetectionCache.load(cls.td / "cache")
        cls.bench = load_bench()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def online(self):
        names = ["spana"] + (["bytetrack", "botsort_cmc"] if trackers_ready() else []) + \
            (["deep_eiou"] if external_ready(self.bench, "deep_eiou") else [])
        return [c for c in self.bench.candidates if c.name in names]


class CacheTests(Fixture):
    def test_cache_is_byte_identical_and_hashed(self):
        self.assertEqual(Path(self.cache.root / "detections.json").read_bytes(), self.det_path.read_bytes())
        again = DetectionCache.load(self.td / "cache")
        self.assertEqual(again.fingerprint(), self.cache.fingerprint())
        self.assertEqual(len(again.frames), N)
        self.assertEqual(again.frames[3].embeddings.shape, (4, 8))

    def test_tampered_cache_refused(self):
        bad = self.td / "bad"
        shutil.copytree(self.td / "cache", bad)
        doc = json.loads((bad / "detections.json").read_text())
        doc["frames"][0]["detections"][0][0] += 1.0
        (bad / "detections.json").write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            DetectionCache.load(bad)

    def test_other_video_refused(self):
        with self.assertRaises(SystemExit):
            build_cache(self.td / "c2", "SYN", self.det_path, "f" * 64)

    def test_every_tracker_receives_identical_input(self):
        seen = {}
        for cand in self.online():
            rec = Recorder(adapters.make_tracker(cand, DetectionCache.load(self.td / "cache"), self.bench.external_dir))
            run_online(rec, self.cache, str(self.video), cand.needs_image)
            seen[cand.name] = rec.h.hexdigest()
        self.assertGreaterEqual(len(seen), 1)
        self.assertEqual(len(set(seen.values())), 1, seen)


class AdapterTests(Fixture):
    def _check(self, run):
        self.assertEqual([f for f, _ in run], [cf.index for cf in self.cache.frames])
        for (f, outs), cf in zip(run, self.cache.frames):
            self.assertTrue(all(isinstance(o, TrackOutput) and isinstance(o.track_id, str) for o in outs))
            check_outputs(outs, cf.detections)
        ids = {o.track_id for _, outs in run for o in outs}
        self.assertGreaterEqual(len(ids), 3)                        # the three players are tracked
        late = run[-1][1]
        self.assertGreaterEqual(len(late), 3)

    def test_online_adapters_follow_the_interface(self):
        for cand in self.online():
            with self.subTest(cand.name):
                run, secs = run_online(adapters.make_tracker(cand, self.cache, self.bench.external_dir), self.cache,
                                       str(self.video), cand.needs_image)
                self.assertGreater(secs, 0)
                self._check(run)

    @unittest.skipUnless(importlib.util.find_spec("trackers") is not None, "trackers not installed")
    def test_image_trackers_refuse_missing_frames(self):
        for cand in self.online():
            if cand.needs_image:
                t = adapters.make_tracker(cand, self.cache, self.bench.external_dir)
                cf = self.cache.frames[0]
                with self.assertRaises(ValueError):
                    t.update(cf.detections, FrameInput(cf.index, cf.t, None, cf.embeddings))

    def test_gta_link_relabels_on_the_same_detections(self):
        if not (trackers_ready() and external_ready(self.bench, "gta_link")):
            self.skipTest("trackers / GTA-Link checkout not available")
        botsort = next(c for c in self.bench.candidates if c.name == "botsort_cmc")
        base, _ = run_online(adapters.make_tracker(botsort, self.cache, self.bench.external_dir), self.cache,
                             str(self.video), True)
        gta = next(c for c in self.bench.candidates if c.name == "gta_link")
        run = adapters.refine(gta, base, self.cache, self.bench.external_dir)
        self._check(run)
        before = sorted((f, o.detection_index) for f, outs in base for o in outs)
        after = sorted((f, o.detection_index) for f, outs in run for o in outs)
        self.assertEqual(before, after)                              # relinking never adds or drops a box

    def test_spana_reference_refuses_other_detection_config(self):
        cand = next(c for c in self.bench.candidates if c.name == "spana")
        from bench.adapters.spana_ref import SpanaReference
        with self.assertRaises(ValueError):
            SpanaReference(cand, FPS, (W, H), {"score_low": 0.5})


class MetricTests(unittest.TestCase):
    """Known answers on a 4-frame toy: two players, label step 1."""
    def gt(self):
        g = GroundTruth()
        for f in range(4):
            g.players[f] = {"1": (0.0, 0.0, 10.0, 30.0), "2": (100.0, 0.0, 110.0, 30.0)}
        return g

    def out(self, f, tid, box, certain=True):
        return TrackOutput(tid, 0, box, certain)

    @unittest.skipUnless(importlib.util.find_spec("trackeval") is not None, "trackeval not installed (bench venv only)")
    def test_perfect_run(self):
        g = self.gt()
        run = [(f, [self.out(f, "a", g.players[f]["1"]), self.out(f, "b", g.players[f]["2"])]) for f in range(4)]
        from bench.metrics import trackeval_row, trackeval_sequence
        r = trackeval_row(trackeval_sequence(trackeval_data(g, run)))
        self.assertEqual((r["idf1"], r["idsw"], r["mota"]), (1.0, 0, 1.0))
        s = target_stats(g, run, 1, 1.0).row()
        self.assertEqual((s["target_coverage"], s["target_wrong_frames"], s["target_idsw"], s["target_fragments"]),
                         (1.0, 0, 0, 0))
        self.assertEqual(s["target_longest_seg_s_median"], 4.0)

    def test_swap_counts_wrong_and_silent_frames(self):
        g = self.gt()
        p1, p2 = g.players[0]["1"], g.players[0]["2"]
        run = [(0, [self.out(0, "a", p1), self.out(0, "b", p2)]),
               (1, [self.out(1, "a", p1), self.out(1, "b", p2)]),
               (2, [self.out(2, "b", p1), self.out(2, "a", p2, certain=False)]),      # swap; "a" flagged
               (3, [self.out(3, "b", p1), self.out(3, "a", p2)])]                     # "a" now silent
        s = target_stats(g, run, 1, 1.0)
        self.assertEqual(s.id_switches, 2)                            # each player changes track once
        self.assertEqual(s.fragments, 2)
        self.assertEqual(s.wrong, 4)                                  # frames 2, 3 for each tapped track
        self.assertEqual(s.wrong_unflagged, 3)                        # "a" on frame 2 was flagged
        self.assertEqual(s.covered, 4)
        self.assertEqual(s.longest_s, [2.0, 2.0])


class EndToEnd(Fixture):
    """cache -> track -> score through bench.run on synthetic DEV-style frozen labels (every 3rd frame)."""
    @unittest.skipUnless(importlib.util.find_spec("trackeval") is not None, "trackeval not installed (bench venv only)")
    def test_track_and_score_write_the_csv(self):
        lab = self.td / "labels"
        lab.mkdir()
        step = 3
        lines = [f"{i + 1},{p + 1},{b[0]},{b[1]},{b[2] - b[0]},{b[3] - b[1]},1,1,1"
                 for i in range(0, N, step) for p, b in enumerate(boxes_at(i))]
        gt = ("\n".join(lines) + "\n").encode()
        sha = hashlib.sha256(gt).hexdigest()
        (lab / "gt.txt").write_bytes(gt)
        (lab / "labels_frozen.json").write_text(json.dumps({"frames_labelled": list(range(0, N, step))}))
        (lab / "FROZEN.json").write_text(json.dumps({
            "split": "DEV", "gt_sha256": sha, "frame_step": step, "annotator": "test",
            "clip": {"width": W, "height": H, "sha256": self.doc["video"]["sha256"], "fps": FPS}}))
        names = ["spana.yaml"] + (["bytetrack.yaml", "botsort.yaml"] if trackers_ready() else [])
        cfg = self.td / "bench.yaml"
        src = (REPO / "config" / "tracking" / "bench_v1.yaml").read_text()
        src = re.sub(r"clips:\n(  - .*\n)+", f"clips:\n  - {{name: SYN, labels: {lab}}}\n", src)
        src = re.sub(r"candidates: .*\n", "candidates: [" + ", ".join(
            str(REPO / "config" / "tracking" / n) for n in names) + "]\n", src)
        cfg.write_text(src)
        from bench import run as R
        common = ["--bench", str(cfg), "--cache-root", str(self.td), "--out-root", str(self.td / "out")]
        (self.td / "SYN").symlink_to(self.td / "cache")
        self.assertEqual(R.main(common + ["track", "--video", f"SYN={self.video}"]), 0)
        out = self.td / "res.csv"
        self.assertEqual(R.main(common + ["score", "--csv", str(out)]), 0)
        import csv
        rows = list(csv.DictReader(out.open()))
        self.assertEqual(len(rows), 2 * len(names))                  # per clip + pooled, per tracker
        self.assertEqual(set(rows[0]), set(R.CSV_COLUMNS))
        for r in rows:
            self.assertEqual(r["target_players"], "3")
            self.assertEqual(r["target_wrong_frames_unflagged"], "0", r)   # no silent swap on a clean clip
        with self.assertRaises(SystemExit):                                # never overwritten
            R.main(common + ["score", "--csv", str(out)])


class Boundaries(unittest.TestCase):
    def test_product_never_imports_bench(self):
        for p in (REPO / "spana").rglob("*.py"):
            self.assertIsNone(re.search(r"^\s*(from|import)\s+bench\b", p.read_text(), re.M), p)

    def test_unlicensed_code_is_not_vendored(self):
        bench = load_bench()
        for c in bench.candidates:
            if not c.commercial_ok:
                self.assertIn("EVAL-ONLY", (REPO / "bench" / "adapters" / f"{c.name}.py").read_text())
        ignored = (REPO / ".gitignore").read_text()
        self.assertIn("bench/external/", ignored)

    def test_candidate_configs_are_strict(self):
        from bench.config import BenchConfigError, load_candidate
        d = Path(tempfile.mkdtemp())
        try:
            src = (REPO / "config" / "tracking" / "bytetrack.yaml").read_text()
            (d / "x.yaml").write_text(src + "surprise: 1\n")
            with self.assertRaises(BenchConfigError):
                load_candidate(d / "x.yaml")
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
