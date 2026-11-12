"""Detection-dataset tooling: config, licensing / split policy, deterministic sampling, manifests, the Roboflow
detector adapter and the DEV baseline. No network: the Roboflow project and inference client are fakes."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from spana.config import ConfigError
from spana.dataset.config import PolicyError, authorize, dataset_config_from_dict, load_dataset_config
from spana.dataset.manifest import ManifestError, validate_manifest
from spana.dataset.sampling import FrameSignal, scan_signals, select_frames
from spana.detection.roboflow_backend import HttpInferenceClient, RoboflowDetector, RoboflowError
from spana.records import Detection
from spana.secrets import SecretError, load_secret
from spana.video import VideoSource
from tools import roboflow_ingest
from tools.dev import roboflow_baseline

REPO = Path(__file__).resolve().parent.parent
CFG_PATH = REPO / "config" / "roboflow.yaml"
SECRET = "rf_TEST_SECRET_never_leaks_123"
NOW = datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc)


def raw() -> dict:
    return yaml.safe_load(CFG_PATH.read_text())


def cfg_with(matches=None, **over):
    d = raw()
    if matches is not None:
        d["licensed_matches"] = matches
    for k, v in over.items():
        sec, key = k.split("__")
        d[sec][key] = v
    return dataset_config_from_dict(d)


def match(mid, split, confirmed):
    return {"match_id": mid, "split": split, "permission": "test record", "third_party_processing_confirmed": confirmed}


MATCHES = [match("M_TRAIN", "train", True), match("M_DEV", "dev", True), match("M_DEV_UNCONF", "dev", False),
           match("M_EVAL", "eval", True)]


def write_video(path: Path, fps: float = 25.0, seconds: float = 10.0) -> None:
    """Static texture; fast panning 2-3 s; a cut to a different scene at 5 s."""
    rng = np.random.default_rng(0)
    a = cv2.resize(rng.integers(0, 255, (45, 225, 3), dtype=np.uint8), (900, 180), interpolation=cv2.INTER_NEAREST)
    b = np.zeros_like(a)
    b[..., 0], b[..., 1], b[..., 2] = 40, 160, 40
    b = cv2.add(b, (a // 4))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 90))
    for i in range(int(fps * seconds)):
        t = i / fps
        src = a if t < 5.0 else b
        x = 0 if not 2.0 <= t < 3.0 else int((t - 2.0) * fps) * 20
        vw.write(cv2.resize(src[:, x:x + 320], (160, 90)))
    vw.release()


class FakeProject:
    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def single_upload(self, **kw):
        self.calls.append(kw)
        if kw["metadata"]["frame_index"] in self.fail_on:
            raise RuntimeError(f"HTTP 500 for https://api.roboflow.com/upload?api_key={SECRET}")
        return {"image": {"id": f"img{len(self.calls):04d}"}}


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        c = load_dataset_config(CFG_PATH)
        self.assertEqual(c.classes, ("player", "goalkeeper", "referee", "ball"))
        self.assertEqual((c.roboflow.workspace, c.roboflow.project), ("nathans-workspace-9brdc", "spana-elite"))
        self.assertEqual(c.split, "dev")
        for k in ("model_id", "confidence", "iou"):
            self.assertTrue(hasattr(c.baseline, k))
        self.assertEqual(c.roboflow_split("dev"), "valid")
        self.assertNotIn("api_key:", CFG_PATH.read_text())   # only the variable *name* is configured

    def test_fails_closed(self):
        bad = []
        d = raw(); del d["sampling"]["interval_s"]; bad.append(d)
        d = raw(); d["sampling"]["extra"] = 1; bad.append(d)
        d = raw(); d["split"] = "eval"; bad.append(d)
        d = raw(); d["split"] = "train"; bad.append(d)
        d = raw(); d["classes"] = "player"; bad.append(d)
        d = raw(); d["classes"] = ["player", "player"]; bad.append(d)
        d = raw(); d["baseline"]["class_map"] = ["person:human"]; bad.append(d)
        d = raw(); d["baseline"]["api_url"] = "serverless.roboflow.com"; bad.append(d)
        d = raw(); d["baseline"]["confidence"] = 1.5; bad.append(d)
        d = raw(); d["sampling"]["analysis_step_s"] = 5.0; bad.append(d)
        d = raw(); d["roboflow"]["dev_split_name"] = "train"; bad.append(d)
        d = raw(); d["licensed_matches"] = [match("X", "train", True), match("X", "dev", True)]; bad.append(d)
        d = raw(); d["licensed_matches"] = [match("X", "test", True)]; bad.append(d)
        d = raw(); d["licensed_matches"] = [dict(match("X", "train", True), third_party_processing_confirmed="yes")]
        bad.append(d)
        d = raw(); d["schema_version"] = "0.1.0"; bad.append(d)
        for b in bad:
            with self.assertRaises(ConfigError):
                dataset_config_from_dict(b)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_with(MATCHES)

    def test_eval_always_refused(self):
        for tp in (True, False):
            with self.assertRaisesRegex(PolicyError, "eval"):
                authorize(self.cfg, "M_EVAL", "eval", third_party=tp)

    def test_unlicensed_and_wrong_split_refused(self):
        with self.assertRaisesRegex(PolicyError, "not in licensed_matches"):
            authorize(self.cfg, "UNKNOWN", "train", third_party=False)
        with self.assertRaisesRegex(PolicyError, "registered for split"):
            authorize(self.cfg, "M_TRAIN", "dev", third_party=False)

    def test_third_party_needs_confirmation(self):
        with self.assertRaisesRegex(PolicyError, "third-party"):
            authorize(self.cfg, "M_DEV_UNCONF", "dev", third_party=True)
        self.assertEqual(authorize(self.cfg, "M_DEV_UNCONF", "dev", third_party=False).match_id, "M_DEV_UNCONF")

    def test_ingest_refuses_before_reading_video(self):
        missing = "/nonexistent/clip.mp4"   # a VideoError would mean the video was opened first
        with tempfile.TemporaryDirectory() as td:
            for mid, split in (("M_EVAL", "eval"), ("UNKNOWN", "train"), ("M_TRAIN", "dev"), ("M_DEV_UNCONF", "dev")):
                with self.assertRaises(PolicyError):
                    roboflow_ingest.run_ingest(self.cfg, video=missing, match_id=mid, split=split, dry_run=False,
                                               project=FakeProject(), root=Path(td))
            self.assertFalse((Path(td) / "runs").exists())

    def test_cli_exit_code_for_eval(self):
        r = subprocess.run([os.sys.executable, "-m", "tools.roboflow_ingest", "--video", "x.mp4", "--match-id", "A",
                            "--split", "eval", "--dry-run"], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("eval", r.stderr)

    def test_default_config_matches_are_train_only_and_unique(self):
        cfg = load_dataset_config(CFG_PATH)
        self.assertEqual({m.split for m in cfg.licensed_matches}, {"train"})   # DEV / EVAL matches come separately
        for m in cfg.licensed_matches:
            with self.assertRaisesRegex(PolicyError, "registered for split"):
                authorize(cfg, m.match_id, "dev", third_party=False)

    def test_baseline_dev_only_and_hosted_needs_confirmation(self):
        with self.assertRaisesRegex(PolicyError, "registered for split"):
            roboflow_baseline.run_baseline(self.cfg, video="/nonexistent.mp4", match_id="M_TRAIN")
        with self.assertRaisesRegex(PolicyError, "third-party"):
            roboflow_baseline.run_baseline(self.cfg, video="/nonexistent.mp4", match_id="M_DEV_UNCONF")
        local = cfg_with(MATCHES, baseline__api_url="http://localhost:9001")
        self.assertTrue(local.is_local_api())
        with self.assertRaises(Exception) as cm:   # passes policy, then fails on the missing video
            roboflow_baseline.run_baseline(local, video="/nonexistent.mp4", match_id="M_DEV_UNCONF")
        self.assertNotIsInstance(cm.exception, PolicyError)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.s = cfg_with(MATCHES).sampling

    def _signals(self, n=200, step=0.2, motion_at=(), cut_at=()):
        return [FrameSignal(i * 5, i * step, 0.5 if i in motion_at else 0.0, 0.9 if i in cut_at else 0.0)
                for i in range(n)]

    def test_interval_motion_cut_and_gap(self):
        sel = select_frames(self._signals(n=50, motion_at={11, 12, 30}, cut_at={20}), self.s)
        by = {f.frame_index // 5: f.reasons for f in sel}
        self.assertEqual([k for k, r in by.items() if "interval" in r], [0, 10, 20, 30, 40])
        self.assertEqual(by[11], ("motion",))
        self.assertNotIn(12, by)                        # within event_min_gap_s of the frame at 11
        self.assertEqual(by[20], ("cut", "interval"))
        self.assertEqual(by[30], ("interval", "motion"))

    def test_deterministic_and_capped_evenly(self):
        sig = self._signals(n=3000, motion_at=set(range(0, 3000, 7)))
        a, b = select_frames(sig, self.s), select_frames(list(reversed(sig)), self.s)
        self.assertEqual(a, b)
        self.assertEqual(len(a), self.s.max_frames_per_video)
        self.assertEqual(a[0].frame_index, 0)
        idx = [f.frame_index for f in a]
        self.assertEqual(idx, sorted(set(idx)))

    def test_video_scan_is_deterministic_and_finds_cut_and_motion(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "v.avi"
            write_video(p)
            s1 = scan_signals(VideoSource(p).frames(), self.s)
            s2 = scan_signals(VideoSource(p).frames(), self.s)
            self.assertEqual(s1, s2)
            sel = select_frames(s1, self.s)
            self.assertEqual(sel, select_frames(s2, self.s))
            cuts = [f.timestamp_s for f in sel if "cut" in f.reasons]
            motion = [f.timestamp_s for f in sel if "motion" in f.reasons]
            self.assertTrue(cuts and all(4.9 <= t <= 5.3 for t in cuts), cuts)
            self.assertTrue(motion and all(1.9 <= t <= 3.3 or t in cuts for t in motion), motion)


class IngestManifestTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_with(MATCHES)
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.video = self.root / "clip.avi"
        write_video(self.video)

    def tearDown(self):
        self.td.cleanup()

    def test_dry_run_manifest_structure(self):
        out = roboflow_ingest.run_ingest(self.cfg, video=str(self.video), match_id="M_DEV_UNCONF", split="dev",
                                         dry_run=True, root=self.root, now=NOW, argv=["--video", str(self.video)])
        doc = json.loads(out.read_text())
        validate_manifest(doc)
        self.assertTrue(out.is_relative_to(self.root / self.cfg.paths.frames_dir))
        self.assertEqual(doc["tags"], ["match:M_DEV_UNCONF", "split:dev"])
        self.assertEqual(doc["roboflow_split"], "valid")
        self.assertEqual(doc["video"]["sha256"], VideoSource(self.video).meta.sha256)
        self.assertNotIn(str(self.root), out.read_text())          # no local directory names
        self.assertGreater(len(doc["frames"]), 3)
        for f in doc["frames"]:
            self.assertEqual(f["status"], "not_uploaded")
            img = out.parent / f["image_file"]
            self.assertTrue(img.is_file())
            self.assertEqual(f["tags"], doc["tags"])

    def test_upload_tags_split_trace_and_secret_redaction(self):
        dry = json.loads(roboflow_ingest.run_ingest(self.cfg, video=str(self.video), match_id="M_TRAIN",
                                                    split="train", dry_run=True, root=self.root, now=NOW).read_text())
        fail_idx = dry["frames"][1]["frame_index"]
        proj = FakeProject(fail_on={fail_idx})
        later = NOW.replace(minute=1)
        out = roboflow_ingest.run_ingest(self.cfg, video=str(self.video), match_id="M_TRAIN", split="train",
                                         dry_run=False, project=proj, api_key=SECRET, root=self.root, now=later)
        text = out.read_text()
        doc = json.loads(text)
        validate_manifest(doc)
        self.assertTrue(out.is_relative_to(self.root / self.cfg.paths.manifest_dir))
        self.assertNotIn(SECRET, text)
        self.assertEqual(len(proj.calls), len(doc["frames"]))
        self.assertEqual({c["split"] for c in proj.calls}, {"train"})
        self.assertEqual({tuple(c["tag_names"]) for c in proj.calls}, {("match:M_TRAIN", "split:train")})
        self.assertEqual({c["batch_name"] for c in proj.calls}, {"spana_M_TRAIN_train"})
        self.assertEqual([f["frame_index"] for f in doc["frames"]], [f["frame_index"] for f in dry["frames"]])
        for f, call in zip(doc["frames"], proj.calls):
            self.assertEqual(call["metadata"]["frame_index"], f["frame_index"])
            self.assertEqual(call["metadata"]["timestamp_s"], f["timestamp_s"])
            if f["frame_index"] == fail_idx:
                self.assertEqual((f["status"], f["upload_id"]), ("failed", None))
                self.assertIn("***", f["error"])
            else:
                self.assertEqual(f["status"], "uploaded")
                self.assertTrue(f["upload_id"].startswith("img"))
        self.assertEqual(doc["summary"]["n_failed"], 1)

    def test_match_recorded_under_another_split_is_refused(self):
        roboflow_ingest.run_ingest(self.cfg, video=str(self.video), match_id="M_TRAIN", split="train",
                                   dry_run=False, project=FakeProject(), api_key=SECRET, root=self.root, now=NOW)
        moved = cfg_with([match("M_TRAIN", "dev", True)])   # registry edited after an upload
        with self.assertRaisesRegex(PolicyError, "never spans splits"):
            roboflow_ingest.run_ingest(moved, video=str(self.video), match_id="M_TRAIN", split="dev",
                                       dry_run=False, project=FakeProject(), root=self.root, now=NOW.replace(hour=13))

    def test_validate_manifest_rejects_bad_documents(self):
        out = roboflow_ingest.run_ingest(self.cfg, video=str(self.video), match_id="M_TRAIN", split="train",
                                         dry_run=False, project=FakeProject(), api_key=SECRET, root=self.root, now=NOW)
        good = json.loads(out.read_text())
        mutations = [
            lambda d: d.update(split="eval"),
            lambda d: d.update(tags=["match:OTHER", "split:train"]),
            lambda d: d["frames"][0].update(tags=["match:M_TRAIN", "split:dev"]),
            lambda d: d["frames"][0].update(status="failed"),
            lambda d: d["frames"][0].update(reasons=["random"]),
            lambda d: d["frames"].reverse(),
            lambda d: d["summary"].update(n_frames=0),
            lambda d: d["video"].update(sha256="abc"),
            lambda d: d.pop("permission"),
        ]
        for m in mutations:
            d = copy.deepcopy(good)
            m(d)
            with self.assertRaises(ManifestError):
                validate_manifest(d)


class FakeClient:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def infer(self, image_jpeg, *, model_id, confidence, iou):
        self.calls.append((image_jpeg, model_id, confidence, iou))
        return self.resp


def pred(x, y, w, h, c, label):
    return {"x": x, "y": y, "width": w, "height": h, "confidence": c, "class": label, "class_id": 0}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_with(MATCHES, baseline__class_map=["player:player", "goalkeeper:goalkeeper",
                                                          "referee:referee", "ball:ball"])
        self.frame = np.zeros((90, 160, 3), np.uint8)

    def _det(self, preds, w=160, h=90):
        client = FakeClient({"image": {"width": w, "height": h}, "predictions": preds})
        return RoboflowDetector.from_config(self.cfg, api_key=SECRET, client=client), client

    def test_output_schema_and_config_passthrough(self):
        det, client = self._det([pred(50, 40, 10, 20, 0.9, "player"), pred(5, 5, 20, 20, 0.8, "ball"),
                                 pred(80, 40, 10, 20, 0.6, "referee"), pred(100, 40, 10, 20, 0.1, "player"),
                                 pred(30, 30, 5, 5, 0.9, "spectator")])
        out = det.detect(self.frame)
        self.assertEqual(len(client.calls), 1)
        jpeg, model_id, conf, iou = client.calls[0]
        self.assertEqual((model_id, conf, iou), (self.cfg.baseline.model_id, self.cfg.baseline.confidence,
                                                 self.cfg.baseline.iou))
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
        self.assertTrue(all(isinstance(d, Detection) for d in out))
        names = [det.class_names[d.class_id] for d in out]
        self.assertEqual(names, ["player", "referee", "ball"])      # sorted by class; low score + unmapped dropped
        self.assertEqual(out[0].box, (45.0, 30.0, 55.0, 50.0))      # centre/size -> x1,y1,x2,y2
        self.assertEqual(out[2].box, (0.0, 0.0, 15.0, 15.0))        # clipped to the frame
        for d in out:
            self.assertEqual(len(d.box), 4)
            self.assertIsInstance(d.score, float)
            self.assertFalse(d.multi_body)
        self.assertEqual(dict(det.dropped_labels), {"spectator": 1})
        self.assertEqual(det.model_id["model_id"], self.cfg.baseline.model_id)

    def test_fails_closed_on_bad_responses(self):
        for resp in ({"predictions": "x", "image": {"width": 160, "height": 90}},
                     {"image": {"width": 320, "height": 180}, "predictions": []},
                     {"image": {"width": 160, "height": 90}, "predictions": [{"x": 1}]},
                     {"image": {"width": 160, "height": 90}, "predictions": [pred(1, 1, 1, float("nan"), .9, "ball")]}):
            det = RoboflowDetector.from_config(self.cfg, api_key=SECRET, client=FakeClient(resp))
            with self.assertRaises(RoboflowError):
                det.detect(self.frame)

    def test_http_client_key_in_body_never_in_url_or_errors(self):
        session = mock.Mock()
        session.post.return_value = mock.Mock(status_code=200, json=lambda: {"predictions": []})
        c = HttpInferenceClient("https://example.invalid/", SECRET, 5.0, session=session)
        c.infer(b"jpg", model_id="m/1", confidence=0.3, iou=0.5)
        (url,), kw = session.post.call_args
        self.assertEqual(url, "https://example.invalid/infer/object_detection")
        self.assertNotIn(SECRET, url)
        self.assertEqual(kw["json"]["api_key"], SECRET)
        self.assertEqual((kw["json"]["confidence"], kw["json"]["iou_threshold"]), (0.3, 0.5))
        session.post.return_value = mock.Mock(status_code=401, text=f"bad key {SECRET}")
        with self.assertRaises(RoboflowError) as cm:
            c.infer(b"jpg", model_id="m/1", confidence=0.3, iou=0.5)
        self.assertNotIn(SECRET, str(cm.exception))
        session.post.side_effect = ConnectionError(f"https://x?api_key={SECRET}")
        with self.assertRaises(RoboflowError) as cm:
            c.infer(b"jpg", model_id="m/1", confidence=0.3, iou=0.5)
        self.assertNotIn(SECRET, str(cm.exception))


class SecretTests(unittest.TestCase):
    def test_env_then_dotenv_then_fail(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / ".env"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SPANA_TEST_KEY", None)
                with self.assertRaises(SecretError):
                    load_secret("SPANA_TEST_KEY", env_file=f)
                f.write_text("SPANA_TEST_KEY=from_file\n")
                self.assertEqual(load_secret("SPANA_TEST_KEY", env_file=f), "from_file")
                os.environ["SPANA_TEST_KEY"] = "from_env"
                self.assertEqual(load_secret("SPANA_TEST_KEY", env_file=f), "from_env")

    def test_dotenv_is_gitignored_and_example_is_not(self):
        self.assertEqual(subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO).returncode, 0)
        self.assertEqual(subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=REPO).returncode, 1)
        self.assertEqual((REPO / ".env.example").read_text().count("ROBOFLOW_API_KEY="), 1)


class FakeDetector:
    class_names = ("player", "goalkeeper", "referee", "ball")
    dropped_labels = {}

    def __init__(self):
        self.n = 0

    def detect(self, frame):
        self.n += 1
        dets = [Detection((10, 10, 20, 40), 0.91, 0), Detection((40, 10, 50, 40), 0.55, 0)]
        if self.n % 2:
            dets.append(Detection((60, 60, 64, 64), 0.42, 3))
        return dets


class BaselineTests(unittest.TestCase):
    def test_outputs_cache_and_no_overwrite(self):
        cfg = cfg_with(MATCHES, baseline__api_url="http://localhost:9001")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "dev.avi"
            write_video(video)
            det = FakeDetector()
            out = roboflow_baseline.run_baseline(cfg, video=str(video), match_id="M_DEV_UNCONF", detector=det,
                                                 root=root)
            doc = json.loads((out / "detections.json").read_text())
            self.assertEqual(doc["split"], "dev")
            n = len(doc["frames"])
            self.assertEqual(det.n, n)
            for f in doc["frames"]:
                self.assertEqual(set(f), {"frame_index", "timestamp_s", "reasons", "detections"})
                for d in f["detections"]:
                    self.assertEqual(set(d), {"box", "class", "class_id", "score"})
            s = json.loads((out / "summary.json").read_text())
            self.assertEqual(s["per_class"]["player"]["total"], 2 * n)
            self.assertEqual(s["per_class"]["ball"]["frame_share"], round(((n + 1) // 2) / n, 4))
            self.assertEqual(s["per_class"]["goalkeeper"]["frame_share"], 0.0)
            self.assertEqual(sum(s["confidence_histogram_all"]), 2 * n + (n + 1) // 2)
            self.assertEqual(len(s["per_frame"]), n)
            samples = sorted((out / "samples").glob("*.jpg"))
            self.assertEqual(len(samples), min(cfg.baseline.render_samples, n))
            self.assertIn("ball", (out / "summary.md").read_text())
            # cached: identical request does not call the detector again
            det2 = FakeDetector()
            roboflow_baseline.run_baseline(cfg, video=str(video), match_id="M_DEV_UNCONF", detector=det2, root=root)
            self.assertEqual(det2.n, 0)
            # different thresholds never overwrite the cached detections
            other = cfg_with(MATCHES, baseline__api_url="http://localhost:9001", baseline__confidence=0.5)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                roboflow_baseline.run_baseline(other, video=str(video), match_id="M_DEV_UNCONF",
                                               detector=FakeDetector(), root=root)


if __name__ == "__main__":
    unittest.main()
