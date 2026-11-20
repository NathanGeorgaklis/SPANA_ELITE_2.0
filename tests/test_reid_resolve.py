"""Re-ID evidence for IDENTITY_UNCERTAIN (config reid_resolve, spana.identity.reid): off = no change and no embedding
requests; on = a crossing is cleared by the calibrated margin only when the embeddings say so (resolved_by REID),
a swapped crossing is never cleared, identical embeddings never clear; tracks 2.3.0 resolved_by; the follower
resumes only after a REID resolution with target.resume_after_reid; the model is pinned by the config's sha256
(record and ONNX file) and box embeddings use the training crop."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spana.artifacts import TRACKS_SCHEMA, ArtifactError, tracklet_rows, tracks_frame, validate_tracks
from spana.config import load_config
from spana.identity.follower import Selection, follow
from spana.records import IdentityState
from tests.synth import Player, run
from tests.test_target_follow import AU, A, N, tracks_doc

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
ON = dataclasses.replace(CFG, reid_resolve=dataclasses.replace(CFG.reid_resolve, enabled=True))


def unit(*v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def crossing(cfg, reid=None, calls=None):
    """Two same-looking players (no colour) run through each other: the tracker cannot tell them apart."""
    a = Player("A", (300, 500), (240, 0))
    b = Player("B", (700, 500), (-240, 0))
    return run([a, b], fps=30, duration_s=3.0, use_appearance=False, config=cfg, reid=reid, reid_calls=calls)


def same_identity_throughout(res) -> bool:
    """No CERTAIN track ever sits on a different player than its first observation."""
    first = {}
    for ft, gids in zip(res.frames, res.gt_of_detection):
        for o in ft.tracks:
            if o.detection_index is None:
                continue
            g = gids[o.detection_index]
            first.setdefault(o.track_id, g)
            if o.identity == IdentityState.CERTAIN and g != first[o.track_id]:
                return False
    return True


class TrackerReid(unittest.TestCase):
    def test_off_changes_nothing_and_requests_nothing(self):
        calls = []
        base = crossing(CFG)
        off = crossing(CFG, reid={"A": unit(1, 0), "B": unit(0, 1)}, calls=calls)
        self.assertEqual(calls, [])
        self.assertEqual([[dataclasses.astuple(o) for o in f.tracks] for f in base.frames],
                         [[dataclasses.astuple(o) for o in f.tracks] for f in off.frames])
        self.assertTrue(any(o.identity != IdentityState.CERTAIN for o in base.frames[-1].tracks))

    def test_distinct_embeddings_clear_the_episode_and_never_switch_identity(self):
        res = crossing(ON, reid={"A": unit(1, 0.05), "B": unit(0.05, 1)})
        resolved = [d for f in res.frames for d in f.decisions if d["kind"] == "identity_resolved"]
        self.assertTrue(resolved)
        self.assertTrue(all(d["resolved_by"] == "REID" for d in resolved))
        last = res.frames[-1].tracks
        self.assertTrue(all(o.identity == IdentityState.CERTAIN and o.resolved_by == "REID" for o in last))
        self.assertTrue(same_identity_throughout(res))

    def test_identical_or_missing_embeddings_never_clear(self):
        base = crossing(CFG)
        for reid in ({"A": unit(1, 1), "B": unit(1, 1)}, {}):
            res = crossing(ON, reid=reid)
            self.assertFalse([d for f in res.frames for d in f.decisions if d["kind"] == "identity_resolved"])
            self.assertEqual([o.identity for o in res.frames[-1].tracks], [o.identity for o in base.frames[-1].tracks])

    def test_pair_rule_is_the_calibrated_margin(self):
        from spana.tracking.tracker import MultiPlayerTracker
        t = MultiPlayerTracker(ON, 1920, 1080)
        own, rival = unit(1, 0), unit(0, 1)
        m = ON.reid_resolve.margin
        self.assertIs(t.reid_verdict(own, own, rival), True)                       # crop looks like itself
        self.assertIs(t.reid_verdict(rival, own, rival), False)                    # swapped: evidence against
        between = unit(1, 1)
        self.assertIsNone(t.reid_verdict(between, own, rival))                     # no margin: no evidence
        e = unit(np.cos(0.1), np.sin(0.1))
        self.assertEqual(t.reid_verdict(e, own, rival), float(e @ own) - float(e @ rival) >= m)
        self.assertIsNone(t.reid_verdict(None, own, rival))
        self.assertIsNone(t.reid_verdict(own, None, rival))

    def test_tracks_2_3_0_document(self):
        res = crossing(ON, reid={"A": unit(1, 0.05), "B": unit(0.05, 1)})
        doc = {"schema": "spana.tracks", "schema_version": TRACKS_SCHEMA[1], "provenance": {}, "video": {},
               "detections_sha256": "0" * 64, "frames": [tracks_frame(f, CFG.output) for f in res.frames],
               "tracklets": tracklet_rows(res.tracker.summaries())}
        validate_tracks(doc)
        self.assertEqual(TRACKS_SCHEMA[1], "2.3.0")
        bad = copy.deepcopy(doc)
        row = next(t for fr in bad["frames"] for t in fr["tracks"] if t["identity"] == "IDENTITY_UNCERTAIN")
        row["resolved_by"] = "REID"
        with self.assertRaises(ArtifactError):
            validate_tracks(bad)
        old = copy.deepcopy(doc)                                        # 2.2.0 = no resolved_by
        old["schema_version"] = "2.2.0"
        for fr in old["frames"]:
            for t in fr["tracks"]:
                t.pop("resolved_by")
        validate_tracks(old)
        old["schema_version"] = "2.1.0"                                 # 2.1.0 = no identity_doubt either
        for fr in old["frames"]:
            for t in fr["tracks"]:
                t.pop("identity_doubt")
        validate_tracks(old)
        stale = copy.deepcopy(doc)                                      # 2.2.0 must not carry resolved_by
        stale["schema_version"] = "2.2.0"
        with self.assertRaises(ArtifactError):
            validate_tracks(stale)


def with_resolution(doc: dict, by) -> dict:
    """tracks_doc (2.1.0) -> 2.3.0, every row that is CERTAIN after an episode resolved_by `by`."""
    d = copy.deepcopy(doc)
    d["schema_version"] = "2.3.0"
    for fr in d["frames"]:
        for t in fr["tracks"]:
            t["identity_doubt"] = False
            t["resolved_by"] = by if t["identity"] == "CERTAIN" and t["uncertain_episodes"] > 0 else None
    validate_tracks(d)
    return d


class FollowerResume(unittest.TestCase):
    FRAMES = [{"t1": N, "t2": N}, {"t1": A, "t2": A}, {"t1": AU, "t2": AU}, {"t1": A, "t2": A}, {"t1": A, "t2": A}]

    def answers(self, by, resume):
        cfg = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, resume_after_reid=resume))
        frames, _ = follow(with_resolution(tracks_doc(self.FRAMES), by), Selection.track_at_frame("t1", 1), cfg)
        return [(f["state"], f["reason"]) for f in frames]

    def test_resume_only_after_reid_and_only_when_allowed(self):
        self.assertEqual(self.answers("REID", True)[-2:], [("TARGET", "TRACK_OBSERVED")] * 2)
        self.assertEqual(self.answers("REID", False)[-2:], [("ABSTAIN", "AFTER_UNCERTAIN")] * 2)
        self.assertEqual(self.answers("APPEARANCE", True)[-2:], [("ABSTAIN", "AFTER_UNCERTAIN")] * 2)
        self.assertEqual(self.answers(None, True)[-2:], [("ABSTAIN", "AFTER_UNCERTAIN")] * 2)
        self.assertEqual(self.answers("REID", True)[1], ("ABSTAIN", "IDENTITY_UNCERTAIN"))


def tiny_onnx(path: Path):
    """A stand-in model: mean colour per channel of the 224x112 input -> 3-d 'embedding'."""
    import onnx
    from onnx import TensorProto, helper
    node = helper.make_node("ReduceMean", ["x"], ["m"], axes=[2, 3], keepdims=0)
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3, 224, 112])],
                          [helper.make_tensor_value_info("m", TensorProto.FLOAT, [None, 3])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8                                                    # readable by older onnxruntime builds
    onnx.save(m, str(path))


class PinnedModel(unittest.TestCase):
    """Hashed model registry: config pins the sha256; a record or ONNX file with another hash is refused."""
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        (self.td / "m").mkdir()
        tiny_onnx(self.td / "m" / "e.onnx")
        self.sha = hashlib.sha256((self.td / "m" / "e.onnx").read_bytes()).hexdigest()
        self.rec = {"name": "test", "source": "test", "version": "t1", "licence": "test", "onnx_path": "m/e.onnx",
                    "onnx_sha256": self.sha, "input_size_hw": [224, 112], "crop_pad": 0.05, "color_order": "rgb",
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "embedding_dim": 3,
                    "validation": "test"}
        (self.td / "rec.json").write_text(json.dumps(self.rec))

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_crop_is_the_training_crop(self):
        from spana.identity.reid import cut_crop
        from tools.reid_crops import cut_crop as train_crop
        img = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        for b in ((100, 50, 140, 150), (-5, 300, 30, 380), (600, 10, 660, 90)):
            self.assertTrue(np.array_equal(cut_crop(img, b, 0.05, (224, 112)), train_crop(img, b, 0.05, (224, 112))))

    def test_embed_boxes(self):
        from spana.identity.reid import ReidEmbedder, embed_boxes, load_pinned
        e = ReidEmbedder(load_pinned(self.td / "rec.json", self.sha))
        img = np.zeros((360, 640, 3), np.uint8)
        img[50:150, 100:140] = (0, 0, 255)                              # red player (BGR)
        img[50:150, 300:340] = (255, 0, 0)                              # blue player
        v = embed_boxes(e, img, [(100, 50, 140, 150), (300, 50, 340, 150), (700, 50, 720, 80)])
        self.assertIsNone(v[2])                                         # box outside the frame: no crop
        self.assertTrue(all(abs(np.linalg.norm(x) - 1) < 1e-6 for x in v[:2]))
        self.assertLess(float(v[0] @ v[1]), 0.99)

    def test_hash_mismatch_is_refused(self):
        from spana.identity.reid import ReidModelError, load_pinned
        with self.assertRaises(ReidModelError):
            load_pinned(self.td / "rec.json", "0" * 64)                 # record is not the calibrated model
        (self.td / "m" / "e.onnx").write_bytes(b"tampered")
        with self.assertRaises(ReidModelError):
            load_pinned(self.td / "rec.json", self.sha)                 # file differs from the record

    def test_registry_records_pin_v2_and_v3(self):
        """models/spana_reid_v2.json / v3.json name the hashes recorded for them; config pins v2."""
        v2 = json.loads((ROOT / "models" / "spana_reid_v2.json").read_text())
        v3 = json.loads((ROOT / "models" / "spana_reid_v3.json").read_text())
        self.assertEqual(v2["onnx_sha256"], "c269450e871cb28b4020b715618edc9dcda328faf951503a9e09f8e7a3032f84")
        self.assertEqual(v3["onnx_sha256"], "56a1b4c8e24aeb449c963974a75c90e9cf2dfa8b1a49c1517fe825f10d07e8f6")
        self.assertEqual(CFG.reid_resolve.model_record, "models/spana_reid_v2.json")
        self.assertEqual(CFG.reid_resolve.model_sha256, v2["onnx_sha256"])
        self.assertFalse(CFG.reid_resolve.enabled)
        self.assertEqual(CFG.reid_resolve.margin, 0.517)


if __name__ == "__main__":
    unittest.main()
