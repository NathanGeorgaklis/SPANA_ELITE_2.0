"""Re-ID signatures per track (spana.track_embeddings 1.0.0) for re-entry offers (spana.identity.follower).

A pinned SPANA re-ID model (ONNX, `models/spana_reid_v2.json`) embeds clean crops of each confirmed track: the first
and the last `reid.samples` observations that are confirmed, CERTAIN, uncontested, at least `reid.min_crop_height_px`
tall, at least `reid.sample_min_gap_s` apart, and overlap no other observed box by more than `reid.clean_max_iou`
(a crop mixing two players is not evidence about either). A track's `first` / `last` signature is the normalised mean
of those embeddings. A track without such a crop has no signature (`null`): it is never offered, never guessed.

The signatures are evidence for an OFFER only. Similarity never makes an answer by itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..artifacts import EMBEDDINGS_SCHEMA, sha256_doc, validate_track_embeddings
from ..config import SpanaConfig
from ..provenance import build_provenance
from ..geometry import iou_matrix
from ..video import sha256_file

DECIMALS = 5                       # stored embedding precision (size bound; cosine error < 1e-4)


class ReidModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReidModelRecord:
    name: str
    source: str
    version: str
    licence: str
    onnx_path: str                 # relative to the record file
    onnx_sha256: str
    input_size_hw: Tuple[int, int]
    crop_pad: float                # box padding per side, share of width / height (as in training crops)
    color_order: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    embedding_dim: int
    validation: str

    @staticmethod
    def load(path: str | Path) -> "ReidModelRecord":
        p = Path(path)
        if not p.is_file():
            raise ReidModelError(f"re-ID model record not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        names = list(ReidModelRecord.__dataclass_fields__)
        if set(data) != set(names):
            raise ReidModelError(f"re-ID model record {p}: keys {sorted(data)} != {sorted(names)}")
        if data["color_order"] not in ("rgb", "bgr") or len(data["onnx_sha256"]) != 64 or \
                not data["licence"].strip() or not data["version"].strip():
            raise ReidModelError("re-ID model record needs rgb|bgr, a licence, an exact version and a sha256")
        rec = ReidModelRecord(**{**data, "input_size_hw": tuple(data["input_size_hw"]), "mean": tuple(data["mean"]),
                                 "std": tuple(data["std"])})
        weights = (p.parent / rec.onnx_path).resolve()
        if not weights.is_file():
            raise ReidModelError(f"re-ID weights not found: {weights}")
        if sha256_file(weights) != rec.onnx_sha256:
            raise ReidModelError(f"re-ID weights sha256 differs from the pinned {rec.onnx_sha256}")
        object.__setattr__(rec, "onnx_path", str(weights))
        return rec


def cut_crop(img: np.ndarray, box: Sequence[float], pad: float, size_hw: Tuple[int, int]) -> np.ndarray:
    """Padded box, clipped to the image, resized to size_hw (BGR in, BGR out); the training crop convention."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    xa, ya = int(max(0, round(x1 - pad * w))), int(max(0, round(y1 - pad * h)))
    xb, yb = int(min(W, round(x2 + pad * w))), int(min(H, round(y2 + pad * h)))
    c = img[ya:yb, xa:xb]
    if c.size == 0:
        raise ValueError(f"empty crop for box {box}")
    return cv2.resize(c, (size_hw[1], size_hw[0]),
                      interpolation=cv2.INTER_AREA if c.shape[0] > size_hw[0] else cv2.INTER_LINEAR)


class ReidEmbedder:
    def __init__(self, record: ReidModelRecord, batch: int = 32):
        import onnxruntime as ort
        self.rec, self.batch = record, batch
        self.session = ort.InferenceSession(record.onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def model_id(self) -> dict:
        return {"name": self.rec.name, "version": self.rec.version, "sha256": self.rec.onnx_sha256}

    def __call__(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings (N x D) of model-size BGR crops."""
        if not len(crops_bgr):
            return np.zeros((0, self.rec.embedding_dim))
        mean, std = np.array(self.rec.mean), np.array(self.rec.std)
        x = np.stack([(((c[:, :, ::-1] if self.rec.color_order == "rgb" else c).astype(np.float32) / 255.0 - mean)
                       / std).transpose(2, 0, 1) for c in crops_bgr]).astype(np.float32)
        y = np.concatenate([self.session.run(None, {self.input: x[k:k + self.batch]})[0]
                            for k in range(0, len(x), self.batch)]).reshape(len(x), -1).astype(np.float64)
        if y.shape[1] != self.rec.embedding_dim:
            raise ReidModelError(f"model output dim {y.shape[1]} != record {self.rec.embedding_dim}")
        return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def select_samples(tracks_doc: dict, cfg: SpanaConfig) -> Dict[str, Dict[str, List[Tuple[int, List[float]]]]]:
    """track -> {"first": [(frame, box)], "last": [(frame, box)]}: the clean observations that will be embedded."""
    rc = cfg.reid
    clean: Dict[str, List[Tuple[int, float, List[float]]]] = {}
    for fr in tracks_doc["frames"]:
        obs = [t for t in fr["tracks"] if t["source"] == "OBSERVED"]
        boxes = np.array([t["box"] for t in obs], dtype=float).reshape(-1, 4)
        ov = iou_matrix(boxes, boxes) if len(obs) else np.zeros((0, 0))
        if len(obs):
            np.fill_diagonal(ov, 0.0)
        for k, t in enumerate(obs):
            b = t["box"]
            if not t["confirmed"] or t["identity"] != "CERTAIN" or t["contested"] or \
                    b[3] - b[1] < rc.min_crop_height_px or (len(obs) > 1 and ov[k].max() > rc.clean_max_iou):
                continue
            clean.setdefault(t["track_id"], []).append((fr["frame"], fr["t"], list(b)))

    def spaced(rows):
        out, last_t = [], None
        for f, t, b in rows:
            if last_t is None or abs(t - last_t) >= rc.sample_min_gap_s - 1e-9:
                out.append((f, b))
                last_t = t
            if len(out) == rc.samples:
                break
        return out

    return {tid: {"first": spaced(rows), "last": sorted(spaced(list(reversed(rows))))} for tid, rows in clean.items()}


def signature(emb: np.ndarray) -> Optional[List[float]]:
    if not len(emb):
        return None
    m = emb.mean(axis=0)
    m = m / max(float(np.linalg.norm(m)), 1e-12)
    return [round(float(v), DECIMALS) for v in m]


def track_embeddings(video, tracks_doc: dict, embedder: ReidEmbedder, cfg: SpanaConfig) -> Dict[str, dict]:
    """Per track {first, last, first_frames, last_frames}; reads only the frames that hold a sample."""
    samples = select_samples(tracks_doc, cfg)
    need: Dict[int, List[Tuple[str, str, List[float]]]] = {}
    for tid, s in samples.items():
        for end in ("first", "last"):
            for f, b in s[end]:
                need.setdefault(f, []).append((tid, end, b))
    crops: Dict[Tuple[str, str], List[np.ndarray]] = {}
    if need:
        first, last = min(need), max(need)
        for i, _, img in video.frames(first, last + 1):
            for tid, end, b in need.get(i, ()):
                crops.setdefault((tid, end), []).append(
                    cut_crop(img, b, embedder.rec.crop_pad, embedder.rec.input_size_hw))
    out = {}
    for tid in sorted(samples):
        row = {}
        for end in ("first", "last"):
            c = crops.get((tid, end), [])
            row[end] = signature(embedder(c)) if c else None
            row[f"{end}_frames"] = [f for f, _ in samples[tid][end]]
        out[tid] = row
    return out


def cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    x, y = np.asarray(a, float), np.asarray(b, float)
    return float(x @ y / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-12))


def embeddings_document(video, tracks_doc: dict, embedder: ReidEmbedder, cfg: SpanaConfig,
                        argv: Sequence[str] = ()) -> dict:
    """Validated spana.track_embeddings 1.0.0 document for a validated spana.tracks document and its video."""
    import dataclasses
    import sys
    if video.meta.sha256 != tracks_doc["video"]["sha256"]:
        raise ReidModelError("the video differs from the tracks document's video")
    tsha = sha256_doc(tracks_doc)
    doc = {"schema": EMBEDDINGS_SCHEMA[0], "schema_version": EMBEDDINGS_SCHEMA[1],
           "provenance": build_provenance(config_sha256=cfg.sha256(), argv=list(argv) or sys.argv,
                                          inputs={"video": video.meta.sha256, "tracks": tsha,
                                                  "model": embedder.rec.onnx_sha256}),
           "video": tracks_doc["video"], "tracks_sha256": tsha,
           "model": {**embedder.model_id(), "embedding_dim": embedder.rec.embedding_dim},
           "reid_config": dataclasses.asdict(cfg.reid),
           "tracks": track_embeddings(video, tracks_doc, embedder, cfg)}
    validate_track_embeddings(doc, tracks_doc)
    return doc
