"""ONNX Runtime detector backend driven by a pinned model record.

A model record (JSON) is mandatory. The weights file is hash-verified before loading; any missing field,
unsupported output format or hash mismatch raises ModelRecordError (fail closed, no substitute model).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np

from ..config import DetectionConfig
from ..records import Detection
from ..video import sha256_file
from .postprocess import filter_detections, letterbox

SUPPORTED_OUTPUT_FORMATS = ("cxcywh_class_scores",)   # per candidate: cx, cy, w, h, C class scores; no objectness
SUPPORTED_OUTPUT_LAYOUTS = ("channels_first", "channels_last")  # (1, 4+C, N) / (1, N, 4+C)
SUPPORTED_COLOR_ORDERS = ("RGB", "BGR")


class ModelRecordError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRecord:
    name: str
    source: str
    version: str
    sha256: str
    licence: str
    weights_path: str          # relative to the record file
    input_size: int            # square input, pixels
    color_order: str
    scale_to_unit: bool        # divide pixel values by 255
    pad_value: int
    output_format: str
    output_layout: str
    person_class_id: int

    @staticmethod
    def load(path: str | Path) -> "ModelRecord":
        p = Path(path)
        if not p.is_file():
            raise ModelRecordError(f"model record not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        names = list(ModelRecord.__dataclass_fields__)
        missing = [n for n in names if n not in data]
        unknown = [k for k in data if k not in names]
        if missing or unknown:
            raise ModelRecordError(f"model record {p}: missing {missing}, unknown {unknown}")
        rec = ModelRecord(**data)
        if rec.output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ModelRecordError(f"unsupported output_format {rec.output_format!r}")
        if rec.output_layout not in SUPPORTED_OUTPUT_LAYOUTS:
            raise ModelRecordError(f"unsupported output_layout {rec.output_layout!r}")
        if rec.color_order not in SUPPORTED_COLOR_ORDERS:
            raise ModelRecordError(f"unsupported color_order {rec.color_order!r}")
        if not rec.licence.strip() or not rec.version.strip() or len(rec.sha256) != 64:
            raise ModelRecordError("model record needs a licence, an exact version and a sha256")
        weights = (p.parent / rec.weights_path).resolve()
        if not weights.is_file():
            raise ModelRecordError(f"weights not found: {weights}")
        actual = sha256_file(weights)
        if actual != rec.sha256:
            raise ModelRecordError(f"weights sha256 {actual} != pinned {rec.sha256}")
        object.__setattr__(rec, "weights_path", str(weights))
        return rec


def decode_cxcywh_class_scores(raw: np.ndarray, layout: str, num_coords: int = 4):
    """Return (boxes_xyxy, best_score, best_class) in model-input pixels."""
    a = np.asarray(raw, dtype=float)
    if a.ndim != 3 or a.shape[0] != 1:
        raise ModelRecordError(f"unexpected output shape {a.shape}")
    if layout not in SUPPORTED_OUTPUT_LAYOUTS:
        raise ModelRecordError(f"unsupported output_layout {layout!r}")
    a = a[0].T if layout == "channels_first" else a[0]
    if a.shape[1] <= num_coords:
        raise ModelRecordError(f"output has no class scores: {a.shape}")
    cx, cy, w, h = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    cls = a[:, num_coords:]
    best = cls.argmax(axis=1)
    score = cls[np.arange(len(cls)), best]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, score, best


class OnnxPersonDetector:
    def __init__(self, record: ModelRecord, cfg: DetectionConfig, providers: List[str] | None = None):
        import onnxruntime as ort  # imported here so the rest of SPANA does not depend on it

        self._rec = record
        self._cfg = cfg
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0
        self._sess = ort.InferenceSession(record.weights_path, sess_options=opts,
                                          providers=providers or ["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    @property
    def model_id(self) -> dict:
        d = asdict(self._rec)
        d["weights_path"] = Path(d["weights_path"]).name
        return d

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        r = self._rec
        img, lb = letterbox(frame_bgr, r.input_size, r.pad_value)
        if r.color_order == "RGB":
            img = img[:, :, ::-1]
        x = img.astype(np.float32)
        if r.scale_to_unit:
            x /= 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        raw = self._sess.run(None, {self._input: x})[0]
        boxes, scores, classes = decode_cxcywh_class_scores(raw, r.output_layout)
        h, w = frame_bgr.shape[:2]
        return filter_detections(lb.to_source(boxes), scores, classes, frame_w=w, frame_h=h, cfg=self._cfg,
                                 keep_class=r.person_class_id)
