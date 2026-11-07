"""ONNX Runtime detector backend driven by a pinned model record.

A model record (JSON) is mandatory. The weights file is hash-verified before loading; any missing field,
unsupported format or hash mismatch raises ModelRecordError (fail closed, no substitute model).

Supported output formats (the record says which; nothing is inferred from tensor shapes):
  cxcywh_class_scores : per candidate cx, cy, w, h, C class scores (already in input pixels); no objectness.
  yolox_grid_obj_cls  : YOLOX raw head output (1, N, 5+C): grid offsets / log-sizes per stride, sigmoid
                        objectness and class scores. Decoded as in YOLOX demo_postprocess; score = obj * cls.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import DetectionConfig
from ..records import Detection
from ..video import sha256_file
from .postprocess import letterbox
from .tiling import detect_tiled

SUPPORTED_OUTPUT_FORMATS = ("cxcywh_class_scores", "yolox_grid_obj_cls")
SUPPORTED_OUTPUT_LAYOUTS = ("channels_first", "channels_last")  # (1, K, N) / (1, N, K)
SUPPORTED_COLOR_ORDERS = ("RGB", "BGR")
SUPPORTED_ANCHORS = ("center", "top_left")


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
    letterbox_anchor: str      # where the resized image sits on the padded canvas
    output_format: str
    output_layout: str
    strides: List[int]         # feature-map strides (grid formats only; empty otherwise)
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
        checks = [(rec.output_format in SUPPORTED_OUTPUT_FORMATS, f"output_format {rec.output_format!r}"),
                  (rec.output_layout in SUPPORTED_OUTPUT_LAYOUTS, f"output_layout {rec.output_layout!r}"),
                  (rec.color_order in SUPPORTED_COLOR_ORDERS, f"color_order {rec.color_order!r}"),
                  (rec.letterbox_anchor in SUPPORTED_ANCHORS, f"letterbox_anchor {rec.letterbox_anchor!r}"),
                  ((rec.output_format == "yolox_grid_obj_cls") == bool(rec.strides), f"strides {rec.strides!r}"),
                  (all(isinstance(s, int) and s > 0 and rec.input_size % s == 0 for s in rec.strides),
                   f"strides {rec.strides!r} vs input_size {rec.input_size}")]
        for ok, what in checks:
            if not ok:
                raise ModelRecordError(f"unsupported or inconsistent {what}")
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


def _candidates(raw: np.ndarray, layout: str) -> np.ndarray:
    a = np.asarray(raw, dtype=float)
    if a.ndim != 3 or a.shape[0] != 1:
        raise ModelRecordError(f"unexpected output shape {a.shape}")
    if layout not in SUPPORTED_OUTPUT_LAYOUTS:
        raise ModelRecordError(f"unsupported output_layout {layout!r}")
    return a[0].T if layout == "channels_first" else a[0].copy()


def _xyxy(cx, cy, w, h) -> np.ndarray:
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def decode_cxcywh_class_scores(raw: np.ndarray, layout: str):
    """Return (boxes_xyxy, best_score, best_class) in model-input pixels."""
    a = _candidates(raw, layout)
    if a.shape[1] <= 4:
        raise ModelRecordError(f"output has no class scores: {a.shape}")
    cls = a[:, 4:]
    best = cls.argmax(axis=1)
    return _xyxy(a[:, 0], a[:, 1], a[:, 2], a[:, 3]), cls[np.arange(len(cls)), best], best


def decode_yolox(raw: np.ndarray, layout: str, input_size: int, strides: List[int]):
    a = _candidates(raw, layout)
    grids, mult = [], []
    for s in strides:
        n = input_size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack([xv, yv], 2).reshape(-1, 2))
        mult.append(np.full((n * n, 1), s))
    grid, stride = np.concatenate(grids), np.concatenate(mult).astype(float)
    if a.shape[0] != len(grid) or a.shape[1] <= 5:
        raise ModelRecordError(f"YOLOX output {a.shape} does not match {len(grid)} grid cells")
    xy = (a[:, :2] + grid) * stride
    wh = np.exp(np.clip(a[:, 2:4], None, 20.0)) * stride
    cls = a[:, 5:] * a[:, 4:5]
    best = cls.argmax(axis=1)
    return _xyxy(xy[:, 0], xy[:, 1], wh[:, 0], wh[:, 1]), cls[np.arange(len(cls)), best], best


class OnnxPersonDetector:
    def __init__(self, record: ModelRecord, cfg: DetectionConfig, providers: List[str] | None = None):
        import onnxruntime as ort  # imported here so the rest of SPANA does not depend on it

        self._rec = record
        self._cfg = cfg
        opts = ort.SessionOptions()
        self._sess = ort.InferenceSession(record.weights_path, sess_options=opts,
                                          providers=providers or ["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    @property
    def model_id(self) -> dict:
        d = asdict(self._rec)
        d["weights_path"] = Path(d["weights_path"]).name
        return d

    def infer(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Raw person candidates for one image: (boxes in image pixels, scores). No NMS, no size filter."""
        r = self._rec
        img, lb = letterbox(image_bgr, r.input_size, r.pad_value, r.letterbox_anchor)
        if r.color_order == "RGB":
            img = img[:, :, ::-1]
        x = img.astype(np.float32)
        if r.scale_to_unit:
            x /= 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        raw = self._sess.run(None, {self._input: x})[0]
        if r.output_format == "yolox_grid_obj_cls":
            boxes, scores, classes = decode_yolox(raw, r.output_layout, r.input_size, r.strides)
        else:
            boxes, scores, classes = decode_cxcywh_class_scores(raw, r.output_layout)
        keep = (classes == r.person_class_id) & (scores >= self._cfg.score_low)
        return lb.to_source(boxes[keep]), scores[keep]

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return detect_tiled(self.infer, frame_bgr, self._cfg, self._rec.person_class_id)
