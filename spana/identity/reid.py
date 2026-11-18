"""SPANA re-ID embedder: L2-normalised appearance embeddings of detection boxes (ONNX, CPU).

Used only as evidence for clearing IDENTITY_UNCERTAIN (spana.tracking.tracker, config `reid`), never to force an
association. The model record pins the ONNX file by sha256 and states the preprocessing used in training
(tools/reid_crops.cut_crop + tools/reid_train.to_input); a file with another hash is refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ReidModelError(RuntimeError):
    pass


def crop(img: np.ndarray, box: Sequence[float], pad: float, size_hw) -> Optional[np.ndarray]:
    """The training crop: padded box clipped to the image, resized to size_hw (None for an empty crop)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    xa, ya = int(max(0, round(x1 - pad * w))), int(max(0, round(y1 - pad * h)))
    xb, yb = int(min(W, round(x2 + pad * w))), int(min(H, round(y2 + pad * h)))
    c = img[ya:yb, xa:xb]
    if c.size == 0:
        return None
    return cv2.resize(c, (int(size_hw[1]), int(size_hw[0])),
                      interpolation=cv2.INTER_AREA if c.shape[0] > size_hw[0] else cv2.INTER_LINEAR)


class ReidEmbedder:
    def __init__(self, record_path: str | Path, expected_sha256: str, root: Path = REPO):
        rp = Path(record_path)
        rp = rp if rp.is_absolute() else root / rp
        if not rp.is_file():
            raise ReidModelError(f"re-ID model record not found: {rp}")
        rec = json.loads(rp.read_text())
        if rec.get("onnx_sha256") != expected_sha256:
            raise ReidModelError("re-ID model record does not name the calibrated model (config reid.model_sha256)")
        onnx = rp.parent / rec["onnx_path"]
        if not onnx.is_file():
            raise ReidModelError(f"re-ID ONNX not found: {onnx}")
        h = hashlib.sha256(onnx.read_bytes()).hexdigest()
        if h != expected_sha256:
            raise ReidModelError(f"re-ID ONNX sha256 {h} != {expected_sha256}")
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.size_hw, self.pad, self.sha256 = tuple(rec["input_hw"]), float(rec["crop_pad"]), h

    def embed(self, img_bgr: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[Optional[np.ndarray]]:
        crops = [crop(img_bgr, b, self.pad, self.size_hw) for b in boxes]
        ok = [k for k, c in enumerate(crops) if c is not None]
        out: List[Optional[np.ndarray]] = [None] * len(boxes)
        if not ok:
            return out
        x = np.stack([((crops[k][:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                      for k in ok]).astype(np.float32)
        y = np.concatenate([self.session.run(None, {self.input: x[j:j + 32]})[0] for j in range(0, len(x), 32)])
        y = y.astype(np.float64)
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        for k, v in zip(ok, y):
            out[k] = v
        return out
