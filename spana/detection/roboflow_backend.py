"""Roboflow detector backend (inference-server HTTP API).

Outputs SPANA `Detection` records only (box in source pixels, score, class index into the configured class list);
no vendor objects and no football logic leave this module. Model id, thresholds and the label -> class map come from
config/roboflow.yaml.

Transport: POST {api_url}/infer/object_detection with a JSON body (the key travels in the body, never in the URL,
so it cannot end up in access logs or error messages). The same API is served by Roboflow's hosted endpoint and by a
local inference server. `inference-sdk` is not used: it pins numpy<2.4 and the non-headless opencv-python, which
conflicts with the engine's pinned numpy / OpenCV (see records/roboflow_2026-11-12/RECORD.md).
"""
from __future__ import annotations

import base64
from collections import Counter
from typing import Dict, List, Optional, Protocol, Sequence

import cv2
import numpy as np

from ..records import Detection
from ..secrets import redact


class RoboflowError(RuntimeError):
    pass


class InferenceClient(Protocol):
    def infer(self, image_jpeg: bytes, *, model_id: str, confidence: float, iou: float) -> dict:
        """Raw response: {"image": {"width", "height"}, "predictions": [{"x","y","width","height","confidence","class"}]}"""


class HttpInferenceClient:
    def __init__(self, api_url: str, api_key: str, timeout_s: float, session=None):
        import requests  # local import: tests inject a fake session or a fake client
        self._url = api_url.rstrip("/") + "/infer/object_detection"
        self._key = api_key
        self._timeout = timeout_s
        self._session = session if session is not None else requests.Session()

    def infer(self, image_jpeg: bytes, *, model_id: str, confidence: float, iou: float) -> dict:
        body = {"model_id": model_id, "api_key": self._key, "confidence": confidence, "iou_threshold": iou,
                "image": {"type": "base64", "value": base64.b64encode(image_jpeg).decode("ascii")}}
        try:
            r = self._session.post(self._url, json=body, timeout=self._timeout)
        except Exception as e:  # network errors may echo request details: never let the key through
            raise RoboflowError(redact(f"inference request failed: {type(e).__name__}: {e}", self._key)) from None
        if r.status_code != 200:
            raise RoboflowError(redact(f"inference HTTP {r.status_code}: {r.text[:300]}", self._key))
        try:
            return r.json()
        except ValueError:
            raise RoboflowError("inference response is not JSON") from None


class RoboflowDetector:
    """`Detector` implementation. Fails closed on malformed responses or a size mismatch."""

    def __init__(self, *, client: InferenceClient, model_id: str, class_names: Sequence[str],
                 class_map: Dict[str, str], confidence: float, iou: float, jpeg_quality: int = 95):
        unknown = set(class_map.values()) - set(class_names)
        if unknown:
            raise RoboflowError(f"class_map targets {sorted(unknown)} are not in the class list")
        self._client = client
        self._model_id = model_id
        self._class_names = tuple(class_names)
        self._label_to_id = {lbl: self._class_names.index(c) for lbl, c in class_map.items()}
        self._confidence = confidence
        self._iou = iou
        self._jpeg_quality = jpeg_quality
        self.dropped_labels: Counter = Counter()   # model labels outside class_map (reported, not silently lost)

    @classmethod
    def from_config(cls, cfg, api_key: str, client: Optional[InferenceClient] = None) -> "RoboflowDetector":
        """`cfg` is a spana.dataset.config.DatasetConfig."""
        b = cfg.baseline
        client = client or HttpInferenceClient(b.api_url, api_key, b.timeout_s)
        return cls(client=client, model_id=b.model_id, class_names=cfg.classes, class_map=cfg.class_map(),
                   confidence=b.confidence, iou=b.iou, jpeg_quality=cfg.sampling.jpeg_quality)

    @property
    def model_id(self) -> dict:
        return {"backend": "roboflow", "model_id": self._model_id, "confidence": self._confidence, "iou": self._iou}

    @property
    def class_names(self) -> tuple:
        return self._class_names

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            raise RoboflowError("JPEG encoding failed")
        resp = self._client.infer(buf.tobytes(), model_id=self._model_id, confidence=self._confidence, iou=self._iou)
        return self.parse(resp, w, h)

    def parse(self, resp: dict, width: int, height: int) -> List[Detection]:
        if not isinstance(resp, dict) or not isinstance(resp.get("predictions"), list):
            raise RoboflowError("malformed response: no predictions list")
        img = resp.get("image") or {}
        if (img.get("width"), img.get("height")) != (width, height):
            raise RoboflowError(f"response image size {img.get('width')}x{img.get('height')} != frame {width}x{height}")
        out: List[Detection] = []
        for p in resp["predictions"]:
            try:
                x, y, bw, bh, score = (float(p[k]) for k in ("x", "y", "width", "height", "confidence"))
                label = str(p["class"])
            except (KeyError, TypeError, ValueError):
                raise RoboflowError(f"malformed prediction: {p!r}") from None
            if not all(np.isfinite([x, y, bw, bh, score])):
                raise RoboflowError(f"non-finite prediction: {p!r}")
            if label not in self._label_to_id:
                self.dropped_labels[label] += 1
                continue
            if score < self._confidence:
                continue
            x1, y1 = max(0.0, x - bw / 2), max(0.0, y - bh / 2)
            x2, y2 = min(float(width), x + bw / 2), min(float(height), y + bh / 2)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection((x1, y1, x2, y2), score, self._label_to_id[label]))
        out.sort(key=lambda d: (d.class_id, -d.score, d.box))
        return out
