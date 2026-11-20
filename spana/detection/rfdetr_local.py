"""Local RF-DETR detector: the trained football detector run on this machine (no hosted calls, frames stay here).

Model record (`models/rfdetr_v1.json`): pinned weights (PyTorch checkpoint) and their ONNX export, both sha256-checked
before loading; input resolution; `output_classes` maps each model output index to a configured class name ("" =
an output the model never uses). Two interchangeable runners give the same raw outputs (validated within 3e-4 logits,
records/local_inference_2026-11-16/RECORD.md):
  * "torch": rfdetr + PyTorch on `device` ("mps" = Apple GPU, "cpu"); optional dependencies
    (requirements-local-detector.txt), imported only when this runner is chosen. On the GPU, config `inference`
    may run it in fp16 and / or torch.compile it (records/identity_speed_2026-11-18); the CPU stays fp32, eager;
  * "onnx": ONNX Runtime on CPU, using only the engine's pinned dependencies.
Pre-processing: BGR -> RGB, plain resize to resolution x resolution (bilinear), / 255, ImageNet mean / std.
Post-processing: sigmoid over (queries x outputs), the top `queries` scores, unused outputs dropped, boxes (normalised
cx, cy, w, h) scaled to source pixels and clipped to the frame, score >= the requested confidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..profiling import StageTimer
from ..records import Detection

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
VARIANTS = ("nano", "small", "medium", "base", "large")
RUNTIMES = ("torch", "onnx")


class LocalModelError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class LocalModelRecord:
    name: str
    source: str
    version: str
    licence: str
    variant: str
    num_classes: int
    resolution: int
    weights_path: str          # relative to the record file
    weights_sha256: str
    onnx_path: str             # relative to the record file
    onnx_sha256: str
    output_classes: Tuple[str, ...]   # model output index -> configured class name, "" = unused
    validation: str
    base_dir: str = ""

    @staticmethod
    def load(path) -> "LocalModelRecord":
        p = Path(path)
        if not p.is_file():
            raise LocalModelError(f"model record not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        names = [f.name for f in fields(LocalModelRecord) if f.name != "base_dir"]
        missing, unknown = [n for n in names if n not in data], [k for k in data if k not in names]
        if missing or unknown:
            raise LocalModelError(f"model record {p}: missing {missing}, unknown {unknown}")
        rec = LocalModelRecord(**{**data, "output_classes": tuple(data["output_classes"]),
                                  "base_dir": str(p.parent.resolve())})
        if rec.variant not in VARIANTS or rec.resolution <= 0 or rec.num_classes <= 0:
            raise LocalModelError(f"model record {p}: bad variant / resolution / num_classes")
        if len(rec.output_classes) != rec.num_classes + 1:
            raise LocalModelError(f"model record {p}: output_classes must have num_classes + 1 entries")
        used = [c for c in rec.output_classes if c]
        if len(used) != rec.num_classes or len(set(used)) != len(used):
            raise LocalModelError(f"model record {p}: output_classes must name each class once")
        if any(len(s) != 64 for s in (rec.weights_sha256, rec.onnx_sha256)) or not rec.licence.strip():
            raise LocalModelError(f"model record {p}: needs a licence and sha256 for weights and ONNX")
        return rec

    def file(self, which: str) -> Path:
        """Verified path of the 'weights' or 'onnx' file. Fails closed on a missing file or a hash mismatch."""
        rel, sha = (self.weights_path, self.weights_sha256) if which == "weights" else (self.onnx_path, self.onnx_sha256)
        f = Path(self.base_dir) / rel
        if not f.is_file():
            raise LocalModelError(f"{which} file not found: {f} (download / export it; see the model record)")
        got = _sha256(f)
        if got != sha:
            raise LocalModelError(f"{which} file {f.name}: sha256 {got[:12]}… != pinned {sha[:12]}…")
        return f


def preprocess(frame_bgr: np.ndarray, resolution: int, height: Optional[int] = None) -> np.ndarray:
    """(1, 3, H, R) float32 network input: RGB, resized to R wide x H high (H = R unless `height` is given),
    ((u8 / 255) - MEAN) / STD per channel.

    The same float32 operations per element as the plain formula `((x / 255) - MEAN) / STD` on an HWC array,
    done per channel plane in place: bit-identical output, without the length-3 broadcast inner loop and the
    full-size temporaries (records/identity_speed_2026-11-20: 23 -> 4.5 ms per 1280 frame)."""
    h = resolution if height is None else height
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    u = cv2.resize(rgb, (resolution, h), interpolation=cv2.INTER_LINEAR)
    out = np.empty((1, 3, h, resolution), np.float32)
    for c in range(3):
        p = out[0, c]
        np.divide(u[:, :, c], np.float32(255.0), out=p, dtype=np.float32)
        np.subtract(p, MEAN[c], out=p)
        np.divide(p, STD[c], out=p)
    return out


def decode(boxes: np.ndarray, logits: np.ndarray, width: int, height: int, floor: float,
           output_to_class_id: Sequence[int]) -> List[Detection]:
    """boxes (Q, 4) normalised cx, cy, w, h; logits (Q, K). output_to_class_id[k] = class id or -1 (unused)."""
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    flat = prob.reshape(-1)
    k = prob.shape[1]
    out = []
    for t in np.argsort(-flat, kind="stable")[: prob.shape[0]]:
        score = float(flat[t])
        if score < floor:
            break
        q, c = divmod(int(t), k)
        cid = output_to_class_id[c]
        if cid < 0:
            continue
        cx, cy, bw, bh = (float(v) for v in boxes[q])
        x1, y1 = max(0.0, (cx - bw / 2) * width), max(0.0, (cy - bh / 2) * height)
        x2, y2 = min(float(width), (cx + bw / 2) * width), min(float(height), (cy + bh / 2) * height)
        if x2 > x1 and y2 > y1:
            out.append(Detection((x1, y1, x2, y2), score, cid))
    out.sort(key=lambda d: (d.class_id, -d.score, d.box))
    return out


class OnnxRunner:
    def __init__(self, record: LocalModelRecord):
        import onnxruntime as ort
        self._s = ort.InferenceSession(str(record.file("onnx")), providers=["CPUExecutionProvider"])
        self._in = self._s.get_inputs()[0].name
        self.describe = {"runtime": "onnx", "device": "cpu", "onnxruntime": ort.__version__}

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        boxes, logits = self._s.run(None, {self._in: x})
        return boxes[0], logits[0]


class TorchRunner:
    """precision / compile apply on the GPU (mps) only; on the CPU the network runs fp32, eager (unchanged)."""

    def __init__(self, record: LocalModelRecord, device: str, precision: str = "fp32", compile: bool = False):
        try:
            import rfdetr
            import torch
        except ImportError as e:
            raise LocalModelError(f"runtime 'torch' needs requirements-local-detector.txt installed ({e})") from None
        if device == "mps" and not torch.backends.mps.is_available():
            raise LocalModelError("device 'mps' requested but the Apple GPU backend is not available")
        cls = getattr(rfdetr, f"RFDETR{record.variant.capitalize()}")
        model = cls(pretrain_weights=str(record.file("weights")), num_classes=record.num_classes,
                    resolution=record.resolution, device=device)
        if device != "mps":
            precision, compile = "fp32", False
        if precision not in ("fp32", "fp16"):
            raise LocalModelError(f"precision must be fp32 or fp16, got {precision!r}")
        self._torch, self._dev = torch, device
        self._dtype = {"fp32": torch.float32, "fp16": torch.float16}[precision]
        self._net = model.model.model.eval().to(device=device, dtype=self._dtype)
        self._fn = torch.compile(self._net) if compile else self._net
        self.describe = {"runtime": "torch", "device": device, "torch": torch.__version__,
                         "rfdetr": getattr(rfdetr, "__version__", "1.12.1"), "precision": precision,
                         "compiled": bool(compile), "batch_frames": 1}

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with self._torch.no_grad():
            out = self._fn(self._torch.from_numpy(x).to(device=self._dev, dtype=self._dtype))
        return out["pred_boxes"][0].float().cpu().numpy(), out["pred_logits"][0].float().cpu().numpy()


class LocalRFDETRDetector:
    """`Detector`: class_ids index `class_names` (the configured class list)."""

    def __init__(self, record: LocalModelRecord, runner, class_names: Sequence[str], confidence: float):
        unknown = [c for c in record.output_classes if c and c not in class_names]
        if unknown:
            raise LocalModelError(f"model outputs {unknown} are not configured classes {list(class_names)}")
        self._rec, self._run, self._conf = record, runner, confidence
        self._names = tuple(class_names)
        self._map = [self._names.index(c) if c else -1 for c in record.output_classes]
        self.timer = StageTimer(False)   # spana.profiling; a no-op unless the caller sets an enabled timer

    @property
    def class_names(self) -> tuple:
        return self._names

    @property
    def model_id(self) -> dict:
        return {"backend": "rfdetr_local", "name": self._rec.name, "version": self._rec.version,
                "weights_sha256": self._rec.weights_sha256, "onnx_sha256": self._rec.onnx_sha256,
                "resolution": self._rec.resolution, "confidence": self._conf, **self._run.describe}

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        with self.timer.stage("detect.preprocess"):
            x = preprocess(frame_bgr, self._rec.resolution)
        with self.timer.stage("detect.model"):       # ends with the host copy of the outputs (GPU sync)
            boxes, logits = self._run(x)
        with self.timer.stage("detect.decode"):
            return decode(boxes, logits, w, h, self._conf, self._map)


def make_runner(record: LocalModelRecord, runtime: str, device: str, precision: str = "fp32", compile: bool = False):
    """precision / compile: the GPU settings (config `inference`); ignored on the CPU and by the onnx runtime."""
    if runtime not in RUNTIMES:
        raise LocalModelError(f"runtime must be one of {RUNTIMES}")
    if runtime == "onnx":
        if device != "cpu":
            raise LocalModelError("runtime 'onnx' runs on device 'cpu' only (CoreML fails on this model; RECORD)")
        return OnnxRunner(record)
    return TorchRunner(record, device, precision, compile)
