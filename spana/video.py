"""Video input. Fails closed on unreadable files or invalid frame rates."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMeta:
    path: str
    sha256: str
    fps: float
    width: int
    height: int
    frame_count_reported: int     # container metadata; may be inaccurate
    timestamp_source: str         # how frame times are derived

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class VideoSource:
    """Sequential frame reader. Frame time = index / fps (constant-frame-rate assumption, recorded in meta)."""

    def __init__(self, path: str | Path, expected_sha256: Optional[str] = None):
        p = Path(path)
        if not p.is_file():
            raise VideoError(f"video not found: {p}")
        digest = sha256_file(p)
        if expected_sha256 is not None and digest != expected_sha256:
            raise VideoError(f"video sha256 mismatch: {digest} != {expected_sha256}")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise VideoError(f"cannot open video: {p}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not (math.isfinite(fps) and fps > 0):
            raise VideoError(f"invalid fps {fps!r} in {p}")
        if w <= 0 or h <= 0:
            raise VideoError(f"invalid frame size {w}x{h} in {p}")
        self.meta = VideoMeta(str(p), digest, fps, w, h, n, "index_over_fps")

    def frames(self, start: int = 0, stop: Optional[int] = None) -> Iterator[Tuple[int, float, np.ndarray]]:
        cap = cv2.VideoCapture(self.meta.path)
        try:
            if start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            idx = start
            while stop is None or idx < stop:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != self.meta.width or frame.shape[0] != self.meta.height:
                    raise VideoError(f"frame {idx} size {frame.shape[1]}x{frame.shape[0]} differs from metadata")
                yield idx, idx / self.meta.fps, frame
                idx += 1
        finally:
            cap.release()
