"""Shared pieces for the dataset-preparation tools: clients, verified image cache, deterministic ordering, tag tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DatasetConfig
from .rf_project import ProjectImage, RoboflowApiError, RoboflowProject


class SkipFrame(Exception):
    """A frame that cannot be processed safely; the reason is logged and the run continues."""


def open_project(cfg: DatasetConfig, api_key: str) -> RoboflowProject:
    r = cfg.roboflow
    return RoboflowProject(api_url=r.api_url, workspace=r.workspace, project=r.project, api_key=api_key,
                           timeout_s=cfg.audit.fetch_timeout_s)


def match_tag(img: ProjectImage) -> Optional[str]:
    m = [t[len("match:"):] for t in img.tags if t.startswith("match:")]
    return m[0] if len(m) == 1 else None


def sort_key(img: ProjectImage) -> Tuple:
    """(match, frame_index, name, id): stable order for selection and reports."""
    fi = img.metadata.get("frame_index")
    return (match_tag(img) or "", fi if isinstance(fi, int) else -1, img.name, img.id)


def evenly_spaced(items: Sequence, k: int) -> list:
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 0:
        return []
    if k == 1:
        return [items[0]]
    return [items[i] for i in sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})]


def load_image(project: RoboflowProject, img: ProjectImage, cache_dir: Path) -> np.ndarray:
    """Original pixels of a project image, verified against the sha256 recorded at upload. Raises SkipFrame."""
    sha = img.metadata.get("image_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise SkipFrame("no image_sha256 recorded at upload (cannot verify the bytes)")
    if not img.url:
        raise SkipFrame("no image url")
    cached = cache_dir / f"{img.id}.jpg"
    data = None
    if cached.is_file():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            data = None
    if data is None:
        try:
            data = project.fetch_image(img.url, sha)
        except RoboflowApiError as e:
            raise SkipFrame(f"image fetch failed: {e}") from None
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(cached)
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise SkipFrame("image does not decode")
    if img.width and img.height and (frame.shape[1], frame.shape[0]) != (img.width, img.height):
        raise SkipFrame(f"decoded size {frame.shape[1]}x{frame.shape[0]} != recorded {img.width}x{img.height}")
    return frame
