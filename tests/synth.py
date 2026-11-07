"""Synthetic multi-player scenes with ground truth, for tracker tests.

A scene is a set of players with pitch-plane trajectories expressed in body heights and seconds, rendered to
image boxes at any fps through an optional per-frame camera transform. Generic: no team, kit or clip notion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from spana.config import SpanaConfig, load_config
from spana.detection.postprocess import nms
from spana.records import CameraObservation, CameraStatus, Detection
from spana.tracking.tracker import MultiPlayerTracker

from pathlib import Path

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config" / "default.yaml")


def cfg() -> SpanaConfig:
    return load_config(CONFIG_PATH)


@dataclass
class Player:
    gid: str
    start: Tuple[float, float]       # image px at t=0 (world == image when the camera is static)
    velocity: Tuple[float, float]    # px / s
    height: float = 80.0
    aspect: float = 0.4
    visible: Callable[[float], bool] = lambda t: True
    score: Callable[[float], float] = lambda t: 0.9
    appearance: Optional[np.ndarray] = None   # torso descriptor (normalised histogram); None = unknown

    def box_at(self, t: float) -> np.ndarray:
        cx = self.start[0] + self.velocity[0] * t
        cy = self.start[1] + self.velocity[1] * t
        w = self.height * self.aspect
        return np.array([cx - w / 2, cy - self.height / 2, cx + w / 2, cy + self.height / 2])


def kit(index: int, bins: int = 128, spread: int = 6, seed: int = 0) -> np.ndarray:
    """A generic synthetic descriptor: a smooth bump at a bin position. Same index = same-looking players."""
    x = np.arange(bins)
    v = np.exp(-0.5 * ((x - (index * 37) % bins) / spread) ** 2) + 1e-3
    if seed:
        v = v * (1 + 0.05 * np.random.default_rng(seed).standard_normal(bins)).clip(0.5)
    return v / v.sum()


@dataclass
class RunResult:
    frames: list
    gt_of_detection: List[List[Optional[str]]]          # per frame: detection index -> gid
    gt_boxes: List[Dict[str, np.ndarray]] = field(default_factory=list)  # per frame: visible gid -> box
    det_boxes: List[List[tuple]] = field(default_factory=list)          # per frame: detection boxes (after NMS)
    tracker: Optional[MultiPlayerTracker] = None


def run(players: Sequence[Player], *, fps: float, duration_s: float, frame_w: int = 1920, frame_h: int = 1080,
        camera: Optional[Callable[[int], CameraObservation]] = None,
        world_to_image: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
        config: Optional[SpanaConfig] = None, extra: Optional[Callable[[int, float], List[Detection]]] = None,
        use_appearance: bool = True) -> RunResult:
    c = config or cfg()
    trk = MultiPlayerTracker(c, frame_w, frame_h)
    n = int(round(duration_s * fps))
    res = RunResult([], [], tracker=trk)
    for i in range(n):
        t = i / fps
        dets: List[Detection] = []
        gids: List[Optional[str]] = []
        app: List[Optional[np.ndarray]] = []
        gtb: Dict[str, np.ndarray] = {}
        for p in players:
            b = p.box_at(t)
            if world_to_image is not None:
                b = world_to_image(i, b)
            if not p.visible(t):
                continue
            gtb[p.gid] = b
            dets.append(Detection(tuple(float(v) for v in b), float(p.score(t)), 0))
            gids.append(p.gid)
            app.append(p.appearance if use_appearance else None)
        if extra is not None:
            for d in extra(i, t):
                dets.append(d)
                gids.append(None)
                app.append(None)
        if dets:   # the real detector applies NMS; synthetic detections must not be cleaner or dirtier than that
            keep = sorted(nms(np.array([d.box for d in dets]), np.array([d.score for d in dets]),
                              c.detection.nms_iou))
            dets, gids, app = [dets[k] for k in keep], [gids[k] for k in keep], [app[k] for k in keep]
        cam = camera(i) if camera else (CameraObservation.identity(CameraStatus.FIRST) if i == 0
                                        else CameraObservation.identity(CameraStatus.OK, inliers=100))
        ft = trk.update(i, t, dets, cam, app)
        res.frames.append(ft)
        res.gt_of_detection.append(gids)
        res.gt_boxes.append(gtb)
        res.det_boxes.append([d.box for d in dets])
    return res
