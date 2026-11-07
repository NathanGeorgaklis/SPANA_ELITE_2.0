"""Synthetic multi-player scenes with ground truth, for tracker tests.

A scene is a set of players with pitch-plane trajectories expressed in body heights and seconds, rendered to
image boxes at any fps through an optional per-frame camera transform. Generic: no team, kit or clip notion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from spana.config import SpanaConfig, load_config
from spana.records import CameraObservation, CameraStatus, Detection
from spana.tracking.tracker import MultiPlayerTracker

CONFIG_PATH = "config/default.yaml"


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

    def box_at(self, t: float) -> np.ndarray:
        cx = self.start[0] + self.velocity[0] * t
        cy = self.start[1] + self.velocity[1] * t
        w = self.height * self.aspect
        return np.array([cx - w / 2, cy - self.height / 2, cx + w / 2, cy + self.height / 2])


@dataclass
class RunResult:
    frames: list
    gt_of_detection: List[List[Optional[str]]]          # per frame: detection index -> gid
    track_gids: Dict[str, Set[str]] = field(default_factory=dict)   # track_id -> gids it was OBSERVED on
    gid_tracks: Dict[str, List[str]] = field(default_factory=dict)  # gid -> ordered distinct track ids

    @property
    def false_continuity(self) -> Dict[str, Set[str]]:
        """Tracks observed on more than one ground-truth player (identity switch through the track)."""
        return {k: v for k, v in self.track_gids.items() if len(v) > 1}


def run(players: Sequence[Player], *, fps: float, duration_s: float, frame_w: int = 1920, frame_h: int = 1080,
        camera: Optional[Callable[[int], CameraObservation]] = None,
        world_to_image: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
        config: Optional[SpanaConfig] = None, extra: Optional[Callable[[int, float], List[Detection]]] = None
        ) -> RunResult:
    c = config or cfg()
    trk = MultiPlayerTracker(c, frame_w, frame_h)
    n = int(round(duration_s * fps))
    res = RunResult([], [])
    for i in range(n):
        t = i / fps
        dets: List[Detection] = []
        gids: List[Optional[str]] = []
        for p in players:
            if not p.visible(t):
                continue
            b = p.box_at(t)
            if world_to_image is not None:
                b = world_to_image(i, b)
            dets.append(Detection(tuple(float(v) for v in b), float(p.score(t)), 0))
            gids.append(p.gid)
        if extra is not None:
            for d in extra(i, t):
                dets.append(d)
                gids.append(None)
        cam = camera(i) if camera else (CameraObservation.identity(CameraStatus.FIRST) if i == 0
                                        else CameraObservation.identity(CameraStatus.OK, inliers=100))
        ft = trk.update(i, t, dets, cam)
        res.frames.append(ft)
        res.gt_of_detection.append(gids)
        for o in ft.tracks:
            if o.detection_index is not None and gids[o.detection_index] is not None:
                g = gids[o.detection_index]
                res.track_gids.setdefault(o.track_id, set()).add(g)
                seq = res.gid_tracks.setdefault(g, [])
                if not seq or seq[-1] != o.track_id:
                    seq.append(o.track_id)
    return res
