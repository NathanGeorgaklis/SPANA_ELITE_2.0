"""Deterministic frame sampling for dataset building.

Two passes over a video:
  1. `scan_signals`: at every `analysis_step_s`, a motion score (mean absolute grey-level change against the previous
     analysis frame, 0..1) and an HSV-histogram distance (shot / camera cut) on a compute-only downscale.
  2. `select_frames` (pure): regular frames every `interval_s`, plus extra frames where motion is heavy or a cut starts,
     at least `event_min_gap_s` apart. If more than `max_frames_per_video` are chosen, they are thinned evenly.

No randomness and nothing specific to a match: the same video and config always give the same frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import cv2
import numpy as np

from .config import SamplingSection

_EPS = 1e-9


@dataclass(frozen=True)
class FrameSignal:
    frame_index: int
    timestamp_s: float
    motion: float        # 0..1; 0 on the first analysis frame
    hist_distance: float  # 0..1; 0 on the first analysis frame


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_s: float
    reasons: Tuple[str, ...]   # subset of ("interval", "motion", "cut"), sorted

    def to_dict(self) -> dict:
        return {"frame_index": self.frame_index, "timestamp_s": round(self.timestamp_s, 6),
                "reasons": list(self.reasons)}


def _small(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= width:
        return frame_bgr
    return cv2.resize(frame_bgr, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def _hs_hist(small_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).astype(np.float32)


def scan_signals(frames: Iterable[Tuple[int, float, np.ndarray]], cfg: SamplingSection) -> List[FrameSignal]:
    """Motion / cut signals on analysis frames (the first frame and then every analysis_step_s of frame time)."""
    out: List[FrameSignal] = []
    next_t = None
    prev_grey = prev_hist = None
    for idx, t, frame in frames:
        if next_t is not None and t + _EPS < next_t:
            continue
        next_t = t + cfg.analysis_step_s
        small = _small(frame, cfg.analysis_width_px)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hist = _hs_hist(small)
        if prev_grey is None:
            motion, dist = 0.0, 0.0
        else:
            motion = float(np.mean(np.abs(grey - prev_grey)))
            dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        out.append(FrameSignal(int(idx), float(t), motion, dist))
        prev_grey, prev_hist = grey, hist
    return out


def _thin_evenly(items: Sequence[SampledFrame], cap: int) -> List[SampledFrame]:
    n = len(items)
    if n <= cap:
        return list(items)
    keep = sorted({int(round(i * (n - 1) / (cap - 1))) for i in range(cap)}) if cap > 1 else [0]
    return [items[i] for i in keep]


def select_frames(signals: Sequence[FrameSignal], cfg: SamplingSection) -> List[SampledFrame]:
    """Pure, deterministic selection from analysis-frame signals (see module docstring)."""
    reasons: dict = {}
    times: dict = {}
    next_interval_t = None
    last_event_t = None
    for s in sorted(signals, key=lambda x: x.frame_index):
        r = set()
        if next_interval_t is None or s.timestamp_s + _EPS >= next_interval_t:
            r.add("interval")
            base = s.timestamp_s if next_interval_t is None else next_interval_t
            next_interval_t = base + cfg.interval_s
            while next_interval_t <= s.timestamp_s + _EPS:   # long gaps (dropped frames) do not cause bursts
                next_interval_t += cfg.interval_s
        ev = set()
        if s.hist_distance > cfg.cut_hist_distance:
            ev.add("cut")
        if s.motion > cfg.motion_threshold:
            ev.add("motion")
        if ev and (last_event_t is None or s.timestamp_s - last_event_t + _EPS >= cfg.event_min_gap_s):
            r |= ev
            last_event_t = s.timestamp_s
        if r:
            reasons[s.frame_index] = r
            times[s.frame_index] = s.timestamp_s
    chosen = [SampledFrame(i, times[i], tuple(sorted(reasons[i]))) for i in sorted(reasons)]
    return _thin_evenly(chosen, cfg.max_frames_per_video)


def extract_frames(frames: Iterable[Tuple[int, float, np.ndarray]],
                   selected: Sequence[SampledFrame]) -> Iterator[Tuple[SampledFrame, np.ndarray]]:
    """Second pass: yield the selected frames' pixels (stops after the last selected frame)."""
    wanted = {s.frame_index: s for s in selected}
    last = max(wanted) if wanted else -1
    for idx, _t, frame in frames:
        if idx in wanted:
            yield wanted[idx], frame
        if idx >= last:
            break
