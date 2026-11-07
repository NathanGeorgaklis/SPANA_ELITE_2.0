"""Generic multi-player tracker (all people, no target knowledge, no football semantics).

Per frame:
  1. CUT or UNKNOWN camera motion -> every live track ends (CAMERA_CUT / CAMERA_MOTION_UNKNOWN) and a new
     continuity epoch begins. Without a known inter-frame transform, a neighbour can land on a track's predicted
     position and silently inherit it, so motion continuity is not carried across such a frame. Re-linking across
     epochs is the identity layer's job (appearance), not this layer's.
  2. OK camera motion -> every track state is mapped through the inter-frame transform.
  3. Motion prediction to the frame time (seconds). Beyond predict_max_s the velocity is frozen (HELD).
  4. Association, three Hungarian stages:
        S1 high-score detections  <-> CONFIRMED + LOST tracks (centre gate in body heights, height ratio, IoU/dist cost)
        S2 low-score detections   <-> CONFIRMED tracks observed on the previous frame (IoU only, strict)
        S3 remaining high-score   <-> TENTATIVE tracks
     A match is CONTESTED when another candidate (track for the detection, or detection for the track) is gated
     and within contest_margin of the chosen cost. Contested matches are kept but flagged for downstream layers.
  5. Lifecycle: TENTATIVE -> CONFIRMED (observed span >= confirm_s), CONFIRMED -> LOST when unobserved,
     LOST -> REMOVED after lost_max_s, TENTATIVE -> REMOVED after tentative_max_miss_s, any unobserved track whose
     prediction leaves the image -> REMOVED (LEFT_FRAME).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import SpanaConfig
from ..geometry import box_to_cxcywh, cxcywh_to_box, iou_matrix
from ..records import (CameraObservation, CameraStatus, Detection, EndReason, FrameTracks, PositionSource,
                       TrackObservation, TrackState)
from .kalman import BoxKalman

_INVALID = 1e6


@dataclass
class _Track:
    track_id: str
    kf: BoxKalman
    state: TrackState
    epoch: int
    first_observed_t: float
    last_observed_t: float
    last_update_t: float
    observed_frames: int
    observed_last_frame: bool
    velocity_frozen: bool = False

    def box(self) -> np.ndarray:
        return cxcywh_to_box(self.kf.x)


class MultiPlayerTracker:
    def __init__(self, cfg: SpanaConfig, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.tc = cfg.tracking
        self.frame_w, self.frame_h = frame_w, frame_h
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._epoch = 0
        self._last_frame: Optional[int] = None

    # ------------------------------------------------------------------ public
    def update(self, frame_index: int, time_s: float, detections: Sequence[Detection],
               camera: CameraObservation) -> FrameTracks:
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise ValueError(f"frames must be strictly increasing ({frame_index} after {self._last_frame})")
        self._last_frame = frame_index
        out = FrameTracks(frame_index, time_s, camera)
        out.decisions.append({"kind": "camera", "status": camera.status.value, "inliers": camera.inliers,
                              "hist_distance": camera.hist_distance})

        if camera.status in (CameraStatus.CUT, CameraStatus.UNKNOWN):
            reason = EndReason.CAMERA_CUT if camera.status == CameraStatus.CUT else EndReason.CAMERA_MOTION_UNKNOWN
            for t in self._tracks:
                out.ended.append((t.track_id, reason))
                out.decisions.append({"kind": "end", "track_id": t.track_id, "reason": reason.value})
            self._tracks = []
            self._epoch += 1
        elif camera.status == CameraStatus.OK:
            for t in self._tracks:
                t.kf.apply_camera(camera.matrix())

        for t in self._tracks:
            t.kf.predict(time_s - t.last_update_t)
            t.last_update_t = time_s
            if not t.velocity_frozen and time_s - t.last_observed_t > self.tc.predict_max_s:
                t.kf.freeze_velocity()
                t.velocity_frozen = True

        dets = list(detections)
        high = [i for i, d in enumerate(dets) if d.score >= self.cfg.detection.score_high]
        low = [i for i, d in enumerate(dets) if d.score < self.cfg.detection.score_high]
        matched: Dict[int, Tuple[int, bool]] = {}  # track list index -> (detection index, contested)

        s1_tracks = [k for k, t in enumerate(self._tracks) if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        rem_high = self._stage("S1", s1_tracks, high, dets, time_s, matched, out, iou_only=False)
        s2_tracks = [k for k in s1_tracks if k not in matched and self._tracks[k].observed_last_frame
                     and self._tracks[k].state == TrackState.CONFIRMED]
        self._stage("S2", s2_tracks, low, dets, time_s, matched, out, iou_only=True)
        s3_tracks = [k for k, t in enumerate(self._tracks) if t.state == TrackState.TENTATIVE]
        rem_high = self._stage("S3", s3_tracks, rem_high, dets, time_s, matched, out, iou_only=False)

        self._apply_matches(matched, dets, time_s)
        self._lifecycle(matched, time_s, out)
        self._spawn(rem_high, dets, matched, time_s, out)
        out.tracks = self._emit(matched, dets, time_s)
        return out

    # ----------------------------------------------------------- association
    def _gate_radius_h(self, t: _Track, time_s: float) -> float:
        unobserved = max(0.0, time_s - t.last_observed_t)
        return min(self.tc.gate_max_h, self.tc.gate_base_h + self.tc.max_speed_h_per_s * unobserved)

    def _cost_matrix(self, tracks: Sequence[int], det_idx: Sequence[int], dets: Sequence[Detection],
                     time_s: float, iou_only: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tb = np.array([self._tracks[k].box() for k in tracks]).reshape(-1, 4)
        db = np.array([dets[i].as_array() for i in det_idx]).reshape(-1, 4)
        ious = iou_matrix(tb, db)
        cost = np.full((len(tracks), len(det_idx)), _INVALID)
        dist_h = np.full_like(cost, np.inf)
        for a, k in enumerate(tracks):
            t = self._tracks[k]
            ph = t.kf.height
            pc = t.kf.x[:2]
            radius = self._gate_radius_h(t, time_s)
            for b, i in enumerate(det_idx):
                d = dets[i]
                if iou_only:
                    if ious[a, b] >= self.tc.low_score_min_iou:
                        cost[a, b] = 1.0 - ious[a, b]
                    continue
                ratio = max(d.height, 1.0) / ph
                if not (1.0 / self.tc.max_height_ratio <= ratio <= self.tc.max_height_ratio):
                    continue
                dc = box_to_cxcywh(d.as_array())[:2]
                dh = float(np.hypot(*(dc - pc))) / ph
                dist_h[a, b] = dh
                if dh > radius:  # implausible centre displacement for the unobserved time
                    continue
                w = self.tc.iou_weight
                cost[a, b] = w * (1.0 - ious[a, b]) + (1.0 - w) * (dh / radius)
        return cost, ious, dist_h

    def _stage(self, name: str, tracks: List[int], det_idx: List[int], dets: Sequence[Detection], time_s: float,
               matched: Dict[int, Tuple[int, bool]], out: FrameTracks, iou_only: bool) -> List[int]:
        if not tracks or not det_idx:
            return list(det_idx)
        cost, ious, dist_h = self._cost_matrix(tracks, det_idx, dets, time_s, iou_only)
        rows, cols = linear_sum_assignment(cost)
        used: Set[int] = set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= _INVALID:
                continue
            chosen = cost[r, c]
            margin = self.tc.contest_margin
            rival_tracks = [self._tracks[tracks[a]].track_id for a in range(len(tracks))
                            if a != r and cost[a, c] < _INVALID and cost[a, c] - chosen <= margin]
            rival_dets = [det_idx[b] for b in range(len(det_idx))
                          if b != c and cost[r, b] < _INVALID and cost[r, b] - chosen <= margin]
            contested = bool(rival_tracks or rival_dets)
            k, i = tracks[r], det_idx[c]
            matched[k] = (i, contested)
            used.add(i)
            out.decisions.append({
                "kind": "match", "stage": name, "track_id": self._tracks[k].track_id, "detection": i,
                "cost": round(float(chosen), 4), "iou": round(float(ious[r, c]), 4),
                "dist_h": None if not np.isfinite(dist_h[r, c]) else round(float(dist_h[r, c]), 4),
                "contested": contested, "rival_tracks": rival_tracks, "rival_detections": rival_dets,
                "prior_state": self._tracks[k].state.value,
            })
        return [i for i in det_idx if i not in used]

    # ------------------------------------------------------------- lifecycle
    def _apply_matches(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection], time_s: float) -> None:
        for k, t in enumerate(self._tracks):
            if k in matched:
                d = dets[matched[k][0]]
                t.kf.update(box_to_cxcywh(d.as_array()))
                t.last_observed_t = time_s
                t.observed_frames += 1
                t.velocity_frozen = False
                t.observed_last_frame = True
            else:
                t.observed_last_frame = False

    def _left_frame(self, t: _Track) -> bool:
        cx, cy = t.kf.x[:2]
        return not (0.0 <= cx <= self.frame_w and 0.0 <= cy <= self.frame_h)

    def _lifecycle(self, matched: Dict[int, Tuple[int, bool]], time_s: float, out: FrameTracks) -> None:
        keep: List[_Track] = []
        remap: Dict[int, int] = {}
        for k, t in enumerate(self._tracks):
            unobserved = time_s - t.last_observed_t
            reason: Optional[EndReason] = None
            if k in matched:
                if t.state == TrackState.TENTATIVE:
                    if t.observed_frames >= 2 and t.last_observed_t - t.first_observed_t >= self.tc.confirm_s - 1e-9:
                        t.state = TrackState.CONFIRMED
                elif t.state == TrackState.LOST:
                    t.state = TrackState.CONFIRMED
            else:
                if t.state == TrackState.TENTATIVE and unobserved > self.tc.tentative_max_miss_s + 1e-9:
                    reason = EndReason.TENTATIVE_EXPIRED
                elif t.state == TrackState.CONFIRMED:
                    t.state = TrackState.LOST
                if reason is None and t.state == TrackState.LOST and unobserved > self.tc.lost_max_s + 1e-9:
                    reason = EndReason.LOST_EXPIRED
                if reason is None and self._left_frame(t):
                    reason = EndReason.LEFT_FRAME
            if reason is not None:
                out.ended.append((t.track_id, reason))
                out.decisions.append({"kind": "end", "track_id": t.track_id, "reason": reason.value})
            else:
                remap[k] = len(keep)
                keep.append(t)
        self._tracks = keep
        new_matched = {remap[k]: v for k, v in matched.items() if k in remap}
        matched.clear()
        matched.update(new_matched)

    def _spawn(self, rem_high: List[int], dets: Sequence[Detection], matched: Dict[int, Tuple[int, bool]],
               time_s: float, out: FrameTracks) -> None:
        observed_boxes = np.array([dets[i].as_array() for i, _ in matched.values()]).reshape(-1, 4)
        for i in rem_high:
            d = dets[i]
            if len(observed_boxes):
                overlap = float(iou_matrix(d.as_array()[None], observed_boxes).max())
                if overlap > self.tc.duplicate_iou:
                    out.decisions.append({"kind": "spawn_refused", "detection": i, "reason": "DUPLICATE_OF_MATCHED",
                                          "iou": round(overlap, 4)})
                    continue
            tid = f"T{self._next_id:06d}"
            self._next_id += 1
            t = _Track(tid, BoxKalman(box_to_cxcywh(d.as_array()), self.cfg.kalman), TrackState.TENTATIVE,
                       self._epoch, time_s, time_s, time_s, 1, True)
            self._tracks.append(t)
            matched[len(self._tracks) - 1] = (i, False)
            observed_boxes = np.vstack([observed_boxes, d.as_array()[None]])
            out.decisions.append({"kind": "spawn", "track_id": tid, "detection": i})

    def _emit(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection],
              time_s: float) -> List[TrackObservation]:
        obs: List[TrackObservation] = []
        boxes = [np.asarray(dets[matched[k][0]].box if k in matched else t.box(), dtype=float)
                 for k, t in enumerate(self._tracks)]
        overlaps = iou_matrix(np.array(boxes).reshape(-1, 4), np.array(boxes).reshape(-1, 4))
        np.fill_diagonal(overlaps, 0.0)
        for k, t in enumerate(self._tracks):
            if k in matched:
                i, contested = matched[k]
                box = dets[i].box
                src, det_i, score = PositionSource.OBSERVED, i, dets[i].score
            else:
                box = tuple(float(v) for v in t.box())
                src = PositionSource.HELD if t.velocity_frozen else PositionSource.PREDICTED
                det_i, score, contested = None, None, False
            j = int(np.argmax(overlaps[k])) if len(boxes) > 1 else -1
            ov = float(overlaps[k, j]) if j >= 0 else 0.0
            obs.append(TrackObservation(t.track_id, t.state, src, box, det_i, score, contested,
                                        round(time_s - t.last_observed_t, 6), t.observed_frames, t.epoch,
                                        round(ov, 6), self._tracks[j].track_id if ov > 0 else None))
        return obs
