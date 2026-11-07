"""Generic multi-player tracker: every person, no target knowledge, no football semantics.

States: ACTIVE (observed now), OCCLUDED (unobserved, covered by another observed player), LOST (unobserved,
unexplained), TERMINATED (ended; reported in FrameTracks.ended). `confirmed` is separate: a new track is
uncertain until it has been observed over confirm_s, and an uncertain track that misses a frame is terminated.

Per frame:
  1. CUT or UNKNOWN camera motion -> every track TERMINATED, new continuity epoch. Without a known inter-frame
     transform a neighbour can land on a track's prediction and inherit it; re-linking across epochs is the
     identity layer's job.
  2. OK camera motion -> every track state mapped through the inter-frame transform.
  3. Prediction to the frame time (seconds); beyond predict_max_s a LOST track's velocity is frozen (HELD).
  4. Association, in priority order (Hungarian within a stage). Every pair must pass BOTH the max-speed centre
     gate (body heights) and the Kalman gate (squared Mahalanobis distance of the centre <= gate_chi2): tight for
     a track with an established velocity, wide for a new or long-unobserved one. In every stage a known track
     appearance that is clearly dissimilar to a known detection descriptor vetoes the pair.
       S1 high-score detections  <-> confirmed tracks observed on the previous frame
       S2 low-score detections   <-> remaining tracks observed on the previous frame (IoU only, strict):
                                     continuation through weak frames, never a recovery or a spawn
       S3 remaining high-score   <-> unconfirmed tracks observed on the previous frame
       Hidden-rival check: a continuing match is refused when an unobserved confirmed track (any gap) explains the
          detection better by more than contest_margin (the continuing player may have dropped while the hidden
          one re-emerged), or equally well (within the margin) unless appearance favours the continuing track by
          min_margin: two same-looking players at the same place cannot be told apart. Kept matches within the
          margin are CONTESTED. A refused detection goes through S4 and, if refused there, spawns a new track
          naming both candidates as possible_predecessors.
       S4 recovery of an unobserved track: LOST or unconfirmed (gap <= rematch_max_s), or confirmed OCCLUDED
          (gap <= occlusion_max_s). The max-speed gate must hold exactly one unexplained high-score detection, no
          detection whose association was contested, and that detection must lie in no other unmatched track's
          gate. It must also fit the hidden player's own motion: within gate_base_h + 0.5 * max_accel_h_per_s2 *
          gap^2 body heights of the constant-velocity prediction (OCCLUDED tracks keep their velocity). A gap
          longer than rematch_max_s (occlusion only) additionally REQUIRES appearance that matches the track and
          beats every other track by min_margin, so same-looking players are never recovered on a guess.
     Refused recovery is safe: the old track stays unobserved and expires; a new track is spawned and lists
     the unobserved tracks whose gate it lies in as `possible_predecessors` (evidence, not a merge).
     Known limit: a never-tracked player who looks the same and appears exactly on a hidden player's predicted
     path is indistinguishable from him; every bridged gap is therefore recorded (`recoveries`, `last_gap_s`).
  5. Spawn: a high-score detection starts an unconfirmed track unless it duplicates an observed box
     (IoU > duplicate_iou, or inside it by duplicate_ioa of its own area: a partial-body duplicate).
     Confirmation needs an observed span >= confirm_s AND the box not nested (IoA >= confirm_nested_ioa) inside
     another observed player's box: a shifted partial-body duplicate can never become a confirmed track. An
     unconfirmed track older than unconfirmed_max_s is terminated.
  6. Unmatched: OCCLUDED if >= occlusion_min_ioa of the predicted box is covered by an observed box and the gap is
     <= occlusion_max_s (sticky while that occluder is still observed: a re-emerging body is often missed),
     else LOST; an unconfirmed track unobserved longer than rematch_max_s is TERMINATED; TERMINATED after lost_max_s or when the prediction leaves the image.

Track confidence (documented, uncalibrated, not a probability): over the last confidence_window_s of the track's
frames, the mean detection score with unobserved frames counted as 0. Range [0, 1]; high = detected strongly and
consistently recently. Association ambiguity is NOT folded in: it is the separate `contested` flag.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..appearance import similarity
from ..config import SpanaConfig
from ..geometry import box_to_cxcywh, cxcywh_to_box, iou_matrix
from ..records import (CameraObservation, CameraStatus, Detection, EndReason, FrameTracks, HistoryEntry,
                       PositionSource, TrackletSummary, TrackObservation, TrackState)
from .kalman import BoxKalman

_INVALID = 1e6
_EPS = 1e-9


@dataclass
class _Track:
    track_id: str
    kf: BoxKalman
    epoch: int
    created_frame: int
    first_observed_t: float
    last_observed_t: float
    last_observed_frame: int
    last_update_t: float
    state: TrackState = TrackState.ACTIVE
    confirmed: bool = False
    observed_frames: int = 1
    observed_last_frame: bool = True
    missed_frames: int = 0
    velocity_frozen: bool = False
    occluder: Optional[str] = None
    appearance: Optional[np.ndarray] = None
    appearance_t: Optional[float] = None
    recoveries: int = 0
    last_gap_s: float = 0.0
    predecessors: Tuple[str, ...] = ()
    scores: Deque[Tuple[float, float]] = field(default_factory=deque)      # (time, score or 0)
    history: Deque[HistoryEntry] = field(default_factory=deque)

    def box(self) -> np.ndarray:
        return cxcywh_to_box(self.kf.x)


class MultiPlayerTracker:
    def __init__(self, cfg: SpanaConfig, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.tc = cfg.tracking
        self.ac = cfg.appearance
        self.frame_w, self.frame_h = frame_w, frame_h
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._epoch = 0
        self._last_frame: Optional[int] = None
        self._summaries: Dict[str, TrackletSummary] = {}

    # ------------------------------------------------------------------ public
    def update(self, frame_index: int, time_s: float, detections: Sequence[Detection], camera: CameraObservation,
               appearance: Optional[Sequence[Optional[np.ndarray]]] = None) -> FrameTracks:
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise ValueError(f"frames must be strictly increasing ({frame_index} after {self._last_frame})")
        if (camera.status == CameraStatus.FIRST) != (self._last_frame is None):
            raise ValueError("camera status FIRST is required on the first frame and only there")
        dets = list(detections)
        desc: List[Optional[np.ndarray]] = list(appearance) if appearance is not None else [None] * len(dets)
        if len(desc) != len(dets):
            raise ValueError("appearance descriptors must align with detections")
        self._last_frame = frame_index
        out = FrameTracks(frame_index, time_s, camera)
        out.decisions.append({"kind": "camera", "status": camera.status.value, "inliers": camera.inliers,
                              "hist_distance": camera.hist_distance})

        if camera.status in (CameraStatus.CUT, CameraStatus.UNKNOWN):
            reason = EndReason.CAMERA_CUT if camera.status == CameraStatus.CUT else EndReason.CAMERA_MOTION_UNKNOWN
            for t in self._tracks:
                self._end(t, reason, frame_index, out)
            self._tracks = []
            self._epoch += 1
        elif camera.status == CameraStatus.OK:
            for t in self._tracks:
                t.kf.apply_camera(camera.matrix())

        for t in self._tracks:
            t.kf.predict(time_s - t.last_update_t)
            t.last_update_t = time_s
            if not t.velocity_frozen and t.state != TrackState.OCCLUDED and \
                    time_s - t.last_observed_t > self.tc.predict_max_s:
                t.kf.freeze_velocity()
                t.velocity_frozen = True

        high = [i for i, d in enumerate(dets) if d.score >= self.cfg.detection.score_high]
        low = [i for i, d in enumerate(dets) if d.score < self.cfg.detection.score_high]
        matched: Dict[int, Tuple[int, bool]] = {}   # track index -> (detection index, contested)

        cont = [k for k, t in enumerate(self._tracks) if t.observed_last_frame]
        s1 = [k for k in cont if self._tracks[k].confirmed]
        s3 = [k for k in cont if not self._tracks[k].confirmed]
        rem_high = self._stage("S1", s1, high, dets, time_s, matched, out, False, desc)
        self._stage("S2", [k for k in cont if k not in matched], low, dets, time_s, matched, out, True, desc)
        rem_high = self._stage("S3", [k for k in s3 if k not in matched], rem_high, dets, time_s, matched, out,
                               False, desc)
        rem_high = self._refuse_hidden_rivals(rem_high, dets, desc, time_s, matched, out)
        rem_high = self._recover(rem_high, dets, desc, time_s, matched, out)   # low scores never recover/spawn

        self._apply_matches(matched, dets, desc, frame_index, time_s)
        self._lifecycle(matched, dets, frame_index, time_s, out)
        self._spawn(rem_high, dets, desc, matched, frame_index, time_s, out)
        out.tracks = self._emit(matched, dets, frame_index, time_s)
        return out

    def history(self, track_id: str) -> List[HistoryEntry]:
        for t in self._tracks:
            if t.track_id == track_id:
                return list(t.history)
        raise KeyError(track_id)

    def summaries(self) -> List[TrackletSummary]:
        return [self._summaries[k] for k in sorted(self._summaries)]

    # ----------------------------------------------------------- association
    def _gate_radius_h(self, t: _Track, time_s: float) -> float:
        unobserved = max(0.0, time_s - t.last_observed_t)
        return min(self.tc.gate_max_h, self.tc.gate_base_h + self.tc.max_speed_h_per_s * unobserved)

    def _cost_matrix(self, tracks: Sequence[int], det_idx: Sequence[int], dets: Sequence[Detection],
                     time_s: float, iou_only: bool, desc: Optional[Sequence[Optional[np.ndarray]]] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gated costs. A known track appearance that is clearly dissimilar to a known detection descriptor
        (similarity < appearance.min_similarity) vetoes the pair in every stage."""
        tb = np.array([self._tracks[k].box() for k in tracks]).reshape(-1, 4)
        db = np.array([dets[i].as_array() for i in det_idx]).reshape(-1, 4)
        ious = iou_matrix(tb, db)
        cost = np.full((len(tracks), len(det_idx)), _INVALID)
        dist_h = np.full_like(cost, np.inf)
        for a, k in enumerate(tracks):
            t = self._tracks[k]
            ph, pc = t.kf.height, t.kf.x[:2]
            radius = self._gate_radius_h(t, time_s)
            for b, i in enumerate(det_idx):
                d = dets[i]
                if desc is not None:
                    sim = similarity(desc[i], t.appearance)
                    if sim is not None and sim < self.ac.min_similarity:
                        continue
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
                if dh > radius or t.kf.position_mahalanobis2(dc) > self.tc.gate_chi2:
                    continue
                w = self.tc.iou_weight
                cost[a, b] = w * (1.0 - ious[a, b]) + (1.0 - w) * (dh / self.tc.gate_max_h)
        return cost, ious, dist_h

    def _stage(self, name: str, tracks: List[int], det_idx: List[int], dets: Sequence[Detection], time_s: float,
               matched: Dict[int, Tuple[int, bool]], out: FrameTracks, iou_only: bool,
               desc: Sequence[Optional[np.ndarray]]) -> List[int]:
        if not tracks or not det_idx:
            return list(det_idx)
        cost, ious, dist_h = self._cost_matrix(tracks, det_idx, dets, time_s, iou_only, desc)
        rows, cols = linear_sum_assignment(cost)
        used: Set[int] = set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= _INVALID:
                continue
            chosen, margin = cost[r, c], self.tc.contest_margin
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
            })
        return [i for i in det_idx if i not in used]

    def _recoverable(self, t: _Track, time_s: float) -> bool:
        """An unobserved track that a detection may still be re-associated to on this frame."""
        if t.observed_last_frame:
            return False
        gap = time_s - t.last_observed_t
        if t.confirmed and t.state == TrackState.OCCLUDED:
            return gap <= self.tc.occlusion_max_s + _EPS
        return gap <= self.tc.rematch_max_s + _EPS

    def _refuse_hidden_rivals(self, rem_high: List[int], dets: Sequence[Detection],
                              desc: Sequence[Optional[np.ndarray]], time_s: float,
                              matched: Dict[int, Tuple[int, bool]], out: FrameTracks) -> List[int]:
        """A continuing track's match is refused when a recoverable hidden track explains the detection better
        (by more than contest_margin): the continuing player may have dropped while the hidden one re-emerged.
        Within the margin the match stands but is CONTESTED. A refused high-score detection goes to S4."""
        # Rivals are ALL unobserved confirmed tracks (not only recoverable ones): being a rival can only refuse.
        hidden = [k for k, t in enumerate(self._tracks)
                  if k not in matched and not t.observed_last_frame and t.confirmed]
        if not matched:
            return rem_high
        refused: List[int] = []
        for k, (i, contested) in list(matched.items()):
            # The ambiguity with a track's own recorded possible predecessors was settled when it was spawned.
            rivals = [j for j in hidden if self._tracks[j].track_id not in self._tracks[k].predecessors]
            if not rivals:
                continue
            own = self._cost_matrix([k], [i], dets, time_s, False, desc)[0][0, 0]
            hc = self._cost_matrix(rivals, [i], dets, time_s, False, desc)[0][:, 0]
            best = float(hc.min())
            if best >= _INVALID:
                continue
            h = self._tracks[rivals[int(hc.argmin())]]
            if best > own + self.tc.contest_margin:
                continue
            s_own, s_hid = similarity(desc[i], self._tracks[k].appearance), similarity(desc[i], h.appearance)
            own_distinct = s_own is not None and s_hid is not None and s_own - s_hid >= self.ac.min_margin
            if best + self.tc.contest_margin < own or not own_distinct:
                del matched[k]
                refused.append(i)
                out.decisions.append({
                    "kind": "match_refused", "track_id": self._tracks[k].track_id, "detection": i,
                    "reason": "HIDDEN_TRACK_EXPLAINS_BETTER" if best + self.tc.contest_margin < own
                    else "AMBIGUOUS_WITH_HIDDEN_TRACK",
                    "own_cost": None if own >= _INVALID else round(float(own), 4), "hidden_track": h.track_id,
                    "hidden_cost": round(best, 4), "own_similarity": s_own, "hidden_similarity": s_hid})
            elif not contested:
                matched[k] = (i, True)
                out.decisions.append({"kind": "contested_by_hidden", "track_id": self._tracks[k].track_id,
                                      "detection": i, "hidden_track": h.track_id})
        high = set(rem_high) | {i for i in refused if dets[i].score >= self.cfg.detection.score_high}
        return sorted(high)

    def _recover(self, rem_high: List[int], dets: Sequence[Detection], desc: Sequence[Optional[np.ndarray]],
                 time_s: float, matched: Dict[int, Tuple[int, bool]], out: FrameTracks) -> List[int]:
        cand = [k for k, t in enumerate(self._tracks) if k not in matched and self._recoverable(t, time_s)]
        if not cand or not rem_high:
            return rem_high
        all_t, all_d = list(range(len(self._tracks))), list(range(len(dets)))
        gated = self._cost_matrix(all_t, all_d, dets, time_s, iou_only=False)[0] < _INVALID
        matched_det = {i: c for i, c in matched.values()}
        unmatched_tracks = [k for k in all_t if k not in matched]
        accepted: Dict[int, int] = {}
        for k in cand:
            t = self._tracks[k]
            gap = time_s - t.last_observed_t
            in_gate = [i for i in all_d if gated[k, i]]
            if not in_gate:
                continue
            reason, sim, margin = None, None, None
            unexplained = [i for i in in_gate if i not in matched_det]
            if any(matched_det.get(i) for i in in_gate):
                reason = "CONTESTED_NEIGHBOUR"
            elif len(unexplained) != 1:
                reason = "NO_UNEXPLAINED_DETECTION" if not unexplained else "AMBIGUOUS_DETECTIONS"
            elif unexplained[0] not in rem_high:
                reason = "LOW_SCORE"
            else:
                i = unexplained[0]
                rivals = [self._tracks[j].track_id for j in unmatched_tracks if j != k and gated[j, i]]
                sim = similarity(desc[i], t.appearance)
                if rivals:
                    reason = "RIVAL_TRACK"
                elif sim is not None and sim < self.ac.min_similarity:
                    reason = "APPEARANCE_MISMATCH"
                elif not self._motion_consistent(t, dets[i], gap):
                    reason = "INCONSISTENT_WITH_MOTION"
                elif gap > self.tc.rematch_max_s + _EPS:
                    others = [similarity(desc[i], o.appearance) for j, o in enumerate(self._tracks) if j != k]
                    others = [o for o in others if o is not None]
                    margin = None if sim is None else sim - max(others, default=0.0)
                    if sim is None:
                        reason = "APPEARANCE_UNKNOWN"
                    elif margin < self.ac.min_margin:
                        reason = "APPEARANCE_NOT_DISTINCT"
            entry = {"kind": "recovery_refused" if reason else "recovery", "stage": "S4", "track_id": t.track_id,
                     "prior_state": t.state.value, "gap_s": round(gap, 6), "detections_in_gate": in_gate,
                     "similarity": None if sim is None else round(sim, 4),
                     "margin": None if margin is None else round(margin, 4)}
            if reason:
                entry["reason"] = reason
            else:
                accepted[k] = unexplained[0]
            out.decisions.append(entry)
        used: Set[int] = set()
        for k, i in accepted.items():
            t = self._tracks[k]
            t.recoveries += 1
            t.last_gap_s = round(time_s - t.last_observed_t, 6)
            matched[k] = (i, False)
            used.add(i)
        return [i for i in rem_high if i not in used]

    def _motion_consistent(self, t: _Track, d: Detection, gap: float) -> bool:
        ph = t.kf.height
        dh = float(np.hypot(*(box_to_cxcywh(d.as_array())[:2] - t.kf.x[:2]))) / ph
        return dh <= self.tc.gate_base_h + 0.5 * self.tc.max_accel_h_per_s2 * gap * gap

    # ------------------------------------------------------------- lifecycle
    def _clean(self, i: int, dets: Sequence[Detection], matched: Dict[int, Tuple[int, bool]]) -> bool:
        others = np.array([dets[j].as_array() for j, _ in matched.values() if j != i]).reshape(-1, 4)
        return not len(others) or float(iou_matrix(dets[i].as_array()[None], others).max()) <= self.ac.clean_max_iou

    def _update_appearance(self, t: _Track, d: Optional[np.ndarray], time_s: float) -> None:
        if d is None:
            return
        if t.appearance is None or t.appearance_t is None:
            t.appearance = d.copy()
        else:
            a = 1.0 - float(np.exp(-(time_s - t.appearance_t) / self.ac.tau_s))
            t.appearance = (1.0 - a) * t.appearance + a * d
            t.appearance /= t.appearance.sum()
        t.appearance_t = time_s

    def _apply_matches(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection],
                       desc: Sequence[Optional[np.ndarray]], frame_index: int, time_s: float) -> None:
        for k, t in enumerate(self._tracks):
            if k in matched:
                i, contested = matched[k]
                t.kf.update(box_to_cxcywh(dets[i].as_array()))
                t.last_observed_t, t.last_observed_frame = time_s, frame_index
                t.observed_frames += 1
                t.velocity_frozen, t.observed_last_frame, t.missed_frames = False, True, 0
                t.state, t.occluder = TrackState.ACTIVE, None
                if not contested and self._clean(i, dets, matched):
                    self._update_appearance(t, desc[i], time_s)
            else:
                t.observed_last_frame = False
                t.missed_frames += 1

    def _nested(self, b: np.ndarray, k: int, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection]) -> bool:
        area = max((b[2] - b[0]) * (b[3] - b[1]), _EPS)
        for j, (i, _) in matched.items():
            if j == k:
                continue
            o = dets[i].box
            iw = max(0.0, min(b[2], o[2]) - max(b[0], o[0]))
            ih = max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
            if iw * ih / area >= self.tc.confirm_nested_ioa:
                return True
        return False

    def _left_frame(self, t: _Track) -> bool:
        cx, cy = t.kf.x[:2]
        return not (0.0 <= cx <= self.frame_w and 0.0 <= cy <= self.frame_h)

    def _occluder(self, t: _Track, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection]
                  ) -> Optional[str]:
        b = t.box()
        area = max((b[2] - b[0]) * (b[3] - b[1]), _EPS)
        best, best_id = 0.0, None
        for k, (i, _) in matched.items():
            o = dets[i].box
            iw = max(0.0, min(b[2], o[2]) - max(b[0], o[0]))
            ih = max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
            if iw * ih / area > best:
                best, best_id = iw * ih / area, self._tracks[k].track_id
        return best_id if best >= self.tc.occlusion_min_ioa else None

    def _lifecycle(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection], frame_index: int,
                   time_s: float, out: FrameTracks) -> None:
        keep: List[_Track] = []
        remap: Dict[int, int] = {}
        for k, t in enumerate(self._tracks):
            reason: Optional[EndReason] = None
            gap = time_s - t.last_observed_t
            if k in matched:
                if not t.confirmed:
                    if t.observed_frames >= 2 and t.last_observed_t - t.first_observed_t >= self.tc.confirm_s - _EPS \
                            and not self._nested(dets[matched[k][0]].as_array(), k, matched, dets):
                        t.confirmed = True
                    elif time_s - t.first_observed_t > self.tc.unconfirmed_max_s + _EPS:
                        reason = EndReason.UNCONFIRMED_TIMEOUT
            elif not t.confirmed and gap > self.tc.rematch_max_s + _EPS:
                reason = EndReason.UNCONFIRMED_MISSED
            else:
                occ = None
                if gap <= self.tc.occlusion_max_s + _EPS:
                    # Sticky while the occluder is still observed: an occluded body is often missed while it
                    # re-emerges (partly covered). Recovery rules, not the state, keep it safe.
                    occ = self._occluder(t, matched, dets)
                    if occ is None and t.state == TrackState.OCCLUDED and \
                            any(self._tracks[j].track_id == t.occluder for j in matched):
                        occ = t.occluder
                t.state, t.occluder = (TrackState.OCCLUDED, occ) if occ else (TrackState.LOST, None)
                if gap > self.tc.lost_max_s + _EPS:
                    reason = EndReason.LOST_EXPIRED
                elif self._left_frame(t):
                    reason = EndReason.LEFT_FRAME
            if reason is not None:
                self._end(t, reason, frame_index, out)
            else:
                remap[k] = len(keep)
                keep.append(t)
        self._tracks = keep
        new = {remap[k]: v for k, v in matched.items() if k in remap}
        matched.clear()
        matched.update(new)

    def _end(self, t: _Track, reason: EndReason, frame_index: int, out: FrameTracks) -> None:
        out.ended.append((t.track_id, reason))
        out.decisions.append({"kind": "end", "track_id": t.track_id, "reason": reason.value})
        s = self._summaries[t.track_id]
        self._summaries[t.track_id] = TrackletSummary(
            s.track_id, s.epoch, s.first_frame, s.last_frame, s.last_observed_frame, s.observed_frames,
            s.confirmed, reason.value, s.recoveries, s.possible_predecessors)

    def _spawn(self, rem_high: List[int], dets: Sequence[Detection], desc: Sequence[Optional[np.ndarray]],
               matched: Dict[int, Tuple[int, bool]], frame_index: int, time_s: float, out: FrameTracks) -> None:
        observed = [dets[i].as_array() for i, _ in matched.values()]
        unobserved = [k for k in range(len(self._tracks)) if k not in matched]
        for i in rem_high:
            d = dets[i].as_array()
            dup = self._duplicate_of(d, observed)
            if dup is not None:
                out.decisions.append({"kind": "spawn_refused", "detection": i, **dup})
                continue
            preds: Tuple[str, ...] = ()
            if unobserved:
                g = self._cost_matrix(unobserved, [i], dets, time_s, iou_only=False)[0][:, 0] < _INVALID
                preds = tuple(self._tracks[unobserved[a]].track_id for a in np.flatnonzero(g))
            tid = f"T{self._next_id:06d}"
            self._next_id += 1
            t = _Track(tid, BoxKalman(box_to_cxcywh(d), self.cfg.kalman), self._epoch, frame_index, time_s, time_s,
                       frame_index, time_s, predecessors=preds)
            if self._clean(i, dets, matched):
                self._update_appearance(t, desc[i], time_s)
            self._tracks.append(t)
            matched[len(self._tracks) - 1] = (i, False)
            observed.append(d)
            out.decisions.append({"kind": "spawn", "track_id": tid, "detection": i,
                                  "possible_predecessors": list(preds)})

    def _duplicate_of(self, d: np.ndarray, observed: List[np.ndarray]) -> Optional[dict]:
        if not observed:
            return None
        ob = np.array(observed).reshape(-1, 4)
        iou = iou_matrix(d[None], ob)[0]
        area = max((d[2] - d[0]) * (d[3] - d[1]), _EPS)
        iw = np.clip(np.minimum(d[2], ob[:, 2]) - np.maximum(d[0], ob[:, 0]), 0, None)
        ih = np.clip(np.minimum(d[3], ob[:, 3]) - np.maximum(d[1], ob[:, 1]), 0, None)
        ioa = iw * ih / area
        if iou.max() > self.tc.duplicate_iou:
            return {"reason": "DUPLICATE_OF_OBSERVED", "iou": round(float(iou.max()), 4)}
        if ioa.max() >= self.tc.duplicate_ioa:
            return {"reason": "INSIDE_OBSERVED_BOX", "ioa": round(float(ioa.max()), 4)}
        return None

    # ------------------------------------------------------------------ output
    def _confidence(self, t: _Track, time_s: float, score: Optional[float]) -> float:
        t.scores.append((time_s, 0.0 if score is None else float(score)))
        while t.scores and time_s - t.scores[0][0] > self.tc.confidence_window_s + _EPS:
            t.scores.popleft()
        return float(sum(s for _, s in t.scores) / len(t.scores))

    def _emit(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection], frame_index: int,
              time_s: float) -> List[TrackObservation]:
        boxes = [np.asarray(dets[matched[k][0]].box if k in matched else t.box(), dtype=float)
                 for k, t in enumerate(self._tracks)]
        ov = iou_matrix(np.array(boxes).reshape(-1, 4), np.array(boxes).reshape(-1, 4))
        np.fill_diagonal(ov, 0.0)
        obs: List[TrackObservation] = []
        for k, t in enumerate(self._tracks):
            if k in matched:
                i, contested = matched[k]
                box, src, det_i, score = dets[i].box, PositionSource.OBSERVED, i, dets[i].score
            else:
                box = tuple(float(v) for v in t.box())
                src = PositionSource.HELD if t.velocity_frozen else PositionSource.PREDICTED
                det_i, score, contested = None, None, False
            j = int(np.argmax(ov[k])) if len(boxes) > 1 else -1
            o = float(ov[k, j]) if j >= 0 else 0.0
            conf = self._confidence(t, time_s, score)
            rec = TrackObservation(
                t.track_id, t.state, t.confirmed, src, box, det_i, score, round(conf, 6), contested,
                frame_index - t.created_frame, t.missed_frames, round(time_s - t.last_observed_t, 6),
                t.observed_frames, t.epoch, t.occluder, round(o, 6), self._tracks[j].track_id if o > 0 else None,
                t.recoveries, t.last_gap_s, t.predecessors)
            obs.append(rec)
            t.history.append(HistoryEntry(frame_index, time_s, t.state, src, box, score))
            while t.history and time_s - t.history[0].time_s > self.tc.history_s + _EPS:
                t.history.popleft()
            self._summaries[t.track_id] = TrackletSummary(
                t.track_id, t.epoch, t.created_frame, frame_index, t.last_observed_frame, t.observed_frames,
                t.confirmed, None, t.recoveries, t.predecessors)
        return obs
