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
       S2 low-score detections   <-> remaining CONFIRMED tracks observed on the previous frame (IoU only, strict):
                                     continuation through weak frames, never a recovery or a spawn. Unconfirmed
                                     tracks never take low-score boxes (real DEV finding: a new track took its
                                     player's low-score partial box ahead of its own high-score full box in S3)
       S3 remaining high-score   <-> unconfirmed tracks observed on the previous frame
       S3b appearance doubt link (tracking.appearance_doubt_link): a track observed on the previous frame and still
          unmatched whose ONLY motion-gated detection (appearance ignored) is a remaining high-score detection that
          no other track gates, overlapping its predicted box by >= appearance_doubt_min_iou, takes it even though
          its appearance vetoed the pair. The link is never certain: the track becomes IDENTITY_UNCERTAIN
          (identity_doubt, appearance frozen) and stays so until clean crops match its pre-doubt appearance for
          identity_resolve_s. Real DEV finding (records/continuity_attribution_2026-11-18): the torso-colour veto
          ended 25 continuing tracks on their own player's next box, each re-spawned one frame later.
          Every later check (hidden rivals, multi-body, identity ambiguity) still applies to the link.
       S3c extent-change link (tracking.extent_change_link): a confirmed track observed within rematch_max_s and
          still unmatched may take a remaining high-score detection whose box changed EXTENT rather than position
          (part of the body hidden behind another player, or re-emerging): the smaller of the detection and the
          predicted box lies in the larger by >= confirm_nested_ioa, their top or bottom edges differ by
          <= extent_edge_max_h body heights, the height ratio is within extent_max_ratio, the horizontal centre is
          within the gate radius, and appearance does not veto. Exclusive only: no other track gates or extent-fits
          the detection, and the track has no other candidate. Real DEV finding
          (records/gate_break_diagnosis_2026-11-18): such boxes failed the height or centre (Kalman) gate and
          spawned a new track. Every later check still applies. A link across an unobserved gap (track not observed
          on the previous frame) is a bridged gap: it must also fit the track's own motion (S4 rule; S4's
          no-contested-neighbour rule is implied by the exclusive gate) and is recorded like an S4 recovery
          (`recovery` decision, rule EXTENT_CHANGE; recoveries / last_gap_s). The linked box replaces the track's
          box state (BoxKalman.rebase_extent, velocity kept) instead of being averaged with the old extent.
       Hidden-rival check: a continuing match is refused when an unobserved confirmed track (gap <= occlusion_max_s,
          and whose own motion can reach the detection) explains the
          detection better by more than contest_margin (the continuing player may have dropped while the hidden
          one re-emerged), or equally well (within the margin) unless appearance favours the continuing track by
          min_margin: two same-looking players at the same place cannot be told apart. Kept matches within the
          margin are CONTESTED. A refused detection goes through S4 and, if refused there, spawns a new track
          naming both candidates as possible_predecessors.
       S4 recovery of an unobserved track: LOST or unconfirmed (gap <= rematch_max_s), or confirmed OCCLUDED
          (gap <= occlusion_max_s). The max-speed gate must hold exactly one unexplained high-score detection (with
          tracking.low_score_recovery a low-score one also counts for a confirmed track within rematch_max_s whose
          prediction it overlaps by >= low_score_min_iou), no
          detection whose association was contested, and that detection must lie in no other unmatched track's
          gate. It must also fit the hidden player's own motion: within gate_base_h + 0.5 * max_accel_h_per_s2 *
          gap^2 body heights of the constant-velocity prediction (OCCLUDED tracks keep their velocity). A gap
          longer than rematch_max_s (occlusion only) additionally REQUIRES appearance that matches the track and
          beats every other track by min_margin, so same-looking players are never recovered on a guess.
       Claimants (tracking.rival_eligible_only, on by default since 2026-11-19): S3c's "no other track gates or extent-fits the
          detection" and S4's "no other unmatched track's gate" then count only tracks that could take a detection on
          this frame (observed on the previous frame, or still recoverable); a stale track past its recovery window
          no longer blocks (records/rival_eligible_2026-11-19).
     Refused recovery is safe: the old track stays unobserved and expires; a new track is spawned and lists
     the unobserved tracks whose gate it lies in as `possible_predecessors` (evidence, not a merge).
     Known limit: a never-tracked player who looks the same and appears exactly on a hidden player's predicted
     path is indistinguishable from him; every bridged gap is therefore recorded (`recoveries`, `last_gap_s`).
  Identity: a contested match of a confirmed track that a rival confirmed track also passes, with appearance
     unable to separate them, marks both IDENTITY_UNCERTAIN (appearance frozen) until clean appearance evidence
     favours the track over every recorded rival for identity_resolve_s. With config reid_resolve on, calibrated re-ID
     evidence (spana.identity.reid) counts as well; the track records how the episode was cleared (resolved_by).
  5. Spawn: a high-score detection starts an unconfirmed track unless it duplicates an observed box
     (IoU > duplicate_iou, or inside it by duplicate_ioa of its own area: a partial-body duplicate).
     Confirmation needs an observed span >= confirm_s AND the box not nested (IoA >= confirm_nested_ioa) inside
     another observed player's box: a shifted partial-body duplicate can never become a confirmed track. An
     unconfirmed track older than unconfirmed_max_s is terminated.
  6. Unmatched: OCCLUDED if >= occlusion_min_ioa of the predicted box is covered by an observed box and the gap is
     <= occlusion_max_s (sticky while that occluder is still observed: a re-emerging body is often missed),
     else LOST; an unconfirmed track unobserved longer than rematch_max_s is TERMINATED; TERMINATED after lost_max_s.
     Leaving the picture: an unconfirmed track whose prediction leaves the image is TERMINATED (LEFT_FRAME). A
     confirmed one is HELD (LOST, velocity frozen where it left, still camera-compensated) for exit_hold_s and stays
     recoverable in S4 under the long-gap rules (appearance required beyond rematch_max_s); after exit_hold_s out
     of the picture it is TERMINATED (LEFT_FRAME). exit_hold_s = 0 (the default) ends it on leaving, as before:
     on DEV the hold cost more recoveries than it gained, because torso colour cannot tell teammates apart
     (records/exit_hold_2026-11-18).

Track confidence (documented, uncalibrated, not a probability): over the last confidence_window_s of the track's
frames, the mean detection score with unobserved frames counted as 0. Range [0, 1]; high = detected strongly and
consistently recently. Association ambiguity is NOT folded in: it is the separate `contested` flag.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..appearance import similarity
from ..config import SpanaConfig
from ..geometry import box_to_cxcywh, cxcywh_to_box, iou_matrix
from ..records import (CameraObservation, CameraStatus, Detection, EndReason, FrameTracks, HistoryEntry,
                       IdentityState, PositionSource, TrackletSummary, TrackObservation, TrackState)
from .kalman import BoxKalman

_INVALID = 1e6
_EPS = 1e-9


@dataclass
class _Rival:
    """One unresolved identity ambiguity between a track and a rival track."""
    appearance: Optional[np.ndarray]    # the rival's stored appearance when the ambiguity (re)started, or None
    since_t: float                      # time of the latest contest between the two (evidence must come after it)
    reid: Optional[np.ndarray] = None   # the rival's re-ID reference when the ambiguity (re)started, or None


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
    exited_t: Optional[float] = None       # time the prediction left the picture (held, see exit_hold_s)
    appearance: Optional[np.ndarray] = None
    appearance_t: Optional[float] = None
    recoveries: int = 0
    last_gap_s: float = 0.0
    predecessors: Tuple[str, ...] = ()
    identity_rivals: Dict[str, "_Rival"] = field(default_factory=dict)   # rival id -> what is known about it
    uncertain_episodes: int = 0
    resolve_since_t: Optional[float] = None    # start of the current run of clean evidence favouring this track
    appearance_doubt_since: Optional[float] = None   # latest S3b doubt link (None = no appearance doubt)
    doubt_appearance: Optional[np.ndarray] = None    # self-consistent resolution: appearance seen during the doubt
    resolve_run_reid: bool = False             # every frame of that run cleared every rival by the re-ID margin
    resolved_by: Optional[str] = None          # rule that cleared the latest episode (APPEARANCE / REID)
    reid_ref: Optional[np.ndarray] = None      # re-ID reference (clean, uncontested, CERTAIN observation)
    reid_ref_t: Optional[float] = None
    scores: Deque[Tuple[float, float]] = field(default_factory=deque)      # (time, score or 0)
    history: Deque[HistoryEntry] = field(default_factory=deque)

    def box(self) -> np.ndarray:
        return cxcywh_to_box(self.kf.x)


class MultiPlayerTracker:
    def __init__(self, cfg: SpanaConfig, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.tc = cfg.tracking
        self.ac = cfg.appearance
        self.rc = cfg.reid_resolve
        self.frame_w, self.frame_h = frame_w, frame_h
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._clean_crops: Dict[str, Tuple[float, np.ndarray]] = {}   # track id -> latest clean, uncontested crop
        self._epoch = 0
        self._last_frame: Optional[int] = None
        self._summaries: Dict[str, TrackletSummary] = {}
        self._pending_doubt: Dict[str, int] = {}   # S3b links of this frame: track id -> detection index
        self._pending_extent: Dict[str, int] = {}  # S3c links of this frame: track id -> detection index

    # ------------------------------------------------------------------ public
    def update(self, frame_index: int, time_s: float, detections: Sequence[Detection], camera: CameraObservation,
               appearance: Optional[Sequence[Optional[np.ndarray]]] = None,
               reid_embed: Optional[Callable[[List[int]], List[Optional[np.ndarray]]]] = None) -> FrameTracks:
        """reid_embed (config reid_resolve.enabled only): detection indices -> L2-normalised re-ID embeddings, computed on
        demand for the few detections that need one (spana.identity.reid)."""
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
                              "hist_distance": camera.hist_distance, "motion_std_px": round(camera.motion_std_px, 3)})

        if camera.status in (CameraStatus.CUT, CameraStatus.UNKNOWN):
            reason = EndReason.CAMERA_CUT if camera.status == CameraStatus.CUT else EndReason.CAMERA_MOTION_UNKNOWN
            for t in self._tracks:
                self._end(t, reason, frame_index, out)
            self._tracks = []
            self._epoch += 1
        elif camera.status == CameraStatus.OK:
            for t in self._tracks:
                t.kf.apply_camera(camera.matrix(), camera.motion_std_px)

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
        self._stage("S2", [k for k in s1 if k not in matched], low, dets, time_s, matched, out, True, desc)
        rem_high = self._stage("S3", [k for k in s3 if k not in matched], rem_high, dets, time_s, matched, out,
                               False, desc)
        rem_high = self._doubt_links(rem_high, dets, desc, time_s, matched, out)
        rem_high = self._extent_links(rem_high, dets, desc, time_s, matched, out)
        rem_high = self._refuse_hidden_rivals(rem_high, dets, desc, time_s, matched, out)
        rem_high = self._recover(rem_high, dets, desc, time_s, matched, out)   # low scores never recover/spawn
        self._log_unmatched(dets, desc, time_s, matched, out)                    # read-only diagnostics
        self._contest_multi_body(matched, dets, out)
        self._identity_ambiguity(matched, dets, desc, time_s, out)
        emb = self._embeddings(matched, dets, time_s, reid_embed)
        self._identity_resolution(matched, dets, desc, time_s, out, emb)

        self._apply_matches(matched, dets, desc, frame_index, time_s, emb)
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

    def _doubt_links(self, rem_high: List[int], dets: Sequence[Detection], desc: Sequence[Optional[np.ndarray]],
                     time_s: float, matched: Dict[int, Tuple[int, bool]], out: FrameTracks) -> List[int]:
        """S3b (see module docstring). Only pairs that the appearance veto alone refused; the link is flagged."""
        self._pending_doubt = {}
        if not self.tc.appearance_doubt_link:
            return rem_high
        cont = [k for k, t in enumerate(self._tracks) if t.observed_last_frame and k not in matched]
        if not cont or not rem_high:
            return rem_high
        all_t, all_d = list(range(len(self._tracks))), list(range(len(dets)))
        cost, ious, _ = self._cost_matrix(all_t, all_d, dets, time_s, iou_only=False)   # motion only
        gated = cost < _INVALID
        used: Set[int] = set()
        for k in cont:
            t = self._tracks[k]
            in_gate = [i for i in all_d if gated[k, i]]
            if len(in_gate) != 1 or in_gate[0] not in rem_high or in_gate[0] in used:
                continue
            i = in_gate[0]
            sim = similarity(desc[i], t.appearance)
            if sim is None or sim >= self.ac.min_similarity:
                continue                      # not an appearance veto: the ordinary stages decided
            if any(gated[j, i] for j in all_t if j != k) or ious[k, i] < self.tc.appearance_doubt_min_iou:
                continue                      # not exclusive, or too loose to call the same body
            matched[k] = (i, False)
            used.add(i)
            self._pending_doubt[t.track_id] = i
            out.decisions.append({"kind": "doubt_link", "stage": "S3b", "track_id": t.track_id, "detection": i,
                                  "iou": round(float(ious[k, i]), 4), "similarity": round(sim, 4)})
        return [i for i in rem_high if i not in used]

    def _extent_fit(self, t: _Track, d: Detection, time_s: float) -> bool:
        """The detection could be this track's body with part of it hidden or re-emerging (S3c geometry only)."""
        p, b = t.box(), d.as_array()
        hp, hb = max(p[3] - p[1], _EPS), max(b[3] - b[1], _EPS)
        if not 1.0 / self.tc.extent_max_ratio <= hb / hp <= self.tc.extent_max_ratio:
            return False
        iw = max(0.0, min(p[2], b[2]) - max(p[0], b[0]))
        ih = max(0.0, min(p[3], b[3]) - max(p[1], b[1]))
        small = min((p[2] - p[0]) * hp, (b[2] - b[0]) * hb)
        if iw * ih / max(small, _EPS) < self.tc.confirm_nested_ioa:
            return False
        if min(abs(b[1] - p[1]), abs(b[3] - p[3])) / hp > self.tc.extent_edge_max_h:
            return False
        return abs((b[0] + b[2]) / 2 - (p[0] + p[2]) / 2) / hp <= self._gate_radius_h(t, time_s)

    def _extent_links(self, rem_high: List[int], dets: Sequence[Detection], desc: Sequence[Optional[np.ndarray]],
                      time_s: float, matched: Dict[int, Tuple[int, bool]], out: FrameTracks) -> List[int]:
        """S3c (see module docstring). Certain links, exclusive pairs only."""
        self._pending_extent = {}
        if not self.tc.extent_change_link or not rem_high:
            return rem_high
        cand = [k for k, t in enumerate(self._tracks) if k not in matched and t.confirmed
                and time_s - t.last_observed_t <= self.tc.rematch_max_s + _EPS]
        if not cand:
            return rem_high
        all_t, all_d = list(range(len(self._tracks))), list(range(len(dets)))
        gated = self._cost_matrix(all_t, all_d, dets, time_s, iou_only=False)[0] < _INVALID   # motion gates
        fit = {(k, i): self._extent_fit(self._tracks[k], dets[i], time_s) for k in all_t for i in rem_high}
        used: Set[int] = set()
        for k in cand:
            t = self._tracks[k]
            mine = [i for i in rem_high if fit[(k, i)] and i not in used]
            if len(mine) != 1:
                continue
            i = mine[0]
            if any(gated[k, j] for j in all_d if j != i):
                continue                      # the track's own gate holds another detection
            if any((gated[j, i] or fit[(j, i)]) and self._may_claim(self._tracks[j], time_s) for j in all_t if j != k):
                continue                      # another track could take this box
            sim = similarity(desc[i], t.appearance)
            if sim is not None and sim < self.ac.min_similarity:
                continue
            gap = time_s - t.last_observed_t
            bridged = not t.observed_last_frame
            if bridged and not self._motion_consistent(t, dets[i], gap):
                continue                      # a bridged gap must also fit the track's own motion (S4 rule)
            matched[k] = (i, False)
            used.add(i)
            self._pending_extent[t.track_id] = i
            p, b = t.box(), dets[i].as_array()
            entry = {"kind": "recovery" if bridged else "extent_link", "stage": "S3c", "track_id": t.track_id,
                     "detection": i, "height_ratio": round(float((b[3] - b[1]) / max(p[3] - p[1], _EPS)), 4),
                     "gap_s": round(gap, 6), "similarity": None if sim is None else round(sim, 4)}
            if bridged:                       # every bridged gap is recorded, as in S4
                t.recoveries += 1
                t.last_gap_s = round(gap, 6)
                entry.update(prior_state=t.state.value, rule="EXTENT_CHANGE")
            out.decisions.append(entry)
        return [i for i in rem_high if i not in used]

    def _uncertain(self, t: _Track) -> bool:
        return bool(t.identity_rivals) or t.appearance_doubt_since is not None

    def _contest_multi_body(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection], out: FrameTracks) -> None:
        """A match through a box that may hold more than one person is CONTESTED (so it never updates appearance),
        whichever stage made it."""
        for k, (i, contested) in list(matched.items()):
            if dets[i].multi_body and not contested:
                matched[k] = (i, True)
                out.decisions.append({"kind": "contested_multi_body", "track_id": self._tracks[k].track_id,
                                      "detection": i})

    def _identity_ambiguity(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection],
                            desc: Sequence[Optional[np.ndarray]], time_s: float, out: FrameTracks) -> None:
        """IDENTITY_UNCERTAIN trigger (real DEV finding: a crossing swapped two identities with only per-frame
        contest flags, so the swap stayed silent). A contested match of a confirmed track whose detection a rival
        confirmed track also passes (every gate + its own motion), with appearance unable to separate the two by
        min_margin, marks BOTH tracks uncertain. Evaluated on predicted states, before any update (read-only)."""
        for k, (i, contested) in list(matched.items()):
            t = self._tracks[k]
            if not (contested and t.confirmed):
                continue
            for r in self._tracks:
                if r is t or not r.confirmed or not self._recoverable_or_live(r, time_s):
                    continue
                rep = self._gate_report(r, dets[i], desc[i], time_s)
                if rep["failed"] or not rep["motion_consistent"]:
                    continue
                s_own, s_riv = similarity(desc[i], t.appearance), similarity(desc[i], r.appearance)
                if s_own is not None and s_riv is not None and abs(s_own - s_riv) >= self.ac.min_margin:
                    continue                      # appearance tells them apart: an ordinary contest
                for a, b in ((t, r), (r, t)):
                    if not self._uncertain(a):
                        a.uncertain_episodes += 1
                    prev = a.identity_rivals.get(b.track_id)
                    snap = prev.appearance if prev is not None else (None if b.appearance is None
                                                                      else b.appearance.copy())
                    rsnap = prev.reid if prev is not None else (None if b.reid_ref is None else b.reid_ref.copy())
                    a.identity_rivals[b.track_id] = _Rival(snap, time_s, rsnap)
                    a.resolve_since_t, a.resolved_by = None, None
                out.decisions.append({"kind": "identity_uncertain", "track_id": t.track_id, "rival": r.track_id,
                                      "detection": i, "own_similarity": s_own, "rival_similarity": s_riv})

    def _recoverable_or_live(self, t: _Track, time_s: float) -> bool:
        return t.observed_last_frame or time_s - t.last_observed_t <= self.tc.occlusion_max_s + _EPS

    def _embeddings(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection], time_s: float,
                    reid_embed) -> Dict[int, Optional[np.ndarray]]:
        """Re-ID embeddings for the clean, uncontested matches that need one: uncertain tracks (evidence) and
        CERTAIN confirmed tracks whose reference is due (reid_resolve.ref_update_s). Empty when it is off."""
        if not self.rc.enabled or reid_embed is None:
            return {}
        need = set()
        for k, (i, contested) in matched.items():
            t = self._tracks[k]
            if contested or not self._clean(i, dets, matched):
                continue
            if t.identity_rivals or (not self._uncertain(t) and t.confirmed and
                                     (t.reid_ref is None or time_s - t.reid_ref_t >= self.rc.ref_update_s - _EPS)):
                need.add(i)
        if not need:
            return {}
        idx = sorted(need)
        vecs = list(reid_embed(idx))
        if len(vecs) != len(idx):
            raise ValueError("reid_embed must return one embedding (or None) per requested detection")
        return dict(zip(idx, vecs))

    def reid_verdict(self, e: Optional[np.ndarray], own: Optional[np.ndarray],
                     rival: Optional[np.ndarray]) -> Optional[bool]:
        """One rival pair: the crop is nearer the track's own reference than the rival's by reid_resolve.margin
        (True), nearer the rival's by the margin (False: evidence of a swap, never a reason to switch), or neither / unknown (None)."""
        if e is None or own is None or rival is None:
            return None
        m = float(e @ own) - float(e @ rival)
        return True if m >= self.rc.margin else (False if m <= -self.rc.margin else None)

    def _identity_resolution(self, matched: Dict[int, Tuple[int, bool]], dets: Sequence[Detection],
                             desc: Sequence[Optional[np.ndarray]], time_s: float, out: FrameTracks,
                             emb: Optional[Dict[int, Optional[np.ndarray]]] = None) -> None:
        """An uncertain track becomes CERTAIN only after clean evidence (uncontested match, clean crop) clears EVERY
        rival pair on the same frame and no pair has been contradicted for identity_resolve_s. Per pair:
          TWO_SIDED  both appearances stored: own crop matches own (>= min_similarity) and beats the rival's by
                     min_margin (the original rule);
          RIVAL_ONLY only the rival's stored: own crop is clearly unlike the rival (<= min_similarity - min_margin);
          OWN_ONLY   only own stored: own crop matches own, and the rival's latest clean crop taken after the
                     contest is clearly unlike own (<= min_similarity - min_margin); no such crop = no evidence;
          neither stored: never evidence.
        An appearance doubt (S3b) is one more condition: own crop matches the track's frozen pre-doubt appearance
        (>= min_similarity); evidence must come after the latest doubt link. With
        tracking.doubt_resolve_self_consistent the doubt also passes (SELF_CONSISTENT) when the track is exclusive on
        this frame (its motion gate holds only this detection, no other track gates it) and the crop matches the
        appearance seen since the doubt began (the first such crop only starts that reference); a non-exclusive
        frame or a disagreeing crop is evidence against. On resolution the track adopts that appearance.
        With reid_resolve on, each rival pair also gets a REID verdict from the re-ID embedding of the clean crop:
        cleared if sim(own reference) - sim(rival reference at the contest) >= reid_resolve.margin, against if
        <= -margin, else no evidence. A pair is against if either cue is against, cleared if either clears it. An
        appearance doubt has no rival reference: only appearance clears it. The track is resolved_by REID only if
        re-ID cleared every rival on every frame of the run and there was no appearance doubt, else APPEARANCE.
        Any pair with clean evidence against it resets the run; frames without evidence for a pair hold it."""
        clear_max = self.ac.min_similarity - self.ac.min_margin
        gated: Optional[np.ndarray] = None
        emb = emb or {}
        for k, (i, contested) in matched.items():
            t = self._tracks[k]
            if not self._uncertain(t) or contested or not self._clean(i, dets, matched):
                continue
            e = emb.get(i)
            if desc[i] is None and e is None:
                continue
            s_own = similarity(desc[i], t.appearance) if desc[i] is not None else None
            verdicts: Dict[str, Tuple[str, Optional[bool]]] = {}
            if t.appearance_doubt_since is not None:
                v = None if s_own is None or time_s <= t.appearance_doubt_since + _EPS \
                    else s_own >= self.ac.min_similarity
                verdicts["(appearance_doubt)"] = ("APPEARANCE_DOUBT", v)
                if v is False and self.tc.doubt_resolve_self_consistent and not t.identity_rivals:
                    if gated is None:
                        gated = self._cost_matrix(list(range(len(self._tracks))), list(range(len(dets))), dets,
                                                  time_s, iou_only=False)[0] < _INVALID
                    exclusive = int(gated[:, i].sum()) == 1 and int(gated[k].sum()) == 1 and bool(gated[k, i])
                    if not exclusive:
                        t.doubt_appearance = None
                    elif t.doubt_appearance is None:
                        t.doubt_appearance, v = desc[i].copy(), None        # the reference starts here
                    else:
                        v = similarity(desc[i], t.doubt_appearance) >= self.ac.min_similarity
                        if v:
                            t.doubt_appearance = 0.5 * (t.doubt_appearance + desc[i])
                            t.doubt_appearance /= t.doubt_appearance.sum()
                        else:
                            t.doubt_appearance = desc[i].copy()             # restart the reference
                    verdicts["(appearance_doubt)"] = ("SELF_CONSISTENT", v)
            reid_v: Dict[str, Optional[bool]] = {}
            for rid, rv in t.identity_rivals.items():
                reid_v[rid] = self.reid_verdict(e, t.reid_ref, rv.reid)
                if desc[i] is None:
                    verdicts[rid] = ("NO_CROP", None)
                elif t.appearance is not None and rv.appearance is not None:
                    ok = s_own >= self.ac.min_similarity and s_own - similarity(desc[i], rv.appearance) >= \
                        self.ac.min_margin
                    verdicts[rid] = ("TWO_SIDED", ok)
                elif rv.appearance is not None:
                    verdicts[rid] = ("RIVAL_ONLY", similarity(desc[i], rv.appearance) <= clear_max)
                elif t.appearance is not None:
                    crop = self._clean_crops.get(rid)
                    if crop is None or crop[0] <= rv.since_t:
                        verdicts[rid] = ("OWN_ONLY", None if s_own >= self.ac.min_similarity else False)
                    else:
                        verdicts[rid] = ("OWN_ONLY", s_own >= self.ac.min_similarity and
                                         similarity(crop[1], t.appearance) <= clear_max)
                else:
                    verdicts[rid] = ("NONE_STORED", None)
            for rid, (rule, v) in list(verdicts.items()):           # combine with the re-ID verdict per pair
                r = reid_v.get(rid)               # None for the appearance doubt (no rival reference)
                if v is False or r is False:
                    verdicts[rid] = (rule + ("+REID_AGAINST" if r is False else ""), False)
                elif v or r:
                    verdicts[rid] = ("REID" if not v else (rule + "+REID" if r else rule), True)
            all_reid = bool(reid_v) and all(reid_v.values()) and "(appearance_doubt)" not in verdicts
            if any(v is False for _, v in verdicts.values()):
                if t.resolve_since_t is not None:
                    out.decisions.append({"kind": "identity_resolution_reset", "track_id": t.track_id,
                                          "detection": i, "own_similarity": s_own,
                                          "against": sorted(r for r, (_, v) in verdicts.items() if v is False)})
                t.resolve_since_t = None
                continue
            if not all(v for _, v in verdicts.values()):
                continue                      # some pair has no evidence on this frame: hold, do not advance
            if t.resolve_since_t is None:
                t.resolve_since_t, t.resolve_run_reid = time_s, True
            t.resolve_run_reid = t.resolve_run_reid and all_reid
            if time_s - t.resolve_since_t >= self.tc.identity_resolve_s - _EPS:
                by = "REID" if t.resolve_run_reid else "APPEARANCE"
                out.decisions.append({"kind": "identity_resolved", "track_id": t.track_id, "detection": i,
                                      "rivals": {r: rule for r, (rule, _) in sorted(verdicts.items())},
                                      "own_similarity": s_own, "resolved_by": by})
                if verdicts.get("(appearance_doubt)", ("",))[0] == "SELF_CONSISTENT":
                    t.appearance, t.appearance_t = t.doubt_appearance.copy(), time_s   # the look seen since
                t.identity_rivals, t.resolve_since_t, t.appearance_doubt_since = {}, None, None
                t.doubt_appearance, t.resolved_by = None, by
                self._pending_doubt.pop(t.track_id, None)   # this frame's doubt link is what just resolved

    def _may_claim(self, t: _Track, time_s: float) -> bool:
        """Could t take a detection on this frame (S3c / S4 claimant)? With tracking.rival_eligible_only, only a track
        observed on the previous frame or still recoverable; otherwise any track (the behaviour before)."""
        return not self.tc.rival_eligible_only or t.observed_last_frame or self._recoverable(t, time_s)

    def _recoverable(self, t: _Track, time_s: float) -> bool:
        """An unobserved track that a detection may still be re-associated to on this frame."""
        if t.observed_last_frame:
            return False
        gap = time_s - t.last_observed_t
        if t.confirmed and t.exited_t is not None and time_s - t.exited_t <= self.tc.exit_hold_s + _EPS:
            return True                       # held after leaving the picture (long-gap rules apply in S4)
        if t.confirmed and t.state == TrackState.OCCLUDED:
            return gap <= self.tc.occlusion_max_s + _EPS
        return gap <= self.tc.rematch_max_s + _EPS

    def _refuse_hidden_rivals(self, rem_high: List[int], dets: Sequence[Detection],
                              desc: Sequence[Optional[np.ndarray]], time_s: float,
                              matched: Dict[int, Tuple[int, bool]], out: FrameTracks) -> List[int]:
        """A continuing track's match is refused when a recoverable hidden track explains the detection better
        (by more than contest_margin): the continuing player may have dropped while the hidden one re-emerged.
        Within the margin the match stands but is CONTESTED. A refused high-score detection goes to S4."""
        # Rivals: unobserved confirmed tracks within occlusion_max_s (wider than the recovery window: being a rival
        # can only refuse). Per detection, a rival must also fit the detection with its OWN motion (acceleration
        # bound): real DEV finding, stale predictions left in a crowd otherwise refused live continuations and the
        # resulting new lost tracks refused more (a fragmentation cascade).
        hidden = [k for k, t in enumerate(self._tracks)
                  if k not in matched and not t.observed_last_frame and t.confirmed
                  and time_s - t.last_observed_t <= self.tc.occlusion_max_s + _EPS]
        if not matched:
            return rem_high
        refused: List[int] = []
        for k, (i, contested) in list(matched.items()):
            # The ambiguity with a track's own recorded possible predecessors was settled when it was spawned.
            rivals = [j for j in hidden if self._tracks[j].track_id not in self._tracks[k].predecessors
                      and self._motion_consistent(self._tracks[j], dets[i], time_s - self._tracks[j].last_observed_t)]
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
        if not cand or not (rem_high or self.tc.low_score_recovery):
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
            reason, sim, margin, rivals = None, None, None, []
            unexplained = [i for i in in_gate if i not in matched_det]
            if any(matched_det.get(i) for i in in_gate):
                reason = "CONTESTED_NEIGHBOUR"
            elif len(unexplained) != 1:
                reason = "NO_UNEXPLAINED_DETECTION" if not unexplained else "AMBIGUOUS_DETECTIONS"
            elif unexplained[0] not in rem_high and not self._low_score_recoverable(t, dets[unexplained[0]], gap):
                reason = "LOW_SCORE"
            else:
                i = unexplained[0]
                rivals = [self._tracks[j].track_id for j in unmatched_tracks if j != k and gated[j, i]
                          and self._may_claim(self._tracks[j], time_s)]
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
                if reason == "RIVAL_TRACK":
                    entry["rival_tracks"] = sorted(rivals)          # diagnostic: who the rival was
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

    def _low_score_recoverable(self, t: _Track, d: Detection, gap: float) -> bool:
        """tracking.low_score_recovery: a confirmed track's short gap may close on a low-score box that overlaps
        its prediction (ByteTrack's second pass, restricted to S4's exclusive rules)."""
        return self.tc.low_score_recovery and t.confirmed and gap <= self.tc.rematch_max_s + _EPS and \
            float(iou_matrix(t.box()[None], d.as_array()[None])[0, 0]) >= self.tc.low_score_min_iou

    def _motion_consistent(self, t: _Track, d: Detection, gap: float) -> bool:
        ph = t.kf.height
        dh = float(np.hypot(*(box_to_cxcywh(d.as_array())[:2] - t.kf.x[:2]))) / ph
        return dh <= self.tc.gate_base_h + 0.5 * self.tc.max_accel_h_per_s2 * gap * gap

    # ----------------------------------------------------------- diagnostics
    LOG_CANDIDATES = 3   # nearest detections reported per unmatched track (log size bound, not a decision value)

    def _gate_report(self, t: _Track, d: Detection, desc_i: Optional[np.ndarray], time_s: float) -> dict:
        """Every association check for one (track, detection) pair, evaluated read-only."""
        ph, pc = t.kf.height, t.kf.x[:2]
        dc = box_to_cxcywh(d.as_array())[:2]
        dh = float(np.hypot(*(dc - pc))) / ph
        radius = self._gate_radius_h(t, time_s)
        m2 = t.kf.position_mahalanobis2(dc)
        ratio = max(d.height, 1.0) / ph
        iou = float(iou_matrix(t.box()[None], d.as_array()[None])[0, 0])
        sim = similarity(desc_i, t.appearance)
        high = d.score >= self.cfg.detection.score_high
        r = {"score": round(d.score, 4), "dist_h": round(dh, 4), "gate_radius_h": round(radius, 4),
             "centre_ok": dh <= radius, "mahalanobis2": round(m2, 3), "mahalanobis_ok": m2 <= self.tc.gate_chi2,
             "height_ratio": round(ratio, 3),
             "height_ok": 1.0 / self.tc.max_height_ratio <= ratio <= self.tc.max_height_ratio,
             "iou": round(iou, 4), "low_score": not high,
             "low_score_iou_ok": (not high) and iou >= self.tc.low_score_min_iou,
             "appearance_similarity": None if sim is None else round(sim, 4),
             "appearance_veto": sim is not None and sim < self.ac.min_similarity,
             "motion_consistent": self._motion_consistent(t, d, time_s - t.last_observed_t)}
        failed = [k for k in ("centre_ok", "mahalanobis_ok", "height_ok") if high and not r[k]]
        if not high and not r["low_score_iou_ok"]:
            failed.append("low_score_iou_ok")
        if r["appearance_veto"]:
            failed.append("appearance_veto")
        r["failed"] = failed
        return r

    def _log_unmatched(self, dets: Sequence[Detection], desc: Sequence[Optional[np.ndarray]], time_s: float,
                       matched: Dict[int, Tuple[int, bool]], out: FrameTracks) -> None:
        """One 'unmatched' decision per existing track left without a detection on this frame (read-only)."""
        owner = {i: (self._tracks[k].track_id, c) for k, (i, c) in matched.items()}
        refusals: Dict[str, dict] = {}
        for d in out.decisions:
            if d["kind"] in ("match_refused", "recovery_refused") and "track_id" in d:
                refusals.setdefault(d["track_id"], d)
        for k, t in enumerate(self._tracks):
            if k in matched:
                continue
            gap = time_s - t.last_observed_t
            stage = "continuing" if t.observed_last_frame else \
                "recoverable" if self._recoverable(t, time_s) else "not_eligible"
            ph, pc = t.kf.height, t.kf.x[:2]
            near = sorted((float(np.hypot(*(box_to_cxcywh(d.as_array())[:2] - pc))) / ph, i)
                          for i, d in enumerate(dets))
            near = [(dist, i) for dist, i in near if dist <= self.tc.gate_max_h][: self.LOG_CANDIDATES]
            cands = []
            for _, i in near:
                rep = self._gate_report(t, dets[i], desc[i], time_s)
                rep["detection"] = i
                rep["taken_by"] = owner[i][0] if i in owner else None
                rep["taken_contested"] = owner[i][1] if i in owner else None
                cands.append(rep)
            passing = [c for c in cands if not c["failed"]]
            if t.track_id in refusals:
                ref = refusals[t.track_id]
                reason = f"REFUSED:{ref['kind']}:{ref.get('reason')}"
            elif not cands:
                reason = "NO_DETECTION_NEARBY"
            elif stage == "not_eligible":
                reason = "NOT_ELIGIBLE_GAP"
            elif passing and all(c["taken_by"] for c in passing):
                reason = "TAKEN_BY_OTHER"
            elif passing:
                reason = "PASSING_CANDIDATE_UNUSED" if stage == "continuing" else "NO_RECOVERY_ATTEMPT"
            else:
                reason = "GATE_FAILED:" + "+".join(cands[0]["failed"])
            out.decisions.append({"kind": "unmatched", "track_id": t.track_id, "confirmed": t.confirmed,
                                  "prior_state": t.state.value, "stage": stage, "gap_s": round(gap, 6),
                                  "reason": reason, "candidates": cands})

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
                       desc: Sequence[Optional[np.ndarray]], frame_index: int, time_s: float,
                       emb: Optional[Dict[int, Optional[np.ndarray]]] = None) -> None:
        emb = emb or {}
        for k, t in enumerate(self._tracks):
            if k in matched:
                i, contested = matched[k]
                if self._pending_extent.get(t.track_id) == i:
                    t.kf.rebase_extent(box_to_cxcywh(dets[i].as_array()))   # S3c: a new visible extent
                else:
                    t.kf.update(box_to_cxcywh(dets[i].as_array()))
                t.last_observed_t, t.last_observed_frame = time_s, frame_index
                t.observed_frames += 1
                t.velocity_frozen, t.observed_last_frame, t.missed_frames = False, True, 0
                t.state, t.occluder, t.exited_t = TrackState.ACTIVE, None, None
                if self._pending_doubt.get(t.track_id) == i:
                    starting = t.appearance_doubt_since is None
                    if not self._uncertain(t):
                        t.uncertain_episodes += 1
                    t.appearance_doubt_since = time_s
                    if starting:
                        t.doubt_appearance = None
                    if starting or not self.tc.doubt_resolve_self_consistent:
                        t.resolve_since_t = None        # a continuing doubt keeps a self-consistent run going
                if not contested and self._clean(i, dets, matched) and desc[i] is not None:
                    self._clean_crops[t.track_id] = (time_s, desc[i])   # evidence only, never the model
                    if not self._uncertain(t):
                        self._update_appearance(t, desc[i], time_s)     # frozen while IDENTITY_UNCERTAIN
                if not self._uncertain(t) and t.confirmed and emb.get(i) is not None:
                    t.reid_ref, t.reid_ref_t = emb[i], time_s           # only requested when clean / uncontested
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
        # Matched tracks first: confirmation, or the end of an unconfirmed track that ran out of time. Only matched
        # tracks that survive this frame may be an occluder (real DEV finding: an occluder ending on the same frame
        # left a hidden track naming a track that was no longer live).
        ending: Dict[int, EndReason] = {}
        for k in matched:
            t = self._tracks[k]
            if not t.confirmed:
                if t.observed_frames >= 2 and t.last_observed_t - t.first_observed_t >= self.tc.confirm_s - _EPS \
                        and not self._nested(dets[matched[k][0]].as_array(), k, matched, dets):
                    t.confirmed = True
                elif time_s - t.first_observed_t > self.tc.unconfirmed_max_s + _EPS:
                    ending[k] = EndReason.UNCONFIRMED_TIMEOUT
        survivors = {k: v for k, v in matched.items() if k not in ending}
        for k, t in enumerate(self._tracks):
            reason: Optional[EndReason] = None
            gap = time_s - t.last_observed_t
            if k in matched:
                reason = ending.get(k)
            elif not t.confirmed and gap > self.tc.rematch_max_s + _EPS:
                reason = EndReason.UNCONFIRMED_MISSED
            else:
                occ = None
                if gap <= self.tc.occlusion_max_s + _EPS:
                    # Sticky while the occluder is still observed: an occluded body is often missed while it
                    # re-emerges (partly covered). Recovery rules, not the state, keep it safe.
                    occ = self._occluder(t, survivors, dets)
                    if occ is None and t.state == TrackState.OCCLUDED and \
                            any(self._tracks[j].track_id == t.occluder for j in survivors):
                        occ = t.occluder
                t.state, t.occluder = (TrackState.OCCLUDED, occ) if occ else (TrackState.LOST, None)
                if gap > self.tc.lost_max_s + _EPS:
                    reason = EndReason.LOST_EXPIRED
                elif self._left_frame(t):
                    if t.confirmed and t.exited_t is None:          # first frame out of the picture: hold it
                        t.exited_t = time_s
                        t.kf.freeze_velocity()
                        t.velocity_frozen = True
                        out.decisions.append({"kind": "exit_hold", "track_id": t.track_id})
                    if not t.confirmed or time_s - t.exited_t >= self.tc.exit_hold_s - _EPS:
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
            if dets[i].multi_body:            # a possibly merged box must not start a track
                out.decisions.append({"kind": "spawn_refused", "detection": i, "reason": "MULTI_BODY_DETECTION"})
                continue
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
                t.recoveries, t.last_gap_s, t.predecessors,
                IdentityState.IDENTITY_UNCERTAIN if self._uncertain(t) else IdentityState.CERTAIN,
                tuple(sorted(t.identity_rivals)), t.uncertain_episodes, t.appearance_doubt_since is not None,
                None if self._uncertain(t) else t.resolved_by)
            obs.append(rec)
            t.history.append(HistoryEntry(frame_index, time_s, t.state, src, box, score))
            while t.history and time_s - t.history[0].time_s > self.tc.history_s + _EPS:
                t.history.popleft()
            self._summaries[t.track_id] = TrackletSummary(
                t.track_id, t.epoch, t.created_frame, frame_index, t.last_observed_frame, t.observed_frames,
                t.confirmed, None, t.recoveries, t.predecessors)
        return obs
