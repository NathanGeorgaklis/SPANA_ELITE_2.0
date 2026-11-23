"""Ball tracker v1: which candidate, if any, is the ball on each frame. No player, team or event knowledge.

Input per frame: ball candidates (the detector's ball view, spana.detection.ball_view) and the camera observation.
Several hypotheses live at once, because a ball view also fires on heads, boots, pitch spots and spare balls:
  1. Camera: on OK every hypothesis is carried by the inter-frame transform (position, velocity, diameter); on CUT
     or UNKNOWN every hypothesis ends (motion across the frame is meaningless) and the answer is LOST.
  2. Association: a candidate may join a hypothesis within (gate_base_d + max_speed_d_per_s x gap) diameters of
     its last observed position (a speed bound, so a kick may change the velocity at once); pairs are taken
     greedily, confirmed hypotheses first, then by distance to the constant-velocity prediction. Deterministic.
  3. Unmatched candidates start hypotheses. A hypothesis is confirmed after confirm_min_obs observations within
     confirm_window_s (a lone blip never becomes the ball); unconfirmed ones end after confirm_window_s unobserved,
     confirmed ones after lost_max_s.
  4. Answer (fail closed):
     - OBSERVED only for the selected ball observed again, or for the single confirmed hypothesis observed while
       no selected ball is still within predict_max_s (no silent switch to another ball);
     - PREDICTED for the selected ball unobserved <= predict_max_s with its extrapolation inside the picture;
     - with player boxes (the observed player tracks of the frame; optional): a candidate inside a player's box
       above its feet zone is ON THE BODY (the ball view fires on jerseys, shorts, boots, gloves). It cannot start
       a hypothesis. A hypothesis not observed on the previous processed frame cannot rejoin it unless it lies within
       reacquire_max_d diameters of the predicted position (a real ball that merely overlaps a player's box in the
       picture is where it was expected; a jersey 200 px away is not). A ball followed frame to frame across a
       player, and any candidate off the body, are treated as before;
     - a hypothesis observed on fewer than min_seen_share of its frames over the last seen_share_window_s is never
       the answer (LOST SPARSE_OBSERVATIONS): a false object detected now and then is kept alive by each sporadic
       re-detection, while the ball being followed is detected on most frames. It still counts as a rival;
     - AMBIGUOUS when several confirmed hypotheses could be the ball and continuity does not decide;
     - LOST otherwise.
Every association, start, confirmation, end and selection change is returned as a decision for the log.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import BallConfig
from ..records import (BallEndReason, BallFrame, BallHypothesisSummary, BallReason, BallState, CameraObservation,
                       CameraStatus, Detection)

_DISCONTINUOUS = (CameraStatus.CUT, CameraStatus.UNKNOWN)


@dataclass
class _Hypothesis:
    hid: str
    pos: np.ndarray                  # last observed centre, carried into the current frame's coordinates
    vel: np.ndarray                  # px / s in the current frame's coordinates
    diameters: Deque[float]
    last_obs_t: float
    last_obs_frame: int
    last_score: float
    first_frame: int
    obs_times: Deque[float]
    observed: int = 1
    has_velocity: bool = False
    history: Deque[Tuple[float, bool]] = None    # (t, observed) per processed frame, last seen_share_window_s
    confirmed_frame: Optional[int] = None

    @property
    def confirmed(self) -> bool:
        return self.confirmed_frame is not None

    def diameter(self, floor: float) -> float:
        return max(floor, float(np.median(self.diameters)))


@dataclass
class _Assoc:
    cost: float
    dist_d: float
    gate_d: float


def _center(d: Detection) -> np.ndarray:
    x1, y1, x2, y2 = d.box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def _diameter(d: Detection) -> float:
    x1, y1, x2, y2 = d.box
    return 0.5 * ((x2 - x1) + (y2 - y1))


class BallTracker:
    def __init__(self, cfg: BallConfig, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.w, self.h = float(frame_w), float(frame_h)
        self._hyps: Dict[str, _Hypothesis] = {}
        self._selected: Optional[str] = None
        self._next = 1
        self._summaries: Dict[str, BallHypothesisSummary] = {}
        self._prev_frame: Optional[int] = None

    # ------------------------------------------------------------------ public
    def update(self, frame: int, t: float, candidates: Sequence[Detection], camera: CameraObservation,
               players: Optional[Sequence[Sequence[float]]] = None) -> BallFrame:
        """players: boxes [x1, y1, x2, y2] of the player tracks OBSERVED on this frame, or None (no player
        knowledge: every candidate is treated as before)."""
        log: List[dict] = []
        on_body = [self._on_body(d, players) for d in candidates]
        discontinuous = camera.status in _DISCONTINUOUS
        if discontinuous:
            for hid in sorted(self._hyps):
                self._end(hid, frame, BallEndReason.CAMERA_DISCONTINUITY, log)
        elif camera.status is CameraStatus.OK:
            self._carry(camera.matrix())

        matched = self._associate(frame, t, candidates, log, on_body)
        for i, d in enumerate(candidates):
            if i in matched:
                continue
            if on_body[i]:
                log.append({"kind": "blocked", "candidate": i, "reason": "ON_BODY", "score": round(d.score, 4)})
            else:
                self._spawn(frame, t, d, i, log)
        self._record_seen(t, set(matched.values()))
        self._expire(frame, t, log)
        self._prev_frame = frame
        return self._answer(frame, t, camera.status, discontinuous, matched, candidates, log)

    def summaries(self) -> List[BallHypothesisSummary]:
        out = dict(self._summaries)
        for h in self._hyps.values():
            out[h.hid] = self._summary(h, None, None)
        return [out[k] for k in sorted(out)]

    # ------------------------------------------------------------------ steps
    def _record_seen(self, t: float, observed: set) -> None:
        for hid, h in self._hyps.items():
            if h.history is None:
                h.history = deque()
            h.history.append((t, hid in observed))
            while h.history and t - h.history[0][0] > self.cfg.seen_share_window_s + 1e-9:
                h.history.popleft()

    def seen_share(self, h: _Hypothesis) -> float:
        n = len(h.history) if h.history else 0
        return 1.0 if n == 0 else sum(1 for _, o in h.history if o) / n

    def _eligible(self, h: _Hypothesis) -> bool:
        return self.seen_share(h) >= self.cfg.min_seen_share - 1e-9

    def _carry(self, a: np.ndarray) -> None:
        lin, off = a[:, :2], a[:, 2]
        scale = float(np.sqrt(abs(np.linalg.det(lin))))
        for h in self._hyps.values():
            h.pos = lin @ h.pos + off
            h.vel = lin @ h.vel
            h.diameters = deque((v * scale for v in h.diameters), maxlen=h.diameters.maxlen)

    def _on_body(self, d: Detection, players: Optional[Sequence[Sequence[float]]]) -> bool:
        if not players:
            return False
        cx, cy = _center(d)
        return any(x1 <= cx <= x2 and y1 <= cy < y2 - self.cfg.feet_zone_h * (y2 - y1) for x1, y1, x2, y2 in players)

    def _associate(self, frame: int, t: float, cands: Sequence[Detection], log: List[dict],
                   on_body: Sequence[bool]) -> Dict[int, str]:
        c = self.cfg
        pairs: List[Tuple[int, float, str, int, _Assoc]] = []
        for hid, h in self._hyps.items():
            gap = t - h.last_obs_t
            d = h.diameter(c.min_diameter_px)
            pred = h.pos + h.vel * gap
            gate_d = c.gate_base_d + c.max_speed_d_per_s * gap
            followed = h.last_obs_frame == self._prev_frame          # observed on the previous processed frame
            for i, cand in enumerate(cands):
                p = _center(cand)
                dist_d = float(np.linalg.norm(p - h.pos)) / d
                if dist_d <= gate_d:
                    cost = float(np.linalg.norm(p - pred)) / d
                    if not followed and on_body[i] and cost > c.reacquire_max_d:
                        continue                 # no re-acquisition on a player's body away from the prediction
                    pairs.append((0 if h.confirmed else 1, cost, hid, i, _Assoc(cost, dist_d, gate_d)))
        pairs.sort(key=lambda r: r[:4])
        used_h, matched = set(), {}
        for _, _, hid, i, ev in pairs:
            if hid in used_h or i in matched:
                continue
            used_h.add(hid)
            matched[i] = hid
            self._observe(self._hyps[hid], frame, t, cands[i], log)
            log.append({"kind": "match", "hypothesis": hid, "candidate": i, "cost_d": round(ev.cost, 3),
                        "dist_d": round(ev.dist_d, 3), "gate_d": round(ev.gate_d, 3)})
        return matched

    def _observe(self, h: _Hypothesis, frame: int, t: float, d: Detection, log: List[dict]) -> None:
        c = self.cfg
        p = _center(d)
        gap = t - h.last_obs_t
        if gap > 0:
            v = (p - h.pos) / gap
            h.vel = v if not h.has_velocity else c.velocity_smoothing * v + (1 - c.velocity_smoothing) * h.vel
            h.has_velocity = True
        h.pos, h.last_obs_t, h.last_obs_frame, h.last_score = p, t, frame, d.score
        h.diameters.append(_diameter(d))
        h.observed += 1
        h.obs_times.append(t)
        self._maybe_confirm(h, frame, t, log)

    def _maybe_confirm(self, h: _Hypothesis, frame: int, t: float, log: List[dict]) -> None:
        c = self.cfg
        if h.confirmed:
            return
        recent = sum(1 for s in h.obs_times if t - s <= c.confirm_window_s + 1e-9)
        if recent >= c.confirm_min_obs:
            h.confirmed_frame = frame
            log.append({"kind": "confirm", "hypothesis": h.hid, "observations": recent})

    def _spawn(self, frame: int, t: float, d: Detection, i: int, log: List[dict]) -> None:
        c = self.cfg
        hid = f"B{self._next:06d}"
        self._next += 1
        h = _Hypothesis(hid, _center(d), np.zeros(2), deque([_diameter(d)], maxlen=c.diameter_window), t, frame,
                        d.score, frame, deque([t], maxlen=c.confirm_min_obs))
        self._hyps[hid] = h
        log.append({"kind": "spawn", "hypothesis": hid, "candidate": i, "score": round(d.score, 4)})
        self._maybe_confirm(h, frame, t, log)

    def _expire(self, frame: int, t: float, log: List[dict]) -> None:
        c = self.cfg
        for hid in sorted(self._hyps):
            h = self._hyps[hid]
            gap = t - h.last_obs_t
            if not h.confirmed and gap > c.confirm_window_s:
                self._end(hid, frame, BallEndReason.UNCONFIRMED_EXPIRED, log)
            elif h.confirmed and gap > c.lost_max_s:
                self._end(hid, frame, BallEndReason.LOST_EXPIRED, log)

    def _end(self, hid: str, frame: int, reason: BallEndReason, log: List[dict]) -> None:
        h = self._hyps.pop(hid)
        self._summaries[hid] = self._summary(h, frame, reason.value)
        log.append({"kind": "end", "hypothesis": hid, "reason": reason.value})
        if self._selected == hid:
            self._selected = None

    @staticmethod
    def _summary(h: _Hypothesis, end_frame: Optional[int], reason: Optional[str]) -> BallHypothesisSummary:
        return BallHypothesisSummary(h.hid, h.first_frame, h.last_obs_frame, h.observed, h.confirmed_frame,
                                     end_frame, reason)

    # ------------------------------------------------------------------ answer
    def _select(self, hid: Optional[str], log: List[dict]) -> None:
        if hid != self._selected:
            log.append({"kind": "select", "from": self._selected, "to": hid})
            self._selected = hid

    def _answer(self, frame: int, t: float, cam: CameraStatus, discontinuous: bool, matched: Dict[int, str],
                cands: Sequence[Detection], log: List[dict]) -> BallFrame:
        c = self.cfg
        seen = {hid: i for i, hid in matched.items() if self._hyps[hid].confirmed}
        sel = self._hyps.get(self._selected) if self._selected else None
        sel_hidden = sel is not None and sel.hid not in seen and t - sel.last_obs_t <= c.predict_max_s

        def out(state, reason, h=None, conf=None, center=None, contested=False):
            return BallFrame(frame, t, cam, state, reason, h.hid if h else None,
                             None if center is None else (float(center[0]), float(center[1])),
                             h.diameter(c.min_diameter_px) if h else None, conf, contested, len(cands),
                             sum(1 for x in self._hyps.values() if x.confirmed), log)

        if discontinuous and not seen:
            return out(BallState.LOST, BallReason.CAMERA_DISCONTINUITY)
        sparse = lambda h: (log.append({"kind": "sparse", "hypothesis": h.hid,   # noqa: E731
                                        "seen_share": round(self.seen_share(h), 3)}),
                            out(BallState.LOST, BallReason.SPARSE_OBSERVATIONS))[1]
        if sel is not None and sel.hid in seen:
            if not self._eligible(sel):
                return sparse(sel)
            reason = BallReason.CONTINUING if len(seen) > 1 else BallReason.SINGLE_CONFIRMED
            return out(BallState.OBSERVED, reason, sel, cands[seen[sel.hid]].score, sel.pos, len(seen) > 1)
        if seen and sel_hidden:
            return out(BallState.AMBIGUOUS, BallReason.SELECTED_HIDDEN_RIVAL)
        if len(seen) > 1:
            self._select(None, log)
            return out(BallState.AMBIGUOUS, BallReason.RIVAL_CANDIDATES)
        if len(seen) == 1:
            hid, i = next(iter(seen.items()))
            h = self._hyps[hid]
            if not self._eligible(h):
                return sparse(h)
            self._select(hid, log)
            return out(BallState.OBSERVED, BallReason.SINGLE_CONFIRMED, h, cands[i].score, h.pos)
        if sel is None:
            return out(BallState.LOST, BallReason.NO_CONFIRMED_BALL)
        gap = t - sel.last_obs_t
        if gap > c.predict_max_s:
            return out(BallState.LOST, BallReason.GAP_TOO_LONG)
        if not self._eligible(sel):
            return sparse(sel)
        p = sel.pos + sel.vel * gap
        if not (0.0 <= p[0] < self.w and 0.0 <= p[1] < self.h):
            return out(BallState.LOST, BallReason.PREDICTED_OUTSIDE)
        conf = sel.last_score * (1.0 - gap / c.predict_max_s)
        return out(BallState.PREDICTED, BallReason.SHORT_GAP, sel, conf, p)
