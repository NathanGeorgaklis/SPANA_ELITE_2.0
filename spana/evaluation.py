"""Detection + multi-player tracking evaluation against human ground truth (one implementation, used by the
synthetic QA suite and by the DEV validation tool).

Inputs per frame: visible ground-truth boxes {player_id: box} (ignored / not-annotated regions excluded), the
frame's detections, the tracker's OBSERVED track boxes, and the tracker's recovery decisions.

Matching: a versioned MatchRule decides which pairs are eligible (IOU_050: IoU >= 0.5; NESTED_V1 adds same-body
pairs drawn in a different box style); Hungarian picks a one-to-one assignment preferring larger overlap.
NESTED_V2 keeps NESTED_V1 eligibility but, for a box eligible for two labelled players that overlap each other,
attributes it by foot point (bottom-centre) instead of IoU, and marks it AMBIGUOUS when the foot points do not
separate the players or when any of them is labelled occluded (an occluded label's feet are an estimate): an
ambiguous track match counts as coverage but never as an identity change.

Detection layer
  missed_players      ground-truth boxes with no detection matched (by height bin: small players are the risk)
  duplicate_dets      unmatched detections that overlap a matched player (eligible under the match rule, or lie inside his
                      box by dup_ioa: partial-body boxes)
Tracking layer
  identity_transfers  a track matched to player A and later to B (the unsafe error)
  unflagged_transfers SILENT transfers: the track was not IDENTITY_UNCERTAIN on any frame within the flag tolerance
                      (seconds, config/benchmark.yaml) of the transfer frame, whatever its recoveries / links. Only the
                      uncertain state is a flag a later layer acts on (rule of 2026-11-20; before it, recoveries,
                      possible_predecessors and any episode since the previous attribution also counted as flags)
  recovery_caused_*   transfers whose track had a tracker recovery after its previous attribution (all / silent)
  link_caused_*       transfers whose track was continued by an offline link after its previous attribution
  transfers_on_linked_tracks  transfers on any track that received a link (2026-11-18 linked-track rule)
  id_switches         player-centric: changes of a player's matched track id between matched frames
  wrong_recoveries    recovery decisions whose detection is a different player than the track's last one
  duplicate_tracks    frames where an extra OBSERVED track is eligible for a matched player under the rule, or an
                      extra CONFIRMED one lies inside his box (IoA >= dup_ioa)
  fragments           per player: distinct track ids - 1
  coverage            share of visible player-frames with a matched OBSERVED track
  ambiguous_attributions  (NESTED_V2) track matches whose player could not be told apart from an overlapping one
"""
from __future__ import annotations

from dataclasses import dataclass, field
import bisect
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import iou_matrix

Box = Sequence[float]
HEIGHT_BINS_PX = (0, 24, 48, 96, 1_000_000)   # reporting bins only (not used in any decision)


@dataclass(frozen=True)
class TrackBox:
    track_id: str
    box: Box
    confirmed: bool
    detection_index: Optional[int]
    uncertain: bool = False       # IDENTITY_UNCERTAIN on this frame


class TransferFlags:
    """What the transfer rule needs from EVERY processed frame of a run (scored frames are only the labelled ones):
    when each track was IDENTITY_UNCERTAIN, when the tracker recovered it, and when an offline link continued it.

    flagged(track, frame): uncertain on some frame within `tolerance_s` of `frame` (either side).
    Without a TransferFlags, evaluate() uses tolerance 0 on the scored frames themselves."""

    def __init__(self, frames: Iterable[Tuple[int, float, Iterable[Tuple[str, bool, int]]]],
                 link_frames: Mapping[str, Iterable[int]], tolerance_s: float):
        """frames: (frame, t, [(track_id, uncertain, recoveries count)]) for every processed frame, in order.
        link_frames: track id -> frames on which a link continued it (its recoveries count rises there too)."""
        if not tolerance_s >= 0:
            raise ValueError(f"flag tolerance must be >= 0, got {tolerance_s!r}")
        self.tolerance_s = float(tolerance_s)
        self.t: Dict[int, float] = {}
        self.uncertain: Dict[str, List[float]] = {}
        self.recovery: Dict[str, List[int]] = {}
        self.link = {k: sorted(v) for k, v in link_frames.items()}
        last_rec: Dict[str, int] = {}
        for f, t, rows in frames:
            self.t[f] = float(t)
            for tid, unc, rec in rows:
                if unc:
                    self.uncertain.setdefault(tid, []).append(float(t))
                if rec > last_rec.get(tid, 0) and f not in self.link.get(tid, ()):
                    self.recovery.setdefault(tid, []).append(f)
                last_rec[tid] = rec

    def flagged(self, tid: str, frame: int) -> bool:
        t = self.t[frame]
        ts = self.uncertain.get(tid, [])
        k = bisect.bisect_left(ts, t - self.tolerance_s - 1e-9)
        return k < len(ts) and ts[k] <= t + self.tolerance_s + 1e-9

    @staticmethod
    def _between(xs: Sequence[int], lo: int, hi: int) -> bool:
        k = bisect.bisect_right(xs, lo)
        return k < len(xs) and xs[k] <= hi

    def recovery_caused(self, tid: str, prev_frame: int, frame: int) -> bool:
        return self._between(self.recovery.get(tid, []), prev_frame, frame)

    def link_caused(self, tid: str, prev_frame: int, frame: int) -> bool:
        return self._between(self.link.get(tid, []), prev_frame, frame)

    def linked(self, tid: str) -> bool:
        return bool(self.link.get(tid))

    @classmethod
    def from_tracks(cls, tracks_doc: dict, decisions: Iterable[dict], tolerance_s: float) -> "TransferFlags":
        """From a spana.tracks document and its decision log (`link` rows come from the offline linker)."""
        links: Dict[str, List[int]] = {}
        for d in decisions:
            if d.get("kind") == "link":
                links.setdefault(d["track_id"], []).append(d["frame"])
        return cls(((fr["frame"], fr["t"], [(t["track_id"], t["identity"] != "CERTAIN", t["recoveries"])
                                             for t in fr["tracks"]]) for fr in tracks_doc["frames"]),
                   links, tolerance_s)


@dataclass(frozen=True)
class EvalFrame:
    frame: int
    gt: Dict[str, Box]
    detections: Sequence[Box]
    tracks: Sequence[TrackBox]                 # OBSERVED tracks only
    recoveries: Sequence[Tuple[str, int]]      # (track_id, detection index) accepted on this frame
    occluded: frozenset = frozenset()          # player ids labelled occluded on this frame (NESTED_V2 only)


@dataclass
class Report:
    gt_boxes: int = 0
    missed_players: int = 0
    missed_by_height: Dict[str, int] = field(default_factory=dict)
    gt_by_height: Dict[str, int] = field(default_factory=dict)
    duplicate_dets: int = 0
    identity_transfers: int = 0
    unflagged_transfers: int = 0
    recovery_caused_transfers: Optional[int] = None    # None without TransferFlags (cause unknown)
    recovery_caused_silent: Optional[int] = None
    link_caused_transfers: Optional[int] = None
    link_caused_silent: Optional[int] = None
    transfers_on_linked_tracks: Optional[int] = None
    flag_tolerance_s: Optional[float] = None
    id_switches: int = 0
    wrong_recoveries: int = 0
    recoveries: int = 0
    duplicate_frames: int = 0
    fragments: Dict[str, int] = field(default_factory=dict)
    covered: int = 0
    transfer_events: List[dict] = field(default_factory=list)
    wrong_recovery_events: List[dict] = field(default_factory=list)
    missed_events: List[dict] = field(default_factory=list)
    duplicate_events: List[dict] = field(default_factory=list)
    ambiguous_attributions: int = 0
    ambiguous_events: List[dict] = field(default_factory=list)

    @property
    def transfer_tracks(self) -> Set[str]:
        return {e["track_id"] for e in self.transfer_events}

    @property
    def total_fragments(self) -> int:
        return sum(self.fragments.values())

    @property
    def coverage(self) -> float:
        return self.covered / self.gt_boxes if self.gt_boxes else 0.0

    @property
    def detection_recall(self) -> float:
        return 1 - self.missed_players / self.gt_boxes if self.gt_boxes else 0.0

    def summary(self) -> dict:
        return {"gt_boxes": self.gt_boxes, "detection_recall": round(self.detection_recall, 4),
                "missed_players": self.missed_players, "missed_by_height_px": self.missed_by_height,
                "gt_by_height_px": self.gt_by_height, "duplicate_detections": self.duplicate_dets,
                "identity_transfers": self.identity_transfers, "unflagged_transfers": self.unflagged_transfers,
                "flag_tolerance_s": self.flag_tolerance_s,
                "recovery_caused_transfers": self.recovery_caused_transfers,
                "recovery_caused_silent": self.recovery_caused_silent,
                "link_caused_transfers": self.link_caused_transfers, "link_caused_silent": self.link_caused_silent,
                "transfers_on_linked_tracks": self.transfers_on_linked_tracks,
                "id_switches": self.id_switches, "wrong_recoveries": self.wrong_recoveries,
                "recoveries": self.recoveries, "duplicate_track_frames": self.duplicate_frames,
                "fragments_total": self.total_fragments, "players": len(self.fragments),
                "track_coverage": round(self.coverage, 4), "ambiguous_attributions": self.ambiguous_attributions}


def _ioa(inner: Box, outer: Box) -> float:
    iw = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    ih = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return iw * ih / max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)


@dataclass(frozen=True)
class MatchRule:
    """When a detection/track box counts as the same body as a ground-truth box. Frozen rules are versioned."""
    name: str
    min_iou: float
    nested: bool = False              # also accept the same body drawn tighter or looser (see NESTED_V1)
    nested_min_ioa: float = 0.0
    nested_min_height_ratio: float = 0.0
    nested_min_width_ratio: float = 0.0
    overlap_min_iou: float = 0.0      # > 0: overlapping-player attribution by foot point (see NESTED_V2)
    foot_ambiguity_h: float = 0.0     # foot-distance margin (label heights) below which attribution is ambiguous


IOU_050 = MatchRule("IOU_050", 0.5)
# NESTED_V1 (frozen 2026-11-08, records/MATCHING_RULE_NESTED_V1.md): IoU >= 0.5, OR all of:
#   each box's centre lies inside the other; one box lies inside the other by >= 0.70 of its own area;
#   height ratio (small/large) >= 0.60; width ratio >= 0.40.
# A superset of IOU_050: it never removes a standard match; it only adds same-body pairs that differ in box style.
NESTED_V1 = MatchRule("NESTED_V1", 0.5, True, 0.70, 0.60, 0.40)
# NESTED_V2 (frozen 2026-11-08, records/MATCHING_RULE_NESTED_V2.md): NESTED_V1 eligibility. A box eligible for
#   two or more labelled players whose boxes overlap each other (IoU >= 0.30) is scored by foot-point distance
#   (bottom-centre, in each player's label height) instead of IoU; if its two nearest such players' foot distances
#   differ by < 0.10 label heights, or any of them is labelled occluded, its attribution is AMBIGUOUS (coverage
#   yes, identity no; the one-to-one assignment then keeps the NESTED_V1 IoU preference).
NESTED_V2 = MatchRule("NESTED_V2", 0.5, True, 0.70, 0.60, 0.40, overlap_min_iou=0.30, foot_ambiguity_h=0.10)
RULES = {r.name: r for r in (IOU_050, NESTED_V1, NESTED_V2)}


def eligible_matrix(gt_boxes: np.ndarray, boxes: np.ndarray, rule: MatchRule) -> Tuple[np.ndarray, np.ndarray]:
    """(eligible[g, d], iou[g, d])."""
    ious = iou_matrix(gt_boxes, boxes)
    ok = ious >= rule.min_iou
    if rule.nested and ious.size:
        g, d = gt_boxes[:, None, :], boxes[None, :, :]
        gc = (g[..., :2] + g[..., 2:4]) / 2
        dc = (d[..., :2] + d[..., 2:4]) / 2
        inside = lambda c, b: (c[..., 0] >= b[..., 0]) & (c[..., 0] <= b[..., 2]) & \
                              (c[..., 1] >= b[..., 1]) & (c[..., 1] <= b[..., 3])        # noqa: E731
        iw = np.clip(np.minimum(g[..., 2], d[..., 2]) - np.maximum(g[..., 0], d[..., 0]), 0, None)
        ih = np.clip(np.minimum(g[..., 3], d[..., 3]) - np.maximum(g[..., 1], d[..., 1]), 0, None)
        ga = np.maximum((g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1]), 1e-9)
        da = np.maximum((d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1]), 1e-9)
        ioa = np.maximum(iw * ih / ga, iw * ih / da)
        gh, dh = g[..., 3] - g[..., 1], d[..., 3] - d[..., 1]
        gw, dw = g[..., 2] - g[..., 0], d[..., 2] - d[..., 0]
        hr = np.minimum(gh, dh) / np.maximum(np.maximum(gh, dh), 1e-9)
        wr = np.minimum(gw, dw) / np.maximum(np.maximum(gw, dw), 1e-9)
        ok |= inside(dc, g) & inside(gc, d) & (ioa >= rule.nested_min_ioa) & \
            (hr >= rule.nested_min_height_ratio) & (wr >= rule.nested_min_width_ratio)
    return ok, ious


def _foot_distance(gt_boxes: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """[g, d] distance between foot points (bottom-centre), in units of the label's height."""
    gf = np.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2, gt_boxes[:, 3]], axis=1)
    df = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    gh = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 1e-9)
    return np.hypot(*(df[None, :, :] - gf[:, None, :]).transpose(2, 0, 1)) / gh[:, None]


def _match(gids: List[str], gt: Dict[str, Box], boxes: Sequence[Box], rule: MatchRule,
           occluded: frozenset = frozenset()) -> Tuple[Dict[int, str], np.ndarray, Set[int]]:
    """(box index -> player, eligibility, box indices whose attribution is ambiguous)."""
    if not gids or not len(boxes):
        return {}, np.zeros((len(gids), len(boxes)), dtype=bool), set()
    G, B = np.array([gt[g] for g in gids], float), np.array(boxes, dtype=float)
    ok, ious = eligible_matrix(G, B, rule)
    score = np.where(ok, ious + 1.0, 0.0)          # eligible pairs only; prefer the larger overlap
    ambiguous: Set[int] = set()
    if rule.overlap_min_iou > 0 and len(gids) > 1:
        mutual = iou_matrix(G, G)
        np.fill_diagonal(mutual, 0.0)
        foot = _foot_distance(G, B)
        for d in range(len(boxes)):
            cand = np.flatnonzero(ok[:, d])
            group = [g for g in cand if any(mutual[g, o] >= rule.overlap_min_iou for o in cand if o != g)]
            if len(group) < 2:
                continue
            if any(gids[g] in occluded for g in group):      # no reliable foot point: keep IoU, attribute nothing
                ambiguous.add(d)
                continue
            score[group, d] = 1.0 + np.clip(1.0 - foot[group, d], 0.0, 1.0)   # nearer foot point wins
            fd = np.sort(foot[group, d])
            if fd[1] - fd[0] < rule.foot_ambiguity_h:
                ambiguous.add(d)
    rows, cols = linear_sum_assignment(-score)
    return {int(c): gids[r] for r, c in zip(rows, cols) if ok[r, c]}, ok, ambiguous


def _bin(h: float) -> str:
    for lo, hi in zip(HEIGHT_BINS_PX, HEIGHT_BINS_PX[1:]):
        if lo <= h < hi:
            return f"{lo}-{hi if hi < 1_000_000 else 'inf'}"
    return "inf"


def evaluate(frames: Iterable[EvalFrame], rule: MatchRule = IOU_050, dup_ioa: float = 0.6,
             flags: Optional[TransferFlags] = None) -> Report:
    r = Report()
    if flags is not None:
        r.flag_tolerance_s = flags.tolerance_s
        r.recovery_caused_transfers = r.recovery_caused_silent = 0
        r.link_caused_transfers = r.link_caused_silent = r.transfers_on_linked_tracks = 0
    track_gt: Dict[str, str] = {}
    gt_last: Dict[str, str] = {}
    gt_tracks: Dict[str, List[str]] = {}
    attr_frame: Dict[str, int] = {}                # track -> frame of its last clear attribution
    for ef in frames:
        prev_track_gt = dict(track_gt)
        gids = sorted(ef.gt)
        r.gt_boxes += len(gids)
        for g in gids:
            b = _bin(ef.gt[g][3] - ef.gt[g][1])
            r.gt_by_height[b] = r.gt_by_height.get(b, 0) + 1

        det_gt, det_ok, _ = _match(gids, ef.gt, ef.detections, rule, ef.occluded)
        hit = set(det_gt.values())
        for g in gids:
            if g not in hit:
                r.missed_players += 1
                b = _bin(ef.gt[g][3] - ef.gt[g][1])
                r.missed_by_height[b] = r.missed_by_height.get(b, 0) + 1
                r.missed_events.append({"frame": ef.frame, "player": g, "height_px": round(ef.gt[g][3] - ef.gt[g][1], 1)})
        for d in range(len(ef.detections)):
            if d in det_gt or not gids:
                continue
            over = [g for gi, g in enumerate(gids) if g in hit and (det_ok[gi, d] or
                                                                  _ioa(ef.detections[d], ef.gt[g]) >= dup_ioa)]
            if over:
                r.duplicate_dets += 1
                r.duplicate_events.append({"frame": ef.frame, "detection": d, "player": over[0]})

        tr_gt, tr_ok, tr_amb = _match(gids, ef.gt, [t.box for t in ef.tracks], rule, ef.occluded)
        for c, g in tr_gt.items():
            t = ef.tracks[c]
            r.covered += 1
            if c in tr_amb:            # player not identifiable: no identity bookkeeping on this frame
                r.ambiguous_attributions += 1
                r.ambiguous_events.append({"frame": ef.frame, "track_id": t.track_id, "assigned": g,
                                           "last_clear": track_gt.get(t.track_id)})
                continue
            if t.track_id in track_gt and track_gt[t.track_id] != g:
                prev = attr_frame[t.track_id]
                flagged = flags.flagged(t.track_id, ef.frame) if flags is not None else t.uncertain
                ev = {"frame": ef.frame, "previous_frame": prev, "track_id": t.track_id, "from": track_gt[t.track_id],
                      "to": g, "flagged": flagged, "identity_uncertain": flagged}
                r.identity_transfers += 1
                r.unflagged_transfers += 0 if flagged else 1
                if flags is not None:
                    rc = flags.recovery_caused(t.track_id, prev, ef.frame)
                    lc = flags.link_caused(t.track_id, prev, ef.frame)
                    ev.update(recovery_caused=rc, link_caused=lc, on_linked_track=flags.linked(t.track_id))
                    r.recovery_caused_transfers += rc
                    r.recovery_caused_silent += rc and not flagged
                    r.link_caused_transfers += lc
                    r.link_caused_silent += lc and not flagged
                    r.transfers_on_linked_tracks += flags.linked(t.track_id)
                r.transfer_events.append(ev)
            track_gt[t.track_id] = g
            attr_frame[t.track_id] = ef.frame
            if g in gt_last and gt_last[g] != t.track_id:
                r.id_switches += 1
            gt_last[g] = t.track_id
            seq = gt_tracks.setdefault(g, [])
            if t.track_id not in seq:
                seq.append(t.track_id)
        extra = [c for c in range(len(ef.tracks)) if c not in tr_gt]
        if gids and any(tr_ok[:, c].any() or
                        (ef.tracks[c].confirmed and max(_ioa(ef.tracks[c].box, ef.gt[g]) for g in gids) >= dup_ioa)
                        for c in extra):
            r.duplicate_frames += 1

        for tid, d in ef.recoveries:
            r.recoveries += 1
            g_now = det_gt.get(d)
            if g_now is not None and prev_track_gt.get(tid) not in (None, g_now):
                r.wrong_recoveries += 1
                r.wrong_recovery_events.append({"frame": ef.frame, "track_id": tid, "was": prev_track_gt[tid],
                                                "recovered_onto": g_now})
    r.fragments = {g: len(v) - 1 for g, v in gt_tracks.items()}
    return r


def frames_from_artifacts(tracks_doc: dict, detections_doc: dict, gt: Dict[int, Dict[str, Box]],
                          decisions: Iterable[dict]) -> List[EvalFrame]:
    # (pair with TransferFlags.from_tracks(tracks_doc, decisions, tolerance) in evaluate())
    """Build evaluation frames from a run's artifacts and per-frame ground truth (frame index -> boxes)."""
    dets = {fr["frame"]: [r[:4] for r in fr["detections"]] for fr in detections_doc["frames"]}
    rec: Dict[int, List[Tuple[str, int]]] = {}
    tracks_by_frame = {fr["frame"]: fr for fr in tracks_doc["frames"]}
    for d in decisions:
        if d.get("kind") == "recovery":
            fr = tracks_by_frame[d["frame"]]
            t = next(t for t in fr["tracks"] if t["track_id"] == d["track_id"])
            rec.setdefault(d["frame"], []).append((d["track_id"], t["detection_index"]))
    out = []
    for fr in tracks_doc["frames"]:
        obs = [TrackBox(t["track_id"], t["box"], t["confirmed"], t["detection_index"], t["identity"] != "CERTAIN")
               for t in fr["tracks"] if t["source"] == "OBSERVED"]
        out.append(EvalFrame(fr["frame"], gt.get(fr["frame"], {}), dets.get(fr["frame"], []), obs,
                             rec.get(fr["frame"], [])))
    return out
