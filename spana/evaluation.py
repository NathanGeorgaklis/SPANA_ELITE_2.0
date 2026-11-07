"""Detection + multi-player tracking evaluation against human ground truth (one implementation, used by the
synthetic QA suite and by the DEV validation tool).

Inputs per frame: visible ground-truth boxes {player_id: box} (ignored / not-annotated regions excluded), the
frame's detections, the tracker's OBSERVED track boxes, and the tracker's recovery decisions.

Matching: Hungarian on IoU, a pair counts when IoU >= match_iou.

Detection layer
  missed_players      ground-truth boxes with no detection matched (by height bin: small players are the risk)
  duplicate_dets      unmatched detections that overlap a matched player (IoU >= match_iou, or lie inside his
                      box by dup_ioa: partial-body boxes)
Tracking layer
  identity_transfers  a track matched to player A and later to B (the unsafe error)
  unflagged_transfers transfers on tracks with no ambiguity flag (no possible_predecessors, no recoveries)
  id_switches         player-centric: changes of a player's matched track id between matched frames
  wrong_recoveries    recovery decisions whose detection is a different player than the track's last one
  duplicate_tracks    frames where an extra OBSERVED track overlaps a matched player (IoU >= match_iou), or an
                      extra CONFIRMED one lies inside his box (IoA >= dup_ioa)
  fragments           per player: distinct track ids - 1
  coverage            share of visible player-frames with a matched OBSERVED track
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    flagged: bool                 # has possible_predecessors or recoveries > 0
    detection_index: Optional[int]


@dataclass(frozen=True)
class EvalFrame:
    frame: int
    gt: Dict[str, Box]
    detections: Sequence[Box]
    tracks: Sequence[TrackBox]                 # OBSERVED tracks only
    recoveries: Sequence[Tuple[str, int]]      # (track_id, detection index) accepted on this frame


@dataclass
class Report:
    gt_boxes: int = 0
    missed_players: int = 0
    missed_by_height: Dict[str, int] = field(default_factory=dict)
    gt_by_height: Dict[str, int] = field(default_factory=dict)
    duplicate_dets: int = 0
    identity_transfers: int = 0
    unflagged_transfers: int = 0
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
                "id_switches": self.id_switches, "wrong_recoveries": self.wrong_recoveries,
                "recoveries": self.recoveries, "duplicate_track_frames": self.duplicate_frames,
                "fragments_total": self.total_fragments, "players": len(self.fragments),
                "track_coverage": round(self.coverage, 4)}


def _ioa(inner: Box, outer: Box) -> float:
    iw = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    ih = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return iw * ih / max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)


def _match(gids: List[str], gt: Dict[str, Box], boxes: Sequence[Box], match_iou: float
           ) -> Tuple[Dict[int, str], np.ndarray]:
    if not gids or not len(boxes):
        return {}, np.zeros((len(gids), len(boxes)))
    ious = iou_matrix(np.array([gt[g] for g in gids]), np.array(boxes, dtype=float))
    rows, cols = linear_sum_assignment(-ious)
    return {int(c): gids[r] for r, c in zip(rows, cols) if ious[r, c] >= match_iou}, ious


def _bin(h: float) -> str:
    for lo, hi in zip(HEIGHT_BINS_PX, HEIGHT_BINS_PX[1:]):
        if lo <= h < hi:
            return f"{lo}-{hi if hi < 1_000_000 else 'inf'}"
    return "inf"


def evaluate(frames: Iterable[EvalFrame], match_iou: float = 0.5, dup_ioa: float = 0.6) -> Report:
    r = Report()
    track_gt: Dict[str, str] = {}
    gt_last: Dict[str, str] = {}
    gt_tracks: Dict[str, List[str]] = {}
    for ef in frames:
        prev_track_gt = dict(track_gt)
        gids = sorted(ef.gt)
        r.gt_boxes += len(gids)
        for g in gids:
            b = _bin(ef.gt[g][3] - ef.gt[g][1])
            r.gt_by_height[b] = r.gt_by_height.get(b, 0) + 1

        det_gt, det_iou = _match(gids, ef.gt, ef.detections, match_iou)
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
            over = [g for gi, g in enumerate(gids) if g in hit and (det_iou[gi, d] >= match_iou or
                                                                  _ioa(ef.detections[d], ef.gt[g]) >= dup_ioa)]
            if over:
                r.duplicate_dets += 1
                r.duplicate_events.append({"frame": ef.frame, "detection": d, "player": over[0]})

        tr_gt, tr_iou = _match(gids, ef.gt, [t.box for t in ef.tracks], match_iou)
        for c, g in tr_gt.items():
            t = ef.tracks[c]
            r.covered += 1
            if t.track_id in track_gt and track_gt[t.track_id] != g:
                r.identity_transfers += 1
                r.unflagged_transfers += 0 if t.flagged else 1
                r.transfer_events.append({"frame": ef.frame, "track_id": t.track_id, "from": track_gt[t.track_id],
                                          "to": g, "flagged": t.flagged})
            track_gt[t.track_id] = g
            if g in gt_last and gt_last[g] != t.track_id:
                r.id_switches += 1
            gt_last[g] = t.track_id
            seq = gt_tracks.setdefault(g, [])
            if t.track_id not in seq:
                seq.append(t.track_id)
        extra = [c for c in range(len(ef.tracks)) if c not in tr_gt]
        if gids and any(tr_iou[:, c].max() >= match_iou or
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
        obs = [TrackBox(t["track_id"], t["box"], t["confirmed"], bool(t["possible_predecessors"]) or t["recoveries"] > 0,
                        t["detection_index"]) for t in fr["tracks"] if t["source"] == "OBSERVED"]
        out.append(EvalFrame(fr["frame"], gt.get(fr["frame"], {}), dets.get(fr["frame"], []), obs,
                             rec.get(fr["frame"], [])))
    return out
