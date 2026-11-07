"""QA tracking metrics on synthetic ground truth (box-based, so they do not trust the tracker's own indices).

Per frame, OBSERVED track boxes are matched to visible ground-truth boxes by Hungarian on IoU (>= MATCH_IOU).

  identity_transfers : a track matched to ground-truth player A and later to B (A != B). The unsafe error:
                       one track id carrying two physical players (ID switch through the track / false merge).
  id_switches        : ground-truth-centric (MOT-style): times a player's matched track id changes between
                       matched frames. Includes safe fragmentation restarts and unsafe transfers.
  duplicate_frames   : frames where an extra OBSERVED track overlaps an already-matched player (IoU >= MATCH_IOU):
                       one player represented by two simultaneous tracks.
  fragments          : per player, number of distinct track ids minus 1 (false splits).
  wrong_recoveries   : recovery decisions whose detection belongs to a player other than the track's last one.
  coverage           : share of visible player-frames with a matched observed track.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from spana.geometry import iou_matrix
from spana.records import PositionSource

MATCH_IOU = 0.5


@dataclass
class Metrics:
    identity_transfers: int = 0
    id_switches: int = 0
    duplicate_frames: int = 0
    fragments: Dict[str, int] = field(default_factory=dict)
    wrong_recoveries: int = 0
    recoveries: int = 0
    coverage: float = 0.0
    transfer_tracks: Set[str] = field(default_factory=set)

    @property
    def total_fragments(self) -> int:
        return sum(self.fragments.values())


def evaluate(res) -> Metrics:
    m = Metrics()
    track_gt: Dict[str, str] = {}
    gt_last_track: Dict[str, str] = {}
    gt_tracks: Dict[str, List[str]] = {}
    visible = covered = 0
    for f, ft in enumerate(res.frames):
        prev_track_gt = dict(track_gt)
        gt = res.gt_boxes[f]
        gids = sorted(gt)
        obs = [o for o in ft.tracks if o.source == PositionSource.OBSERVED]
        visible += len(gids)
        if not gids or not obs:
            continue
        ious = iou_matrix(np.array([gt[g] for g in gids]), np.array([o.box for o in obs]))
        rows, cols = linear_sum_assignment(-ious)
        matched_obs = set()
        for r, c in zip(rows, cols):
            if ious[r, c] < MATCH_IOU:
                continue
            g, tid = gids[r], obs[c].track_id
            matched_obs.add(c)
            covered += 1
            if tid in track_gt and track_gt[tid] != g:
                m.identity_transfers += 1
                m.transfer_tracks.add(tid)
            track_gt[tid] = g
            if g in gt_last_track and gt_last_track[g] != tid:
                m.id_switches += 1
            gt_last_track[g] = tid
            seq = gt_tracks.setdefault(g, [])
            if tid not in seq:
                seq.append(tid)
        extra = [c for c in range(len(obs)) if c not in matched_obs]
        if any(ious[:, c].max() >= MATCH_IOU for c in extra):
            m.duplicate_frames += 1
        for d in ft.decisions:
            if d["kind"] == "recovery":
                m.recoveries += 1
                det = next(o for o in ft.tracks if o.track_id == d["track_id"]).detection_index
                g_now = res.gt_of_detection[f][det]
                if prev_track_gt.get(d["track_id"]) not in (None, g_now):
                    m.wrong_recoveries += 1
    m.fragments = {g: len(v) - 1 for g, v in gt_tracks.items()}
    m.coverage = covered / visible if visible else 0.0
    return m
