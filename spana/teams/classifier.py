"""Team classification v1: TEAM_A / TEAM_B / GOALKEEPER / REFEREE / UNKNOWN per track segment, per match.

Design: records/design/TEAM_CLASSIFICATION_V1_DESIGN.md (Nathan's go-ahead 2026-11-22). No kit colour or team is
known in advance:

1. Evidence: a track-frame gives evidence only when the track is confirmed, the row is OBSERVED and identity CERTAIN,
   the camera is OK / FIRST, the box is not flagged multi-body, is at least `appearance.min_crop_height_px` tall and
   overlaps no other detection above `appearance.clean_max_iou`. Sampled every `team.sample_every_s` per track.
2. Segments: a track is split wherever it is not (confirmed and CERTAIN): a rival swap inside an uncertainty episode
   changes the team and the tracker cannot tell it from a teammate swap. Each segment is labelled on its own; a
   segment never inherits another's label.
3. Match model: a weighted, trimmed 2-means of the segments' median torso histograms (Hellinger distance, computed as
   the Euclidean distance of square-root histograms / sqrt 2). Deterministic: density seeds (outliers cannot seed a team), ties by
   (clip, frame, track). It abstains when the centres are too close, when the clusters overlap (separation not
   above `team.min_separation_radius_ratio` x the larger cluster radius), or when a cluster is too small.
4. Team label: every evidence frame votes; a segment is TEAM_X only with enough votes and vote share for its nearest
   cluster X, and only when it is not a colour outlier.
5. Roles (`team.roles_enabled`, OFF in v1, decision 2026-11-22): a colour outlier is UNKNOWN (COLOUR_OUTLIER),
   never the nearest team. With roles on, only a colour outlier (far from both centres) can be REFEREE / GOALKEEPER, and only with a detector
   role-hint majority. The detector hint alone never decides (it confuses kits with roles on DEV).

Fail closed: anything else is UNKNOWN with a reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..appearance import torso_descriptor
from ..config import AppearanceConfig, TeamConfig, frames_for
from ..geometry import iou_matrix
from ..records import CameraStatus, IdentityState, PositionSource, TeamLabel, TeamReason

_EVIDENCE_CAMERA = (CameraStatus.OK.value, CameraStatus.FIRST.value)
_SQRT2 = math.sqrt(2.0)


@dataclass(eq=False)   # identity semantics: segments hold arrays
class Segment:
    clip: str
    track_id: str
    frame: int
    frame_end: int
    evidence: List[Tuple[int, float, np.ndarray]] = field(default_factory=list)  # (frame, t, sqrt-histogram)
    role_rows: Dict[str, int] = field(default_factory=dict)   # detector class name -> observed rows
    observed_rows: int = 0

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.clip, self.frame, self.track_id)

    def eligible(self, cfg: TeamConfig) -> bool:
        if len(self.evidence) < cfg.min_segment_frames:
            return False
        return self.evidence[-1][1] - self.evidence[0][1] >= cfg.min_segment_s

    def descriptor(self) -> np.ndarray:
        """Per-bin median of the evidence histograms, renormalised, as a unit square-root histogram."""
        hist = np.median(np.stack([u * u for _, _, u in self.evidence]), axis=0)
        total = float(hist.sum())
        return np.sqrt(hist / total) if total > 0 else np.sqrt(np.full_like(hist, 1.0 / hist.size))


def hellinger(u: np.ndarray, v: np.ndarray) -> float:
    """Hellinger distance of two square-root histograms (unit L2 vectors), in [0, 1]."""
    return float(min(1.0, np.linalg.norm(u - v) / _SQRT2))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def segments_of_clip(clip: str, tracks: dict, detections: dict, frames: Iterable[Tuple[int, np.ndarray]],
                     acfg: AppearanceConfig, tcfg: TeamConfig) -> List[Segment]:
    """Segments of one clip with their evidence. `frames` yields (frame index, BGR image) for at least every tracks
    frame (others are skipped)."""
    fps = float(tracks["video"]["fps"])
    step = frames_for(tcfg.sample_every_s, fps)
    names = list(detections["class_names"])
    det = {f["frame"]: f["detections"] for f in detections["frames"]}
    open_: Dict[str, Segment] = {}
    done: List[Segment] = []
    last_sample: Dict[str, int] = {}
    tframes = {fr["frame"]: fr for fr in tracks["frames"]}
    for fi, img in frames:
        fr = tframes.get(fi)
        if fr is None:
            continue
        rows = det.get(fi, [])
        boxes = np.asarray([r[:4] for r in rows], dtype=float).reshape(-1, 4)
        ious = iou_matrix(boxes, boxes) if len(rows) else np.zeros((0, 0))
        if len(rows):
            np.fill_diagonal(ious, 0.0)
        live = set()
        for r in fr["tracks"]:
            tid = r["track_id"]
            ok = r["confirmed"] and r["identity"] == IdentityState.CERTAIN.value
            if not ok:
                if tid in open_:
                    done.append(open_.pop(tid))
                continue
            live.add(tid)
            seg = open_.get(tid)
            if seg is None:
                seg = open_[tid] = Segment(clip, tid, fi, fi)
            seg.frame_end = fi
            if r["source"] != PositionSource.OBSERVED.value or r["detection_index"] is None:
                continue
            k = r["detection_index"]
            seg.observed_rows += 1
            cname = names[rows[k][5]]
            seg.role_rows[cname] = seg.role_rows.get(cname, 0) + 1
            if fr["camera"]["status"] not in _EVIDENCE_CAMERA or bool(rows[k][6]):
                continue
            if len(rows) > 1 and ious[k].max() > acfg.clean_max_iou:
                continue
            if tid in last_sample and fi - last_sample[tid] < step:
                continue
            h = torso_descriptor(img, rows[k][:4], acfg)
            if h is None:
                continue
            last_sample[tid] = fi
            seg.evidence.append((fi, float(fr["t"]), np.sqrt(h)))
        for tid in [t for t in open_ if t not in live]:   # track ended or left the document
            done.append(open_.pop(tid))
    done.extend(open_.values())
    return sorted(done, key=lambda s: s.key)


def _fit(us: List[np.ndarray], ws: List[float], radius: float, max_iter: int
         ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Weighted 2-means on unit vectors (Hellinger geometry), deterministic.

    Seeds by density, not by the farthest pair (which picks two colour outliers, e.g. a referee and a keeper): the
    first seed is the segment with the most evidence within `radius`, the second the densest segment farther than
    `radius` from the first (the farthest one when none is). Ties by input order (sorted segment keys)."""
    n = len(us)
    dist = [[hellinger(us[i], us[j]) for j in range(n)] for i in range(n)]
    dens = [sum(ws[j] for j in range(n) if dist[i][j] <= radius) for i in range(n)]
    first = max(range(n), key=lambda i: (dens[i], -i))
    far = [i for i in range(n) if dist[first][i] > radius]
    second = max(far, key=lambda i: (dens[i], -i)) if far else max(range(n), key=lambda i: (dist[first][i], -i))
    ca, cb = us[first].copy(), us[second].copy()
    assign: List[int] = []
    for _ in range(max_iter):
        new = [0 if hellinger(u, ca) <= hellinger(u, cb) else 1 for u in us]
        if new == assign:
            break
        assign = new
        for c in (0, 1):
            idx = [i for i in range(n) if assign[i] == c]
            if idx:
                m = _unit(sum(ws[i] * us[i] for i in idx))
                if c == 0:
                    ca = m
                else:
                    cb = m
    return ca, cb, assign


def model_verdict(separation: float, radii: Sequence[float], sizes: Sequence[int], tcfg: TeamConfig
                  ) -> Optional[str]:
    """None when the two-team model may answer, else why the whole match abstains."""
    if separation < tcfg.min_cluster_separation:
        return "CLUSTERS_NOT_SEPARATED"
    if separation <= tcfg.min_separation_radius_ratio * max(radii):
        return "CLUSTERS_OVERLAP"          # a member is as far from its own centre as the two centres are apart
    if min(sizes) < tcfg.min_cluster_segments:
        return "CLUSTER_TOO_SMALL"
    return None


def classify(segments: Sequence[Segment], tcfg: TeamConfig, decimals: int) -> Tuple[dict, List[dict]]:
    """(model, tracks) parts of a spana.teams document for all segments of one match."""
    r = lambda v: None if v is None else round(float(v), decimals)   # noqa: E731
    segs = sorted(segments, key=lambda s: s.key)
    elig = [s for s in segs if s.eligible(tcfg)]
    desc = {id(s): s.descriptor() for s in elig}
    model = {"status": "ABSTAIN", "reason": None, "clusters": [], "separation": None, "trimmed_segments": 0,
             "outlier_segments": 0}
    centres: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if len(elig) < 2 * tcfg.min_cluster_segments:
        model["reason"] = "TOO_FEW_SEGMENTS"
    else:
        kept = list(elig)
        for _ in range(tcfg.trim_rounds):
            ca, cb, _a = _fit([desc[id(s)] for s in kept], [len(s.evidence) for s in kept],
                              tcfg.outlier_distance, tcfg.max_kmeans_iter)
            nxt = [s for s in kept if min(hellinger(desc[id(s)], ca), hellinger(desc[id(s)], cb))
                   <= tcfg.outlier_distance]
            if len(nxt) == len(kept) or len(nxt) < 2:
                break
            kept = nxt
        model["trimmed_segments"] = len(elig) - len(kept)
        ca, cb, assign = _fit([desc[id(s)] for s in kept], [len(s.evidence) for s in kept],
                              tcfg.outlier_distance, tcfg.max_kmeans_iter)
        members = {0: [s for s, a in zip(kept, assign) if a == 0 and
                       hellinger(desc[id(s)], ca) <= tcfg.outlier_distance],
                   1: [s for s, a in zip(kept, assign) if a == 1 and
                       hellinger(desc[id(s)], cb) <= tcfg.outlier_distance]}
        first = min((s for c in (0, 1) for s in members[c]), key=lambda s: s.key, default=None)
        if first is not None and first in members[1]:      # TEAM_A holds the earliest-starting segment
            ca, cb, members = cb, ca, {0: members[1], 1: members[0]}
        sep = hellinger(ca, cb)
        radii = [max((hellinger(desc[id(s)], cen) for s in members[c]), default=0.0)
                 for c, cen in ((0, ca), (1, cb))]
        model["separation"] = r(sep)
        model["clusters"] = [{"label": lab, "segments": len(members[c]),
                              "evidence_frames": sum(len(s.evidence) for s in members[c]), "radius": r(radii[c])}
                             for c, lab in ((0, TeamLabel.TEAM_A.value), (1, TeamLabel.TEAM_B.value))]
        model["reason"] = model_verdict(sep, radii, (len(members[0]), len(members[1])), tcfg)
        if model["reason"] is None:
            model["status"] = "FITTED"
            centres = (ca, cb)
    tracks: Dict[Tuple[str, str], List[dict]] = {}
    outliers = 0
    for s in segs:
        row = _segment_row(s, s in elig, desc.get(id(s)), centres, tcfg, r)
        outliers += row.pop("_outlier")
        tracks.setdefault((s.clip, s.track_id), []).append(row)
    model["outlier_segments"] = outliers
    out = []
    for (clip, tid), rows in sorted(tracks.items()):
        decided = {x["label"] for x in rows if x["label"] != TeamLabel.UNKNOWN.value}
        out.append({"clip": clip, "track_id": tid, "segments": rows,
                    "flags": ["TEAM_CHANGE"] if len(decided) > 1 else []})
    return model, out


def _segment_row(s: Segment, eligible: bool, d: Optional[np.ndarray], centres, tcfg: TeamConfig, r) -> dict:
    share = {k: r(v / s.observed_rows) for k, v in sorted(s.role_rows.items())} if s.observed_rows else {}
    row = {"frame": s.frame, "frame_end": s.frame_end, "label": TeamLabel.UNKNOWN.value, "reasons": [],
           "evidence_frames": len(s.evidence), "votes": {"TEAM_A": 0, "TEAM_B": 0, "abstain": 0},
           "dist": {"TEAM_A": None, "TEAM_B": None}, "role_hint_share": share, "_outlier": False}
    if centres is not None and s.evidence:
        ca, cb = centres
        for _, _, u in s.evidence:
            da, db = hellinger(u, ca), hellinger(u, cb)
            k = "TEAM_A" if da + tcfg.vote_margin < db else "TEAM_B" if db + tcfg.vote_margin < da else "abstain"
            row["votes"][k] += 1
        if d is None:
            d = s.descriptor()
        row["dist"] = {"TEAM_A": r(hellinger(d, ca)), "TEAM_B": r(hellinger(d, cb))}
    if not eligible:
        row["reasons"] = [TeamReason.TOO_FEW_CLEAN_FRAMES.value]
        return row
    if centres is None:
        row["reasons"] = [TeamReason.MATCH_MODEL_ABSTAIN.value]
        return row
    da, db = hellinger(d, centres[0]), hellinger(d, centres[1])
    if min(da, db) <= tcfg.outlier_distance:
        x = "TEAM_A" if da <= db else "TEAM_B"
        n = len(s.evidence)
        if row["votes"][x] >= tcfg.min_votes and row["votes"][x] / n >= tcfg.min_vote_share:
            row["label"] = x
        else:
            row["reasons"] = [TeamReason.LOW_VOTE_SHARE.value]
        return row
    row["_outlier"] = True
    if not tcfg.roles_enabled:          # v1 (decision 2026-11-22): a colour outlier is never put in the nearest team
        row["reasons"] = [TeamReason.COLOUR_OUTLIER.value]
        return row
    obs = s.observed_rows
    ref, gk = s.role_rows.get(tcfg.referee_class, 0), s.role_rows.get(tcfg.goalkeeper_class, 0)
    def strong(c: int) -> bool:
        return c >= tcfg.role_min_frames and c / obs >= tcfg.role_min_share
    if strong(ref) and gk / obs < tcfg.role_conflict_share:
        row["label"] = TeamLabel.REFEREE.value
    elif strong(gk) and ref / obs < tcfg.role_conflict_share:
        row["label"] = TeamLabel.GOALKEEPER.value
    elif (strong(ref) or strong(gk)) or (ref / obs >= tcfg.role_conflict_share and gk / obs >= tcfg.role_conflict_share):
        row["reasons"] = [TeamReason.ROLE_HINT_CONFLICT.value]
    else:
        row["reasons"] = [TeamReason.COLOUR_OUTLIER_NO_ROLE.value]
    return row


def label_at(doc: dict, clip: str, track_id: str, frame: int) -> Tuple[str, Optional[str]]:
    """(label, reason) of one track-frame in a spana.teams document; OUTSIDE_SEGMENT when no segment holds it."""
    for t in doc["tracks"]:
        if t["clip"] == clip and t["track_id"] == track_id:
            for sg in t["segments"]:
                if sg["frame"] <= frame <= sg["frame_end"]:
                    return sg["label"], (sg["reasons"][0] if sg["reasons"] else None)
    return TeamLabel.UNKNOWN.value, TeamReason.OUTSIDE_SEGMENT.value


def iter_frames(video, first: int, last: int) -> Iterator[Tuple[int, np.ndarray]]:
    for i, _t, img in video.frames(first, last + 1):
        yield i, img
