"""Candidate D: GTA-Link offline tracklet splitting + connecting (MIT code, pinned checkout in bench/external/),
applied to a finished online run. Features = the cache's SPANA re-ID v2 embeddings (their OSNet weights unused)."""
from __future__ import annotations

import sys
from typing import Dict, List, Tuple

from spana.tracking.protocol import TrackOutput

from ..config import Candidate
from ..external import verified_path


def gta_link(cand: Candidate, run: list, cache, external_dir) -> list:
    """run: [(frame index, [TrackOutput])] aligned with cache.frames. Returns the relinked run, same shape."""
    path = str(verified_path(external_dir, cand))
    if path not in sys.path:
        sys.path.insert(0, path)
    import refine_tracklets as R                                   # their module
    from Tracklet import Tracklet
    if not cache.has_embeddings:
        raise ValueError("GTA-Link needs per-detection embeddings (build the cache with --embed)")
    p = cand.params
    if not p["use_connect"]:
        raise ValueError("GTA-Link without the connector is not the published method")
    int_id: Dict[str, int] = {}
    tracklets: Dict[int, object] = {}
    where: Dict[Tuple[int, Tuple[float, ...]], Tuple[int, bool]] = {}
    for (f, outs), cf in zip(run, cache.frames):
        if f != cf.index:
            raise ValueError("run and cache frames differ")
        for o in outs:
            tid = int_id.setdefault(o.track_id, len(int_id) + 1)
            x1, y1, x2, y2 = cf.detections[o.detection_index].box
            tlwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            key = (f, tuple(round(v, 3) for v in tlwh))
            if key in where:
                raise ValueError(f"frame {f}: two tracks on identical boxes; cannot map GTA-Link output back")
            where[key] = (o.detection_index, o.identity_certain)
            feat = cf.embeddings[o.detection_index].astype(float)
            if tid not in tracklets:
                tracklets[tid] = Tracklet(tid, f, 1.0, tlwh)
            else:
                tracklets[tid].append_det(f, 1.0, tlwh)
            tracklets[tid].append_feat(feat)
    if not tracklets:
        return [(f, []) for f, _ in run]
    max_x, max_y = R.get_spatial_constraints(tracklets, p["spatial_factor"])
    if p["use_split"]:
        tracklets = R.split_tracklets(tracklets, eps=p["eps"], max_k=p["max_k"], min_samples=p["min_samples"],
                                      len_thres=p["min_len"])
    dist = R.get_distance_matrix(tracklets)
    merged = R.merge_tracklets(tracklets, {}, dist, seq_name=None, max_x_range=max_x, max_y_range=max_y,
                               merge_dist_thres=p["merge_dist_thres"])
    per_frame: Dict[int, List[TrackOutput]] = {f: [] for f, _ in run}
    for n, tid in enumerate(sorted(merged), 1):                 # their save_results numbering
        tr = merged[tid]
        for f, b in zip(tr.times, tr.bboxes):
            k, certain = where[(int(f), tuple(round(float(v), 3) for v in b[:4]))]
            per_frame[int(f)].append((f"G{n:06d}", k, certain))
    by_index = {cf.index: cf for cf in cache.frames}
    return [(f, [TrackOutput(t, k, tuple(by_index[f].detections[k].box), c) for t, k, c in per_frame[f]])
            for f, _ in run]
