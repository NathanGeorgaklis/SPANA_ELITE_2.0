"""Adapter: synthetic RunResult -> spana.evaluation (the same scorer the DEV validation tool uses)."""
from __future__ import annotations

from spana.evaluation import EvalFrame, Report, TrackBox
from spana.evaluation import evaluate as _evaluate
from spana.records import PositionSource

Metrics = Report


def evaluate(res) -> Report:
    frames = []
    for f, ft in enumerate(res.frames):
        obs = [TrackBox(o.track_id, o.box, o.confirmed, bool(o.possible_predecessors) or o.recoveries > 0,
                        o.detection_index) for o in ft.tracks if o.source == PositionSource.OBSERVED]
        idx = {o.track_id: o.detection_index for o in ft.tracks}
        rec = [(d["track_id"], idx[d["track_id"]]) for d in ft.decisions if d["kind"] == "recovery"]
        dets = [d_box for d_box in res.det_boxes[f]]
        frames.append(EvalFrame(f, res.gt_boxes[f], dets, obs, rec))
    return _evaluate(frames)
