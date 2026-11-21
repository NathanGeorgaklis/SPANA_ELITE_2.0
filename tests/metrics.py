"""Adapter: synthetic RunResult -> spana.evaluation (the same scorer the DEV validation tool uses)."""
from __future__ import annotations

from pathlib import Path

from spana.evaluation import EvalFrame, Report, TrackBox, TransferFlags
from spana.evaluation import evaluate as _evaluate
from spana.records import PositionSource
from tools.benchmark import DEFAULT_BENCH_CONFIG, load_bench_config

TOLERANCE_S = load_bench_config(DEFAULT_BENCH_CONFIG).transfer_flag_tolerance_s

Metrics = Report


def evaluate(res) -> Report:
    frames = []
    for f, ft in enumerate(res.frames):
        obs = [TrackBox(o.track_id, o.box, o.confirmed, o.detection_index, o.identity.value != "CERTAIN")
               for o in ft.tracks if o.source == PositionSource.OBSERVED]
        idx = {o.track_id: o.detection_index for o in ft.tracks}
        rec = [(d["track_id"], idx[d["track_id"]]) for d in ft.decisions if d["kind"] == "recovery"]
        dets = [d_box for d_box in res.det_boxes[f]]
        frames.append(EvalFrame(f, res.gt_boxes[f], dets, obs, rec))
    flags = TransferFlags(((f, ft.time_s, [(o.track_id, o.identity.value != "CERTAIN", o.recoveries)
                                           for o in ft.tracks]) for f, ft in enumerate(res.frames)), {}, TOLERANCE_S)
    return _evaluate(frames, flags=flags)
