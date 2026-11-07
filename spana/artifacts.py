"""Versioned, validated JSON artifacts passed between layers.

  spana.detections 1.0.0 : per-frame person detections (Pass: detection)
  spana.tracks     1.0.0 : per-frame multi-player tracks (Pass: tracking), consumes a detections artifact

Any change to keys, types, units or meaning increments the version. Writers are atomic and never overwrite.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import OutputConfig
from .records import CameraStatus, Detection, EndReason, FrameTracks, PositionSource, TrackState

DETECTIONS_SCHEMA = ("spana.detections", "1.0.0")
TRACKS_SCHEMA = ("spana.tracks", "1.0.0")


class ArtifactError(ValueError):
    pass


def canonical_bytes(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_doc(doc: Any) -> str:
    return hashlib.sha256(canonical_bytes(doc)).hexdigest()


def write_json_atomic(path: str | Path, doc: Any) -> str:
    """Write canonical JSON; refuse to overwrite. Returns the sha256 of the written bytes."""
    p = Path(path)
    if p.exists():
        raise ArtifactError(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(doc)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return hashlib.sha256(data).hexdigest()


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> str:
    p = Path(path)
    if p.exists():
        raise ArtifactError(f"refusing to overwrite {p}")
    tmp = p.with_suffix(p.suffix + ".tmp")
    h = hashlib.sha256()
    with open(tmp, "wb") as fh:
        for r in rows:
            line = json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
            h.update(line)
            fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return h.hexdigest()


def read_json(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return json.loads(fh.read())


# ----------------------------------------------------------------------------- detections
def detection_frame(frame: int, t: float, dets: Sequence[Detection], oc: OutputConfig) -> dict:
    return {"frame": frame, "t": round(t, 6),
            "detections": [[*(round(v, oc.box_decimals) for v in d.box), round(d.score, oc.score_decimals),
                            d.class_id] for d in dets]}


def detections_from_frame(fr: dict) -> List[Detection]:
    return [Detection((float(r[0]), float(r[1]), float(r[2]), float(r[3])), float(r[4]), int(r[5]))
            for r in fr["detections"]]


def validate_detections(doc: Any) -> None:
    _header(doc, DETECTIONS_SCHEMA, ("schema", "schema_version", "provenance", "video", "model", "frames"))
    prev = -1
    for fr in doc["frames"]:
        _keys(fr, ("frame", "t", "detections"), "frame")
        prev = _frame_order(fr, prev)
        for r in fr["detections"]:
            if not (isinstance(r, list) and len(r) == 6 and all(_num(v) for v in r[:5]) and isinstance(r[5], int)):
                raise ArtifactError(f"frame {fr['frame']}: bad detection row {r!r}")
            if not (r[2] >= r[0] and r[3] >= r[1] and 0.0 <= r[4] <= 1.0):
                raise ArtifactError(f"frame {fr['frame']}: invalid box/score {r!r}")


# ----------------------------------------------------------------------------- tracks
def tracks_frame(ft: FrameTracks, oc: OutputConfig) -> dict:
    cam = ft.camera
    return {
        "frame": ft.frame_index, "t": round(ft.time_s, 6),
        "camera": {"status": cam.status.value, "affine": [list(r) for r in cam.affine], "inliers": cam.inliers,
                   "hist_distance": None if cam.hist_distance is None else round(cam.hist_distance, 6)},
        "tracks": [{
            "track_id": o.track_id, "state": o.state.value, "source": o.source.value,
            "box": [round(v, oc.box_decimals) for v in o.box], "detection_index": o.detection_index,
            "detection_score": None if o.detection_score is None else round(o.detection_score, oc.score_decimals),
            "contested": o.contested, "time_since_observed_s": o.time_since_observed_s,
            "observed_frames": o.observed_frames, "epoch": o.epoch,
            "max_overlap_iou": round(o.max_overlap_iou, oc.score_decimals), "overlap_with": o.overlap_with,
        } for o in ft.tracks],
        "ended": [[tid, r.value] for tid, r in ft.ended],
    }


_TRACK_KEYS = ("track_id", "state", "source", "box", "detection_index", "detection_score", "contested",
               "time_since_observed_s", "observed_frames", "epoch", "max_overlap_iou", "overlap_with")


def validate_tracks(doc: Any) -> None:
    _header(doc, TRACKS_SCHEMA, ("schema", "schema_version", "provenance", "video", "detections_sha256", "frames"))
    states = {s.value for s in TrackState if s != TrackState.REMOVED}
    sources = {s.value for s in PositionSource}
    cams = {s.value for s in CameraStatus}
    reasons = {r.value for r in EndReason}
    prev = -1
    for fr in doc["frames"]:
        _keys(fr, ("frame", "t", "camera", "tracks", "ended"), "frame")
        prev = _frame_order(fr, prev)
        where = f"frame {fr['frame']}"
        if fr["camera"].get("status") not in cams:
            raise ArtifactError(f"{where}: bad camera status")
        ids = set()
        for tr in fr["tracks"]:
            _keys(tr, _TRACK_KEYS, where)
            if tr["track_id"] in ids:
                raise ArtifactError(f"{where}: duplicate track {tr['track_id']}")
            ids.add(tr["track_id"])
            if tr["state"] not in states or tr["source"] not in sources:
                raise ArtifactError(f"{where}: bad state/source for {tr['track_id']}")
            observed = tr["source"] == PositionSource.OBSERVED.value
            if observed != (tr["detection_index"] is not None) or observed != (tr["detection_score"] is not None):
                raise ArtifactError(f"{where}: {tr['track_id']} detection fields disagree with source")
            if not observed and tr["contested"]:
                raise ArtifactError(f"{where}: {tr['track_id']} contested without an observation")
            if (tr["overlap_with"] is None) != (tr["max_overlap_iou"] == 0):
                raise ArtifactError(f"{where}: {tr['track_id']} overlap fields disagree")
            if tr["overlap_with"] is not None and tr["overlap_with"] == tr["track_id"]:
                raise ArtifactError(f"{where}: {tr['track_id']} overlaps itself")
            if observed and tr["time_since_observed_s"] != 0:
                raise ArtifactError(f"{where}: {tr['track_id']} observed but time_since_observed_s != 0")
        for e in fr["ended"]:
            if not (isinstance(e, list) and len(e) == 2 and e[1] in reasons):
                raise ArtifactError(f"{where}: bad ended entry {e!r}")


# ----------------------------------------------------------------------------- helpers
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _keys(obj: Any, keys: Sequence[str], where: str) -> None:
    if not isinstance(obj, dict) or set(obj) != set(keys):
        got = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise ArtifactError(f"{where}: keys {got} != {sorted(keys)}")


def _header(doc: Any, schema: tuple, keys: Sequence[str]) -> None:
    _keys(doc, keys, "document")
    if (doc["schema"], doc["schema_version"]) != schema:
        raise ArtifactError(f"schema {doc['schema']}/{doc['schema_version']} != {schema[0]}/{schema[1]}")


def _frame_order(fr: Dict[str, Any], prev: int) -> int:
    if not isinstance(fr["frame"], int) or fr["frame"] <= prev:
        raise ArtifactError(f"frames not strictly increasing at {fr.get('frame')!r}")
    return fr["frame"]
