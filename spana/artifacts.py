"""Versioned, validated JSON artifacts passed between layers.

  spana.detections 1.3.0 : per-frame person detections (Pass: detection)
  spana.tracks     2.0.0 : per-frame multi-player tracks (Pass: tracking), consumes a detections artifact

Change log:
  1.1.0 detections: + top-level detection_config (tracking refuses detections made under another config).
  1.2.0 detections: + top-level frame_stride (every N-th frame analysed; frames must be a contiguous run).
  1.3.0 detections: + 7th row column multi_body (0/1). 1.2.0 inputs: migrate with tools/refilter_detections.py.
  1.1.0 tracks: + per-track gaps_bridged / last_gap_s; stricter validation (state/source, boxes, camera,
        time, epoch, ended ids). 1.0.0 documents are not accepted.
  2.0.0 tracks: states ACTIVE/OCCLUDED/LOST (TERMINATED via `ended`), + confirmed, confidence, age_frames,
        missed_frames, occluder, recoveries (replaces gaps_bridged), possible_predecessors; + top-level
        `tracklets` summary. Not backward compatible (state vocabulary changed).

Any change to keys, types, units or meaning increments the version. Writers are atomic and never overwrite.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import OutputConfig
from .records import CameraStatus, Detection, EndReason, FrameTracks, PositionSource, TrackletSummary, TrackState

DETECTIONS_SCHEMA = ("spana.detections", "1.3.0")
TRACKS_SCHEMA = ("spana.tracks", "2.0.0")


class ArtifactError(ValueError):
    pass


def canonical_bytes(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_doc(doc: Any) -> str:
    return hashlib.sha256(canonical_bytes(doc)).hexdigest()


def _publish(path: Path, data: bytes) -> str:
    """Write bytes to a unique temp file, then hard-link it into place: atomic and never overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise ArtifactError(f"refusing to overwrite {path}") from None
    finally:
        os.unlink(tmp)
    return hashlib.sha256(data).hexdigest()


def refuse_existing(*paths: str | Path) -> None:
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise ArtifactError(f"refusing to overwrite {existing}")


def write_json_atomic(path: str | Path, doc: Any) -> str:
    """Write canonical JSON; refuse to overwrite. Returns the sha256 of the written bytes."""
    return _publish(Path(path), canonical_bytes(doc))


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> str:
    data = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                    for r in rows)
    return _publish(Path(path), data)


def read_json(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return json.loads(fh.read())


# ----------------------------------------------------------------------------- detections
def detection_frame(frame: int, t: float, dets: Sequence[Detection], oc: OutputConfig) -> dict:
    return {"frame": frame, "t": round(t, 6),
            "detections": [[*(round(v, oc.box_decimals) for v in d.box), round(d.score, oc.score_decimals),
                            d.class_id, int(d.multi_body)] for d in dets]}


def detections_from_frame(fr: dict) -> List[Detection]:
    return [Detection((float(r[0]), float(r[1]), float(r[2]), float(r[3])), float(r[4]), int(r[5]), bool(r[6]))
            for r in fr["detections"]]


def validate_detections(doc: Any) -> None:
    _header(doc, DETECTIONS_SCHEMA, ("schema", "schema_version", "provenance", "video", "model",
                                     "detection_config", "frame_stride", "frames"))
    if not (isinstance(doc["frame_stride"], int) and doc["frame_stride"] >= 1):
        raise ArtifactError("bad frame_stride")
    if not doc["frames"]:
        raise ArtifactError("detections artifact has no frames")
    prev = -1
    for fr in doc["frames"]:
        _keys(fr, ("frame", "t", "detections"), "frame")
        prev = _frame_order(fr, prev)
        for r in fr["detections"]:
            if not (isinstance(r, list) and len(r) == 7 and all(_num(v) for v in r[:5]) and isinstance(r[5], int)
                    and r[6] in (0, 1) and not isinstance(r[6], bool)):
                raise ArtifactError(f"frame {fr['frame']}: bad detection row {r!r}")
            if not (r[2] > r[0] and r[3] > r[1] and 0.0 <= r[4] <= 1.0):
                raise ArtifactError(f"frame {fr['frame']}: invalid box/score {r!r}")


# ----------------------------------------------------------------------------- tracks
def tracks_frame(ft: FrameTracks, oc: OutputConfig) -> dict:
    cam = ft.camera
    return {
        "frame": ft.frame_index, "t": round(ft.time_s, 6),
        "camera": {"status": cam.status.value, "affine": [list(r) for r in cam.affine], "inliers": cam.inliers,
                   "hist_distance": None if cam.hist_distance is None else round(cam.hist_distance, 6)},
        "tracks": [{
            "track_id": o.track_id, "state": o.state.value, "confirmed": o.confirmed, "source": o.source.value,
            "box": [round(v, oc.box_decimals) for v in o.box], "detection_index": o.detection_index,
            "detection_score": None if o.detection_score is None else round(o.detection_score, oc.score_decimals),
            "confidence": round(o.confidence, oc.score_decimals), "contested": o.contested,
            "age_frames": o.age_frames, "missed_frames": o.missed_frames,
            "time_since_observed_s": o.time_since_observed_s, "observed_frames": o.observed_frames,
            "epoch": o.epoch, "occluder": o.occluder,
            "max_overlap_iou": round(o.max_overlap_iou, oc.score_decimals),
            "overlap_with": o.overlap_with if round(o.max_overlap_iou, oc.score_decimals) > 0 else None,
            "recoveries": o.recoveries, "last_gap_s": o.last_gap_s,
            "possible_predecessors": list(o.possible_predecessors),
        } for o in ft.tracks],
        "ended": [[tid, r.value] for tid, r in ft.ended],
    }


def tracklet_rows(summaries: Iterable[TrackletSummary]) -> List[dict]:
    return [{**s.__dict__, "possible_predecessors": list(s.possible_predecessors)} for s in summaries]


_TRACK_KEYS = ("track_id", "state", "confirmed", "source", "box", "detection_index", "detection_score",
               "confidence", "contested", "age_frames", "missed_frames", "time_since_observed_s",
               "observed_frames", "epoch", "occluder", "max_overlap_iou", "overlap_with", "recoveries",
               "last_gap_s", "possible_predecessors")
_TRACKLET_KEYS = ("track_id", "epoch", "first_frame", "last_frame", "last_observed_frame", "observed_frames",
                  "confirmed", "end_reason", "recoveries", "possible_predecessors")
_CAMERA_KEYS = ("status", "affine", "inliers", "hist_distance")
_SOURCES_FOR_STATE = {TrackState.ACTIVE.value: {PositionSource.OBSERVED.value},
                      TrackState.OCCLUDED.value: {PositionSource.PREDICTED.value, PositionSource.HELD.value},
                      TrackState.LOST.value: {PositionSource.PREDICTED.value, PositionSource.HELD.value}}


def validate_tracks(doc: Any, detections: Optional[dict] = None) -> None:
    """Structural and temporal validation. With the source `detections` artifact, every observation is also
    checked against the detection it claims (index in range, same box and score)."""
    _header(doc, TRACKS_SCHEMA, ("schema", "schema_version", "provenance", "video", "detections_sha256", "frames",
                                 "tracklets"))
    det_rows = None
    if detections is not None:
        if sha256_doc(detections) != doc["detections_sha256"]:
            raise ArtifactError("detections artifact does not match detections_sha256")
        det_rows = {fr["frame"]: fr["detections"] for fr in detections["frames"]}
    if not doc["frames"]:
        raise ArtifactError("tracks artifact has no frames")
    cams = {s.value for s in CameraStatus}
    reasons = {r.value for r in EndReason}
    prev, prev_t, prev_epoch = -1, -math.inf, 0
    seen: Dict[str, int] = {}       # track id -> epoch
    live_prev: set = set()
    ended_ids: set = set()
    last: Dict[str, dict] = {}      # track id -> its record on the previous frame it was live
    for n, fr in enumerate(doc["frames"]):
        _keys(fr, ("frame", "t", "camera", "tracks", "ended"), "frame")
        prev = _frame_order(fr, prev)
        where = f"frame {fr['frame']}"
        if not _num(fr["t"]) or fr["t"] <= prev_t:
            raise ArtifactError(f"{where}: time not strictly increasing")
        prev_t = fr["t"]
        cam = fr["camera"]
        _keys(cam, _CAMERA_KEYS, f"{where} camera")
        if cam["status"] not in cams or (cam["status"] == CameraStatus.FIRST.value) != (n == 0):
            raise ArtifactError(f"{where}: bad camera status {cam['status']!r}")
        a = cam["affine"]
        if not (isinstance(a, list) and len(a) == 2 and all(isinstance(r, list) and len(r) == 3
                                                            and all(_num(v) for v in r) for r in a)):
            raise ArtifactError(f"{where}: bad camera affine")
        if not (isinstance(cam["inliers"], int) and cam["inliers"] >= 0):
            raise ArtifactError(f"{where}: bad camera inliers")
        if cam["hist_distance"] is not None and not _num(cam["hist_distance"]):
            raise ArtifactError(f"{where}: bad hist_distance")
        for e in fr["ended"]:
            if not (isinstance(e, list) and len(e) == 2 and e[1] in reasons and e[0] in live_prev):
                raise ArtifactError(f"{where}: bad ended entry {e!r}")
        ended_now = {e[0] for e in fr["ended"]}
        breaks = cam["status"] in (CameraStatus.CUT.value, CameraStatus.UNKNOWN.value)
        if breaks and ended_now != live_prev:
            raise ArtifactError(f"{where}: {cam['status']} must end every live track")
        frame_ids = {t.get("track_id") for t in fr["tracks"] if isinstance(t, dict)}
        if live_prev - frame_ids - ended_now:
            raise ArtifactError(f"{where}: tracks vanished without an end: {sorted(live_prev - frame_ids - ended_now)}")
        used_dets: set = set()
        ids: set = set()
        for tr in fr["tracks"]:
            _keys(tr, _TRACK_KEYS, where)
            tid = tr["track_id"]
            if not isinstance(tid, str) or tid in ids:
                raise ArtifactError(f"{where}: duplicate or bad track id {tid!r}")
            ids.add(tid)
            if tr["source"] not in _SOURCES_FOR_STATE.get(tr["state"], set()):
                raise ArtifactError(f"{where}: {tid} state {tr['state']!r} with source {tr['source']!r}")
            b = tr["box"]
            if not (isinstance(b, list) and len(b) == 4 and all(_num(v) for v in b) and b[2] > b[0] and b[3] > b[1]):
                raise ArtifactError(f"{where}: {tid} bad box")
            observed = tr["source"] == PositionSource.OBSERVED.value
            di, ds = tr["detection_index"], tr["detection_score"]
            if observed != (di is not None) or observed != (ds is not None):
                raise ArtifactError(f"{where}: {tid} detection fields disagree with source")
            if observed and not (isinstance(di, int) and di >= 0 and _num(ds) and 0 <= ds <= 1):
                raise ArtifactError(f"{where}: {tid} bad detection index/score")
            if not observed and tr["contested"]:
                raise ArtifactError(f"{where}: {tid} contested without an observation")
            if not _num(tr["time_since_observed_s"]) or (observed != (tr["time_since_observed_s"] == 0)):
                raise ArtifactError(f"{where}: {tid} time_since_observed_s disagrees with source")
            if (tr["overlap_with"] is None) != (tr["max_overlap_iou"] == 0) or tr["overlap_with"] == tid:
                raise ArtifactError(f"{where}: {tid} overlap fields disagree")
            if not (isinstance(tr["recoveries"], int) and tr["recoveries"] >= 0 and _num(tr["last_gap_s"])
                    and (tr["recoveries"] == 0) == (tr["last_gap_s"] == 0)):
                raise ArtifactError(f"{where}: {tid} bad recovery fields")
            if not (isinstance(tr["confirmed"], bool) and _num(tr["confidence"]) and 0 <= tr["confidence"] <= 1):
                raise ArtifactError(f"{where}: {tid} bad confirmed/confidence")
            if not (isinstance(tr["age_frames"], int) and isinstance(tr["missed_frames"], int)
                    and 0 <= tr["missed_frames"] <= tr["age_frames"] and (tr["missed_frames"] == 0) == observed):
                raise ArtifactError(f"{where}: {tid} bad age/missed frames")
            if (tr["occluder"] is not None) != (tr["state"] == TrackState.OCCLUDED.value) or tr["occluder"] == tid:
                raise ArtifactError(f"{where}: {tid} occluder disagrees with state")
            if not (isinstance(tr["possible_predecessors"], list)
                    and all(isinstance(x, str) and x != tid for x in tr["possible_predecessors"])):
                raise ArtifactError(f"{where}: {tid} bad possible_predecessors")
            ep = tr["epoch"]
            if not isinstance(ep, int) or ep < prev_epoch or seen.get(tid, ep) != ep:
                raise ArtifactError(f"{where}: {tid} epoch inconsistent")
            if observed:
                if di in used_dets:
                    raise ArtifactError(f"{where}: detection {di} used by two tracks")
                used_dets.add(di)
                if det_rows is not None:
                    rows = det_rows.get(fr["frame"])
                    if rows is None or di >= len(rows) or rows[di][:4] != b or rows[di][4] != ds:
                        raise ArtifactError(f"{where}: {tid} does not match detection {di}")
            if tr["occluder"] is not None and tr["occluder"] not in frame_ids:
                raise ArtifactError(f"{where}: {tid} occluder {tr['occluder']} is not live")
            if any(p not in seen for p in tr["possible_predecessors"]):
                raise ArtifactError(f"{where}: {tid} names an unknown predecessor")
            if not isinstance(tr["observed_frames"], int) or tr["observed_frames"] < 1:
                raise ArtifactError(f"{where}: {tid} bad observed_frames")
            if tid not in seen:
                if not (tr["age_frames"] == 0 and tr["observed_frames"] == 1 and observed and not tr["confirmed"]):
                    raise ArtifactError(f"{where}: new track {tid} does not start as a fresh observation")
                if breaks and ep <= prev_epoch and n > 0:
                    raise ArtifactError(f"{where}: {tid} after {cam['status']} must be in a new epoch")
            else:
                p = last[tid]
                step = fr["frame"] - p["frame"]
                if tr["age_frames"] != p["age_frames"] + step or \
                        tr["observed_frames"] != p["observed_frames"] + (1 if observed else 0) or \
                        tr["missed_frames"] != (0 if observed else p["missed_frames"] + 1) or \
                        (p["confirmed"] and not tr["confirmed"]):
                    raise ArtifactError(f"{where}: {tid} counters inconsistent with the previous frame")
            seen[tid] = ep
            last[tid] = {"frame": fr["frame"], "age_frames": tr["age_frames"], "observed_frames": tr["observed_frames"],
                         "missed_frames": tr["missed_frames"], "confirmed": tr["confirmed"]}
        prev_epoch = max([prev_epoch] + [t["epoch"] for t in fr["tracks"]])
        ended_ids |= ended_now
        if ids & ended_ids:
            raise ArtifactError(f"{where}: an ended track id reappears {sorted(ids & ended_ids)}")
        live_prev = ids
    rows = doc["tracklets"]
    if not isinstance(rows, list):
        raise ArtifactError("tracklets must be a list")
    tids = set()
    for r in rows:
        _keys(r, _TRACKLET_KEYS, "tracklet")
        if r["track_id"] in tids or r["track_id"] not in seen:
            raise ArtifactError(f"tracklet {r['track_id']!r} duplicated or never emitted")
        tids.add(r["track_id"])
        if r["end_reason"] is not None and r["end_reason"] not in reasons:
            raise ArtifactError(f"tracklet {r['track_id']}: bad end_reason")
        if (r["end_reason"] is None) != (r["track_id"] in live_prev):
            raise ArtifactError(f"tracklet {r['track_id']}: end_reason disagrees with the last frame")
    if tids != set(seen):
        raise ArtifactError("tracklets do not cover every emitted track")


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
