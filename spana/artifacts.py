"""Versioned, validated JSON artifacts passed between layers.

  spana.detections 1.3.0 : per-frame person detections (Pass: detection)
  spana.tracks     2.0.0 : per-frame multi-player tracks (Pass: tracking), consumes a detections artifact
  spana.target     1.0.0 : per-frame state of the user's selected player (Pass: target identity), consumes tracks

Change log:
  1.1.0 detections: + top-level detection_config (tracking refuses detections made under another config).
  1.2.0 detections: + top-level frame_stride (every N-th frame analysed; frames must be a contiguous run).
  1.3.0 detections: + 7th row column multi_body (0/1). 1.2.0 inputs: migrate with tools/refilter_detections.py.
  1.1.0 tracks: + per-track gaps_bridged / last_gap_s; stricter validation (state/source, boxes, camera,
        time, epoch, ended ids). 1.0.0 documents are not accepted.
  2.0.0 tracks: states ACTIVE/OCCLUDED/LOST (TERMINATED via `ended`), + confirmed, confidence, age_frames,
        missed_frames, occluder, recoveries (replaces gaps_bridged), possible_predecessors; + top-level
        `tracklets` summary. Not backward compatible (state vocabulary changed).
  2.1.0 tracks: + per-track identity (CERTAIN / IDENTITY_UNCERTAIN), identity_rivals, uncertain_episodes.
  1.0.0 target: selection (track_at_frame 1.0.0), target_config, track_end, frames {frame, t, state, reason,
        track_id, box}. CANDIDATE reserved (refused in 1.0.0).
  2.0.0 target: relinks. CANDIDATE (RELINK_PREDECESSOR) allowed; TARGET may be on a confirmed relinked track;
        every frame names the track it refers to; + confirmations, relinks log, track_ends (replaces track_end),
        target_config + relink_max_gap_s / relink_auto_confirm. 1.0.0 documents keep their 1.0.0 meaning.
  2.1.0 target: user_actions (confirm_candidate, confirm_relink, reject_candidate, track_at_frame re-selection)
        replace confirmations; relinks + rejection; + gaps (stretches without a TARGET answer, for the review
        timeline). 2.0.0 documents keep their 2.0.0 meaning.

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
from .records import (CameraStatus, Detection, EndReason, FrameTracks, IdentityState, PositionSource, TargetReason,
                      TargetState, TrackletSummary, TrackState)

DETECTIONS_SCHEMA = ("spana.detections", "1.4.0")   # 1.4.0: class_names (class_id indexes it)
TRACKS_SCHEMA = ("spana.tracks", "2.1.0")
TARGET_SCHEMA = ("spana.target", "2.1.0")   # 1.0.0 / 2.0.0 documents still validate (validate_target)


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
    _header(doc, DETECTIONS_SCHEMA, ("schema", "schema_version", "provenance", "video", "model", "class_names",
                                     "detection_config", "frame_stride", "frames"))
    names = doc["class_names"]
    if not (isinstance(names, list) and names and all(isinstance(n, str) and n for n in names)
            and len(set(names)) == len(names)):
        raise ArtifactError("class_names must be a non-empty list of distinct names")
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
            if not 0 <= r[5] < len(names):
                raise ArtifactError(f"frame {fr['frame']}: class_id {r[5]} has no entry in class_names")


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
            "identity": o.identity.value, "identity_rivals": list(o.identity_rivals),
            "uncertain_episodes": o.uncertain_episodes,
        } for o in ft.tracks],
        "ended": [[tid, r.value] for tid, r in ft.ended],
    }


def tracklet_rows(summaries: Iterable[TrackletSummary]) -> List[dict]:
    return [{**s.__dict__, "possible_predecessors": list(s.possible_predecessors)} for s in summaries]


_IDENTITY = {v.value for v in IdentityState}
_TRACK_KEYS = ("track_id", "state", "confirmed", "source", "box", "detection_index", "detection_score",
               "confidence", "contested", "age_frames", "missed_frames", "time_since_observed_s",
               "observed_frames", "epoch", "occluder", "max_overlap_iou", "overlap_with", "recoveries",
               "last_gap_s", "possible_predecessors", "identity", "identity_rivals",
               "uncertain_episodes")
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
            if tr["identity"] not in _IDENTITY or not isinstance(tr["identity_rivals"], list) or \
                    any(not isinstance(x, str) or x == tid for x in tr["identity_rivals"]) or \
                    (tr["identity"] == IdentityState.CERTAIN.value) != (not tr["identity_rivals"]) or \
                    not (isinstance(tr["uncertain_episodes"], int) and tr["uncertain_episodes"] >= 0) or \
                    (tr["identity"] != IdentityState.CERTAIN.value and tr["uncertain_episodes"] < 1):
                raise ArtifactError(f"{where}: {tid} bad identity fields")
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


# ----------------------------------------------------------------------------- target
_TARGET_REASONS = {  # state -> reasons allowed with it (1.0.0)
    TargetState.TARGET.value: {TargetReason.TRACK_OBSERVED.value},
    TargetState.ABSTAIN.value: {TargetReason.NOT_CONFIRMED.value, TargetReason.IDENTITY_UNCERTAIN.value,
                                TargetReason.AFTER_UNCERTAIN.value, TargetReason.TRACK_OCCLUDED.value},
    TargetState.LOST.value: {TargetReason.TRACK_LOST.value, TargetReason.TRACK_ENDED.value},
}
_SELECTIONS = {("track_at_frame", "1.0.0"): ("kind", "version", "frame", "track_id")}


def validate_target(doc: Any, tracks: Optional[dict] = None) -> None:
    """Validate a spana.target document of any supported version (2.1.0 current; 2.0.0 / 1.0.0 as written then)."""
    if not isinstance(doc, dict):
        raise ArtifactError("target document must be an object")
    version = doc.get("schema_version")
    if version == "1.0.0":
        return _validate_target_v1(doc, tracks)
    if version in ("2.0.0", "2.1.0"):
        return _validate_target_v2(doc, tracks, v21=version == "2.1.0")
    raise ArtifactError(f"unsupported spana.target version {version!r}")


def _validate_target_v1(doc: Any, tracks: Optional[dict] = None) -> None:
    """1.0.0. Structural validation. With the source `tracks` artifact, every answer is also checked against it: same
    frames from the selection on, TARGET only on the selected track while it is ACTIVE, confirmed and CERTAIN."""
    _header(doc, ("spana.target", "1.0.0"), ("schema", "schema_version", "provenance", "video", "tracks_sha256", "selection",
                                 "target_config", "track_end", "frames"))
    sel = doc["selection"]
    if not isinstance(sel, dict) or (sel.get("kind"), sel.get("version")) not in _SELECTIONS:
        raise ArtifactError("unsupported selection")
    _keys(sel, _SELECTIONS[(sel["kind"], sel["version"])], "selection")
    if not (isinstance(sel["frame"], int) and not isinstance(sel["frame"], bool) and isinstance(sel["track_id"], str)):
        raise ArtifactError("bad selection frame / track_id")
    tc = doc["target_config"]
    _keys(tc, ("resume_after_uncertain",), "target_config")
    if not isinstance(tc["resume_after_uncertain"], bool):
        raise ArtifactError("bad target_config")
    frames = doc["frames"]
    if not frames or frames[0].get("frame") != sel["frame"]:
        raise ArtifactError("target frames must start at the selection frame")
    end = doc["track_end"]
    if end is not None:
        _keys(end, ("frame", "reason"), "track_end")
        if end["reason"] not in {r.value for r in EndReason}:
            raise ArtifactError("bad track_end reason")
    prev, prev_t, ended = -1, -math.inf, False
    for fr in frames:
        _keys(fr, ("frame", "t", "state", "reason", "track_id", "box"), "target frame")
        prev = _frame_order(fr, prev)
        where = f"target frame {fr['frame']}"
        if not _num(fr["t"]) or fr["t"] <= prev_t:
            raise ArtifactError(f"{where}: time not strictly increasing")
        prev_t = fr["t"]
        if fr["reason"] not in _TARGET_REASONS.get(fr["state"], set()):
            raise ArtifactError(f"{where}: state {fr['state']!r} with reason {fr['reason']!r}")
        is_target = fr["state"] == TargetState.TARGET.value
        b = fr["box"]
        if is_target:
            if fr["track_id"] != sel["track_id"] or not (isinstance(b, list) and len(b) == 4 and all(_num(v) for v in b)
                                                         and b[2] > b[0] and b[3] > b[1]):
                raise ArtifactError(f"{where}: TARGET needs the selected track id and a valid box")
        elif fr["track_id"] is not None or b is not None:
            raise ArtifactError(f"{where}: no track id or box without an answer")
        if end is not None and fr["frame"] == end["frame"]:
            ended = True
        if (fr["reason"] == TargetReason.TRACK_ENDED.value) != ended:
            raise ArtifactError(f"{where}: TRACK_ENDED disagrees with track_end")
    if end is not None and not ended:
        raise ArtifactError("track_end frame is not among the target frames")
    if tracks is None:
        return
    if sha256_doc(tracks) != doc["tracks_sha256"]:
        raise ArtifactError("tracks artifact does not match tracks_sha256")
    if tracks["video"] != doc["video"]:
        raise ArtifactError("target video differs from the tracks video")
    src = [fr for fr in tracks["frames"] if fr["frame"] >= sel["frame"]]
    if [(f["frame"], f["t"]) for f in src] != [(f["frame"], f["t"]) for f in frames]:
        raise ArtifactError("target frames differ from the tracks frames after the selection")
    for tf, fr in zip(src, frames):
        tr = next((t for t in tf["tracks"] if t["track_id"] == sel["track_id"]), None)
        hit = [r for tid, r in tf["ended"] if tid == sel["track_id"]]
        if hit and (end is None or end != {"frame": tf["frame"], "reason": hit[0]}):
            raise ArtifactError(f"frame {tf['frame']}: selected track ended but track_end disagrees")
        if fr["state"] == TargetState.TARGET.value and not (
                tr is not None and tr["state"] == TrackState.ACTIVE.value and tr["confirmed"]
                and tr["identity"] == IdentityState.CERTAIN.value and tr["box"] == fr["box"]):
            raise ArtifactError(f"frame {tf['frame']}: TARGET not backed by an ACTIVE, confirmed, CERTAIN track box")
    first = next((t for t in src[0]["tracks"] if t["track_id"] == sel["track_id"]), None)
    if first is None or first["state"] != TrackState.ACTIVE.value:
        raise ArtifactError("selection does not name an ACTIVE track on its frame")


_TARGET_REASONS_V2 = {**_TARGET_REASONS,
                      TargetState.CANDIDATE.value: {TargetReason.RELINK_PREDECESSOR.value}}
_RELINK_KEYS = ("from_track", "to_track", "lost_frame", "proposed_frame", "gap_s", "evidence", "confirmation",
                "outcome", "dropped_reason")
_CONFIRM = ("confirm_candidate", "1.0.0")
_ACTION_KEYS = {("confirm_candidate", "1.0.0"): ("kind", "version", "frame", "track_id"),
                ("confirm_relink", "1.0.0"): ("kind", "version", "frame", "from_track", "to_track", "proposed_frame"),
                ("reject_candidate", "1.0.0"): ("kind", "version", "frame", "from_track", "to_track", "proposed_frame"),
                ("track_at_frame", "1.0.0"): ("kind", "version", "frame", "track_id")}


def gaps_of(frames: Sequence[dict], relinks: Sequence[dict]) -> List[dict]:
    """spana.target 2.1.0 gaps: maximal runs of frames without a TARGET answer, for the review timeline. Each gap
    lists the states in it and the offers (relinks) shown as CANDIDATE inside it."""
    out, prev = [], None
    for fr in frames:
        if fr["state"] == TargetState.TARGET.value:
            prev = None
            continue
        if prev is None:
            out.append({"start_frame": fr["frame"], "end_frame": fr["frame"], "start_t": fr["t"], "end_t": fr["t"],
                        "frames": 0, "states": [], "offers": []})
        g = out[-1]
        g["end_frame"], g["end_t"] = fr["frame"], fr["t"]
        g["frames"] += 1
        if fr["state"] not in g["states"]:
            g["states"].append(fr["state"])
        if fr["state"] == TargetState.CANDIDATE.value:
            r = max((r for r in relinks if r["to_track"] == fr["track_id"] and r["proposed_frame"] <= fr["frame"]),
                    key=lambda r: r["proposed_frame"])
            key = [r["from_track"], r["to_track"], r["proposed_frame"]]
            if key not in g["offers"]:
                g["offers"].append(key)
        prev = fr
    for g in out:
        g["states"] = sorted(g["states"])
    return out


def _validate_target_v2(doc: Any, tracks: Optional[dict] = None, v21: bool = True) -> None:
    """2.0.0 / 2.1.0. Structure, then (with the source tracks) every answer and every relink against the tracks."""
    version = "2.1.0" if v21 else "2.0.0"
    _header(doc, ("spana.target", version),
            ("schema", "schema_version", "provenance", "video", "tracks_sha256", "selection",
             "user_actions" if v21 else "confirmations", "target_config", "track_ends", "relinks", "frames")
            + (("gaps",) if v21 else ()))
    sel = doc["selection"]
    if not isinstance(sel, dict) or (sel.get("kind"), sel.get("version")) not in _SELECTIONS:
        raise ArtifactError("unsupported selection")
    _keys(sel, _SELECTIONS[(sel["kind"], sel["version"])], "selection")
    if not (isinstance(sel["frame"], int) and not isinstance(sel["frame"], bool) and isinstance(sel["track_id"], str)):
        raise ArtifactError("bad selection frame / track_id")
    tc = doc["target_config"]
    _keys(tc, ("resume_after_uncertain", "relink_max_gap_s", "relink_auto_confirm"), "target_config")
    if not (isinstance(tc["resume_after_uncertain"], bool) and isinstance(tc["relink_auto_confirm"], bool)
            and _num(tc["relink_max_gap_s"]) and tc["relink_max_gap_s"] > 0):
        raise ArtifactError("bad target_config")
    confs = doc["user_actions" if v21 else "confirmations"]
    if not isinstance(confs, list):
        raise ArtifactError("user actions must be a list")
    allowed = _ACTION_KEYS if v21 else {_CONFIRM: _ACTION_KEYS[_CONFIRM]}
    for c in confs:
        kind = (c.get("kind"), c.get("version")) if isinstance(c, dict) else None
        if kind not in allowed:
            raise ArtifactError(f"bad user action {c!r}")
        _keys(c, allowed[kind], "user action")
        if not isinstance(c["frame"], int) or isinstance(c["frame"], bool):
            raise ArtifactError(f"bad user action {c!r}")
    reselected = {}
    for c in confs:
        if (c["kind"], c["version"]) == ("track_at_frame", "1.0.0"):
            reselected.setdefault(c["track_id"], []).append(c["frame"])
    ends = doc["track_ends"]
    if not isinstance(ends, list):
        raise ArtifactError("track_ends must be a list")
    for e in ends:
        _keys(e, ("track_id", "frame", "reason"), "track_end")
        if e["reason"] not in {r.value for r in EndReason}:
            raise ArtifactError("bad track_end reason")
    rels = doc["relinks"]
    if not isinstance(rels, list):
        raise ArtifactError("relinks must be a list")
    used_confs = [c for c in confs if (c["kind"], c["version"]) == ("track_at_frame", "1.0.0")]
    for r in rels:
        _keys(r, _RELINK_KEYS + (("rejection",) if v21 else ()), "relink")
        _keys(r["evidence"], ("possible_predecessors", "candidates"), "relink evidence")
        late_ok = v21 and r["outcome"] == "CONFIRMED"         # 2.1.0: an offer may be confirmed after it was dropped
        if r["outcome"] not in ("PENDING", "CONFIRMED", "DROPPED") or \
                (r["outcome"] == "DROPPED" and r["dropped_reason"] is None) or \
                (r["outcome"] != "DROPPED" and r["dropped_reason"] is not None and not late_ok) or \
                (r["outcome"] == "CONFIRMED") != (r["confirmation"] is not None):
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: outcome fields disagree")
        key = (r["from_track"], r["to_track"], r["proposed_frame"])
        rej = r.get("rejection")
        if rej is not None:
            if not (rej in confs and (rej["kind"], rej["version"]) == ("reject_candidate", "1.0.0") and
                    (rej["from_track"], rej["to_track"], rej["proposed_frame"]) == key and r["outcome"] != "CONFIRMED"):
                raise ArtifactError(f"relink {key}: rejection not among the user actions")
            used_confs.append(rej)
        if r["from_track"] not in r["evidence"]["possible_predecessors"] or r["to_track"] not in r["evidence"]["candidates"]:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: evidence does not support it")
        if not (_num(r["gap_s"]) and 0 <= r["gap_s"] <= tc["relink_max_gap_s"] + 1e-9):
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: gap outside relink_max_gap_s")
        c = r["confirmation"]
        if c is not None:
            if c == {"kind": "auto_confirm", "frame": c.get("frame")}:
                if not tc["relink_auto_confirm"]:
                    raise ArtifactError("auto-confirmed relink while relink_auto_confirm is false")
            elif c in confs and (c["kind"], c["version"]) == _CONFIRM and c["track_id"] == r["to_track"]:
                used_confs.append(c)
            elif v21 and c in confs and (c["kind"], c["version"]) == ("confirm_relink", "1.0.0") and \
                    (c["from_track"], c["to_track"], c["proposed_frame"]) == key and c["frame"] >= r["proposed_frame"]:
                used_confs.append(c)
            else:
                raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: confirmation not among the user actions")
    if len(used_confs) != len(confs) or any(used_confs.count(c) > 1 for c in used_confs):
        raise ArtifactError("a user action applies to nothing (or to two things)")
    frames = doc["frames"]
    if not frames or frames[0].get("frame") != sel["frame"]:
        raise ArtifactError("target frames must start at the selection frame")
    confirmed_at = {(r["to_track"]): r["confirmation"]["frame"] for r in rels if r["outcome"] == "CONFIRMED"}
    proposed = {}
    for r in rels:
        proposed.setdefault(r["to_track"], []).append(r["proposed_frame"])
    ended_at = {e["track_id"]: e["frame"] for e in ends}
    prev, prev_t = -1, -math.inf
    for fr in frames:
        _keys(fr, ("frame", "t", "state", "reason", "track_id", "box"), "target frame")
        prev = _frame_order(fr, prev)
        where = f"target frame {fr['frame']}"
        if not _num(fr["t"]) or fr["t"] <= prev_t:
            raise ArtifactError(f"{where}: time not strictly increasing")
        prev_t = fr["t"]
        if fr["reason"] not in _TARGET_REASONS_V2.get(fr["state"], set()):
            raise ArtifactError(f"{where}: state {fr['state']!r} with reason {fr['reason']!r}")
        tid, b = fr["track_id"], fr["box"]
        if not isinstance(tid, str):
            raise ArtifactError(f"{where}: every frame names the track it refers to")
        answers = fr["state"] in (TargetState.TARGET.value, TargetState.CANDIDATE.value)
        if answers != (b is not None):
            raise ArtifactError(f"{where}: a box exactly when the answer is TARGET or CANDIDATE")
        if answers and not (isinstance(b, list) and len(b) == 4 and all(_num(v) for v in b) and b[2] > b[0]
                            and b[3] > b[1]):
            raise ArtifactError(f"{where}: bad box")
        if fr["state"] == TargetState.TARGET.value and tid != sel["track_id"] and \
                not (tid in confirmed_at and confirmed_at[tid] <= fr["frame"]) and \
                not any(g <= fr["frame"] for g in reselected.get(tid, [])):
            raise ArtifactError(f"{where}: TARGET on {tid}: not the selection, a re-selection or a confirmed relink")
        if fr["state"] == TargetState.CANDIDATE.value and not any(p <= fr["frame"] for p in proposed.get(tid, [])):
            raise ArtifactError(f"{where}: CANDIDATE {tid} was never proposed")
        if fr["reason"] == TargetReason.TRACK_ENDED.value and not (tid in ended_at and ended_at[tid] <= fr["frame"]):
            raise ArtifactError(f"{where}: TRACK_ENDED for {tid} without a recorded end")
    by_frame = {fr["frame"]: fr for fr in frames}
    for tid, fs in reselected.items():
        for g in fs:
            if g not in by_frame or by_frame[g]["track_id"] != tid:
                raise ArtifactError(f"re-selection of {tid} on frame {g} is not reflected in the answers")
    if v21 and doc["gaps"] != gaps_of(frames, rels):
        raise ArtifactError("gaps do not match the frames")
    if tracks is None:
        return
    if sha256_doc(tracks) != doc["tracks_sha256"]:
        raise ArtifactError("tracks artifact does not match tracks_sha256")
    if tracks["video"] != doc["video"]:
        raise ArtifactError("target video differs from the tracks video")
    src = [fr for fr in tracks["frames"] if fr["frame"] >= sel["frame"]]
    for tid, fs in reselected.items():
        for g in fs:
            tf = next((x for x in src if x["frame"] == g), None)
            t = None if tf is None else next((x for x in tf["tracks"] if x["track_id"] == tid), None)
            if t is None or t["state"] != TrackState.ACTIVE.value:
                raise ArtifactError(f"re-selection of {tid} on frame {g} is not an observed box")
    if [(f["frame"], f["t"]) for f in src] != [(f["frame"], f["t"]) for f in frames]:
        raise ArtifactError("target frames differ from the tracks frames after the selection")
    first = next((t for t in src[0]["tracks"] if t["track_id"] == sel["track_id"]), None)
    if first is None or first["state"] != TrackState.ACTIVE.value:
        raise ArtifactError("selection does not name an ACTIVE track on its frame")
    born: Dict[str, dict] = {}
    for tf in tracks["frames"]:
        for t in tf["tracks"]:
            born.setdefault(t["track_id"], {"frame": tf["frame"], "t": tf["t"], "pp": t["possible_predecessors"]})
    for e in ends:
        tf = next((x for x in src if x["frame"] == e["frame"]), None)
        if tf is None or [e["track_id"], e["reason"]] not in tf["ended"]:
            raise ArtifactError(f"track_end {e} not in the tracks artifact")
    t_of = {tf["frame"]: tf["t"] for tf in tracks["frames"]}
    for r in rels:
        b = born.get(r["to_track"])
        if b is None or r["from_track"] not in b["pp"] or not (r["lost_frame"] < b["frame"] <= r["proposed_frame"]):
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: not supported by the tracks artifact")
        t0 = t_of[r["lost_frame"]]
        if abs((b["t"] - t0) - r["gap_s"]) > 1e-5:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: gap_s disagrees with the tracks")
        rivals = [k for k, v in born.items() if k != r["to_track"] and r["from_track"] in v["pp"]
                  and r["lost_frame"] < v["frame"] <= r["proposed_frame"] and v["t"] - t0 <= tc["relink_max_gap_s"] + 1e-9]
        if rivals:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: not the only candidate ({rivals})")
    for tf, fr in zip(src, frames):
        if fr["state"] in (TargetState.TARGET.value, TargetState.CANDIDATE.value):
            tr = next((t for t in tf["tracks"] if t["track_id"] == fr["track_id"]), None)
            if not (tr is not None and tr["state"] == TrackState.ACTIVE.value and tr["confirmed"]
                    and tr["identity"] == IdentityState.CERTAIN.value and tr["box"] == fr["box"]):
                raise ArtifactError(f"frame {tf['frame']}: {fr['state']} not backed by an ACTIVE, confirmed, CERTAIN "
                                    "track box")


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
