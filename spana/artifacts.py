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
  2.2.0 tracks: + per-track identity_doubt (bool). IDENTITY_UNCERTAIN may now have no rivals: the track continued
        through an appearance disagreement on an exclusive motion link (tracking.appearance_doubt_link).
        CERTAIN <=> no rivals and no doubt. 2.1.0 documents still validate (validate_tracks) with their meaning.
  2.3.0 tracks: + per-track resolved_by (APPEARANCE / REID / null): how the latest uncertainty episode was cleared
        (config reid_resolve). 2.1.0 / 2.2.0 documents still validate with their meaning.
  1.0.0 target: selection (track_at_frame 1.0.0), target_config, track_end, frames {frame, t, state, reason,
        track_id, box}. CANDIDATE reserved (refused in 1.0.0).
  2.0.0 target: relinks. CANDIDATE (RELINK_PREDECESSOR) allowed; TARGET may be on a confirmed relinked track;
        every frame names the track it refers to; + confirmations, relinks log, track_ends (replaces track_end),
        target_config + relink_max_gap_s / relink_auto_confirm. 1.0.0 documents keep their 1.0.0 meaning.
  2.1.0 target: user_actions (confirm_candidate, confirm_relink, reject_candidate, track_at_frame re-selection)
        replace confirmations; relinks + rejection; + gaps (stretches without a TARGET answer, for the review
        timeline). 2.0.0 documents keep their 2.0.0 meaning.
  2.2.0 target: + target_config.tap_vouches_doubt; TARGET may carry reason USER_VOUCHED: on an ACTIVE, confirmed
        track uncertain only by an appearance doubt (tracks 2.2.0 identity_doubt, no rivals), tapped by the user
        (selection or re-selection) earlier in the same uncertainty episode. 2.1.0 documents keep their meaning.
  2.9.0 target: + target_config.resume_after_reid (answer again after an episode the tracker cleared by re-ID,
        tracks 2.3.0 resolved_by REID; config reid_resolve). 2.0.0 - 2.8.0 documents still validate.
  2.8.0 target: + target_config.offer_hold_s (an offer is kept while its track is briefly hidden; CANDIDATE is still
        shown only on frames where the track is ACTIVE, confirmed and CERTAIN). 2.7.0 documents keep their meaning.
  2.7.0 target: exit side. + target_config reentry_exit_edge / reentry_exit_edge_max_frac; + reentry_log
        `exit_excluded` (candidates ruled out because the player left the picture by another side). With the tracks,
        every re-ID offer is re-checked against the exit side (spana.identity.reid.exit_side). 2.6.0 documents keep
        their meaning.
  2.6.0 target: better re-ID signatures (option 2). + target_config reentry_signature ("mean" | "set") /
        reentry_wait_samples / reentry_withdraw. With "set", a reid offer's similarity is the quality-weighted
        best-pair similarity of the crop sets (spana.track_embeddings 1.1.0), the candidate's crops limited to frames
        <= proposed_frame when reentry_wait_samples > 0; an offer may be DROPPED with reason WITHDRAWN.
        2.5.0 documents keep their meaning.
  2.5.0 target: re-entry offers ranked by appearance + position. + top-level reentry_log (the re-entry rule's
        decisions, logged when they change during a loss: offer / abstention reason, appearance, position, combined
        scores, margin, every candidate); + target_config reentry_mutual / reentry_position_weight; reid evidence
        {similarity (appearance), position, combined, position_informative, second_combined, candidates}.
        2.4.0 documents keep their meaning.
  2.4.0 target: re-entry offers by re-ID. + top-level embeddings_sha256 (the spana.track_embeddings document used, or
        null); + target_config reentry_offer / reentry_max_gap_s / reentry_min_similarity / reentry_min_margin; relink
        evidence has a `kind`: "predecessor" {possible_predecessors, candidates} or "reid" {similarity,
        second_similarity, candidates}; CANDIDATE reason REENTRY_REID for a reid offer. 2.3.0 documents keep their
        meaning. A track_embeddings document validates with validate_track_embeddings.
  2.3.0 target: + target_config.resume_after_doubt_resolved (a no-rival doubt episode does not taint the followed
        track). Answers keep the 2.2.0 rules. 2.2.0 documents keep their meaning.
  1.0.0 ball: per-frame ball answer (OBSERVED / PREDICTED / LOST / AMBIGUOUS + reason, selected hypothesis, centre,
        diameter, confidence, contested, candidate counts) + hypotheses summary; consumes a ball-view detections
        artifact (model.ball_view, detection_config = the ball config section).

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
TRACKS_SCHEMA = ("spana.tracks", "2.3.0")   # 2.1.0 / 2.2.0 documents still validate (validate_tracks)
TRACKS_ACCEPTED = ("2.1.0", "2.2.0", "2.3.0")
TARGET_SCHEMA = ("spana.target", "2.9.0")   # 1.0.0 and 2.0.0 - 2.8.0 still validate


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
            "uncertain_episodes": o.uncertain_episodes, "identity_doubt": o.identity_doubt,
            "resolved_by": o.resolved_by,
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
_RESOLVED_BY = (None, "APPEARANCE", "REID")
_TRACKLET_KEYS = ("track_id", "epoch", "first_frame", "last_frame", "last_observed_frame", "observed_frames",
                  "confirmed", "end_reason", "recoveries", "possible_predecessors")
_CAMERA_KEYS = ("status", "affine", "inliers", "hist_distance")
_SOURCES_FOR_STATE = {TrackState.ACTIVE.value: {PositionSource.OBSERVED.value},
                      TrackState.OCCLUDED.value: {PositionSource.PREDICTED.value, PositionSource.HELD.value},
                      TrackState.LOST.value: {PositionSource.PREDICTED.value, PositionSource.HELD.value}}


def validate_tracks(doc: Any, detections: Optional[dict] = None) -> None:
    """Structural and temporal validation. With the source `detections` artifact, every observation is also
    checked against the detection it claims (index in range, same box and score)."""
    version = doc.get("schema_version") if isinstance(doc, dict) else None
    _header(doc, (TRACKS_SCHEMA[0], version if version in TRACKS_ACCEPTED else TRACKS_SCHEMA[1]),
            ("schema", "schema_version", "provenance", "video", "detections_sha256", "frames", "tracklets"))
    track_keys = _TRACK_KEYS + (("identity_doubt",) if version != "2.1.0" else ()) + \
        (("resolved_by",) if version not in ("2.1.0", "2.2.0") else ())
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
            _keys(tr, track_keys, where)
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
                    not isinstance(tr.get("identity_doubt", False), bool) or \
                    (tr["identity"] == IdentityState.CERTAIN.value) != \
                    (not tr["identity_rivals"] and not tr.get("identity_doubt", False)) or \
                    not (isinstance(tr["uncertain_episodes"], int) and tr["uncertain_episodes"] >= 0) or \
                    (tr["identity"] != IdentityState.CERTAIN.value and tr["uncertain_episodes"] < 1):
                raise ArtifactError(f"{where}: {tid} bad identity fields")
            if "resolved_by" in tr and (tr["resolved_by"] not in _RESOLVED_BY or (tr["resolved_by"] is not None and (
                    tr["identity"] != IdentityState.CERTAIN.value or tr["uncertain_episodes"] < 1))):
                raise ArtifactError(f"{where}: {tid} bad resolved_by")
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


def validate_target(doc: Any, tracks: Optional[dict] = None, embeddings: Optional[dict] = None) -> None:
    """Validate a spana.target document of any supported version (2.4.0 current; older versions as written then).
    With `embeddings` (2.4.0), every re-ID offer's similarities are recomputed from the signatures."""
    if not isinstance(doc, dict):
        raise ArtifactError("target document must be an object")
    version = doc.get("schema_version")
    if version == "1.0.0":
        return _validate_target_v1(doc, tracks)
    if version in ("2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.4.0", "2.5.0", "2.6.0", "2.7.0", "2.8.0", "2.9.0"):
        return _validate_target_v2(doc, tracks, version, embeddings)
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
_TARGET_REASONS_V22 = {**_TARGET_REASONS_V2,
                       TargetState.TARGET.value: {TargetReason.TRACK_OBSERVED.value, TargetReason.USER_VOUCHED.value}}
_TARGET_REASONS_V24 = {**_TARGET_REASONS_V22,
                       TargetState.CANDIDATE.value: {TargetReason.RELINK_PREDECESSOR.value,
                                                     TargetReason.REENTRY_REID.value}}
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


def _validate_target_v2(doc: Any, tracks: Optional[dict] = None, version: str = "2.2.0",
                        embeddings: Optional[dict] = None) -> None:
    """2.0.0 - 2.9.0. Structure, then (with the source tracks) every answer and every relink against the
    tracks. 2.9.0 = 2.8.0 + target_config.resume_after_reid."""
    order = ("2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.4.0", "2.5.0", "2.6.0", "2.7.0", "2.8.0", "2.9.0")
    since = lambda v: order.index(version) >= order.index(v)      # noqa: E731
    v21, v22, v23, v24, v25 = since("2.1.0"), since("2.2.0"), since("2.3.0"), since("2.4.0"), since("2.5.0")
    v26, v27, v28, v29 = since("2.6.0"), since("2.7.0"), since("2.8.0"), since("2.9.0")
    _header(doc, ("spana.target", version),
            ("schema", "schema_version", "provenance", "video", "tracks_sha256", "selection",
             "user_actions" if v21 else "confirmations", "target_config", "track_ends", "relinks", "frames")
            + (("gaps",) if v21 else ()) + (("embeddings_sha256",) if v24 else ()) + (("reentry_log",) if v25 else ()))
    if v24 and doc["embeddings_sha256"] is not None and not isinstance(doc["embeddings_sha256"], str):
        raise ArtifactError("embeddings_sha256 must be a string or null")
    sel = doc["selection"]
    if not isinstance(sel, dict) or (sel.get("kind"), sel.get("version")) not in _SELECTIONS:
        raise ArtifactError("unsupported selection")
    _keys(sel, _SELECTIONS[(sel["kind"], sel["version"])], "selection")
    if not (isinstance(sel["frame"], int) and not isinstance(sel["frame"], bool) and isinstance(sel["track_id"], str)):
        raise ArtifactError("bad selection frame / track_id")
    tc = doc["target_config"]
    _keys(tc, ("resume_after_uncertain", "relink_max_gap_s", "relink_auto_confirm")
          + (("tap_vouches_doubt",) if v22 else ()) + (("resume_after_doubt_resolved",) if v23 else ())
          + (("reentry_offer", "reentry_max_gap_s", "reentry_min_similarity", "reentry_min_margin") if v24 else ())
          + (("reentry_mutual", "reentry_position_weight") if v25 else ())
          + (("reentry_signature", "reentry_wait_samples", "reentry_withdraw") if v26 else ())
          + (("reentry_exit_edge", "reentry_exit_edge_max_frac") if v27 else ())
          + (("offer_hold_s",) if v28 else ())
          + (("resume_after_reid",) if v29 else ()),
          "target_config")
    if v28 and not (_num(tc["offer_hold_s"]) and tc["offer_hold_s"] >= 0):
        raise ArtifactError("bad target_config offer_hold_s")
    if v27 and not (isinstance(tc["reentry_exit_edge"], bool) and _num(tc["reentry_exit_edge_max_frac"]) and
                    0.0 <= tc["reentry_exit_edge_max_frac"] <= 1.0):
        raise ArtifactError("bad target_config exit-side keys")
    if v26 and not (tc["reentry_signature"] in ("mean", "set") and isinstance(tc["reentry_wait_samples"], int) and
                    not isinstance(tc["reentry_wait_samples"], bool) and tc["reentry_wait_samples"] >= 0 and
                    isinstance(tc["reentry_withdraw"], bool)):
        raise ArtifactError("bad target_config signature keys")
    if not v26 and any(r.get("dropped_reason") == "WITHDRAWN" for r in doc.get("relinks", [])
                       if isinstance(r, dict)):
        raise ArtifactError("WITHDRAWN offers need spana.target 2.6.0")
    if v24 and not (isinstance(tc["reentry_offer"], bool) and _num(tc["reentry_max_gap_s"]) and
                    _num(tc["reentry_min_similarity"]) and _num(tc["reentry_min_margin"])):
        raise ArtifactError("bad target_config re-entry keys")
    if v29 and not isinstance(tc["resume_after_reid"], bool):
        raise ArtifactError("bad target_config")
    if not (isinstance(tc["resume_after_uncertain"], bool) and isinstance(tc["relink_auto_confirm"], bool)
            and isinstance(tc.get("tap_vouches_doubt", False), bool)
            and isinstance(tc.get("resume_after_doubt_resolved", False), bool)
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
        kind = _relink_kind(r)
        if v24:
            keys = _EVIDENCE_KEYS_V25 if v25 else _EVIDENCE_KEYS
            _keys(r["evidence"], keys.get(kind, ("kind",)), "relink evidence")
            if kind not in keys:
                raise ArtifactError(f"relink evidence kind {kind!r} unknown")
        else:
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
        if kind == "reid":
            ev = r["evidence"]
            sec = ev["second_combined"] if v25 else ev["second_similarity"]
            top = ev["combined"] if v25 else ev["similarity"]
            if v25 and not (_num(ev["position"]) and 0.0 <= ev["position"] <= 1.0 and _num(ev["combined"]) and
                            isinstance(ev["position_informative"], bool) and
                            abs(ev["combined"] - ev["similarity"] - tc["reentry_position_weight"] * ev["position"])
                            <= 1e-5):
                raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: bad re-ID scores")
            if not (tc["reentry_offer"] and doc["embeddings_sha256"] is not None and r["to_track"] in ev["candidates"]
                    and _num(ev["similarity"]) and ev["similarity"] >= tc["reentry_min_similarity"] - 1e-9
                    and (sec is None or (_num(sec) and top - sec >= tc["reentry_min_margin"] - 1e-9))):
                raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: re-ID evidence does not support it")
            if not (_num(r["gap_s"]) and 0 <= r["gap_s"] <= tc["reentry_max_gap_s"] + 1e-9):
                raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: gap outside reentry_max_gap_s")
        else:
            if r["from_track"] not in r["evidence"]["possible_predecessors"] or \
                    r["to_track"] not in r["evidence"]["candidates"]:
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
        proposed.setdefault((r["to_track"], _relink_kind(r)), []).append(r["proposed_frame"])
    ended_at = {e["track_id"]: e["frame"] for e in ends}
    prev, prev_t = -1, -math.inf
    for fr in frames:
        _keys(fr, ("frame", "t", "state", "reason", "track_id", "box"), "target frame")
        prev = _frame_order(fr, prev)
        where = f"target frame {fr['frame']}"
        if not _num(fr["t"]) or fr["t"] <= prev_t:
            raise ArtifactError(f"{where}: time not strictly increasing")
        prev_t = fr["t"]
        if fr["reason"] not in (_TARGET_REASONS_V24 if v24 else _TARGET_REASONS_V22 if v22 else _TARGET_REASONS_V2
                                ).get(fr["state"], set()):
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
        if fr["state"] == TargetState.CANDIDATE.value:
            kind = "reid" if fr["reason"] == TargetReason.REENTRY_REID.value else "predecessor"
            if not any(p <= fr["frame"] for p in proposed.get((tid, kind), [])):
                raise ArtifactError(f"{where}: CANDIDATE {tid} was never proposed ({kind})")
        if fr["reason"] == TargetReason.TRACK_ENDED.value and not (tid in ended_at and ended_at[tid] <= fr["frame"]):
            raise ArtifactError(f"{where}: TRACK_ENDED for {tid} without a recorded end")
    by_frame = {fr["frame"]: fr for fr in frames}
    for tid, fs in reselected.items():
        for g in fs:
            if g not in by_frame or by_frame[g]["track_id"] != tid:
                raise ArtifactError(f"re-selection of {tid} on frame {g} is not reflected in the answers")
    if v21 and doc["gaps"] != gaps_of(frames, rels):
        raise ArtifactError("gaps do not match the frames")
    if v25:
        if not isinstance(doc["reentry_log"], list):
            raise ArtifactError("reentry_log must be a list")
        for e in doc["reentry_log"]:
            _keys(e, _REENTRY_LOG_KEYS + (("exit_excluded",) if v27 else ()), "reentry_log entry")
            if not (isinstance(e["frame"], int) and isinstance(e["offer"], bool) and e["reason"] in _REENTRY_REASONS
                    and (e["offer"] == (e["reason"] is None)) and isinstance(e["candidates"], list)):
                raise ArtifactError(f"bad reentry_log entry at frame {e.get('frame')}")
            if v27 and not (isinstance(e["exit_excluded"], list) and (tc["reentry_exit_edge"] or not e["exit_excluded"])
                            and e["track"] not in e["exit_excluded"]
                            and not set(e["exit_excluded"]) & {c.get("track") for c in e["candidates"]}):
                raise ArtifactError(f"bad reentry_log exit_excluded at frame {e['frame']}")
        if doc["reentry_log"] and not tc["reentry_offer"]:
            raise ArtifactError("reentry_log entries while reentry_offer is false")
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
        reid = _relink_kind(r) == "reid"
        if b is None or (not reid and r["from_track"] not in b["pp"]) or \
                not (r["lost_frame"] < b["frame"] <= r["proposed_frame"]):
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: not supported by the tracks artifact")
        t0 = t_of[r["lost_frame"]]
        if abs((b["t"] - t0) - r["gap_s"]) > 1e-5:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: gap_s disagrees with the tracks")
        if reid:
            continue                      # re-ID offers are checked against the embeddings below
        rivals = [k for k, v in born.items() if k != r["to_track"] and r["from_track"] in v["pp"]
                  and r["lost_frame"] < v["frame"] <= r["proposed_frame"] and v["t"] - t0 <= tc["relink_max_gap_s"] + 1e-9]
        if rivals:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: not the only candidate ({rivals})")
    if embeddings is not None and v24:
        _check_reid_offers(doc, rels, embeddings, v25, tc if v26 else None)
    if v27 and tc["reentry_exit_edge"]:
        _check_exit_side(doc, rels, tracks, tc)
    taps = sorted([(sel["frame"], sel["track_id"])] + [(g, tid) for tid, fs in reselected.items() for g in fs])
    for k, (tf, fr) in enumerate(zip(src, frames)):
        if fr["reason"] == TargetReason.USER_VOUCHED.value:
            _check_vouched(tc, src, frames, k, taps)
        elif fr["state"] in (TargetState.TARGET.value, TargetState.CANDIDATE.value):
            tr = next((t for t in tf["tracks"] if t["track_id"] == fr["track_id"]), None)
            if not (tr is not None and tr["state"] == TrackState.ACTIVE.value and tr["confirmed"]
                    and tr["identity"] == IdentityState.CERTAIN.value and tr["box"] == fr["box"]):
                raise ArtifactError(f"frame {tf['frame']}: {fr['state']} not backed by an ACTIVE, confirmed, CERTAIN "
                                    "track box")


def _relink_kind(r: dict) -> str:
    return r["evidence"].get("kind", "predecessor") if isinstance(r.get("evidence"), dict) else "predecessor"


_EVIDENCE_KEYS = {"predecessor": ("kind", "possible_predecessors", "candidates"),
                  "reid": ("kind", "similarity", "second_similarity", "candidates")}
_EVIDENCE_KEYS_V25 = {"predecessor": ("kind", "possible_predecessors", "candidates"),
                      "reid": ("kind", "similarity", "position", "combined", "position_informative", "second_combined",
                               "candidates")}
_REENTRY_LOG_KEYS = ("frame", "from_track", "offer", "reason", "track", "appearance", "position",
                     "position_informative", "position_reason", "combined", "second_combined", "margin",
                     "reverse_second_combined", "candidates")
_REENTRY_REASONS = (None, "NO_SIGNATURE", "NO_CANDIDATE", "BELOW_MIN_SIMILARITY", "MARGIN", "NOT_MUTUAL")


def _cos(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / max(na * nb, 1e-12)


def _check_exit_side(doc: dict, rels: List[dict], tracks: dict, tc: dict) -> None:
    """2.7.0: every re-ID offer passes the exit-side rule, recomputed from the tracks."""
    from .identity.reid import ReentryContext, exit_side     # lazy: reid imports this module
    ctx = None
    for r in rels:
        if _relink_kind(r) != "reid":
            continue
        ctx = ctx or ReentryContext.of(tracks)
        idx = {fr["frame"]: i for i, fr in enumerate(ctx.frames)}
        li = idx.get(r["lost_frame"])
        if li is None or (r["from_track"], li) not in ctx.box_at or r["to_track"] not in ctx.birth:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: lost frame / offered track not in tracks")
        ex = exit_side(ctx, r["from_track"], li, r["to_track"])
        if ex["status"] == "off_screen" and ex["edge_dist"] > tc["reentry_exit_edge_max_frac"] + 1e-9:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: offered track was born "
                                f"{ex['edge_dist']:.3f} from the side the player left by ({ex['side']})")


def _check_reid_offers(doc: dict, rels: List[dict], emb: dict, v25: bool = False, tc26: Optional[dict] = None
                       ) -> None:
    """Every re-ID offer's appearance similarity (lost track's last vs offered track's first signature); for 2.4.0
    also the appearance runner-up (2.5.0 ranks by appearance + position, checked for consistency in the structure)."""
    if sha256_doc(emb) != doc["embeddings_sha256"]:
        raise ArtifactError("embeddings document does not match embeddings_sha256")
    if emb["tracks_sha256"] != doc["tracks_sha256"]:
        raise ArtifactError("embeddings were made from a different tracks document")
    sig = emb["tracks"]
    for r in rels:
        if _relink_kind(r) != "reid":
            continue
        ev = r["evidence"]
        if tc26 is not None and tc26["reentry_signature"] == "set":
            if emb["schema_version"] == "1.0.0":
                raise ArtifactError("set signatures need spana.track_embeddings 1.1.0")
            from .identity.reid import set_similarity           # lazy: reid imports this module
            a = sig.get(r["from_track"], {}).get("last_crops") or []
            sims = {}
            for c in ev["candidates"]:
                b = sig.get(c, {}).get("first_crops") or []
                if tc26["reentry_wait_samples"] > 0:
                    b = [x for x in b if x["frame"] <= r["proposed_frame"]]
                    if len(b) < tc26["reentry_wait_samples"]:
                        b = []
                sims[c] = set_similarity(a, b)
        else:
            last = sig.get(r["from_track"], {}).get("last")
            sims = {c: _cos(last, sig.get(c, {}).get("first")) for c in ev["candidates"]}
        best = sims.get(r["to_track"])
        if v25:
            if best is None or abs(best - ev["similarity"]) > 1e-4:
                raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: re-ID similarity disagrees with the "
                                    "embeddings")
            continue
        others = [v for c, v in sims.items() if c != r["to_track"] and v is not None]
        second, claimed = (max(others) if others else None), ev["second_similarity"]
        second_ok = (second is None and claimed is None) or \
            (second is not None and claimed is not None and abs(second - claimed) <= 1e-4)
        if best is None or abs(best - ev["similarity"]) > 1e-4 or not second_ok:
            raise ArtifactError(f"relink {r['from_track']}->{r['to_track']}: re-ID similarities disagree with the "
                                "embeddings")


def _doubt_only(t: dict) -> bool:
    return t["identity"] != IdentityState.CERTAIN.value and bool(t.get("identity_doubt")) and not t["identity_rivals"]


def _check_vouched(tc: dict, src: List[dict], frames: List[dict], k: int, taps: List[tuple]) -> None:
    """A USER_VOUCHED answer at index k: the latest tap at or before it is on this track, made while the track was
    doubt-only; from that tap to now the track never had a rival and stayed in the same uncertainty episode, and
    the answered track did not change (CANDIDATE offers in between excepted); now it is ACTIVE, confirmed and
    doubt-only with the answer's box."""
    fr, tid = frames[k], frames[k]["track_id"]
    where = f"frame {fr['frame']}: USER_VOUCHED"
    if not tc.get("tap_vouches_doubt"):
        raise ArtifactError(f"{where} while tap_vouches_doubt is false")
    g = max((g for g, _ in taps if g <= fr["frame"]), default=None)
    if g is None or (g, tid) not in taps:
        raise ArtifactError(f"{where}: the latest tap is not on {tid}")
    first = next(j for j, x in enumerate(src) if x["frame"] == g)
    episodes = None
    for j in range(first, k + 1):
        t = next((x for x in src[j]["tracks"] if x["track_id"] == tid), None)
        if t is None or t["identity_rivals"]:
            raise ArtifactError(f"{where}: {tid} gone or with a rival since the tap")
        if j == first:
            if not _doubt_only(t):
                raise ArtifactError(f"{where}: {tid} was not doubt-only when tapped")
            episodes = t["uncertain_episodes"]
        elif t["uncertain_episodes"] != episodes:
            raise ArtifactError(f"{where}: {tid} entered a new uncertainty episode since the tap")
        if frames[j]["track_id"] != tid and frames[j]["state"] != TargetState.CANDIDATE.value:
            raise ArtifactError(f"{where}: the answered track changed since the tap")
    if not (t["state"] == TrackState.ACTIVE.value and t["confirmed"] and _doubt_only(t) and t["box"] == fr["box"]):
        raise ArtifactError(f"{where}: not backed by an ACTIVE, confirmed, doubt-only track box")


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


EMBEDDINGS_SCHEMA = ("spana.track_embeddings", "1.1.0")
EMBEDDINGS_VERSIONS = ("1.0.0", "1.1.0")


def validate_track_embeddings(doc: Any, tracks: Optional[dict] = None) -> None:
    """spana.track_embeddings 1.0.0 (spana.identity.reid): per track a `first` / `last` re-ID signature (unit vector
    of the model's dimension, or null when the track had no clean crop) and the frames embedded. 1.1.0 adds per end
    `first_crops` / `last_crops`: the crop set [{frame, quality in (0, 1], emb}], frames strictly increasing. With
    the source tracks: same tracks hash, and every embedded frame is a confirmed, CERTAIN observation of that track."""
    _keys(doc, ("schema", "schema_version", "provenance", "video", "tracks_sha256", "model", "reid_config", "tracks"),
          "document")
    if doc["schema"] != EMBEDDINGS_SCHEMA[0] or doc["schema_version"] not in EMBEDDINGS_VERSIONS:
        raise ArtifactError(f"schema {doc['schema']}/{doc['schema_version']} is not a known track_embeddings version")
    v11 = doc["schema_version"] == "1.1.0"
    _keys(doc["model"], ("name", "version", "sha256", "embedding_dim"), "model")
    dim = doc["model"]["embedding_dim"]
    if not (isinstance(dim, int) and dim > 0) or not isinstance(doc["tracks"], dict):
        raise ArtifactError("bad embeddings model / tracks")
    for tid, row in doc["tracks"].items():
        _keys(row, ("first", "last", "first_frames", "last_frames") +
              (("first_crops", "last_crops") if v11 else ()), f"embeddings {tid}")
        for end in ("first", "last"):
            if v11:
                cs = row[f"{end}_crops"]
                if not isinstance(cs, list):
                    raise ArtifactError(f"embeddings {tid}: bad {end}_crops")
                prev = -1
                for c in cs:
                    _keys(c, ("frame", "quality", "emb"), f"embeddings {tid} {end}_crops")
                    if not (isinstance(c["frame"], int) and c["frame"] > prev and _num(c["quality"]) and
                            0.0 < c["quality"] <= 1.0 and isinstance(c["emb"], list) and len(c["emb"]) == dim and
                            all(_num(x) for x in c["emb"])):
                        raise ArtifactError(f"embeddings {tid}: bad crop in {end}_crops")
                    prev = c["frame"]
            v, fs = row[end], row[f"{end}_frames"]
            if not (isinstance(fs, list) and all(isinstance(f, int) for f in fs) and fs == sorted(fs)):
                raise ArtifactError(f"embeddings {tid}: bad {end}_frames")
            if v is None:
                continue
            if not (isinstance(v, list) and len(v) == dim and all(_num(x) for x in v) and fs):
                raise ArtifactError(f"embeddings {tid}: bad {end} signature")
            if abs(math.sqrt(sum(x * x for x in v)) - 1.0) > 1e-3:
                raise ArtifactError(f"embeddings {tid}: {end} signature is not unit length")
    if tracks is None:
        return
    if sha256_doc(tracks) != doc["tracks_sha256"] or tracks["video"] != doc["video"]:
        raise ArtifactError("embeddings were made from a different tracks document")
    obs = {}
    for fr in tracks["frames"]:
        for t in fr["tracks"]:
            if t["source"] == PositionSource.OBSERVED.value and t["confirmed"] and \
                    t["identity"] == IdentityState.CERTAIN.value:
                obs.setdefault(t["track_id"], set()).add(fr["frame"])
    for tid, row in doc["tracks"].items():
        crop_frames = [c["frame"] for e in ("first_crops", "last_crops") for c in row.get(e, [])]
        if not set(row["first_frames"] + row["last_frames"] + crop_frames) <= obs.get(tid, set()):
            raise ArtifactError(f"embeddings {tid}: a sample is not a confirmed, CERTAIN observation of the track")


# ----------------------------------------------------------------------------- ball
BALL_SCHEMA = ("spana.ball", "1.0.0")
_BALL_FRAME_KEYS = ("frame", "t", "camera_status", "state", "reason", "hypothesis", "center", "diameter",
                    "confidence", "contested", "candidates", "confirmed_live")
_BALL_HYP_KEYS = ("hypothesis", "first_frame", "last_observed_frame", "observed_frames", "confirmed_frame",
                  "end_frame", "end_reason")
_BALL_REASONS = {"OBSERVED": {"SINGLE_CONFIRMED", "CONTINUING"}, "PREDICTED": {"SHORT_GAP"},
                 "LOST": {"NO_CONFIRMED_BALL", "GAP_TOO_LONG", "PREDICTED_OUTSIDE", "CAMERA_DISCONTINUITY"},
                 "AMBIGUOUS": {"RIVAL_CANDIDATES", "SELECTED_HIDDEN_RIVAL"}}
_BALL_ENDS = (None, "UNCONFIRMED_EXPIRED", "LOST_EXPIRED", "CAMERA_DISCONTINUITY")


def is_ball_view(ddoc: dict) -> bool:
    return isinstance(ddoc.get("model"), dict) and "ball_view" in ddoc["model"]


def ball_frame(bf, oc: OutputConfig) -> dict:
    r = lambda v, n: None if v is None else round(v, n)   # noqa: E731
    return {"frame": bf.frame_index, "t": round(bf.time_s, 6), "camera_status": bf.camera_status.value,
            "state": bf.state.value, "reason": bf.reason.value, "hypothesis": bf.hypothesis,
            "center": None if bf.center is None else [round(v, oc.box_decimals) for v in bf.center],
            "diameter": r(bf.diameter, oc.box_decimals), "confidence": r(bf.confidence, oc.score_decimals),
            "contested": bf.contested, "candidates": bf.candidates, "confirmed_live": bf.confirmed_live}


def ball_hypothesis_rows(summaries) -> List[dict]:
    return [{k: getattr(s, k) for k in _BALL_HYP_KEYS} for s in summaries]


def validate_ball(doc: Any, detections: Optional[dict] = None) -> None:
    """spana.ball 1.0.0. With the source detections: same hash, same frames, candidate counts match."""
    _header(doc, BALL_SCHEMA, ("schema", "schema_version", "provenance", "video", "detections_sha256",
                               "ball_config", "frames", "hypotheses"))
    if not doc["frames"]:
        raise ArtifactError("ball artifact has no frames")
    hyps = {}
    for h in doc["hypotheses"]:
        _keys(h, _BALL_HYP_KEYS, "hypothesis")
        if h["end_reason"] not in _BALL_ENDS or (h["end_reason"] is None) != (h["end_frame"] is None):
            raise ArtifactError(f"hypothesis {h['hypothesis']}: bad end {h['end_reason']!r} / {h['end_frame']!r}")
        if not (isinstance(h["observed_frames"], int) and h["observed_frames"] >= 1
                and h["first_frame"] <= h["last_observed_frame"]):
            raise ArtifactError(f"hypothesis {h['hypothesis']}: bad observation span")
        hyps[h["hypothesis"]] = h
    if len(hyps) != len(doc["hypotheses"]):
        raise ArtifactError("duplicate hypothesis ids")
    prev, prev_t = -1, -math.inf
    cams = {c.value for c in CameraStatus}
    for fr in doc["frames"]:
        _keys(fr, _BALL_FRAME_KEYS, "ball frame")
        prev = _frame_order(fr, prev)
        if not (_num(fr["t"]) and fr["t"] > prev_t):
            raise ArtifactError(f"frame {fr['frame']}: time not increasing")
        prev_t = fr["t"]
        st, why = fr["state"], fr["reason"]
        if fr["camera_status"] not in cams or st not in _BALL_REASONS or why not in _BALL_REASONS[st]:
            raise ArtifactError(f"frame {fr['frame']}: bad camera / state / reason {fr['camera_status']!r} {st!r} "
                                f"{why!r}")
        answered = st in ("OBSERVED", "PREDICTED")
        fields_ = (fr["hypothesis"], fr["center"], fr["diameter"], fr["confidence"])
        if answered != all(v is not None for v in fields_) or (not answered and any(v is not None for v in fields_)):
            raise ArtifactError(f"frame {fr['frame']}: {st} needs hypothesis / center / diameter / confidence "
                                f"{'set' if answered else 'null'}")
        if answered:
            h = hyps.get(fr["hypothesis"])
            if h is None or h["confirmed_frame"] is None or h["confirmed_frame"] > fr["frame"]:
                raise ArtifactError(f"frame {fr['frame']}: answer on an unknown or unconfirmed hypothesis")
            if not (isinstance(fr["center"], list) and len(fr["center"]) == 2 and all(map(_num, fr["center"]))
                    and _num(fr["diameter"]) and fr["diameter"] > 0 and 0.0 <= fr["confidence"] <= 1.0):
                raise ArtifactError(f"frame {fr['frame']}: bad center / diameter / confidence")
        if fr["contested"] and st != "OBSERVED":
            raise ArtifactError(f"frame {fr['frame']}: only an OBSERVED answer can be contested")
        if not all(isinstance(fr[k], int) and not isinstance(fr[k], bool) and fr[k] >= 0
                   for k in ("candidates", "confirmed_live")):
            raise ArtifactError(f"frame {fr['frame']}: bad counts")
    if detections is not None:
        if doc["detections_sha256"] != sha256_doc(detections):
            raise ArtifactError("ball artifact was made from different detections")
        det = {f["frame"]: len(f["detections"]) for f in detections["frames"]}
        if [f["frame"] for f in doc["frames"]] != sorted(det):
            raise ArtifactError("ball frames differ from the detection frames")
        if any(f["candidates"] != det[f["frame"]] for f in doc["frames"]):
            raise ArtifactError("candidate counts differ from the detections")
