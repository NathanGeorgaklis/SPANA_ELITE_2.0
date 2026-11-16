"""Minimal target follower (spana.target 1.0.0): follow the user's selected track while the tracker is confident.

Selection (versioned, see SELECTION_KINDS): `track_at_frame` 1.0.0 = a track id at a frame, i.e. a tap on a box the
app showed. The track must be live and OBSERVED (ACTIVE) on that frame, otherwise the selection is refused.

Per frame, from the selection frame to the end of the tracks artifact (never earlier, never another track):
  TARGET   TRACK_OBSERVED      the selected track is ACTIVE, confirmed and identity CERTAIN (box = its box)
  ABSTAIN  NOT_CONFIRMED       ... ACTIVE but not yet confirmed
  ABSTAIN  IDENTITY_UNCERTAIN  ... IDENTITY_UNCERTAIN now
  ABSTAIN  AFTER_UNCERTAIN     ... was uncertain at some point after the selection and target.resume_after_uncertain
                               is false (fail closed: the tracker's resolution can be passed by a same-kit rival)
  ABSTAIN  TRACK_OCCLUDED      ... OCCLUDED (hidden behind another player; position only predicted)
  LOST     TRACK_LOST          ... LOST (unobserved, unexplained)
  LOST     TRACK_ENDED         the selected track ended (any EndReason, CAMERA_CUT included). No re-identification
                               yet: LOST for the rest of the clip.
A gap bridged by the tracker itself (same track id, `recoveries`) is followed again: that is the tracker's decision,
not a guess by this layer. CANDIDATE is never emitted by this version.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..artifacts import TARGET_SCHEMA, sha256_doc, validate_target
from ..config import SpanaConfig
from ..provenance import build_provenance
from ..records import IdentityState, TargetReason, TargetState, TrackState

TRACK_AT_FRAME = ("track_at_frame", "1.0.0")
SELECTION_KINDS = {TRACK_AT_FRAME}


class SelectionError(ValueError):
    """The user's selection does not name a player that can be followed: refused, never guessed."""


@dataclass(frozen=True)
class Selection:
    kind: str
    version: str
    frame: int
    track_id: str

    @staticmethod
    def track_at_frame(track_id: str, frame: int) -> "Selection":
        return Selection(TRACK_AT_FRAME[0], TRACK_AT_FRAME[1], frame, track_id)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "version": self.version, "frame": self.frame, "track_id": self.track_id}


def check_selection(tracks_doc: dict, sel: Selection) -> int:
    """Index of the selection frame in tracks_doc['frames']; raises SelectionError if the selection is invalid."""
    if (sel.kind, sel.version) not in SELECTION_KINDS:
        raise SelectionError(f"unsupported selection {sel.kind}/{sel.version}")
    if not (isinstance(sel.frame, int) and not isinstance(sel.frame, bool)) or not isinstance(sel.track_id, str):
        raise SelectionError("selection frame must be an integer and track_id a string")
    idx = next((n for n, fr in enumerate(tracks_doc["frames"]) if fr["frame"] == sel.frame), None)
    if idx is None:
        raise SelectionError(f"frame {sel.frame} is not in the tracks artifact")
    tr = next((t for t in tracks_doc["frames"][idx]["tracks"] if t["track_id"] == sel.track_id), None)
    if tr is None:
        raise SelectionError(f"track {sel.track_id!r} is not live on frame {sel.frame}")
    if tr["state"] != TrackState.ACTIVE.value:
        raise SelectionError(f"track {sel.track_id!r} is {tr['state']} on frame {sel.frame}: select an observed box")
    return idx


def follow(tracks_doc: dict, sel: Selection, cfg: SpanaConfig) -> Tuple[List[dict], Optional[dict]]:
    """(per-frame answers, track_end {frame, reason} or None). tracks_doc must be a validated spana.tracks doc."""
    start = check_selection(tracks_doc, sel)
    resume = cfg.target.resume_after_uncertain
    out: List[dict] = []
    end: Optional[dict] = None
    tainted = False
    episodes0: Optional[int] = None
    for fr in tracks_doc["frames"][start:]:
        state, reason, box = TargetState.LOST, TargetReason.TRACK_ENDED, None
        if end is None:
            ended = {tid: r for tid, r in fr["ended"]}
            if sel.track_id in ended:
                end = {"frame": fr["frame"], "reason": ended[sel.track_id]}
            else:
                tr = next(t for t in fr["tracks"] if t["track_id"] == sel.track_id)   # validated: live until ended
                if episodes0 is None:
                    episodes0 = tr["uncertain_episodes"] - (tr["identity"] != IdentityState.CERTAIN.value)
                uncertain = tr["identity"] != IdentityState.CERTAIN.value
                if uncertain or tr["uncertain_episodes"] > episodes0:
                    tainted = not resume
                if uncertain:
                    state, reason = TargetState.ABSTAIN, TargetReason.IDENTITY_UNCERTAIN
                elif tainted:
                    state, reason = TargetState.ABSTAIN, TargetReason.AFTER_UNCERTAIN
                elif tr["state"] == TrackState.OCCLUDED.value:
                    state, reason = TargetState.ABSTAIN, TargetReason.TRACK_OCCLUDED
                elif tr["state"] == TrackState.LOST.value:
                    state, reason = TargetState.LOST, TargetReason.TRACK_LOST
                elif not tr["confirmed"]:
                    state, reason = TargetState.ABSTAIN, TargetReason.NOT_CONFIRMED
                else:
                    state, reason, box = TargetState.TARGET, TargetReason.TRACK_OBSERVED, list(tr["box"])
        out.append({"frame": fr["frame"], "t": fr["t"], "state": state.value, "reason": reason.value,
                    "track_id": sel.track_id if box is not None else None, "box": box})
    return out, end


def target_document(tracks_doc: dict, sel: Selection, cfg: SpanaConfig, argv: Sequence[str] = ()) -> dict:
    """Validated spana.target document for one selection on one validated spana.tracks document."""
    frames, end = follow(tracks_doc, sel, cfg)
    tsha = sha256_doc(tracks_doc)
    doc = {
        "schema": TARGET_SCHEMA[0], "schema_version": TARGET_SCHEMA[1],
        "provenance": build_provenance(config_sha256=cfg.sha256(), argv=list(argv) or sys.argv,
                                       inputs={"video": tracks_doc["video"]["sha256"], "tracks": tsha}),
        "video": tracks_doc["video"], "tracks_sha256": tsha, "selection": sel.to_dict(),
        "target_config": {"resume_after_uncertain": cfg.target.resume_after_uncertain},
        "track_end": end, "frames": frames,
    }
    validate_target(doc, tracks_doc)
    return doc
