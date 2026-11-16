"""Target follower with relink proposals (spana.target 2.0.0).

Selection (versioned, see SELECTION_KINDS): `track_at_frame` 1.0.0 = a track id at a frame, i.e. a tap on a box the
app showed. The track must be live and OBSERVED (ACTIVE) on that frame, otherwise the selection is refused.

The ANSWERED track starts as the selected one. Per frame, from the selection frame to the end of the tracks artifact
(never earlier), with v1 rules on the answered track:
  TARGET    TRACK_OBSERVED      answered track ACTIVE, confirmed and identity CERTAIN (box = its box)
  ABSTAIN   NOT_CONFIRMED       ... ACTIVE but not yet confirmed
  ABSTAIN   IDENTITY_UNCERTAIN  ... IDENTITY_UNCERTAIN now
  ABSTAIN   AFTER_UNCERTAIN     ... was uncertain since it became the answered track and
                                target.resume_after_uncertain is false
  ABSTAIN   TRACK_OCCLUDED      ... OCCLUDED
  LOST      TRACK_LOST          ... LOST
  LOST      TRACK_ENDED         ... ended (any EndReason)
A gap bridged by the tracker itself (same track id) is followed again: the tracker's decision, not this layer's.

Relink (records/reid_headroom_*: the only clue safe on DEV, 44 / 44). While the answered track A is not observed
(OCCLUDED, LOST or ended), and only if A was confirmed, CERTAIN and not tainted at its last observation, the
candidates are the tracks born after A's last observation within target.relink_max_gap_s of it. A track B is
proposed as
  CANDIDATE RELINK_PREDECESSOR  (box = B's box, track_id = B)
only while ALL hold: B is the only candidate whose possible_predecessors names A; B is ACTIVE, confirmed and CERTAIN
on this frame; no CUT / UNKNOWN camera frame since A's last observation. Otherwise the answer stays A's (fail
closed). A candidate is DROPPED (never proposed again for that loss) when A is observed again, a second candidate
names A, a cut follows, or B stops being ACTIVE / confirmed / CERTAIN; the answer then falls back to A's state.
A CANDIDATE becomes the answered track (TARGET from that frame on, v1 rules, losses may repeat) only on a user
confirmation `confirm_candidate` 1.0.0 {frame, track_id} given on a frame where B is the CANDIDATE, or immediately
when target.relink_auto_confirm is true (off by default; for measurement). A confirmation that does not match a
CANDIDATE frame is refused.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..artifacts import TARGET_SCHEMA, sha256_doc, validate_target
from ..config import SpanaConfig
from ..provenance import build_provenance
from ..records import CameraStatus, IdentityState, TargetReason, TargetState, TrackState

TRACK_AT_FRAME = ("track_at_frame", "1.0.0")
SELECTION_KINDS = {TRACK_AT_FRAME}
CONFIRM_CANDIDATE = ("confirm_candidate", "1.0.0")
_BREAKS = (CameraStatus.CUT.value, CameraStatus.UNKNOWN.value)
_EPS = 1e-9


class SelectionError(ValueError):
    """The user's selection does not name a player that can be followed: refused, never guessed."""


class ConfirmationError(ValueError):
    """A confirmation that does not name a proposed candidate: refused, never applied."""


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


def confirm_candidate(track_id: str, frame: int) -> dict:
    return {"kind": CONFIRM_CANDIDATE[0], "version": CONFIRM_CANDIDATE[1], "frame": frame, "track_id": track_id}


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


def _parse_confirmations(confirmations: Iterable[dict]) -> Dict[Tuple[int, str], dict]:
    out = {}
    for c in confirmations:
        if not isinstance(c, dict) or (c.get("kind"), c.get("version")) != CONFIRM_CANDIDATE or \
                set(c) != {"kind", "version", "frame", "track_id"} or not isinstance(c["track_id"], str) or \
                not isinstance(c["frame"], int) or isinstance(c["frame"], bool):
            raise ConfirmationError(f"not a {CONFIRM_CANDIDATE[0]} {CONFIRM_CANDIDATE[1]} record: {c!r}")
        key = (c["frame"], c["track_id"])
        if key in out:
            raise ConfirmationError(f"duplicate confirmation {c!r}")
        out[key] = dict(c)
    return out


def _usable(tr: Optional[dict]) -> bool:
    return tr is not None and tr["state"] == TrackState.ACTIVE.value and tr["confirmed"] and \
        tr["identity"] == IdentityState.CERTAIN.value


def follow(tracks_doc: dict, sel: Selection, cfg: SpanaConfig, confirmations: Iterable[dict] = ()
           ) -> Tuple[List[dict], dict]:
    """(per-frame answers, {"track_ends": [...], "relinks": [...]}). tracks_doc must be a validated spana.tracks doc.
    Raises SelectionError / ConfirmationError on an invalid selection / confirmation."""
    start = check_selection(tracks_doc, sel)
    tc = cfg.target
    confs = _parse_confirmations(confirmations)
    used: set = set()
    frames = tracks_doc["frames"]
    birth: Dict[str, int] = {}
    preds: Dict[str, Tuple[str, ...]] = {}
    for n, fr in enumerate(frames):
        for t in fr["tracks"]:
            if t["track_id"] not in birth:
                birth[t["track_id"]], preds[t["track_id"]] = n, tuple(t["possible_predecessors"])

    out: List[dict] = []
    track_ends: List[dict] = []
    relinks: List[dict] = []
    cur, cur_ended = sel.track_id, False
    tainted, episodes0 = False, None
    last_obs: Optional[int] = None         # index of cur's last observation
    relinkable = False                     # cur was confirmed, CERTAIN and untainted at its last observation
    search_done = False                    # this loss of cur gets no (further) proposal
    pending: Optional[dict] = None         # relink log entry of the current CANDIDATE

    def drop(reason: str) -> None:
        nonlocal pending, search_done
        if pending is not None:
            pending.update(outcome="DROPPED", dropped_reason=reason)
            pending = None
        search_done = True

    for n in range(start, len(frames)):
        fr = frames[n]
        f, t = fr["frame"], fr["t"]
        live = {tr["track_id"]: tr for tr in fr["tracks"]}
        ended = {tid: r for tid, r in fr["ended"]}

        # ---------------------------------------------------------------- v1 state of the answered track
        if not cur_ended and cur in ended:
            cur_ended = True
            track_ends.append({"track_id": cur, "frame": f, "reason": ended[cur]})
        tr = None if cur_ended else live[cur]                  # validated: live until ended
        observed = tr is not None and tr["state"] == TrackState.ACTIVE.value
        if tr is None:
            state, reason, box = TargetState.LOST, TargetReason.TRACK_ENDED, None
        else:
            uncertain = tr["identity"] != IdentityState.CERTAIN.value
            if episodes0 is None:
                episodes0 = tr["uncertain_episodes"] - int(uncertain)
            if uncertain or tr["uncertain_episodes"] > episodes0:
                tainted = not tc.resume_after_uncertain
            box = None
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
            if observed:
                if pending is not None:
                    drop("ORIGINAL_OBSERVED")
                last_obs = n
                relinkable = tr["confirmed"] and not uncertain and not tainted
                search_done = False
        answer = {"frame": f, "t": t, "state": state.value, "reason": reason.value, "track_id": cur, "box": box}

        # ---------------------------------------------------------------- relink search while cur is not observed
        if not observed and last_obs is not None and relinkable and not search_done:
            t0 = frames[last_obs]["t"]
            if any(frames[k]["camera"]["status"] in _BREAKS for k in range(last_obs + 1, n + 1)):
                drop("CAMERA_BREAK")
            else:
                cands = sorted(c for c, b in birth.items()
                               if last_obs < b <= n and frames[b]["t"] - t0 <= tc.relink_max_gap_s + _EPS)
                listers = [c for c in cands if cur in preds[c]]
                if len(listers) > 1:
                    drop("SEVERAL_CANDIDATES")
                elif pending is not None and not _usable(live.get(pending["to_track"])):
                    drop("CANDIDATE_NOT_USABLE")
                elif pending is None and len(listers) == 1 and _usable(live.get(listers[0])):
                    b = listers[0]
                    pending = {"from_track": cur, "to_track": b, "lost_frame": frames[last_obs]["frame"],
                               "proposed_frame": f, "gap_s": round(frames[birth[b]]["t"] - t0, 6),
                               "evidence": {"possible_predecessors": list(preds[b]), "candidates": cands},
                               "confirmation": None, "outcome": "PENDING", "dropped_reason": None}
                    relinks.append(pending)
            if pending is not None:
                b = pending["to_track"]
                answer.update(state=TargetState.CANDIDATE.value, reason=TargetReason.RELINK_PREDECESSOR.value,
                              track_id=b, box=list(live[b]["box"]))
                key = (f, b)
                if tc.relink_auto_confirm or key in confs:
                    if key in confs:
                        used.add(key)
                    pending.update(confirmation=confs.get(key, {"kind": "auto_confirm", "frame": f}),
                                   outcome="CONFIRMED")
                    cur, cur_ended, pending = b, False, None
                    tainted, episodes0 = False, live[b]["uncertain_episodes"]
                    last_obs, relinkable, search_done = n, True, False
                    answer.update(state=TargetState.TARGET.value, reason=TargetReason.TRACK_OBSERVED.value)
        out.append(answer)
    unused = [confs[k] for k in confs if k not in used]
    if unused:
        raise ConfirmationError(f"confirmations that name no proposed candidate on their frame: {unused}")
    return out, {"track_ends": track_ends, "relinks": relinks}


def target_document(tracks_doc: dict, sel: Selection, cfg: SpanaConfig, argv: Sequence[str] = (),
                    confirmations: Iterable[dict] = ()) -> dict:
    """Validated spana.target 2.0.0 document for one selection (and optional confirmations) on one validated
    spana.tracks document."""
    confirmations = list(confirmations)
    frames, info = follow(tracks_doc, sel, cfg, confirmations)
    tsha = sha256_doc(tracks_doc)
    doc = {
        "schema": TARGET_SCHEMA[0], "schema_version": TARGET_SCHEMA[1],
        "provenance": build_provenance(config_sha256=cfg.sha256(), argv=list(argv) or sys.argv,
                                       inputs={"video": tracks_doc["video"]["sha256"], "tracks": tsha}),
        "video": tracks_doc["video"], "tracks_sha256": tsha, "selection": sel.to_dict(),
        "confirmations": confirmations,
        "target_config": {"resume_after_uncertain": cfg.target.resume_after_uncertain,
                          "relink_max_gap_s": cfg.target.relink_max_gap_s,
                          "relink_auto_confirm": cfg.target.relink_auto_confirm},
        "track_ends": info["track_ends"], "relinks": info["relinks"], "frames": frames,
    }
    validate_target(doc, tracks_doc)
    return doc
