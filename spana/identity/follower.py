"""Target follower with relink proposals (spana.target 2.0.0) and user-vouched doubts (2.2.0).

Selection (versioned, see SELECTION_KINDS): `track_at_frame` 1.0.0 = a track id at a frame, i.e. a tap on a box the
app showed. The track must be live and OBSERVED (ACTIVE) on that frame, otherwise the selection is refused.

The ANSWERED track starts as the selected one. Per frame, from the selection frame to the end of the tracks artifact
(never earlier), with v1 rules on the answered track:
  TARGET    TRACK_OBSERVED      answered track ACTIVE, confirmed and identity CERTAIN (box = its box)
  ABSTAIN   NOT_CONFIRMED       ... ACTIVE but not yet confirmed
  ABSTAIN   IDENTITY_UNCERTAIN  ... IDENTITY_UNCERTAIN now
  ABSTAIN   AFTER_UNCERTAIN     ... was uncertain since it became the answered track and
                                target.resume_after_uncertain is false (unless the latest episode was cleared
                                by the calibrated re-ID margin, resolved_by REID, and target.resume_after_reid,
                                and the track was CERTAIN when it became the answered track: re-ID re-confirms the
                                identity the track had before its episode, which is the user's only if the track
                                was attached while CERTAIN; config 2.14.0)
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

User actions (spana.target 2.1.0; decisions 2026-11-16: non-blocking offer, tap to re-select, after-match review).
Each is versioned, applied on its own frame, and refused (UserActionError) if it does not apply:
  confirm_candidate 1.0.0 {frame, track_id}   accept the CANDIDATE shown on that frame
  confirm_relink    1.0.0 {frame, from_track, to_track, proposed_frame}
                    accept an offer by its relink entry, also after it was dropped (review afterwards); the offered
                    track must still be live on that frame. TARGET on it from that frame on (v1 rules)
  reject_candidate  1.0.0 {frame, from_track, to_track, proposed_frame}
                    decline an offer: withdrawn from that frame, never re-proposed for that loss
  track_at_frame    1.0.0 {frame, track_id}  a tap on a box mid-clip: re-selects the player from that frame on
                    (the track must be ACTIVE there); earlier frames keep their answers
At most one action that changes the answered track (confirm_relink / track_at_frame) per frame.

User-vouched doubt (spana.target 2.2.0, target.tap_vouches_doubt; decision 2026-11-18). A track can be
IDENTITY_UNCERTAIN only because its own appearance disagreed on an exclusive continuation (tracks 2.2.0
`identity_doubt`, no `identity_rivals`). When the user taps such a track (the selection or a `track_at_frame`
re-selection) during that doubt, the tap is the identity evidence the colour model lacks:
  TARGET    USER_VOUCHED        answered track ACTIVE, confirmed, doubt only (no rival), and still in the uncertainty
                                episode the user tapped it in (same uncertain_episodes)
The vouch ends, fail closed, as soon as the track gains a rival (ABSTAIN IDENTITY_UNCERTAIN, then AFTER_UNCERTAIN)
or a new uncertainty episode starts; it never covers a rival ambiguity, and a tap on a CERTAIN track vouches nothing.

Re-entry offer by re-ID (spana.target 2.4.0, target.reentry_offer; Nathan 2026-11-18). After the answered track A
is lost (or has ended) and the predecessor relink window has passed without an offer, the follower may show
  CANDIDATE REENTRY_REID  (box = B's box, track_id = B)
for the track B born after A's last observation, within target.reentry_max_gap_s of it, ACTIVE / confirmed / CERTAIN
now, whose first re-ID signature (spana.track_embeddings) is closest to A's last signature, when that similarity
reaches reentry_min_similarity and beats every other such track's by reentry_min_margin. Tracks without a signature
are never offered. Accepted / declined exactly like a predecessor offer (confirm_relink / confirm_candidate /
reject_candidate); never promoted without the user, whatever relink_auto_confirm says. A dropped or declined re-ID
offer is not repeated for that loss; another track may still be offered. Signatures use the track's first crops,
which may lie after the proposal frame: the offer is meant for the after-match review.

Offer hold (spana.target 2.8.0, target.offer_hold_s; Nathan's FRESH_08 review, records/offer_hold_2026-11-19). An
offer (predecessor or re-ID) whose track is briefly hidden - still live, confirmed and CERTAIN, but OCCLUDED or LOST
(predicted box) - is kept for up to offer_hold_s after its last usable frame instead of being dropped. While held no
CANDIDATE is shown (the answer is the followed track's own state, no box) and nothing can be confirmed or withdrawn;
when the track is usable again the same offer is shown again. Uncertain, unconfirmed, ended or held too long: dropped
(CANDIDATE_NOT_USABLE) as before. offer_hold_s 0 = the behaviour before.

Player confirmations (spana.target 2.10.0; spana.identity.confirmation has the records and rules). A
player_confirmation names a track, a frame range and a verdict, and covers that track's uncertainty episode only:
  IS_ME, applied     the track becomes the answered track from `frame` (like a tap) and its episode is resolved for
                     the target only: TARGET PLAYER_CONFIRMED while the track is IDENTITY_UNCERTAIN in that episode,
                     TRACK_OBSERVED once CERTAIN again; that episode never taints the track. A later episode is
                     uncertain again unless re-ID against the human-confirmed gallery clears it (GALLERY_REID,
                     target.gallery_reid_resolve with reid_resolve.enabled): on clean crops of the answered track the
                     latest gallery crop from before the episode must beat every rival's reference (its latest clean
                     CERTAIN crop before the episode; the target's own earlier tracks are not rivals) by
                     reid_resolve.margin, over a run spanning tracking.identity_resolve_s with
                     reid_resolve.min_evidence_frames clearing frames; a frame against resets the run.
  NOT_ME, applied    the track is no target candidate in that episode: ABSTAIN PLAYER_REJECTED while answered, never
                     offered, never a gallery source.
  CONFLICT           not applied (see the module): logged with its evidence and listed for re-confirmation; a NOT_ME
                     in conflict holds the target's answer on that track uncertain (ABSTAIN CONFIRMATION_CONFLICT).
  revoked            ignored entirely: every output is as if the confirmation had never been made.

Resume after a doubt (spana.target 2.3.0, target.resume_after_doubt_resolved). An uncertainty episode seen ONLY as a
doubt with no rival (every uncertain frame of it doubt-only) does not taint the track: ABSTAIN IDENTITY_UNCERTAIN
while it lasts, TARGET again once the tracker resolves it. Any frame of the episode with a rival taints as before
(target.resume_after_uncertain), and so does an episode the follower never saw.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..artifacts import TARGET_SCHEMA, gaps_of, sha256_doc, validate_target
from ..config import SpanaConfig
from .confirmation import (PLAYER_CONFIRMATION, REVOKE_CONFIRMATION, CropFeatures, appearance_conflicts,
                           gallery_frame_verdict, is_record_ok)
from ..provenance import build_provenance
from ..records import CameraStatus, IdentityState, TargetReason, TargetState, TrackState

TRACK_AT_FRAME = ("track_at_frame", "1.0.0")
SELECTION_KINDS = {TRACK_AT_FRAME}
CONFIRM_CANDIDATE = ("confirm_candidate", "1.0.0")
CONFIRM_RELINK = ("confirm_relink", "1.0.0")
REJECT_CANDIDATE = ("reject_candidate", "1.0.0")
ACTION_KEYS = {CONFIRM_CANDIDATE: ("kind", "version", "frame", "track_id"),
               CONFIRM_RELINK: ("kind", "version", "frame", "from_track", "to_track", "proposed_frame"),
               REJECT_CANDIDATE: ("kind", "version", "frame", "from_track", "to_track", "proposed_frame"),
               TRACK_AT_FRAME: ("kind", "version", "frame", "track_id")}
_BREAKS = (CameraStatus.CUT.value, CameraStatus.UNKNOWN.value)
_EPS = 1e-9


class SelectionError(ValueError):
    """The user's selection does not name a player that can be followed: refused, never guessed."""


class UserActionError(ValueError):
    """A user action (confirmation, rejection, re-selection) that does not apply: refused, never applied."""


ConfirmationError = UserActionError      # name used by spana.target 2.0.0 callers


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


def _relink_action(kind: Tuple[str, str], relink: dict, frame: int) -> dict:
    return {"kind": kind[0], "version": kind[1], "frame": frame, "from_track": relink["from_track"],
            "to_track": relink["to_track"], "proposed_frame": relink["proposed_frame"]}


def confirm_relink(relink: dict, frame: int) -> dict:
    """Accept the offer described by a relink log entry, on `frame` (may be after the offer was dropped)."""
    return _relink_action(CONFIRM_RELINK, relink, frame)


def reject_candidate(relink: dict, frame: int) -> dict:
    return _relink_action(REJECT_CANDIDATE, relink, frame)


def reselect(track_id: str, frame: int) -> dict:
    """A tap on a box mid-clip."""
    return {"kind": TRACK_AT_FRAME[0], "version": TRACK_AT_FRAME[1], "frame": frame, "track_id": track_id}


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


def _parse_actions(actions: Iterable[dict]) -> List[dict]:
    out, seen = [], set()
    for a in actions:
        key = (a.get("kind"), a.get("version")) if isinstance(a, dict) else None
        if key in (PLAYER_CONFIRMATION, REVOKE_CONFIRMATION):
            if not is_record_ok(a):
                raise UserActionError(f"not a valid player confirmation / revocation record: {a!r}")
        elif key not in ACTION_KEYS or set(a) != set(ACTION_KEYS[key]) or not isinstance(a["frame"], int) or \
                isinstance(a["frame"], bool) or any(not isinstance(a[k], str) for k in ("track_id", "from_track",
                                                                                         "to_track") if k in a):
            raise UserActionError(f"not a supported user action record: {a!r}")
        canon = tuple(sorted(a.items()))
        if canon in seen:
            raise UserActionError(f"duplicate user action {a!r}")
        seen.add(canon)
        out.append(dict(a))
    ids = [a["confirmation_id"] for a in out if (a["kind"], a["version"]) == PLAYER_CONFIRMATION]
    if len(ids) != len(set(ids)):
        raise UserActionError("two player confirmations share a confirmation_id")
    revoked = [a["confirmation_id"] for a in out if (a["kind"], a["version"]) == REVOKE_CONFIRMATION]
    if len(revoked) != len(set(revoked)) or not set(revoked) <= set(ids):
        raise UserActionError("a revocation names no confirmation, or one confirmation is revoked twice")
    per_frame: Dict[int, int] = {}
    for a in out:
        if (a["kind"], a["version"]) in (CONFIRM_RELINK, TRACK_AT_FRAME):
            per_frame[a["frame"]] = per_frame.get(a["frame"], 0) + 1
    if any(v > 1 for v in per_frame.values()):
        raise UserActionError("more than one confirm_relink / track_at_frame on one frame")
    return out


def _doubt_only(tr: dict) -> bool:
    """IDENTITY_UNCERTAIN only by an appearance doubt: no rival track to confuse it with (tracks 2.2.0)."""
    return tr["identity"] != IdentityState.CERTAIN.value and bool(tr.get("identity_doubt")) and \
        not tr["identity_rivals"]


def _reentry_offer(cur: str, last_obs: int, n: int, ctx, sig: Dict[str, dict], done: set, live: Dict[str, dict],
                   cfg: SpanaConfig, excluded=()) -> Tuple[Optional[dict], dict]:
    """Re-entry offer by re-ID (spana.target 2.5.0): the shared rule spana.identity.reid.reentry_choice, restricted
    to tracks usable now and not yet offered for this loss. Returns (relink entry or None, the rule's decision)."""
    from .reid import reentry_choice
    frames = ctx.frames
    ch = reentry_choice(cur, last_obs, n, ctx, sig, cfg.target, cfg.tracking,
                        allowed=lambda c: c not in done and _usable(live.get(c)) and
                        (c, live[c]["uncertain_episodes"]) not in excluded, causal=True)
    if not ch["offer"]:
        return None, ch
    best = ch["track"]
    return {"from_track": cur, "to_track": best, "lost_frame": frames[last_obs]["frame"],
            "proposed_frame": frames[n]["frame"], "gap_s": round(frames[ctx.birth[best]]["t"] - frames[last_obs]["t"], 6),
            "evidence": {"kind": "reid", "similarity": ch["appearance"], "position": ch["position"],
                         "combined": ch["combined"], "position_informative": ch["position_informative"],
                         "second_combined": ch["second_combined"], "candidates": sorted(ch["candidates"])},
            "confirmation": None, "rejection": None, "outcome": "PENDING", "dropped_reason": None}, ch


def _log_row(frame: int, cur: str, ch: dict) -> dict:
    """One spana.target reentry_log entry: the re-entry rule's decision with every candidate's scores."""
    return {"frame": frame, "from_track": cur, "offer": ch["offer"], "reason": ch["reason"], "track": ch["track"],
            "appearance": ch["appearance"], "position": ch["position"],
            "position_informative": ch["position_informative"], "position_reason": ch["position_reason"],
            "combined": ch["combined"], "second_combined": ch["second_combined"], "margin": ch["margin"],
            "reverse_second_combined": ch["reverse_second_combined"],
            "candidates": [{"track": c, **v} for c, v in sorted(ch["candidates"].items())],
            "exit_excluded": list(ch["exit_excluded"])}


def _usable(tr: Optional[dict]) -> bool:
    return tr is not None and tr["state"] == TrackState.ACTIVE.value and tr["confirmed"] and \
        tr["identity"] == IdentityState.CERTAIN.value


def _holdable(tr: Optional[dict]) -> bool:
    """Briefly hidden: live, confirmed and CERTAIN, but not observed (OCCLUDED / LOST, predicted box)."""
    return tr is not None and tr["state"] in (TrackState.OCCLUDED.value, TrackState.LOST.value) and \
        tr["confirmed"] and tr["identity"] == IdentityState.CERTAIN.value


def follow(tracks_doc: dict, sel: Selection, cfg: SpanaConfig, actions: Iterable[dict] = (),
           embeddings: Optional[dict] = None, crops: Optional[CropFeatures] = None,
           claims: Optional[Dict[str, Dict[int, str]]] = None) -> Tuple[List[dict], dict]:
    """(per-frame answers, {"track_ends", "relinks", "reentry_log", "confirmation_log", "gallery",
    "gallery_reid_log"}). tracks_doc must be a validated spana.tracks doc; embeddings, if given, a validated
    spana.track_embeddings doc of it (re-entry offers, target.reentry_offer); crops, the clean crop features of the
    same tracks (confirmation gallery, appearance conflicts, gallery re-ID); claims, other followed identities' CERTAIN
    answers {identity: {frame: track id}} (spana.identity.confirmation.claims_from_targets).
    Raises SelectionError / UserActionError on an invalid selection / user action."""
    start = check_selection(tracks_doc, sel)
    tc = cfg.target
    acts = _parse_actions(actions)
    confs = {(a["frame"], a["track_id"]): a for a in acts if (a["kind"], a["version"]) == CONFIRM_CANDIDATE}
    revocations = {a["confirmation_id"]: a for a in acts if (a["kind"], a["version"]) == REVOKE_CONFIRMATION}
    by_frame: Dict[int, List[dict]] = {}
    pc_by_frame: Dict[int, List[dict]] = {}     # player confirmations in force (not revoked), by first frame
    for a in acts:
        kind = (a["kind"], a["version"])
        if kind == PLAYER_CONFIRMATION:
            if a["confirmation_id"] not in revocations:
                pc_by_frame.setdefault(a["frame"], []).append(a)
        elif kind not in (CONFIRM_CANDIDATE, REVOKE_CONFIRMATION):
            by_frame.setdefault(a["frame"], []).append(a)
    used: set = {id(a) for a in acts if (a["kind"], a["version"]) == REVOKE_CONFIRMATION or
                 ((a["kind"], a["version"]) == PLAYER_CONFIRMATION and a["confirmation_id"] in revocations)}
    claims = claims or {}
    frames = tracks_doc["frames"]
    idx_of = {fr["frame"]: n for n, fr in enumerate(frames)}
    birth: Dict[str, int] = {}
    preds: Dict[str, Tuple[str, ...]] = {}
    ep_start: Dict[Tuple[str, int], int] = {}  # (track, uncertain_episodes) -> first frame index with that count
    for n, fr in enumerate(frames):
        for t in fr["tracks"]:
            if t["track_id"] not in birth:
                birth[t["track_id"]], preds[t["track_id"]] = n, tuple(t["possible_predecessors"])
            ep_start.setdefault((t["track_id"], t["uncertain_episodes"]), n)
    clog: Dict[str, dict] = {}             # confirmation id -> log entry
    human: Dict[Tuple[str, int], List[dict]] = {}  # episode -> covered ranges of applied IS_ME confirmations
    rejected: Dict[Tuple[str, int], str] = {}    # episodes removed by an applied NOT_ME -> confirmation id
    held_eps: Dict[Tuple[str, int], str] = {}    # episodes held uncertain by a NOT_ME in conflict -> confirmation id
    gallery: List[dict] = []                     # human-confirmed partition of the target gallery
    episode_scope = tc.confirmation_scope == "episode"   # measurement only: cover the rest of the episode
    gal_runs: Dict[Tuple[str, int], dict] = {}   # gallery re-ID run state + summary per episode
    own_tracks: set = {sel.track_id}             # tracks the target was answered TARGET on (never its own rivals)
    tref = {"reid": None, "colour": None}        # the target's latest clean crop while answered TARGET
    gallery_on = tc.gallery_reid_resolve and cfg.reid_resolve.enabled and crops is not None
    rival_refs: Dict[Tuple[str, int], Optional[object]] = {}

    out: List[dict] = []
    track_ends: List[dict] = []
    relinks: List[dict] = []
    reentry_log: List[dict] = []           # re-entry rule decisions, logged when they change during a loss
    log_key = None
    cur, cur_ended = sel.track_id, False
    tainted, episodes0 = False, None
    anchored: Optional[bool] = None        # cur was CERTAIN when it became the answered track (re-ID resume needs it)
    vouch: Optional[Tuple[str, int]] = None   # (track, uncertainty episode) the user tapped during a doubt
    last_obs: Optional[int] = None         # index of cur's last observation
    relinkable = False                     # cur was confirmed, CERTAIN and untainted at its last observation
    search_done = False                    # this loss of cur gets no (further) proposal
    pending: Optional[dict] = None         # relink log entry of the current CANDIDATE
    reid_done: set = set()                 # tracks already offered by re-ID for this loss (never offered again)
    sig = (embeddings or {}).get("tracks", {}) if tc.reentry_offer else {}
    rctx = None                            # spana.identity.reid.ReentryContext, built on the first re-entry search
    cand_ok_t = None                       # time the pending offer's track was last usable (offer hold)
    held = False

    def usable_cand(tr_c: Optional[dict]) -> bool:
        """An offer candidate: usable, and not removed by the player's NOT_ME for its current episode."""
        return _usable(tr_c) and (tr_c["track_id"], tr_c["uncertain_episodes"]) not in rejected

    def keep(p: dict, t_now: float) -> bool:
        """Keep the pending offer p this frame: its track is usable, or briefly hidden within target.offer_hold_s
        of its last usable frame (held: no CANDIDATE shown)."""
        nonlocal cand_ok_t, held
        tr_c = live.get(p["to_track"])
        if usable_cand(tr_c):
            cand_ok_t = t_now
            return True
        if tc.offer_hold_s > 0 and _holdable(tr_c) and cand_ok_t is not None and \
                t_now - cand_ok_t <= tc.offer_hold_s + _EPS:
            held = True
            return True
        return False

    def drop(reason: str) -> None:
        nonlocal pending, search_done
        if pending is not None:
            if pending["evidence"]["kind"] == "reid":
                reid_done.add(pending["to_track"])
            pending.update(outcome="DROPPED", dropped_reason=reason)
            pending = None
        search_done = True

    def vouch_for(tr: dict) -> Optional[Tuple[str, int]]:
        """The user's tap on tr: a vouch when tr is uncertain only by an appearance doubt (and the policy is on)."""
        if tc.tap_vouches_doubt and _doubt_only(tr):
            return tr["track_id"], tr["uncertain_episodes"]
        return None

    def answer_on(tid: str, why: str, tapped: Optional[dict] = None) -> None:
        """Make tid the answered track from this frame on (a confirmed relink or a re-selection)."""
        nonlocal cur, cur_ended, tainted, episodes0, last_obs, relinkable, search_done, vouch, anchored
        if pending is not None:
            drop(why)
        cur, cur_ended, tainted, episodes0, anchored = tid, False, False, None, None
        last_obs, relinkable, search_done = None, False, False
        reid_done.clear()
        vouch = None if tapped is None else vouch_for(tapped)

    def find_relink(a: dict) -> dict:
        r = next((r for r in relinks if (r["from_track"], r["to_track"], r["proposed_frame"]) ==
                  (a["from_track"], a["to_track"], a["proposed_frame"])), None)
        if r is None or a["frame"] < r["proposed_frame"]:
            raise UserActionError(f"{a['kind']} names no offer made by its frame: {a!r}")
        return r

    def certain_target_now(live_now: Dict[str, dict]) -> bool:
        """The target is answered on machine-CERTAIN evidence (TRACK_OBSERVED) on this frame, before any
        confirmation of this frame is applied."""
        if cur_ended or tainted:
            return False
        tr_c = live_now.get(cur)
        if not _usable(tr_c):
            return False
        e = tr_c["uncertain_episodes"]
        return (episodes0 is None or e == episodes0) and (cur, e) not in rejected and (cur, e) not in held_eps

    def reference_of(ident_claims: Dict[int, str], f0: int) -> Optional[dict]:
        """Another identity's appearance reference: its clean crop on the claimed frame nearest to f0."""
        for fq in sorted(ident_claims, key=lambda x: (abs(x - f0), x)):
            c = crops.get(fq, ident_claims[fq])
            if c is not None:
                return {"reid": c.reid, "colour": c.colour, "frame": fq, "track": ident_claims[fq]}
        return None

    def apply_confirmation(a: dict, n: int, live_now: Dict[str, dict]) -> None:
        """A player_confirmation in force, on its first frame (index n): check it, decide APPLIED / CONFLICT, log it,
        apply its effect. Refused (UserActionError) when it does not name one live episode of an observed track."""
        nonlocal pending
        tid, cid, verdict = a["track_id"], a["confirmation_id"], a["verdict"]
        i1 = idx_of.get(a["frame_end"])
        if i1 is None:
            raise UserActionError(f"player confirmation ends on a frame that is not in the tracks: {a!r}")
        rng = frames[n:i1 + 1]
        trs = [next((t for t in g["tracks"] if t["track_id"] == tid), None) for g in rng]
        if any(t is None for t in trs) or trs[0]["state"] != TrackState.ACTIVE.value:
            raise UserActionError(f"player confirmation must name a track observed on its first frame and live "
                                  f"throughout: {a!r}")
        eps = trs[0]["uncertain_episodes"]
        if any(t["uncertain_episodes"] != eps for t in trs):
            raise UserActionError(f"player confirmation spans two uncertainty episodes of {tid}: {a!r}")
        ep = (tid, eps)
        conflicts: List[dict] = []
        evidence: dict = {}
        certain_here = certain_target_now(live_now)
        if verdict == "IS_ME":
            for q in sorted(claims):
                fs = [g["frame"] for g in rng if claims[q].get(g["frame"]) == tid]
                if fs:
                    conflicts.append({"reason": "OTHER_IDENTITY_CERTAIN", "identity": q, "frames": fs})
            if certain_here and cur != tid:
                conflicts.append({"reason": "TARGET_CERTAIN_ELSEWHERE", "track": cur})
            if ep in rejected:
                conflicts.append({"reason": "PLAYER_CONTRADICTION", "confirmation_id": rejected[ep]})
            if crops is not None:
                cs = [c for c in (crops.get(g["frame"], tid) for g in rng) if c is not None]
                others = {q: r for q in sorted(claims) for r in [reference_of(claims[q], a["frame"])] if r is not None}
                evidence["clean_crops"] = len(cs)
                if cs and others:
                    ac, ev = appearance_conflicts(cs, tref, others, cfg)
                    ev["references"] = {q: [r["frame"], r["track"]] for q, r in others.items()}
                    evidence["appearance"] = ev
                    conflicts += ac
        else:
            if certain_here and cur == tid:
                conflicts.append({"reason": "TARGET_CERTAIN", "track": tid})
            if ep in human:
                conflicts.append({"reason": "PLAYER_CONTRADICTION", "confirmation_id": human[ep][0]["cid"]})
        status = "CONFLICT" if conflicts else "APPLIED"
        entry = {"confirmation_id": cid, "record": dict(a), "status": status, "episode": eps,
                 "track_identity": trs[0]["identity"], "track_rivals": list(trs[0]["identity_rivals"]),
                 "target_track": None if cur_ended else cur, "conflicts": conflicts, "evidence": evidence,
                 "gallery_crops": 0, "revocation": None}
        clog[cid] = entry
        if status == "APPLIED" and verdict == "IS_ME":
            answer_on(tid, "PLAYER_CONFIRMED")
            human.setdefault(ep, []).append(
                {"from": a["frame"], "to": None if episode_scope else a["frame_end"], "cid": cid,
                 "uncertain_at_end": trs[-1]["identity"] != IdentityState.CERTAIN.value})
            for g, t in zip(rng, trs):
                c = None if crops is None else crops.get(g["frame"], tid)
                if c is not None:                          # clean crops only (CropFeatures keeps nothing else)
                    gallery.append({"confirmation_id": cid, "track_id": tid, "frame": g["frame"], "box": list(t["box"])})
                    gallery_crops.append((g["frame"], c))
                    entry["gallery_crops"] += 1
        elif status == "APPLIED":
            rejected[ep] = cid
            if pending is not None and pending["to_track"] == tid:
                drop("PLAYER_REJECTED")
        elif verdict == "NOT_ME":
            held_eps[ep] = cid

    def rival_ref(rid: str, s0: int):
        """A rival's re-ID reference for an episode starting at frame index s0: its latest clean crop while ACTIVE,
        CERTAIN and uncontested before s0 (as the tracker's reference), or None."""
        key = (rid, s0)
        if key not in rival_refs:
            ref = None
            for k in range(s0 - 1, birth.get(rid, s0) - 1, -1):
                t_r = next((x for x in frames[k]["tracks"] if x["track_id"] == rid), None)
                if t_r is not None and t_r["state"] == TrackState.ACTIVE.value and not t_r["contested"] and \
                        t_r["identity"] == IdentityState.CERTAIN.value:
                    c = crops.get(frames[k]["frame"], rid)
                    if c is not None and c.reid is not None:
                        ref = c.reid
                        break
            rival_refs[key] = ref
        return rival_refs[key]

    def gallery_step(n: int, tr: dict, ep: Tuple[str, int]) -> bool:
        """One frame of the gallery re-ID run for the answered track's current episode (target.gallery_reid_resolve).
        True when this frame may be answered: the run has qualified (spans identity_resolve_s with
        min_evidence_frames clearing frames, no frame against since) AND this very frame clears. The track stays
        uncertain, so every answer needs its own evidence; nothing carries past it."""
        run = gal_runs.setdefault(ep, {"track_id": ep[0], "episode": ep[1], "evaluated": 0, "clear": 0, "against": 0,
                                       "hold": {}, "resolved_frame": None, "margins": None, "answered": 0,
                                       "_t0": None, "_k": 0, "_ok": False})
        if tr["state"] != TrackState.ACTIVE.value or tr["contested"]:
            return False
        f, t = frames[n]["frame"], frames[n]["t"]
        s0 = ep_start[ep]
        own = next((c.reid for fg, c in reversed(gallery_crops) if fg < frames[s0]["frame"] and c.reid is not None),
                   None)
        mine = own_tracks | {g["track_id"] for g in gallery}
        rivals = {rid: rival_ref(rid, s0) for rid in tr["identity_rivals"] if rid not in mine}
        status, det = gallery_frame_verdict(crops.get(f, ep[0]), own, rivals, cfg.reid_resolve.margin)
        run["evaluated"] += 1
        if status == "CLEAR":
            run["clear"] += 1
            if run["_t0"] is None:
                run["_t0"], run["_k"] = t, 0
            run["_k"] += 1
            if t - run["_t0"] >= cfg.tracking.identity_resolve_s - _EPS and \
                    run["_k"] >= cfg.reid_resolve.min_evidence_frames:
                run["_ok"] = True
                if run["resolved_frame"] is None:
                    run["resolved_frame"], run["margins"] = f, det["margins"]
            if run["_ok"]:
                run["answered"] += 1
            return run["_ok"]
        if status == "AGAINST":
            run["against"] += 1
            run["_t0"], run["_ok"] = None, False
        else:
            run["hold"][det["reason"]] = run["hold"].get(det["reason"], 0) + 1
        return False

    gallery_crops: List[Tuple[int, object]] = []    # (frame, Crop) of the gallery entries, in order
    vouch = vouch_for(next(t for t in frames[start]["tracks"] if t["track_id"] == sel.track_id))
    for n in range(start, len(frames)):
        fr = frames[n]
        f, t = fr["frame"], fr["t"]
        live = {tr["track_id"]: tr for tr in fr["tracks"]}
        ended = {tid: r for tid, r in fr["ended"]}
        held = False                       # the pending offer's track is briefly hidden this frame (offer hold)

        # ---------------------------------------------------------------- the user's actions on this frame
        deferred: List[dict] = []          # actions naming an offer made on this very frame: applied after it
        for a in sorted(by_frame.get(f, []), key=lambda a: a["kind"] != REJECT_CANDIDATE[0]):   # rejections first
            kind = (a["kind"], a["version"])
            if kind in (REJECT_CANDIDATE, CONFIRM_RELINK) and a["proposed_frame"] == f and not any(
                    (r["from_track"], r["to_track"], r["proposed_frame"]) ==
                    (a["from_track"], a["to_track"], a["proposed_frame"]) for r in relinks):
                deferred.append(a)
                continue
            if kind == REJECT_CANDIDATE:
                r = find_relink(a)
                if r["outcome"] == "CONFIRMED" or r["rejection"] is not None:
                    raise UserActionError(f"cannot reject a confirmed or already rejected offer: {a!r}")
                r["rejection"] = dict(a)
                if r is pending:
                    drop("REJECTED_BY_USER")
            elif kind == CONFIRM_RELINK:
                r = find_relink(a)
                if r["outcome"] == "CONFIRMED" or r["rejection"] is not None:
                    raise UserActionError(f"cannot confirm a confirmed or rejected offer: {a!r}")
                if r["to_track"] not in live:
                    raise UserActionError(f"the offered track is gone on frame {f}; tap the player instead: {a!r}")
                was_pending = r is pending
                pending = None if was_pending else pending
                answer_on(r["to_track"], "OTHER_OFFER_CONFIRMED")
                r.update(confirmation=dict(a), outcome="CONFIRMED")
            else:                                                # TRACK_AT_FRAME: a tap mid-clip
                tr_sel = live.get(a["track_id"])
                if tr_sel is None or tr_sel["state"] != TrackState.ACTIVE.value:
                    raise UserActionError(f"re-selection must tap an observed box on its frame: {a!r}")
                answer_on(a["track_id"], "RESELECTED", tapped=tr_sel)
            used.add(id(a))
        for a in pc_by_frame.get(f, []):             # the player's confirmations, after the other actions
            used.add(id(a))
            apply_confirmation(a, n, live)

        # ---------------------------------------------------------------- v1 state of the answered track
        if not cur_ended and cur in ended:
            cur_ended = True
            track_ends.append({"track_id": cur, "frame": f, "reason": ended[cur]})
        tr = None if cur_ended else live[cur]                  # validated: live until ended
        observed = tr is not None and tr["state"] == TrackState.ACTIVE.value
        if tr is None:
            state, reason, box = TargetState.LOST, TargetReason.TRACK_ENDED, None
        else:
            vouched = vouch is not None and vouch[0] == cur and _doubt_only(tr) and \
                tr["uncertain_episodes"] == vouch[1]
            eps_now = tr["uncertain_episodes"]
            ep_now = (cur, eps_now)
            raw_unc = tr["identity"] != IdentityState.CERTAIN.value
            ranges = human.get(ep_now, [])
            in_range = any(r["from"] <= f and (r["to"] is None or f <= r["to"]) for r in ranges)
            gal_frame = False
            if gallery_on and raw_unc and not vouched and not in_range and gallery_crops and \
                    eps_now == (eps_now - 1 if episodes0 is None else episodes0) + 1:
                gal_frame = gallery_step(n, tr, ep_now)        # only an episode that alone taints the track
            resolved = in_range or gal_frame                   # answered for the target on THIS frame only
            uncertain = raw_unc and not vouched and not resolved
            if vouch is not None and (vouch[0] != cur or tr["identity_rivals"] or tr["uncertain_episodes"] != vouch[1]):
                vouch = None                                   # a rival or a new episode: the tap no longer covers it
            if episodes0 is None:
                episodes0 = tr["uncertain_episodes"] - int(uncertain)
            if anchored is None:
                anchored = tr["identity"] == IdentityState.CERTAIN.value
            if ranges and not in_range:
                # after the confirmed frames: if the track was still uncertain at their end, the rest of the episode
                # is not covered (the tracker's later resolution re-confirms the PRE-episode identity) and taints
                done = [r for r in ranges if r["to"] is not None and r["to"] < f]
                if done and max(done, key=lambda r: r["to"])["uncertain_at_end"] and episodes0 >= eps_now:
                    episodes0 = eps_now - 1
            if uncertain and tc.resume_after_doubt_resolved and _doubt_only(tr):
                episodes0 = tr["uncertain_episodes"]           # a doubt with no rival: abstain now, no taint
            elif uncertain or tr["uncertain_episodes"] > episodes0:
                tainted = not tc.resume_after_uncertain
            if tainted and not uncertain and tc.resume_after_reid and tr.get("resolved_by") == "REID" and anchored:
                tainted, episodes0 = False, tr["uncertain_episodes"]   # cleared by the calibrated re-ID margin
            box = None
            blocked = ep_now in rejected or ep_now in held_eps
            if ep_now in rejected:
                state, reason = TargetState.ABSTAIN, TargetReason.PLAYER_REJECTED
            elif ep_now in held_eps:
                state, reason = TargetState.ABSTAIN, TargetReason.CONFIRMATION_CONFLICT
            elif uncertain:
                state, reason = TargetState.ABSTAIN, TargetReason.IDENTITY_UNCERTAIN
            elif tainted and not resolved:
                state, reason = TargetState.ABSTAIN, TargetReason.AFTER_UNCERTAIN
            elif tr["state"] == TrackState.OCCLUDED.value:
                state, reason = TargetState.ABSTAIN, TargetReason.TRACK_OCCLUDED
            elif tr["state"] == TrackState.LOST.value:
                state, reason = TargetState.LOST, TargetReason.TRACK_LOST
            elif not tr["confirmed"]:
                state, reason = TargetState.ABSTAIN, TargetReason.NOT_CONFIRMED
            else:
                state, box = TargetState.TARGET, list(tr["box"])
                reason = TargetReason.TRACK_OBSERVED
                if raw_unc:
                    reason = TargetReason.PLAYER_CONFIRMED if in_range else \
                        TargetReason.GALLERY_REID if gal_frame else TargetReason.USER_VOUCHED
                own_tracks.add(cur)
                c = None if crops is None else crops.get(f, cur)
                if c is not None:                              # the target's own latest clean appearance
                    tref["reid"] = c.reid if c.reid is not None else tref["reid"]
                    tref["colour"] = c.colour if c.colour is not None else tref["colour"]
            if observed:
                if pending is not None:
                    drop("ORIGINAL_OBSERVED")
                last_obs = n
                relinkable = tr["confirmed"] and not uncertain and not (tainted and not resolved) and not blocked
                search_done = False
                reid_done.clear()
        answer = {"frame": f, "t": t, "state": state.value, "reason": reason.value, "track_id": cur, "box": box}

        # ---------------------------------------------------------------- relink search while cur is not observed
        if not observed and last_obs is not None and relinkable:
            t0 = frames[last_obs]["t"]
            reid_pending = pending is not None and pending["evidence"]["kind"] == "reid"
            if search_done or reid_pending:
                pass                                   # the predecessor search is over for this loss
            elif any(frames[k]["camera"]["status"] in _BREAKS for k in range(last_obs + 1, n + 1)):
                drop("CAMERA_BREAK")
            else:
                cands = sorted(c for c, b in birth.items()
                               if last_obs < b <= n and frames[b]["t"] - t0 <= tc.relink_max_gap_s + _EPS)
                listers = [c for c in cands if cur in preds[c]]
                if len(listers) > 1:
                    drop("SEVERAL_CANDIDATES")
                elif pending is not None and not keep(pending, t):
                    drop("CANDIDATE_NOT_USABLE")
                elif pending is None and len(listers) == 1 and usable_cand(live.get(listers[0])):
                    b = listers[0]
                    pending = {"from_track": cur, "to_track": b, "lost_frame": frames[last_obs]["frame"],
                               "proposed_frame": f, "gap_s": round(frames[birth[b]]["t"] - t0, 6),
                               "evidence": {"kind": "predecessor", "possible_predecessors": list(preds[b]),
                                            "candidates": cands},
                               "confirmation": None, "rejection": None, "outcome": "PENDING",
                               "dropped_reason": None}
                    relinks.append(pending)
                    cand_ok_t = t
            if reid_pending and not keep(pending, t):
                drop("CANDIDATE_NOT_USABLE")
            elif reid_pending and not held and tc.reentry_withdraw and pending["proposed_frame"] < f:
                # option 2: re-run the rule on the evidence so far (the offered track allowed again); an offer it no
                # longer makes for that track is withdrawn, and the search below may offer another track
                from .reid import reentry_choice
                b = pending["to_track"]
                again = reentry_choice(cur, last_obs, n, rctx, sig, tc, cfg.tracking,
                                       allowed=lambda c: (c == b or c not in reid_done) and usable_cand(live.get(c)),
                                       causal=True)
                if not (again["offer"] and again["track"] == b):
                    reentry_log.append(_log_row(f, cur, again))
                    log_key = (cur, last_obs, again["offer"], again["reason"], again["track"])
                    drop("WITHDRAWN")
                    search_done = True
            if pending is None and sig and (search_done or fr["t"] - t0 > tc.relink_max_gap_s + _EPS):
                if rctx is None:
                    from .reid import ReentryContext
                    rctx = ReentryContext.of(tracks_doc)
                pending, decision = _reentry_offer(cur, last_obs, n, rctx, sig, reid_done, live, cfg, rejected)
                key = (cur, last_obs, decision["offer"], decision["reason"], decision["track"])
                if key != log_key:
                    reentry_log.append(_log_row(f, cur, decision))
                    log_key = key
                if pending is not None:
                    relinks.append(pending)
                    cand_ok_t = t
            base = dict(answer)
            if pending is not None and not held:
                b = pending["to_track"]
                answer.update(state=TargetState.CANDIDATE.value,
                              reason=(TargetReason.REENTRY_REID if pending["evidence"]["kind"] == "reid"
                                      else TargetReason.RELINK_PREDECESSOR).value,
                              track_id=b, box=list(live[b]["box"]))
                for a in deferred:                     # the user acted on the offer made on this frame
                    if (a["from_track"], a["to_track"], a["proposed_frame"]) != \
                            (pending["from_track"], b, pending["proposed_frame"]):
                        continue
                    used.add(id(a))
                    if (a["kind"], a["version"]) == REJECT_CANDIDATE:
                        pending["rejection"] = dict(a)
                        drop("REJECTED_BY_USER")
                        answer = base
                    else:
                        confs[(f, b)] = a              # handled as an acceptance below
                    break
            if pending is not None and not held:
                b = pending["to_track"]
                key = (f, b)
                auto = tc.relink_auto_confirm and pending["evidence"]["kind"] == "predecessor"   # never re-ID
                if auto or key in confs:
                    if key in confs:
                        used.add(id(confs[key]))
                    pending.update(confirmation=confs.get(key, {"kind": "auto_confirm", "frame": f}),
                                   outcome="CONFIRMED")
                    cur, cur_ended, pending, vouch = b, False, None, None
                    tainted, episodes0 = False, live[b]["uncertain_episodes"]
                    anchored = live[b]["identity"] == IdentityState.CERTAIN.value
                    last_obs, relinkable, search_done = n, True, False
                    reid_done.clear()
                    answer.update(state=TargetState.TARGET.value, reason=TargetReason.TRACK_OBSERVED.value)
        out.append(answer)
    unused = [a for a in acts if id(a) not in used]
    if unused:
        raise UserActionError(f"user actions that do not apply (no candidate / offer / box on their frame, or "
                              f"before the selection): {unused}")
    log = []
    for a in acts:
        if (a["kind"], a["version"]) != PLAYER_CONFIRMATION:
            continue
        r = revocations.get(a["confirmation_id"])
        log.append(clog[a["confirmation_id"]] if r is None else
                   {"confirmation_id": a["confirmation_id"], "record": dict(a), "status": "REVOKED", "episode": None,
                    "track_identity": None, "track_rivals": [], "target_track": None, "conflicts": [], "evidence": {},
                    "gallery_crops": 0, "revocation": dict(r)})
    greid = [{k: v for k, v in r.items() if not k.startswith("_")} for _, r in sorted(gal_runs.items())]
    return out, {"track_ends": track_ends, "relinks": relinks, "reentry_log": reentry_log, "confirmation_log": log,
                 "gallery": {"human_confirmed": gallery}, "gallery_reid_log": greid}


def gap_duration_s(gap: dict) -> float:
    """Length of a spana.target gap: from its first to its last frame time (a one-frame gap is 0 s)."""
    return gap["end_t"] - gap["start_t"]


def review_list(gaps: Sequence[dict], cfg: SpanaConfig) -> List[dict]:
    """The gaps the after-match review shows the user, in time order: those lasting at least
    target.review_min_gap_s (decision 2026-11-16: the gap list is the primary recovery UI). Shorter gaps stay in the
    document and on the timeline, but are not offered as review items."""
    return [g for g in gaps if gap_duration_s(g) >= cfg.target.review_min_gap_s - _EPS]


def reconfirmation_list(doc_or_log) -> List[dict]:
    """Confirmations the app must show again (spana.target 2.10.0): every one in CONFLICT, with why. Accepts a
    spana.target document or its confirmation_log."""
    log = doc_or_log["confirmation_log"] if isinstance(doc_or_log, dict) else doc_or_log
    return [{"confirmation_id": e["confirmation_id"], "track_id": e["record"]["track_id"],
             "frame": e["record"]["frame"], "verdict": e["record"]["verdict"],
             "reasons": [c["reason"] for c in e["conflicts"]]} for e in log if e["status"] == "CONFLICT"]


def target_document(tracks_doc: dict, sel: Selection, cfg: SpanaConfig, argv: Sequence[str] = (),
                    actions: Iterable[dict] = (), embeddings: Optional[dict] = None,
                    crops: Optional[CropFeatures] = None, claims: Optional[Dict[str, Dict[int, str]]] = None) -> dict:
    """Validated spana.target 2.10.0 document for one selection (and the user's actions, player confirmations
    included) on one validated spana.tracks document (and, for re-entry offers, its validated spana.track_embeddings
    document; for the confirmation gallery / conflicts / gallery re-ID, its clean crop features and other identities'
    certain answers)."""
    actions = [dict(a) for a in actions]
    frames, info = follow(tracks_doc, sel, cfg, actions, embeddings, crops, claims)
    tsha = sha256_doc(tracks_doc)
    esha = None if embeddings is None else sha256_doc(embeddings)
    doc = {
        "schema": TARGET_SCHEMA[0], "schema_version": TARGET_SCHEMA[1],
        "provenance": build_provenance(config_sha256=cfg.sha256(), argv=list(argv) or sys.argv,
                                       inputs={"video": tracks_doc["video"]["sha256"], "tracks": tsha,
                                               **({} if esha is None else {"embeddings": esha})}),
        "video": tracks_doc["video"], "tracks_sha256": tsha, "embeddings_sha256": esha, "selection": sel.to_dict(),
        "user_actions": actions,
        "target_config": {"resume_after_uncertain": cfg.target.resume_after_uncertain,
                          "relink_max_gap_s": cfg.target.relink_max_gap_s,
                          "relink_auto_confirm": cfg.target.relink_auto_confirm,
                          "tap_vouches_doubt": cfg.target.tap_vouches_doubt,
                          "resume_after_doubt_resolved": cfg.target.resume_after_doubt_resolved,
                          "reentry_offer": cfg.target.reentry_offer,
                          "reentry_max_gap_s": cfg.target.reentry_max_gap_s,
                          "reentry_min_similarity": cfg.target.reentry_min_similarity,
                          "reentry_min_margin": cfg.target.reentry_min_margin,
                          "reentry_mutual": cfg.target.reentry_mutual,
                          "reentry_position_weight": cfg.target.reentry_position_weight,
                          "reentry_signature": cfg.target.reentry_signature,
                          "reentry_wait_samples": cfg.target.reentry_wait_samples,
                          "reentry_withdraw": cfg.target.reentry_withdraw,
                          "reentry_exit_edge": cfg.target.reentry_exit_edge,
                          "reentry_exit_edge_max_frac": cfg.target.reentry_exit_edge_max_frac,
                          "offer_hold_s": cfg.target.offer_hold_s,
                          "resume_after_reid": cfg.target.resume_after_reid,
                          "confirm_contradict_reid_margin": cfg.target.confirm_contradict_reid_margin,
                          "confirm_contradict_colour_margin": cfg.target.confirm_contradict_colour_margin,
                          "gallery_reid_resolve": cfg.target.gallery_reid_resolve,
                          "confirmation_scope": cfg.target.confirmation_scope,
                          "reid_resolve_enabled": cfg.reid_resolve.enabled,
                          "reid_margin": cfg.reid_resolve.margin},
        "track_ends": info["track_ends"], "relinks": info["relinks"], "frames": frames,
        "gaps": gaps_of(frames, info["relinks"]), "reentry_log": info["reentry_log"],
        "crop_features": None if crops is None else dict(crops.meta),
        "claims_identities": sorted(claims or {}),
        "confirmation_log": info["confirmation_log"], "gallery": info["gallery"],
        "gallery_reid_log": info["gallery_reid_log"],
    }
    validate_target(doc, tracks_doc, embeddings)
    return doc
