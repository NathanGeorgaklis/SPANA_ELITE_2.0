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
                                by the calibrated re-ID margin, resolved_by REID, and target.resume_after_reid)
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
        if key not in ACTION_KEYS or set(a) != set(ACTION_KEYS[key]) or not isinstance(a["frame"], int) or \
                isinstance(a["frame"], bool) or any(not isinstance(a[k], str) for k in ("track_id", "from_track",
                                                                                         "to_track") if k in a):
            raise UserActionError(f"not a supported user action record: {a!r}")
        canon = tuple(sorted(a.items()))
        if canon in seen:
            raise UserActionError(f"duplicate user action {a!r}")
        seen.add(canon)
        out.append(dict(a))
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
                   cfg: SpanaConfig) -> Tuple[Optional[dict], dict]:
    """Re-entry offer by re-ID (spana.target 2.5.0): the shared rule spana.identity.reid.reentry_choice, restricted
    to tracks usable now and not yet offered for this loss. Returns (relink entry or None, the rule's decision)."""
    from .reid import reentry_choice
    frames = ctx.frames
    ch = reentry_choice(cur, last_obs, n, ctx, sig, cfg.target, cfg.tracking,
                        allowed=lambda c: c not in done and _usable(live.get(c)), causal=True)
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
           embeddings: Optional[dict] = None) -> Tuple[List[dict], dict]:
    """(per-frame answers, {"track_ends": [...], "relinks": [...]}). tracks_doc must be a validated spana.tracks doc;
    embeddings, if given, a validated spana.track_embeddings doc of it (re-entry offers, target.reentry_offer).
    Raises SelectionError / UserActionError on an invalid selection / user action."""
    start = check_selection(tracks_doc, sel)
    tc = cfg.target
    acts = _parse_actions(actions)
    confs = {(a["frame"], a["track_id"]): a for a in acts if (a["kind"], a["version"]) == CONFIRM_CANDIDATE}
    by_frame: Dict[int, List[dict]] = {}
    for a in acts:
        if (a["kind"], a["version"]) != CONFIRM_CANDIDATE:
            by_frame.setdefault(a["frame"], []).append(a)
    used: set = set()                      # ids of applied actions
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
    reentry_log: List[dict] = []           # re-entry rule decisions, logged when they change during a loss
    log_key = None
    cur, cur_ended = sel.track_id, False
    tainted, episodes0 = False, None
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

    def keep(p: dict, t_now: float) -> bool:
        """Keep the pending offer p this frame: its track is usable, or briefly hidden within target.offer_hold_s
        of its last usable frame (held: no CANDIDATE shown)."""
        nonlocal cand_ok_t, held
        tr_c = live.get(p["to_track"])
        if _usable(tr_c):
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
        nonlocal cur, cur_ended, tainted, episodes0, last_obs, relinkable, search_done, vouch
        if pending is not None:
            drop(why)
        cur, cur_ended, tainted, episodes0 = tid, False, False, None
        last_obs, relinkable, search_done = None, False, False
        reid_done.clear()
        vouch = None if tapped is None else vouch_for(tapped)

    def find_relink(a: dict) -> dict:
        r = next((r for r in relinks if (r["from_track"], r["to_track"], r["proposed_frame"]) ==
                  (a["from_track"], a["to_track"], a["proposed_frame"])), None)
        if r is None or a["frame"] < r["proposed_frame"]:
            raise UserActionError(f"{a['kind']} names no offer made by its frame: {a!r}")
        return r

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
            uncertain = tr["identity"] != IdentityState.CERTAIN.value and not vouched
            if vouch is not None and (vouch[0] != cur or tr["identity_rivals"] or tr["uncertain_episodes"] != vouch[1]):
                vouch = None                                   # a rival or a new episode: the tap no longer covers it
            if episodes0 is None:
                episodes0 = tr["uncertain_episodes"] - int(uncertain)
            if uncertain and tc.resume_after_doubt_resolved and _doubt_only(tr):
                episodes0 = tr["uncertain_episodes"]           # a doubt with no rival: abstain now, no taint
            elif uncertain or tr["uncertain_episodes"] > episodes0:
                tainted = not tc.resume_after_uncertain
            if tainted and not uncertain and tc.resume_after_reid and tr.get("resolved_by") == "REID":
                tainted, episodes0 = False, tr["uncertain_episodes"]   # cleared by the calibrated re-ID margin
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
                state, box = TargetState.TARGET, list(tr["box"])
                reason = TargetReason.USER_VOUCHED if vouched else TargetReason.TRACK_OBSERVED
            if observed:
                if pending is not None:
                    drop("ORIGINAL_OBSERVED")
                last_obs = n
                relinkable = tr["confirmed"] and not uncertain and not tainted
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
                elif pending is None and len(listers) == 1 and _usable(live.get(listers[0])):
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
                                       allowed=lambda c: (c == b or c not in reid_done) and _usable(live.get(c)),
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
                pending, decision = _reentry_offer(cur, last_obs, n, rctx, sig, reid_done, live, cfg)
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
                    last_obs, relinkable, search_done = n, True, False
                    reid_done.clear()
                    answer.update(state=TargetState.TARGET.value, reason=TargetReason.TRACK_OBSERVED.value)
        out.append(answer)
    unused = [a for a in acts if id(a) not in used]
    if unused:
        raise UserActionError(f"user actions that do not apply (no candidate / offer / box on their frame, or "
                              f"before the selection): {unused}")
    return out, {"track_ends": track_ends, "relinks": relinks, "reentry_log": reentry_log}


def gap_duration_s(gap: dict) -> float:
    """Length of a spana.target gap: from its first to its last frame time (a one-frame gap is 0 s)."""
    return gap["end_t"] - gap["start_t"]


def review_list(gaps: Sequence[dict], cfg: SpanaConfig) -> List[dict]:
    """The gaps the after-match review shows the user, in time order: those lasting at least
    target.review_min_gap_s (decision 2026-11-16: the gap list is the primary recovery UI). Shorter gaps stay in the
    document and on the timeline, but are not offered as review items."""
    return [g for g in gaps if gap_duration_s(g) >= cfg.target.review_min_gap_s - _EPS]


def target_document(tracks_doc: dict, sel: Selection, cfg: SpanaConfig, argv: Sequence[str] = (),
                    actions: Iterable[dict] = (), embeddings: Optional[dict] = None) -> dict:
    """Validated spana.target 2.8.0 document for one selection (and the user's actions) on one validated
    spana.tracks document (and, for re-entry offers, its validated spana.track_embeddings document)."""
    actions = [dict(a) for a in actions]
    frames, info = follow(tracks_doc, sel, cfg, actions, embeddings)
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
                          "resume_after_reid": cfg.target.resume_after_reid},
        "track_ends": info["track_ends"], "relinks": info["relinks"], "frames": frames,
        "gaps": gaps_of(frames, info["relinks"]), "reentry_log": info["reentry_log"],
    }
    validate_target(doc, tracks_doc, embeddings)
    return doc
