"""Score the target follower against FROZEN human DEV labels: per labelled player, as if the user had tapped them.

    python -m tools.target_score --config config/default.yaml --out REPORT.json \
        --clip data/labels/<clip>/v1 TRK.json [--clip ...] [--artifacts-dir DIR]

For each labelled player of a clip:
  selection  the player's first labelled frame on which an OBSERVED track is attributed to them, unambiguously,
             under the default match rule NESTED_V2 (spana.evaluation, same matching and ignore-region convention as
             tools/benchmark.py). Labelled frames before it are counted (`frames_before_selection`), not scored.
  follow     spana.identity.follower on that track at that frame (one spana.target document per player).
  score      over the player's labelled frames from the selection on (labels are evenly spaced, so frame share =
             time share):
               correct     TARGET box attributed to this player
               wrong       TARGET box attributed to another labelled player (must stay ~0)
               unverified  TARGET box with no clear attribution (NESTED_V2 ambiguous, matched to nobody, or inside
                           an ignore region): neither credited nor counted wrong
               candidate_correct / candidate_wrong / candidate_unverified
                           CANDIDATE proposals (spana.target 2.0.0), judged like answers but never correct time
               abstain / lost   no answer
             plus `answers_while_absent` / `candidates_while_absent`: TARGET / CANDIDATE on a labelled frame where
             the player is not labelled (left the picture or unlabelled), with how many of those are attributed to
             another player.
Relinks: each proposal is judged right / wrong / undecided on the first labelled frame (from its proposal on) where
the answer is on the relinked track and its box is clearly attributed - whether or not the followed player is labelled
on that frame: an offered box that is clearly another player is wrong even while ours is out of the picture
(Nathan's FRESH_08 review, 2026-11-19; records/offer_verdict_2026-11-19). --simulate-user confirms, on its first frame,
every candidate judged right, and nothing else. --simulate-reviewer works through the gap list after the match
(decision 2026-11-16): per gap, confirm a right offer / reject a wrong one, else tap the player on the gap's first
labelled frame where they have an unambiguous track; it reports the actions taken (effort) next to the times. Run with relink_auto_confirm true (config) to see automatic promotion.
--confirmations (with --simulate-reviewer; spana.target 2.10.0): a reviewer tap on a track that is IDENTITY_UNCERTAIN
on that frame issues a player_confirmation IS_ME instead of a re-selection, covering the following labelled frames of
the same gap where the track is still this player's, in the same episode. --review-sim-config sets a share of wrong
taps (config/review_sim.yaml: another labelled player's track, the tapped track's rivals first). The other players'
CERTAIN answers (their reviewer run without confirmations) are the other identities' claims for the conflict rule; a
confirmation in CONFLICT is retried once on a later labelled frame of the same gap (re-confirmation). --crops loads
each clip's clean crop features (<tracks>.crops.npz, spana.cli crops) for the gallery / appearance conflicts / gallery
re-ID. Reports confirmations issued / applied / conflict (by reason) / wrong / wrong blocked / wrong through, and an
undo check per player (confirm then revoke == no confirmation, every output).
Pooled per clip = sums over its players. DEV labels only (refused otherwise); EVAL is never read.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hashlib

from spana.artifacts import read_json, sha256_doc, validate_track_embeddings, validate_tracks, write_json_atomic
from spana.config import load_config, typed_from_dict
from spana.identity.confirmation import CropFeatures, claims_from_targets, player_confirmation, revoke_confirmation
from spana.evaluation import NESTED_V2, _match
from spana.artifacts import gaps_of
from spana.identity.follower import (Selection, confirm_candidate, confirm_relink, follow, reject_candidate, reselect,
                                     review_list, target_document)
from tools.benchmark import IGNORE_IOA, _ioa, load_frozen

RULE = NESTED_V2
OUTPUTS = ("track_ends", "relinks", "reentry_log", "gallery", "gallery_reid_log")


@dataclasses.dataclass(frozen=True)
class ReviewSimConfig:
    schema_version: str
    wrong_is_me_rate: float      # share of confirmation taps that land on another player's track
    seed: int


def load_review_sim(path) -> ReviewSimConfig:
    import yaml
    c = typed_from_dict(ReviewSimConfig, yaml.safe_load(Path(path).read_text()), "review_sim")
    if c.schema_version != "1.0.0" or not 0.0 <= c.wrong_is_me_rate <= 1.0:
        raise SystemExit("review_sim config: schema_version 1.0.0 and wrong_is_me_rate in [0, 1]")
    return c


def draw(sim: ReviewSimConfig, *key) -> float:
    """Deterministic uniform [0, 1) for one tap."""
    h = hashlib.sha256("|".join(str(k) for k in (sim.seed,) + key).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def wrong_track(tracks_frame: dict, attr_f: dict, hit: str, pid: str) -> Optional[str]:
    """The track a mistaken tap lands on: another labelled player's observed track on that frame, the tapped track's
    identity rivals first, else the nearest box (centre distance)."""
    live = {t["track_id"]: t for t in tracks_frame["tracks"]}
    others = [tid for tid, (p, amb) in attr_f.items() if p is not None and p != pid and not amb and tid != hit]
    if not others:
        return None
    rivals = [t for t in live[hit]["identity_rivals"] if t in others]
    if rivals:
        return sorted(rivals)[0]
    hb = live[hit]["box"]
    c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)     # noqa: E731
    return min(others, key=lambda t: (math.dist(c(live[t]["box"]), c(hb)), t))
COUNTS = ("correct", "wrong", "unverified", "candidate_correct", "candidate_wrong", "candidate_unverified", "abstain",
          "lost")


def attributions(gt, tracks_doc: dict) -> Dict[int, Dict[str, Tuple[Optional[str], bool]]]:
    """labelled frame -> {observed track id: (player or None, ambiguous)}; tracks inside an ignore region -> absent."""
    trk = {fr["frame"]: fr for fr in tracks_doc["frames"]}
    out = {}
    for f in gt.frames:
        if f not in trk:
            raise SystemExit(f"labelled frame {f} was not processed by the run")
        ign = gt.ignore.get(f, [])
        obs = [t for t in trk[f]["tracks"] if t["source"] == "OBSERVED"
               and not any(_ioa(t["box"], g) >= IGNORE_IOA for g in ign)]
        players = gt.players.get(f, {})
        m, _, amb = _match(sorted(players), players, [t["box"] for t in obs], RULE, frozenset(gt.occluded.get(f, ())))
        out[f] = {t["track_id"]: (m.get(c), c in amb) for c, t in enumerate(obs)}
    return out


def first_selection(pid: str, gt, attr) -> Optional[Tuple[int, int, str]]:
    """Simulated tap: (labelled frames skipped, frame, track id) of the player's first labelled frame with an
    unambiguous track attribution, or None if there is none."""
    visible = [f for f in gt.frames if pid in gt.players.get(f, {})]
    for n, f in enumerate(visible):
        hit = [tid for tid, (p, amb) in attr[f].items() if p == pid and not amb]
        if hit:
            return n, f, hit[0]
    return None


def judge(attr, f: int, tid: str, pid: str) -> str:
    """correct / wrong / unverified for an answer box of track tid on labelled frame f, for player pid."""
    p, amb = attr[f].get(tid, (None, False))
    return "correct" if (p == pid and not amb) else "wrong" if (p is not None and not amb) else "unverified"


def relink_verdict(r: dict, frames: List[dict], gt, attr, pid: str) -> Tuple[str, Optional[int], Optional[str]]:
    """Right / wrong / undecided for one relink: the first labelled frame from its proposal on where the answer is on
    the relinked track (CANDIDATE or TARGET) and that track's box is attributed clearly - to this player (right) or
    to another labelled player (wrong), whether or not this player is labelled on that frame."""
    labelled = set(gt.frames)
    for a in frames:
        if a["frame"] < r["proposed_frame"] or a["track_id"] != r["to_track"] or \
                a["state"] not in ("CANDIDATE", "TARGET"):
            continue
        if a["frame"] in labelled:
            j = judge(attr, a["frame"], r["to_track"], pid)
            if j != "unverified":
                p, _ = attr[a["frame"]].get(r["to_track"], (None, False))
                return ("right" if j == "correct" else "wrong"), a["frame"], p
    return "undecided", None, None


def simulated_confirmations(tracks_doc: dict, sel: Selection, cfg, gt, attr, pid: str,
                            emb: Optional[dict] = None) -> List[dict]:
    """A user who confirms every candidate that is the right player (judged on the first labelled frame of the
    candidate, NESTED_V2) on the candidate's first frame, and confirms nothing else."""
    confs: List[dict] = []
    judged: set = set()
    while True:
        frames, info = follow(tracks_doc, sel, cfg, confs, emb)
        todo = [r for r in info["relinks"] if r["outcome"] != "CONFIRMED"
                and (r["from_track"], r["to_track"], r["proposed_frame"]) not in judged]
        if not todo:
            return confs
        r = todo[0]
        judged.add((r["from_track"], r["to_track"], r["proposed_frame"]))
        if relink_verdict(r, frames, gt, attr, pid)[0] == "right":
            confs.append(confirm_candidate(r["to_track"], r["proposed_frame"]))


def offer_verdict(r: dict, gt, attr, pid: str) -> str:
    """What a reviewer sees in an offer: who the offered track is, on the first labelled frame from the proposal
    where the offered track is clearly attributed (whether or not it was still offered, and whether or not this
    player is labelled there)."""
    for f in gt.frames:
        if f >= r["proposed_frame"]:
            j = judge(attr, f, r["to_track"], pid)
            if j != "unverified":
                return "right" if j == "correct" else "wrong"
    return "undecided"


def confirmation_end(tracks_doc: dict, gt, attr, pid: str, tid: str, f: int, g: dict) -> int:
    """Last labelled frame of gap g, from f on, up to which the reviewer can vouch for tid: every labelled frame in
    between attributes tid to this player unambiguously, and tid stays live in the same uncertainty episode."""
    trk = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in tracks_doc["frames"]}
    eps = trk[f][tid]["uncertain_episodes"]
    end = f
    for fr in tracks_doc["frames"]:
        x = fr["frame"]
        if x <= f:
            continue
        if x > g["end_frame"] or tid not in trk[x] or trk[x][tid]["uncertain_episodes"] != eps:
            break
        if x in attr:
            p, amb = attr[x].get(tid, (None, False))
            if p != pid or amb:
                break
            end = x
    return end


def simulated_review(tracks_doc: dict, sel: Selection, cfg, gt, attr, pid: str, max_steps: int = 1000,
                     emb: Optional[dict] = None, confirm: bool = False, sim: Optional[ReviewSimConfig] = None,
                     clip: str = "", crops: Optional[CropFeatures] = None, claims: Optional[dict] = None,
                     stats: Optional[dict] = None) -> List[dict]:
    """An after-match reviewer working through the gap list (decision 2026-11-16), knowing the truth: for each gap in
    turn, it confirms an offer judged right or rejects one judged wrong (on the offer's first frame); otherwise it taps
    the player on the gap's first labelled frame where they have an unambiguous track. Best case of the review UI.
    Only the gaps the review list shows (target.review_min_gap_s) are worked through."""
    actions: List[dict] = []
    handled: set = set()
    retried: set = set()
    trk = {fr["frame"]: fr for fr in tracks_doc["frames"]}
    wrong_ids: set = set()
    gap_of: Dict[str, int] = {}            # confirmation id -> start frame of the gap it was made in
    for _ in range(max_steps):
        frames, info = follow(tracks_doc, sel, cfg, actions, emb, crops, claims)
        last = actions[-1] if actions else None
        if confirm and last is not None and last["kind"] == "player_confirmation":
            e = next(x for x in info["confirmation_log"] if x["confirmation_id"] == last["confirmation_id"])
            gs = gap_of[last["confirmation_id"]]
            if e["status"] == "CONFLICT" and gs not in retried:     # surfaced for re-confirmation: once more
                retried.add(gs)
                handled.discard(gs)
        todo = [g for g in review_list(gaps_of(frames, info["relinks"]), cfg) if g["start_frame"] not in handled]
        if not todo:
            if stats is not None:
                stats["wrong_ids"] = wrong_ids
            return actions
        g = todo[0]
        handled.add(g["start_frame"])
        act = None
        for key in g["offers"]:
            r = next(r for r in info["relinks"] if [r["from_track"], r["to_track"], r["proposed_frame"]] == key)
            if r["outcome"] == "CONFIRMED" or r["rejection"] is not None:
                continue
            verdict = offer_verdict(r, gt, attr, pid)
            if verdict == "right":
                act = confirm_relink(r, r["proposed_frame"])
            elif verdict == "wrong":
                act = reject_candidate(r, r["proposed_frame"])
            if act is not None:
                break
        if act is None:
            ans = {a["frame"]: a for a in frames}
            taken = {a["frame"] for a in actions if a["kind"] in ("confirm_relink", "track_at_frame",
                                                                  "player_confirmation")}
            for f in gt.frames:
                if not (g["start_frame"] <= f <= g["end_frame"]) or pid not in gt.players.get(f, {}) or f in taken:
                    continue
                hit = [tid for tid, (p, amb) in attr[f].items() if p == pid and not amb]
                if hit and not (ans[f]["track_id"] == hit[0] and ans[f]["reason"] == "NOT_CONFIRMED"):
                    tr = next(t for t in trk[f]["tracks"] if t["track_id"] == hit[0])
                    if confirm and tr["identity"] != "CERTAIN":
                        cid = f"{pid}@{f}#{len(actions)}"
                        tid, f_end = hit[0], confirmation_end(tracks_doc, gt, attr, pid, hit[0], f, g)
                        if sim is not None and draw(sim, clip, pid, f, len(actions)) < sim.wrong_is_me_rate:
                            w = wrong_track(trk[f], attr[f], hit[0], pid)
                            if w is not None:
                                tid, f_end = w, f
                                wrong_ids.add(cid)
                        act = player_confirmation(cid, tid, f, "IS_ME", f_end)
                        gap_of[cid] = g["start_frame"]
                    else:
                        act = reselect(hit[0], f)
                    break
        if act is not None:
            actions.append(act)
    raise RuntimeError(f"simulated review did not settle for player {pid}")


def same_outputs(a: Tuple[List[dict], dict], b: Tuple[List[dict], dict]) -> bool:
    return a[0] == b[0] and all(a[1][k] == b[1][k] for k in OUTPUTS) and \
        gaps_of(a[0], a[1]["relinks"]) == gaps_of(b[0], b[1]["relinks"])


def undo_check(tracks_doc: dict, sel: Selection, cfg, actions: List[dict], emb, crops, claims) -> dict:
    """Confirm then revoke == no confirmation, on every output: (a) the reviewer's confirmations alone vs no action;
    (b) the full reviewer action list + revocations vs the list without the confirmations (when that list applies)."""
    confs = [a for a in actions if a["kind"] == "player_confirmation"]
    if not confs:
        return {"confirmations": 0}
    revs = [revoke_confirmation(a["confirmation_id"]) for a in confs]
    a_ok = same_outputs(follow(tracks_doc, sel, cfg, confs + revs, emb, crops, claims),
                        follow(tracks_doc, sel, cfg, [], emb, crops, claims))
    rest = [a for a in actions if a["kind"] != "player_confirmation"]
    try:
        ref = follow(tracks_doc, sel, cfg, rest, emb, crops, claims)
    except Exception:              # an action that only applied because of a confirmation: (b) not comparable
        ref = None
    b_ok = None if ref is None else same_outputs(follow(tracks_doc, sel, cfg, actions + revs, emb, crops, claims), ref)
    return {"confirmations": len(confs), "alone_equal": a_ok, "full_equal": b_ok}


def score_player(pid: str, gt, attr, tracks_doc: dict, cfg, art_dir: Optional[Path], clip: str,
                 simulate_user: bool = False, simulate_reviewer: bool = False, emb: Optional[dict] = None,
                 confirm: bool = False, sim: Optional[ReviewSimConfig] = None, crops: Optional[CropFeatures] = None,
                 claims: Optional[dict] = None) -> dict:
    visible = [f for f in gt.frames if pid in gt.players.get(f, {})]
    sel = first_selection(pid, gt, attr)
    row = {"player": pid, "labelled_frames": len(visible), "first_labelled_frame": visible[0] if visible else None}
    if sel is None:
        return {**row, "selected": False}
    n0, f0, tid = sel
    selection = Selection.track_at_frame(tid, f0)
    stats: dict = {}
    confs = simulated_confirmations(tracks_doc, selection, cfg, gt, attr, pid, emb) if simulate_user else \
        simulated_review(tracks_doc, selection, cfg, gt, attr, pid, emb=emb, confirm=confirm, sim=sim, clip=clip,
                         crops=crops, claims=claims, stats=stats) if simulate_reviewer else []
    doc = target_document(tracks_doc, selection, cfg, ["tools.target_score", clip, pid], confs, emb, crops, claims)
    if art_dir is not None:
        write_json_atomic(art_dir / f"{clip}.player{pid}.target.json", doc)
    ans = {fr["frame"]: fr for fr in doc["frames"]}
    c = {k: 0 for k in COUNTS}
    absent = {"TARGET": 0, "CANDIDATE": 0}
    absent_other, absent_other_cand, events = 0, 0, []
    for f in gt.frames:
        if f < f0:
            continue
        a = ans[f]
        answering = a["state"] in ("TARGET", "CANDIDATE")
        if pid not in gt.players.get(f, {}):
            if answering:
                absent[a["state"]] += 1
                p, amb = attr[f].get(a["track_id"], (None, False))
                absent_other += int(a["state"] == "TARGET" and p is not None and not amb)
                absent_other_cand += int(a["state"] == "CANDIDATE" and p is not None and not amb)
                events.append({"frame": f, "kind": f"{a['state'].lower()}_while_absent", "track_id": a["track_id"],
                               "attributed_to": p})
            continue
        if not answering:
            c["abstain" if a["state"] == "ABSTAIN" else "lost"] += 1
            continue
        j = judge(attr, f, a["track_id"], pid)
        c[j if a["state"] == "TARGET" else f"candidate_{j}"] += 1
        if j == "wrong":
            events.append({"frame": f, "kind": "wrong" if a["state"] == "TARGET" else "wrong_candidate",
                           "track_id": a["track_id"], "attributed_to": attr[f][a["track_id"]][0]})
    relinks = []
    for r in doc["relinks"]:
        v, vf, vp = relink_verdict(r, doc["frames"], gt, attr, pid)
        relinks.append({"kind": r["evidence"].get("kind", "predecessor"),
                        "similarity": r["evidence"].get("similarity"),
                        "from_track": r["from_track"], "to_track": r["to_track"], "proposed_frame": r["proposed_frame"],
                        "gap_s": r["gap_s"], "outcome": r["outcome"], "dropped_reason": r["dropped_reason"],
                        "confirmed_by": None if r["confirmation"] is None else r["confirmation"]["kind"],
                        "verdict": v, "verdict_frame": vf, "verdict_player": vp})
    scored = sum(c.values())
    return {**row, "selected": True, "selection": doc["selection"], "frames_before_selection": n0,
            "scored_frames": scored, **c, **pct(c), "answers_while_absent": absent["TARGET"],
            "candidates_while_absent": absent["CANDIDATE"], "answers_while_absent_on_other_player": absent_other,
            "candidates_while_absent_on_other_player": absent_other_cand,
            "track_ends": doc["track_ends"], "confirmations": len(confs), "relinks": relinks,
            "actions": {k: sum(a["kind"] == k for a in confs) for k in
                        ("confirm_candidate", "confirm_relink", "reject_candidate", "track_at_frame",
                         "player_confirmation")},
            "followed_s": round(doc["frames"][-1]["t"] - doc["frames"][0]["t"], 4),
            "gaps": len(doc["gaps"]),
            "confirmation_stats": confirmation_stats(doc, attr, pid, stats.get("wrong_ids", set())),
            "undo": undo_check(tracks_doc, selection, cfg, confs, emb, crops, claims) if confirm else None,
            "gallery_crops": len(doc["gallery"]["human_confirmed"]),
            "gallery_reid": [r for r in doc["gallery_reid_log"]],
            "answers_by_reason": {r: sum(f["reason"] == r for f in doc["frames"] if f["frame"] in attr)
                                  for r in ("PLAYER_CONFIRMED", "GALLERY_REID", "PLAYER_REJECTED",
                                            "CONFIRMATION_CONFLICT")},
            "target_sha256": sha256_doc(doc), "events": events}


def confirmation_stats(doc: dict, attr, pid: str, wrong_ids: set) -> dict:
    """Issued / applied / conflict (by reason) player confirmations; wrong = the confirmed track is another labelled
    player on the confirmation's frame (or a simulated wrong tap); blocked = wrong and in CONFLICT, through = wrong and
    APPLIED; correct_blocked = right and in CONFLICT."""
    out = {"issued": 0, "applied": 0, "conflict": 0, "reasons": {}, "wrong": 0, "wrong_blocked": 0,
           "wrong_through": 0, "correct_blocked": 0, "wrong_events": []}
    for e in doc["confirmation_log"]:
        if e["status"] == "REVOKED":
            continue
        rec = e["record"]
        out["issued"] += 1
        out["applied" if e["status"] == "APPLIED" else "conflict"] += 1
        for c in e["conflicts"]:
            out["reasons"][c["reason"]] = out["reasons"].get(c["reason"], 0) + 1
        p, amb = attr.get(rec["frame"], {}).get(rec["track_id"], (None, False))
        wrong = rec["confirmation_id"] in wrong_ids or (p is not None and p != pid and not amb)
        if wrong:
            out["wrong"] += 1
            out["wrong_blocked" if e["status"] == "CONFLICT" else "wrong_through"] += 1
            out["wrong_events"].append({"confirmation_id": rec["confirmation_id"], "frame": rec["frame"],
                                        "track_id": rec["track_id"], "attributed_to": p, "status": e["status"],
                                        "reasons": [c["reason"] for c in e["conflicts"]]})
        elif e["status"] == "CONFLICT":
            out["correct_blocked"] += 1
    return out


def pct(c: Dict[str, int]) -> dict:
    n = sum(c[k] for k in COUNTS)
    r = lambda k: round(c[k] / n, 4) if n else None      # noqa: E731
    return {"correct_time": r("correct"), "wrong_time": r("wrong"), "unverified_time": r("unverified"),
            "candidate_correct_time": r("candidate_correct"), "candidate_wrong_time": r("candidate_wrong"),
            "candidate_unverified_time": r("candidate_unverified"),
            "no_answer_time": round((c["abstain"] + c["lost"]) / n, 4) if n else None,
            "abstain_time": r("abstain"), "lost_time": r("lost")}


def relink_totals(players: List[dict], kind: Optional[str] = None) -> dict:
    rs = [r for p in players for r in p["relinks"] if kind is None or r["kind"] == kind]
    by_kind = {} if kind is not None else {"by_kind": {k: relink_totals(players, k) for k in ("predecessor", "reid")}}
    return {**by_kind, "proposed": len(rs), "right": sum(r["verdict"] == "right" for r in rs),
            "wrong": sum(r["verdict"] == "wrong" for r in rs), "undecided": sum(r["verdict"] == "undecided" for r in rs),
            "confirmed": sum(r["outcome"] == "CONFIRMED" for r in rs),
            "dropped": sum(r["outcome"] == "DROPPED" for r in rs)}


def score_clip(label_dir: Path, tracks_path: Path, cfg, art_dir: Optional[Path], simulate_user: bool = False,
               simulate_reviewer: bool = False, embeddings_path: Optional[Path] = None, confirm: bool = False,
               sim: Optional[ReviewSimConfig] = None, use_crops: bool = False) -> dict:
    frozen, gt = load_frozen(label_dir)
    tdoc = read_json(tracks_path)
    validate_tracks(tdoc)
    emb = None
    if embeddings_path is not None:
        emb = read_json(embeddings_path)
        validate_track_embeddings(emb, tdoc)
    if tdoc["video"]["sha256"] != frozen["clip"]["sha256"]:
        raise SystemExit("refusing: the tracks were made from a different video than the labels")
    attr = attributions(gt, tdoc)
    ids = sorted({p for f in gt.frames for p in gt.players.get(f, {})}, key=int)
    clip = frozen["clip"]["name"]
    crops = None
    if use_crops:
        crops = CropFeatures.load(tracks_path.with_name(tracks_path.name.replace(".trk.json", ".crops.npz")),
                                  sha256_doc(tdoc))
    claims_of: Dict[str, dict] = {}
    if confirm:
        # other identities' CERTAIN answers: each player's reviewer run WITHOUT confirmations (taps only)
        base: Dict[str, dict] = {}
        for p in ids:
            s = first_selection(p, gt, attr)
            if s is None:
                continue
            sel_p = Selection.track_at_frame(s[2], s[1])
            acts = simulated_review(tdoc, sel_p, cfg, gt, attr, p, emb=emb)
            base[p] = {"frames": follow(tdoc, sel_p, cfg, acts, emb)[0]}
        allc = claims_from_targets(base.items())
        claims_of = {p: {q: c for q, c in allc.items() if q != p} for p in ids}
    players = [score_player(p, gt, attr, tdoc, cfg, art_dir, clip, simulate_user, simulate_reviewer, emb, confirm,
                            sim, crops, claims_of.get(p)) for p in ids]
    sel = [p for p in players if p["selected"]]
    pooled = {k: sum(p[k] for p in sel) for k in COUNTS}
    return {"clip": clip, "labels": {"dir": str(label_dir), "gt_sha256": frozen["gt_sha256"],
                                     "frames": len(gt.frames), "frame_step": frozen["frame_step"]},
            "tracks": {"path": tracks_path.name, "sha256": sha256_doc(tdoc), "commit": tdoc["provenance"].get("git")},
            "embeddings": None if emb is None else {"path": embeddings_path.name, "sha256": sha256_doc(emb)},
            "players": len(players), "players_selected": len(sel),
            "pooled": {**pooled, **pct(pooled),
                       "answers_while_absent": sum(p["answers_while_absent"] for p in sel),
                       "candidates_while_absent": sum(p["candidates_while_absent"] for p in sel),
                       "answers_while_absent_on_other_player": sum(p["answers_while_absent_on_other_player"]
                                                                   for p in sel),
                       "candidates_while_absent_on_other_player": sum(
                           p["candidates_while_absent_on_other_player"] for p in sel),
                       "frames_before_selection": sum(p["frames_before_selection"] for p in sel),
                       "relinks": relink_totals(sel),
                       "actions": {k: sum(p["actions"][k] for p in sel) for k in sel[0]["actions"]} if sel else {},
                       "followed_s": round(sum(p["followed_s"] for p in sel), 4),
                       "confirmations": {k: sum(p["confirmation_stats"][k] for p in sel)
                                         for k in ("issued", "applied", "conflict", "wrong", "wrong_blocked",
                                                   "wrong_through", "correct_blocked")},
                       "conflict_reasons": {r: sum(p["confirmation_stats"]["reasons"].get(r, 0) for p in sel)
                                            for r in sorted({r for p in sel for r in p["confirmation_stats"]["reasons"]})},
                       "undo": {"players_checked": sum(1 for p in sel if p["undo"] and p["undo"]["confirmations"]),
                                "alone_equal": sum(1 for p in sel if p["undo"] and p["undo"].get("alone_equal")),
                                "full_equal": sum(1 for p in sel if p["undo"] and p["undo"].get("full_equal")),
                                "full_not_comparable": sum(1 for p in sel if p["undo"] and p["undo"]["confirmations"]
                                                           and p["undo"].get("full_equal") is None)},
                       "gallery_crops": sum(p["gallery_crops"] for p in sel),
                       "answers_by_reason": {r: sum(p["answers_by_reason"][r] for p in sel)
                                             for r in sel[0]["answers_by_reason"]} if sel else {}},
            "per_player": players}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--clip", nargs="+", action="append", required=True,
                   metavar="LABELS_DIR TRACKS_JSON [EMBEDDINGS_JSON]",
                   help="labels dir, tracks, and optionally the tracks' spana.track_embeddings (re-entry offers)")
    p.add_argument("--out", required=True)
    p.add_argument("--artifacts-dir", default=None, help="also write one spana.target document per player here")
    p.add_argument("--simulate-user", action="store_true",
                   help="confirm each candidate that is the right player (first labelled candidate frame, NESTED_V2) "
                        "on its first frame; confirm nothing else")
    p.add_argument("--confirmations", action="store_true",
                   help="with --simulate-reviewer: taps on IDENTITY_UNCERTAIN tracks issue player_confirmation IS_ME")
    p.add_argument("--review-sim-config", default=None,
                   help="config/review_sim.yaml: share of wrong IS_ME taps (seeded); omit for always-correct taps")
    p.add_argument("--crops", action="store_true",
                   help="load each clip's clean crop features <tracks>.crops.npz (spana.cli crops)")
    p.add_argument("--simulate-reviewer", action="store_true",
                   help="after-match review of the gap list: confirm right offers, reject wrong ones, otherwise tap "
                        "the player on the gap's first labelled frame where they have a track (best case)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    art = Path(a.artifacts_dir) if a.artifacts_dir else None
    if a.simulate_user and a.simulate_reviewer:
        raise SystemExit("choose --simulate-user or --simulate-reviewer, not both")
    if any(len(c) not in (2, 3) for c in a.clip):
        raise SystemExit("--clip takes LABELS_DIR TRACKS_JSON [EMBEDDINGS_JSON]")
    if a.confirmations and not a.simulate_reviewer:
        raise SystemExit("--confirmations needs --simulate-reviewer")
    sim = load_review_sim(a.review_sim_config) if a.review_sim_config else None
    clips = [score_clip(Path(c[0]), Path(c[1]), cfg, art, a.simulate_user, a.simulate_reviewer,
                        Path(c[2]) if len(c) == 3 else None, a.confirmations, sim, a.crops) for c in a.clip]
    report = {"tool": "tools.target_score", "match_rule": RULE.name, "config_sha256": cfg.sha256(),
              "target_config": dataclasses.asdict(cfg.target), "simulate_user": a.simulate_user,
              "simulate_reviewer": a.simulate_reviewer, "confirmations": a.confirmations,
              "review_sim": None if sim is None else dataclasses.asdict(sim), "crops": a.crops, "clips": clips}
    write_json_atomic(a.out, report)
    for c in clips:
        print(c["clip"], {k: c["pooled"][k] for k in ("correct_time", "wrong_time", "unverified_time",
                                                      "candidate_correct_time", "candidate_wrong_time",
                                                      "abstain_time", "lost_time", "answers_while_absent", "relinks",
                                                      "actions")},
              f"players {c['players_selected']}/{c['players']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
