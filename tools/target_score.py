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
             plus `answers_while_absent`: TARGET on a labelled frame where the player is not labelled (left the
             picture or unlabelled), with how many of those are attributed to another player.
Relinks: each proposal is judged right / wrong / undecided on the first labelled frame (from its proposal on) where
the answer is on the relinked track and its box is clearly attributed. --simulate-user confirms, on its first frame,
every candidate judged right, and nothing else. --simulate-reviewer works through the gap list after the match
(decision 2026-11-16): per gap, confirm a right offer / reject a wrong one, else tap the player on the gap's first
labelled frame where they have an unambiguous track; it reports the actions taken (effort) next to the times. Run with relink_auto_confirm true (config) to see automatic promotion.
Pooled per clip = sums over its players. DEV labels only (refused otherwise); EVAL is never read.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spana.artifacts import read_json, sha256_doc, validate_tracks, write_json_atomic
from spana.config import load_config
from spana.evaluation import NESTED_V2, _match
from spana.artifacts import gaps_of
from spana.identity.follower import (Selection, confirm_candidate, confirm_relink, follow, reject_candidate, reselect,
                                     review_list, target_document)
from tools.benchmark import IGNORE_IOA, _ioa, load_frozen

RULE = NESTED_V2
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
    the relinked track (CANDIDATE or TARGET), the player is labelled and the box is attributed clearly."""
    labelled = set(gt.frames)
    for a in frames:
        if a["frame"] < r["proposed_frame"] or a["track_id"] != r["to_track"] or \
                a["state"] not in ("CANDIDATE", "TARGET"):
            continue
        if a["frame"] in labelled and pid in gt.players.get(a["frame"], {}):
            j = judge(attr, a["frame"], r["to_track"], pid)
            if j != "unverified":
                p, _ = attr[a["frame"]].get(r["to_track"], (None, False))
                return ("right" if j == "correct" else "wrong"), a["frame"], p
    return "undecided", None, None


def simulated_confirmations(tracks_doc: dict, sel: Selection, cfg, gt, attr, pid: str) -> List[dict]:
    """A user who confirms every candidate that is the right player (judged on the first labelled frame of the
    candidate, NESTED_V2) on the candidate's first frame, and confirms nothing else."""
    confs: List[dict] = []
    judged: set = set()
    while True:
        frames, info = follow(tracks_doc, sel, cfg, confs)
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
    where the player is labelled and the offered track is clearly attributed (whether or not it was still offered)."""
    for f in gt.frames:
        if f >= r["proposed_frame"] and pid in gt.players.get(f, {}):
            j = judge(attr, f, r["to_track"], pid)
            if j != "unverified":
                return "right" if j == "correct" else "wrong"
    return "undecided"


def simulated_review(tracks_doc: dict, sel: Selection, cfg, gt, attr, pid: str, max_steps: int = 1000) -> List[dict]:
    """An after-match reviewer working through the gap list (decision 2026-11-16), knowing the truth: for each gap in
    turn, it confirms an offer judged right or rejects one judged wrong (on the offer's first frame); otherwise it taps
    the player on the gap's first labelled frame where they have an unambiguous track. Best case of the review UI.
    Only the gaps the review list shows (target.review_min_gap_s) are worked through."""
    actions: List[dict] = []
    handled: set = set()
    for _ in range(max_steps):
        frames, info = follow(tracks_doc, sel, cfg, actions)
        todo = [g for g in review_list(gaps_of(frames, info["relinks"]), cfg) if g["start_frame"] not in handled]
        if not todo:
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
            taken = {a["frame"] for a in actions if a["kind"] in ("confirm_relink", "track_at_frame")}
            for f in gt.frames:
                if not (g["start_frame"] <= f <= g["end_frame"]) or pid not in gt.players.get(f, {}) or f in taken:
                    continue
                hit = [tid for tid, (p, amb) in attr[f].items() if p == pid and not amb]
                if hit and not (ans[f]["track_id"] == hit[0] and ans[f]["reason"] == "NOT_CONFIRMED"):
                    act = reselect(hit[0], f)
                    break
        if act is not None:
            actions.append(act)
    raise RuntimeError(f"simulated review did not settle for player {pid}")


def score_player(pid: str, gt, attr, tracks_doc: dict, cfg, art_dir: Optional[Path], clip: str,
                 simulate_user: bool = False, simulate_reviewer: bool = False) -> dict:
    visible = [f for f in gt.frames if pid in gt.players.get(f, {})]
    sel = first_selection(pid, gt, attr)
    row = {"player": pid, "labelled_frames": len(visible), "first_labelled_frame": visible[0] if visible else None}
    if sel is None:
        return {**row, "selected": False}
    n0, f0, tid = sel
    selection = Selection.track_at_frame(tid, f0)
    confs = simulated_confirmations(tracks_doc, selection, cfg, gt, attr, pid) if simulate_user else \
        simulated_review(tracks_doc, selection, cfg, gt, attr, pid) if simulate_reviewer else []
    doc = target_document(tracks_doc, selection, cfg, ["tools.target_score", clip, pid], confs)
    if art_dir is not None:
        write_json_atomic(art_dir / f"{clip}.player{pid}.target.json", doc)
    ans = {fr["frame"]: fr for fr in doc["frames"]}
    c = {k: 0 for k in COUNTS}
    absent = {"TARGET": 0, "CANDIDATE": 0}
    absent_other, events = 0, []
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
        relinks.append({"from_track": r["from_track"], "to_track": r["to_track"], "proposed_frame": r["proposed_frame"],
                        "gap_s": r["gap_s"], "outcome": r["outcome"], "dropped_reason": r["dropped_reason"],
                        "confirmed_by": None if r["confirmation"] is None else r["confirmation"]["kind"],
                        "verdict": v, "verdict_frame": vf, "verdict_player": vp})
    scored = sum(c.values())
    return {**row, "selected": True, "selection": doc["selection"], "frames_before_selection": n0,
            "scored_frames": scored, **c, **pct(c), "answers_while_absent": absent["TARGET"],
            "candidates_while_absent": absent["CANDIDATE"], "answers_while_absent_on_other_player": absent_other,
            "track_ends": doc["track_ends"], "confirmations": len(confs), "relinks": relinks,
            "actions": {k: sum(a["kind"] == k for a in confs) for k in
                        ("confirm_candidate", "confirm_relink", "reject_candidate", "track_at_frame")},
            "followed_s": round(doc["frames"][-1]["t"] - doc["frames"][0]["t"], 4),
            "gaps": len(doc["gaps"]),
            "target_sha256": sha256_doc(doc), "events": events}


def pct(c: Dict[str, int]) -> dict:
    n = sum(c[k] for k in COUNTS)
    r = lambda k: round(c[k] / n, 4) if n else None      # noqa: E731
    return {"correct_time": r("correct"), "wrong_time": r("wrong"), "unverified_time": r("unverified"),
            "candidate_correct_time": r("candidate_correct"), "candidate_wrong_time": r("candidate_wrong"),
            "candidate_unverified_time": r("candidate_unverified"),
            "no_answer_time": round((c["abstain"] + c["lost"]) / n, 4) if n else None,
            "abstain_time": r("abstain"), "lost_time": r("lost")}


def relink_totals(players: List[dict]) -> dict:
    rs = [r for p in players for r in p["relinks"]]
    return {"proposed": len(rs), "right": sum(r["verdict"] == "right" for r in rs),
            "wrong": sum(r["verdict"] == "wrong" for r in rs), "undecided": sum(r["verdict"] == "undecided" for r in rs),
            "confirmed": sum(r["outcome"] == "CONFIRMED" for r in rs),
            "dropped": sum(r["outcome"] == "DROPPED" for r in rs)}


def score_clip(label_dir: Path, tracks_path: Path, cfg, art_dir: Optional[Path], simulate_user: bool = False,
               simulate_reviewer: bool = False) -> dict:
    frozen, gt = load_frozen(label_dir)
    tdoc = read_json(tracks_path)
    validate_tracks(tdoc)
    if tdoc["video"]["sha256"] != frozen["clip"]["sha256"]:
        raise SystemExit("refusing: the tracks were made from a different video than the labels")
    attr = attributions(gt, tdoc)
    ids = sorted({p for f in gt.frames for p in gt.players.get(f, {})}, key=int)
    clip = frozen["clip"]["name"]
    players = [score_player(p, gt, attr, tdoc, cfg, art_dir, clip, simulate_user, simulate_reviewer) for p in ids]
    sel = [p for p in players if p["selected"]]
    pooled = {k: sum(p[k] for p in sel) for k in COUNTS}
    return {"clip": clip, "labels": {"dir": str(label_dir), "gt_sha256": frozen["gt_sha256"],
                                     "frames": len(gt.frames), "frame_step": frozen["frame_step"]},
            "tracks": {"path": tracks_path.name, "sha256": sha256_doc(tdoc), "commit": tdoc["provenance"].get("git")},
            "players": len(players), "players_selected": len(sel),
            "pooled": {**pooled, **pct(pooled),
                       "answers_while_absent": sum(p["answers_while_absent"] for p in sel),
                       "candidates_while_absent": sum(p["candidates_while_absent"] for p in sel),
                       "answers_while_absent_on_other_player": sum(p["answers_while_absent_on_other_player"]
                                                                   for p in sel),
                       "frames_before_selection": sum(p["frames_before_selection"] for p in sel),
                       "relinks": relink_totals(sel),
                       "actions": {k: sum(p["actions"][k] for p in sel) for k in sel[0]["actions"]} if sel else {},
                       "followed_s": round(sum(p["followed_s"] for p in sel), 4)},
            "per_player": players}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABELS_DIR", "TRACKS_JSON"))
    p.add_argument("--out", required=True)
    p.add_argument("--artifacts-dir", default=None, help="also write one spana.target document per player here")
    p.add_argument("--simulate-user", action="store_true",
                   help="confirm each candidate that is the right player (first labelled candidate frame, NESTED_V2) "
                        "on its first frame; confirm nothing else")
    p.add_argument("--simulate-reviewer", action="store_true",
                   help="after-match review of the gap list: confirm right offers, reject wrong ones, otherwise tap "
                        "the player on the gap's first labelled frame where they have a track (best case)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    art = Path(a.artifacts_dir) if a.artifacts_dir else None
    if a.simulate_user and a.simulate_reviewer:
        raise SystemExit("choose --simulate-user or --simulate-reviewer, not both")
    clips = [score_clip(Path(l), Path(t), cfg, art, a.simulate_user, a.simulate_reviewer) for l, t in a.clip]
    report = {"tool": "tools.target_score", "match_rule": RULE.name, "config_sha256": cfg.sha256(),
              "target_config": dataclasses.asdict(cfg.target), "simulate_user": a.simulate_user,
              "simulate_reviewer": a.simulate_reviewer, "clips": clips}
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
