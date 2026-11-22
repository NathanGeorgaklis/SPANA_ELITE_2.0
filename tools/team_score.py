"""Score spana.teams 1.0.0 (team classification v1) against human side labels. DEV and TRAIN only; EVAL is refused.

    python -m tools.team_score dev --teams G1.teams.json --clip data/labels/<clip>/v1 TRACKS [--clip ...] --out S.json
    python -m tools.team_score train --labels data/reid/TRAIN/<match>/v1 --video V --tracks T --detections D \
        --config config/default.yaml --out S.json

DEV (`dev`): one spana.teams document per match (made by `spana.cli teams` from the same tracks) against the frozen
side labels `data/labels/<clip>/teams_v1` (hashes checked).
- Unit: a labelled frame (every 6th) x an observed track that the NESTED_V2 attribution (tools.target_score)
  gives one GT id unambiguously. Two-player boxes are excluded and counted.
- Truth: A / B -> the per-match team cluster by the one A/B <-> TEAM_A/TEAM_B permutation with the most agreement
  on answered team frames (fixed for every clip of the match, reported with both counts); GK_A / GK_B ->
  GOALKEEPER (v1 has no keeper side); REF -> REFEREE; OTHER -> no answer expected (any answer is wrong).
- An answer is a decided label (not UNKNOWN). Wrong = answered and not the truth, including a REFEREE / GOALKEEPER
  answer on a team player (counted separately as role answers on team players).
- Reported per clip / match: coverage (answered / attributed track-frames, and answered / GT player boxes), wrong
  answered frames with the exact one-sided 95 % (Clopper-Pearson) bound, the confusion, why unanswered frames were
  UNKNOWN, answers on tracks with no GT id, role ids as counts, TEAM_CHANGE flags against the labels.

TRAIN (`train`, the sanity check Nathan approved): the labelled tracklets of a TRAIN re-ID match
(`identities.jsonl`, decision "player", side per player). The classifier runs in-process on the TRAIN tracks the
labels were made on (sha256 checked); every tracks frame inside a player tracklet where the track is observed is a
unit. The TRAIN detections are COCO person-only (spana.detections 1.3.0): no role hint exists there, so REFEREE /
GOALKEEPER cannot be answered and the check covers team sides only.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from spana.artifacts import (TEAMS_SCHEMA, read_json, refuse_existing, sha256_doc, validate_teams,
                             validate_tracks, write_json_atomic)
from spana.records import TeamLabel
from spana.reid_labels import SIDES, split_guard
from tools.benchmark import load_frozen
from tools.reid_cluster import upper_bound
from tools.target_score import attributions

CONFIDENCE = 0.95
BOUND_MAX = 0.005           # acceptance (DECISIONS 2026-11-22): exact one-sided 95 % bound on wrong answered frames
TEAM_SIDES = ("A", "B")
ROLE_TRUTH = {"GK_A": "GOALKEEPER", "GK_B": "GOALKEEPER", "REF": "REFEREE"}
_ROLES = ("GOALKEEPER", "REFEREE")


def load_sides(label_dir: Path) -> Tuple[dict, Dict[str, str]]:
    """(FROZEN.json, {gt id: side}) of the frozen side labels next to a frozen DEV label version."""
    d = label_dir.parent / f"teams_{label_dir.name}"
    frozen = json.loads((d / "FROZEN.json").read_text())
    raw = (d / "sides.json").read_bytes()
    split_guard(frozen["split"], "tune")
    if hashlib.sha256(raw).hexdigest() != frozen["sides_sha256"]:
        raise SystemExit(f"refusing: {d}/sides.json no longer matches its frozen hash")
    sides = json.loads(raw)["sides"]
    if any(v not in SIDES for v in sides.values()):
        raise SystemExit(f"refusing: {d}: unknown side")
    return frozen, sides


def _labels_by_track(teams: dict, clip: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for t in teams["tracks"]:
        if t["clip"] == clip:
            out[t["track_id"]] = t["segments"]
    return out


def _label(segs: Dict[str, List[dict]], tid: str, f: int) -> Tuple[str, str]:
    for sg in segs.get(tid, ()):
        if sg["frame"] <= f <= sg["frame_end"]:
            return sg["label"], (sg["reasons"][0] if sg["reasons"] else "")
    return TeamLabel.UNKNOWN.value, "OUTSIDE_SEGMENT"


def _perm(units: Sequence[dict]) -> Tuple[Dict[str, str], dict]:
    """A/B -> TEAM_A/TEAM_B: the permutation with the most agreement on answered team-player frames (ties: A ->
    TEAM_A)."""
    ident = {"A": "TEAM_A", "B": "TEAM_B"}
    swap = {"A": "TEAM_B", "B": "TEAM_A"}
    agree = lambda m: sum(1 for u in units if u["side"] in m and u["pred"] == m[u["side"]])   # noqa: E731
    a, b = agree(ident), agree(swap)
    return (ident if a >= b else swap), {"A->TEAM_A": a, "A->TEAM_B": b}


def _truth(side: str, perm: Dict[str, str]) -> Optional[str]:
    return perm.get(side) or ROLE_TRUTH.get(side)      # OTHER -> None (no answer expected)


def summarise(units: List[dict], perm: Dict[str, str], gt_boxes: int) -> dict:
    """Counts over scored units (dicts with side, pred, reason, player)."""
    n = len(units)
    ans = [u for u in units if u["pred"] != TeamLabel.UNKNOWN.value]
    wrong = [u for u in ans if u["pred"] != _truth(u["side"], perm)]
    role_on_team = [u for u in ans if u["side"] in TEAM_SIDES and u["pred"] in _ROLES]
    conf = Counter(f"{u['side']}->{u['pred']}" for u in units)
    why = Counter(u["reason"] for u in units if u["pred"] == TeamLabel.UNKNOWN.value)
    team_units = [u for u in units if u["side"] in TEAM_SIDES]
    team_ans = [u for u in team_units if u["pred"] != TeamLabel.UNKNOWN.value]
    return {
        "units": n, "gt_player_boxes": gt_boxes, "answered": len(ans), "wrong": len(wrong),
        "coverage": round(len(ans) / n, 4) if n else None,
        "coverage_of_gt_boxes": round(len(ans) / gt_boxes, 4) if gt_boxes else None,
        "team_player_units": len(team_units), "team_player_answered": len(team_ans),
        "team_player_coverage": round(len(team_ans) / len(team_units), 4) if team_units else None,
        "wrong_rate": round(len(wrong) / len(ans), 5) if ans else None,
        "wrong_bound_95": None if not ans else round(upper_bound(len(wrong), len(ans), CONFIDENCE), 5),
        "role_answers_on_team_players": {"frames": len(role_on_team),
                                         "ids": sorted({u["player"] for u in role_on_team}, key=_idkey)},
        "wrong_ids": dict(sorted(Counter(u["player"] for u in wrong).items(), key=lambda kv: _idkey(kv[0]))),
        "confusion": dict(sorted(conf.items())), "unknown_reasons": dict(sorted(why.items())),
        "players_answered": len({u["player"] for u in ans}), "players": len({u["player"] for u in units}),
    }


def _idkey(p: str):
    return (0, int(p)) if str(p).isdigit() else (1, str(p))


def roles(units: List[dict], sides: Dict[str, Dict[str, str]]) -> List[dict]:
    """Per GK / REF id: counts of what it was answered (frames), not a rate."""
    out = []
    for clip, s in sorted(sides.items()):
        for pid, side in sorted(s.items(), key=lambda kv: _idkey(kv[0])):
            if side not in ROLE_TRUTH:
                continue
            us = [u for u in units if u["clip"] == clip and u["player"] == pid]
            out.append({"clip": clip, "id": pid, "side": side, "truth": ROLE_TRUTH[side], "units": len(us),
                        "answered": dict(sorted(Counter(u["pred"] for u in us).items())),
                        "unknown_reasons": dict(sorted(Counter(u["reason"] for u in us
                                                               if u["pred"] == TeamLabel.UNKNOWN.value).items()))})
    return out


def score_dev(teams: dict, clips: List[Tuple[Path, dict]]) -> dict:
    """One match: its spana.teams document and (label dir, tracks) per clip."""
    by_clip = {i["clip"]: i for i in teams["inputs"]}
    units: List[dict] = []
    extra = Counter()
    gt_boxes: Dict[str, int] = {}
    sides_all: Dict[str, Dict[str, str]] = {}
    flags = []
    tracks_by_clip = {}
    for label_dir, tdoc in clips:
        frozen, gt = load_frozen(label_dir)
        sfrozen, sides = load_sides(label_dir)
        clip = frozen["clip"]["name"]
        if sfrozen["match_id"] != teams["match_id"]:
            raise SystemExit(f"refusing: {clip} is labelled for match {sfrozen['match_id']}, teams doc for "
                             f"{teams['match_id']}")
        if clip not in by_clip or by_clip[clip]["tracks_sha256"] != sha256_doc(tdoc):
            raise SystemExit(f"refusing: the teams document was not made from these {clip} tracks")
        if tdoc["video"]["sha256"] != frozen["clip"]["sha256"]:
            raise SystemExit(f"refusing: {clip} tracks are of another video")
        tracks_by_clip[clip] = tdoc
        sides_all[clip] = sides
        segs = _labels_by_track(teams, clip)
        attr = attributions(gt, tdoc)
        gt_boxes[clip] = sum(len(v) for v in gt.players.values())
        for f in sorted(attr):
            for tid, (pid, amb) in sorted(attr[f].items()):
                pred, why = _label(segs, tid, f)
                if pid is None:
                    if pred != TeamLabel.UNKNOWN.value:
                        extra[clip] += 1
                    continue
                if amb:
                    extra[f"{clip}:two_player_boxes"] += 1
                    continue
                units.append({"clip": clip, "frame": f, "track": tid, "player": pid, "side": sides[pid],
                              "pred": pred, "reason": why})
        for t in teams["tracks"]:
            if t["clip"] == clip and t["flags"]:
                seen = sorted({(u["player"], sides[u["player"]]) for u in units
                               if u["clip"] == clip and u["track"] == t["track_id"]})
                flags.append({"clip": clip, "track_id": t["track_id"], "labels": [s["label"] for s in t["segments"]],
                              "gt_players_on_track": seen})
    validate_teams(teams, tracks_by_clip)
    perm, perm_counts = _perm(units)
    per_clip = {}
    for clip in sorted(tracks_by_clip):
        us = [u for u in units if u["clip"] == clip]
        per_clip[clip] = {**summarise(us, perm, gt_boxes[clip]),
                          "answers_on_tracks_without_gt_id": extra[clip],
                          "two_player_box_units_excluded": extra[f"{clip}:two_player_boxes"]}
    return {"match_id": teams["match_id"], "model": teams["model"], "permutation": perm,
            "permutation_agreement": perm_counts, "match": summarise(units, perm, sum(gt_boxes.values())),
            "clips": per_clip, "roles": roles(units, sides_all), "team_change_flags": flags}


def score_train(label_dir: Path, video: str, tracks_path: str, det_path: str, config: str) -> dict:
    """TRAIN sanity check on the labelled tracklets of one re-ID match (team sides only, see module doc)."""
    from spana.config import load_config
    from spana.reid_labels import load_frozen as load_reid
    from spana.teams.classifier import classify, iter_frames, segments_of_clip
    from spana.video import VideoSource
    lab = load_reid(label_dir, "train")
    cfg = load_config(config)
    tdoc, ddoc = read_json(tracks_path), read_json(det_path)
    frozen = json.loads((label_dir / "FROZEN.json").read_text())
    if sha256_doc(tdoc) != frozen["tracks_sha256"]:
        raise SystemExit("refusing: these are not the tracks the TRAIN labels were made on")
    if tdoc["detections_sha256"] != sha256_doc(ddoc):
        raise SystemExit("refusing: detections are not the ones the tracks were made from")
    validate_tracks(tdoc)
    v = VideoSource(video, expected_sha256=frozen["video_sha256"])
    if ddoc["schema_version"] != "1.3.0" or "class_names" in ddoc:
        raise SystemExit("refusing: expected the TRAIN person-only detections (spana.detections 1.3.0)")
    ddoc = {**ddoc, "class_names": ["person"]}     # COCO person-only: no role hint exists on TRAIN
    first, last = tdoc["frames"][0]["frame"], tdoc["frames"][-1]["frame"]
    match = lab["manifest"]["match_id"]
    segs = segments_of_clip(match, tdoc, ddoc, iter_frames(v, first, last), cfg.appearance, cfg.team)
    model, tracks = classify(segs, cfg.team, cfg.output.score_decimals)
    teams = {"schema": TEAMS_SCHEMA[0], "schema_version": TEAMS_SCHEMA[1], "provenance": {}, "match_id": match,
             "inputs": [{"clip": match, "video_sha256": v.meta.sha256, "tracks_sha256": sha256_doc(tdoc),
                         "detections_sha256": tdoc["detections_sha256"]}],
             "team_config": dataclasses.asdict(cfg.team), "appearance_config": dataclasses.asdict(cfg.appearance),
             "feature": "torso_colour_hs_hist_v1", "model": model, "tracks": tracks}
    validate_teams(teams, {match: tdoc})
    side_of: Dict[str, str] = {}
    for row in lab["identities"]:
        if row["decision"] == "player":
            if side_of.setdefault(row["player"], row["side"]) != row["side"]:
                raise SystemExit(f"refusing: player {row['player']} has two sides")
    segs_by = _labels_by_track(teams, match)
    observed = {(fr["frame"], r["track_id"]) for fr in tdoc["frames"] for r in fr["tracks"]
                if r["source"] == "OBSERVED"}
    units = []
    for row in lab["identities"]:
        if row["decision"] != "player":
            continue
        for f in range(row["first_frame"], row["last_frame"] + 1):
            if (f, row["source_track"]) not in observed:
                continue
            pred, why = _label(segs_by, row["source_track"], f)
            units.append({"clip": match, "frame": f, "track": row["source_track"], "player": row["player"],
                          "side": side_of[row["player"]], "pred": pred, "reason": why})
    perm, perm_counts = _perm(units)
    return {"match_id": match, "model": model, "permutation": perm, "permutation_agreement": perm_counts,
            "match": summarise(units, perm, 0), "roles": roles(units, {match: side_of}),
            "team_config_sha256": hashlib.sha256(json.dumps(teams["team_config"], sort_keys=True).encode()
                                                 ).hexdigest(),
            "note": "TRAIN detections are COCO person-only: no role hint, REFEREE / GOALKEEPER cannot be answered"}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--teams", required=True)
    d.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABEL_DIR", "TRACKS"))
    d.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--labels", required=True)
    t.add_argument("--video", required=True)
    t.add_argument("--tracks", required=True)
    t.add_argument("--detections", required=True)
    t.add_argument("--config", required=True)
    t.add_argument("--out", required=True)
    a = p.parse_args(argv)
    refuse_existing(a.out)
    if a.cmd == "dev":
        res = score_dev(read_json(a.teams), [(Path(l), read_json(tr)) for l, tr in a.clip])
    else:
        res = score_train(Path(a.labels), a.video, a.tracks, a.detections, a.config)
    print(write_json_atomic(a.out, res))
    m = res["match"]
    print(f"{res['match_id']}: units {m['units']} answered {m['answered']} wrong {m['wrong']} "
          f"coverage {m['coverage']} bound {m['wrong_bound_95']} role-on-team {m['role_answers_on_team_players']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
