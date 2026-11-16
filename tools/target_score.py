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
               abstain / lost   no answer
             plus `answers_while_absent`: TARGET on a labelled frame where the player is not labelled (left the
             picture or unlabelled), with how many of those are attributed to another player.
Pooled per clip = sums over its players. DEV labels only (refused otherwise); EVAL is never read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spana.artifacts import read_json, sha256_doc, validate_tracks, write_json_atomic
from spana.config import load_config
from spana.evaluation import NESTED_V2, _match
from spana.identity.follower import Selection, target_document
from tools.benchmark import IGNORE_IOA, _ioa, load_frozen

RULE = NESTED_V2
COUNTS = ("correct", "wrong", "unverified", "abstain", "lost")


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


def score_player(pid: str, gt, attr, tracks_doc: dict, cfg, art_dir: Optional[Path], clip: str) -> dict:
    visible = [f for f in gt.frames if pid in gt.players.get(f, {})]
    sel = first_selection(pid, gt, attr)
    row = {"player": pid, "labelled_frames": len(visible), "first_labelled_frame": visible[0] if visible else None}
    if sel is None:
        return {**row, "selected": False}
    n0, f0, tid = sel
    doc = target_document(tracks_doc, Selection.track_at_frame(tid, f0), cfg, ["tools.target_score", clip, pid])
    if art_dir is not None:
        write_json_atomic(art_dir / f"{clip}.player{pid}.target.json", doc)
    ans = {fr["frame"]: fr for fr in doc["frames"]}
    c = {k: 0 for k in COUNTS}
    absent, absent_other, events = 0, 0, []
    for f in gt.frames:
        if f < f0:
            continue
        a = ans[f]
        if pid not in gt.players.get(f, {}):
            if a["state"] == "TARGET":
                absent += 1
                p, amb = attr[f].get(tid, (None, False))
                absent_other += int(p is not None and not amb)
                events.append({"frame": f, "kind": "answer_while_absent", "attributed_to": p})
            continue
        if a["state"] != "TARGET":
            c["abstain" if a["state"] == "ABSTAIN" else "lost"] += 1
            continue
        p, amb = attr[f].get(tid, (None, False))
        if p == pid and not amb:
            c["correct"] += 1
        elif p is not None and not amb:
            c["wrong"] += 1
            events.append({"frame": f, "kind": "wrong", "attributed_to": p})
        else:
            c["unverified"] += 1
    scored = sum(c.values())
    return {**row, "selected": True, "selection": doc["selection"], "frames_before_selection": n0,
            "scored_frames": scored, **c, **pct(c), "answers_while_absent": absent,
            "answers_while_absent_on_other_player": absent_other, "track_end": doc["track_end"],
            "target_sha256": sha256_doc(doc), "events": events}


def pct(c: Dict[str, int]) -> dict:
    n = sum(c[k] for k in COUNTS)
    r = lambda k: round(c[k] / n, 4) if n else None      # noqa: E731
    return {"correct_time": r("correct"), "wrong_time": r("wrong"), "unverified_time": r("unverified"),
            "no_answer_time": round((c["abstain"] + c["lost"]) / n, 4) if n else None,
            "abstain_time": r("abstain"), "lost_time": r("lost")}


def score_clip(label_dir: Path, tracks_path: Path, cfg, art_dir: Optional[Path]) -> dict:
    frozen, gt = load_frozen(label_dir)
    tdoc = read_json(tracks_path)
    validate_tracks(tdoc)
    if tdoc["video"]["sha256"] != frozen["clip"]["sha256"]:
        raise SystemExit("refusing: the tracks were made from a different video than the labels")
    attr = attributions(gt, tdoc)
    ids = sorted({p for f in gt.frames for p in gt.players.get(f, {})}, key=int)
    clip = frozen["clip"]["name"]
    players = [score_player(p, gt, attr, tdoc, cfg, art_dir, clip) for p in ids]
    sel = [p for p in players if p["selected"]]
    pooled = {k: sum(p[k] for p in sel) for k in COUNTS}
    return {"clip": clip, "labels": {"dir": str(label_dir), "gt_sha256": frozen["gt_sha256"],
                                     "frames": len(gt.frames), "frame_step": frozen["frame_step"]},
            "tracks": {"path": tracks_path.name, "sha256": sha256_doc(tdoc), "commit": tdoc["provenance"].get("git")},
            "players": len(players), "players_selected": len(sel),
            "pooled": {**pooled, **pct(pooled),
                       "answers_while_absent": sum(p["answers_while_absent"] for p in sel),
                       "answers_while_absent_on_other_player": sum(p["answers_while_absent_on_other_player"]
                                                                   for p in sel),
                       "frames_before_selection": sum(p["frames_before_selection"] for p in sel)},
            "per_player": players}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABELS_DIR", "TRACKS_JSON"))
    p.add_argument("--out", required=True)
    p.add_argument("--artifacts-dir", default=None, help="also write one spana.target document per player here")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    art = Path(a.artifacts_dir) if a.artifacts_dir else None
    clips = [score_clip(Path(l), Path(t), cfg, art) for l, t in a.clip]
    report = {"tool": "tools.target_score", "match_rule": RULE.name, "config_sha256": cfg.sha256(),
              "target_config": {"resume_after_uncertain": cfg.target.resume_after_uncertain}, "clips": clips}
    write_json_atomic(a.out, report)
    for c in clips:
        print(c["clip"], {k: c["pooled"][k] for k in ("correct_time", "wrong_time", "unverified_time",
                                                      "abstain_time", "lost_time", "answers_while_absent")},
              f"players {c['players_selected']}/{c['players']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
