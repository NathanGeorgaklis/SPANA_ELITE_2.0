"""Score a detection + tracking run against FROZEN human DEV labels (data/labels/<clip>/v<N>/).

    python -m tools.benchmark --labels data/labels/<clip>/v1 --detections DET.json --tracks TRK.json \
        --decisions TRK.decisions.jsonl --out REPORT.json

Only labelled (done) frames are scored. Detections and track boxes lying inside a human ignore region
(IoA >= IGNORE_IOA) are removed before scoring. Refuses non-DEV labels and labels whose hash no longer matches
FROZEN.json, and refuses runs made from a different video. Uses spana.evaluation (same scorer as synthetic QA).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

from spana.artifacts import read_json, validate_detections, validate_tracks, write_json_atomic
from spana.evaluation import NESTED_V1, RULES, EvalFrame, TrackBox, evaluate
from spana.labels import parse_mot

IGNORE_IOA = 0.5        # scoring convention (not a tracker setting)
DEFAULT_RULE = NESTED_V1.name   # frozen 2026-11-08, records/MATCHING_RULE_NESTED_V1.md


def _ioa(inner, outer) -> float:
    iw = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    ih = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return iw * ih / max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)


def load_frozen(label_dir: Path):
    frozen = json.loads((label_dir / "FROZEN.json").read_text())
    if frozen["split"] != "DEV":
        raise SystemExit("refusing: labels are not DEV")
    gt_bytes = (label_dir / "gt.txt").read_bytes()
    if hashlib.sha256(gt_bytes).hexdigest() != frozen["gt_sha256"]:
        raise SystemExit("refusing: gt.txt no longer matches its frozen hash")
    record = json.loads((label_dir / "labels_frozen.json").read_text())
    gt = parse_mot(gt_bytes.decode(), frozen["clip"]["width"], frozen["clip"]["height"], record["frames_labelled"])
    return frozen, gt


def build_frames(gt, dets_doc: dict, tracks_doc: dict, decisions: List[dict]) -> List[EvalFrame]:
    dets = {fr["frame"]: [r[:4] for r in fr["detections"]] for fr in dets_doc["frames"]}
    trk = {fr["frame"]: fr for fr in tracks_doc["frames"]}
    rec: Dict[int, list] = {}
    for d in decisions:
        if d.get("kind") == "recovery" and d["frame"] in gt.players:
            t = next(t for t in trk[d["frame"]]["tracks"] if t["track_id"] == d["track_id"])
            rec.setdefault(d["frame"], []).append((d["track_id"], t["detection_index"]))
    out = []
    for f in gt.frames:
        if f not in trk:
            raise SystemExit(f"labelled frame {f} was not processed by the run")
        ign = gt.ignore.get(f, [])
        keep = lambda b: not any(_ioa(b, g) >= IGNORE_IOA for g in ign)      # noqa: E731
        d_idx = [k for k, b in enumerate(dets.get(f, [])) if keep(b)]
        remap = {k: n for n, k in enumerate(d_idx)}
        boxes = [dets[f][k] for k in d_idx]
        obs = [TrackBox(t["track_id"], t["box"], t["confirmed"],
                        bool(t["possible_predecessors"]) or t["recoveries"] > 0, remap.get(t["detection_index"]))
               for t in trk[f]["tracks"] if t["source"] == "OBSERVED" and keep(t["box"])]
        recs = [(tid, remap[d]) for tid, d in rec.get(f, []) if d in remap]
        out.append(EvalFrame(f, gt.players.get(f, {}), boxes, obs, recs, frozenset(gt.occluded.get(f, ()))))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    for k in ("labels", "detections", "tracks", "decisions", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--rule", default=DEFAULT_RULE, choices=sorted(RULES))
    a = p.parse_args()
    frozen, gt = load_frozen(Path(a.labels))
    dd, td = read_json(a.detections), read_json(a.tracks)
    validate_detections(dd)
    validate_tracks(td, dd)
    if dd["video"]["sha256"] != frozen["clip"]["sha256"]:
        raise SystemExit("refusing: the run was made from a different video than the labels")
    decisions = [json.loads(l) for l in Path(a.decisions).read_text().splitlines() if l.strip()]
    frames = build_frames(gt, dd, td, decisions)
    r = evaluate(frames, rule=RULES[a.rule])
    occluded = sum(len(gt.occluded.get(ef.frame, ())) for ef in frames)
    miss_occ = sum(1 for e in r.missed_events if e["player"] in gt.occluded.get(e["frame"], ()))
    report = {"match_rule": a.rule, "labels": {"dir": a.labels, "gt_sha256": frozen["gt_sha256"], "frames": len(frames),
                         "frame_step": frozen["frame_step"], "annotator": frozen["annotator"]},
              "run": {"detections_commit": dd["provenance"]["git"], "tracks_commit": td["provenance"]["git"]},
              "summary": r.summary(),
              "recall_visible": round(1 - (r.missed_players - miss_occ) / max(1, r.gt_boxes - occluded), 4),
              "recall_occluded": round(1 - miss_occ / max(1, occluded), 4),
              "gt_occluded_boxes": occluded, "missed_occluded": miss_occ,
              "fragments_by_player": r.fragments,
              "events": {"transfers": r.transfer_events, "wrong_recoveries": r.wrong_recovery_events,
                         "missed": r.missed_events, "duplicates": r.duplicate_events,
                         "ambiguous_attributions": r.ambiguous_events}}
    write_json_atomic(a.out, report)
    print(json.dumps({k: report[k] for k in ("match_rule", "summary", "recall_visible", "recall_occluded")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
