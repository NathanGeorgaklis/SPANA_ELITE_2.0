"""DEV tracker bench (not product code): side-by-side summary of trackers scored on the same clips and detections.

    python -m tools.dev.tracker_bench summary --out SUMMARY.json \
        --tracker NAME TARGET_DEFAULT.json TARGET_REVIEWER.json BENCH.json [BENCH.json ...] [--tracker ...]

Tracking from tools/benchmark.py summaries (NESTED_V2): fragments, ID switches, identity transfers split into flagged
(the tracker marked the identity uncertain) and unflagged (silent). Following from tools/target_score.py, without and
with the simulated after-match reviewer, pooled over clips as player-frame counts: correct / wrong / no answer
(abstain + lost) shares of the scored frames (CANDIDATE frames are scored apart, never correct), and review actions
per followed player-minute. Same pooling as records/bytetrack_compare_2026-11-18.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

TRACK_KEYS = ("fragments_total", "id_switches", "identity_transfers", "unflagged_transfers", "track_coverage")
CAUSE_KEYS = ("recovery_caused_transfers", "recovery_caused_silent", "link_caused_transfers", "link_caused_silent",
              "transfers_on_linked_tracks")      # 2026-11-20 transfer rule; absent (None) in older reports
COUNT_KEYS = ("correct", "wrong", "unverified", "abstain", "lost", "candidate_correct", "candidate_wrong",
              "candidate_unverified")


def tracking(benches: List[dict]) -> dict:
    clips, pooled = {}, {k: 0 for k in TRACK_KEYS if k != "track_coverage"}
    for b in benches:
        s = b["summary"]
        name = Path(b["labels"]["dir"]).parent.name.split("_G")[0]
        row = {k: s[k] for k in TRACK_KEYS}
        row.update({k: s.get(k) for k in CAUSE_KEYS})
        row["flagged_transfers"] = s["identity_transfers"] - s["unflagged_transfers"]
        clips[name] = row
        for k in TRACK_KEYS:
            if k != "track_coverage":
                pooled[k] += s[k]
        for k in CAUSE_KEYS:
            pooled[k] = None if s.get(k) is None or pooled.get(k, 0) is None else pooled.get(k, 0) + s[k]
    pooled["flagged_transfers"] = pooled["identity_transfers"] - pooled["unflagged_transfers"]
    return {"clips": clips, "pooled": pooled}


def following(target: dict) -> dict:
    clips, pooled = {}, {k: 0 for k in COUNT_KEYS}
    pooled.update(actions=0, followed_s=0.0)
    for c in target["clips"]:
        p = c["pooled"]
        row = {k: p[k] for k in COUNT_KEYS}
        row["actions"] = sum(p["actions"].values())
        row["followed_s"] = p["followed_s"]
        clips[c["clip"].split("_G")[0]] = row
        for k in pooled:
            pooled[k] += row[k]
    n = sum(pooled[k] for k in COUNT_KEYS)
    pooled.update(scored=n, correct_share=round(pooled["correct"] / n, 4), wrong_share=round(pooled["wrong"] / n, 4),
                  no_answer_share=round((pooled["abstain"] + pooled["lost"]) / n, 4),
                  actions_per_followed_min=round(pooled["actions"] / (pooled["followed_s"] / 60.0), 2),
                  followed_min=round(pooled["followed_s"] / 60.0, 2))
    for row in clips.values():
        m = sum(row[k] for k in COUNT_KEYS)
        row["wrong_share"] = round(row["wrong"] / m, 4) if m else None
    return {"simulate_reviewer": bool(target.get("simulate_reviewer")), "clips": clips, "pooled": pooled}


def summary(trackers: Dict[str, dict]) -> dict:
    out = {}
    for name, t in trackers.items():
        out[name] = {"tracking": tracking([json.loads(Path(b).read_text()) for b in t["benches"]]),
                     "following_default": following(json.loads(Path(t["default"]).read_text())),
                     "following_reviewer": following(json.loads(Path(t["reviewer"]).read_text()))}
        if out[name]["following_default"]["simulate_reviewer"] or not out[name]["following_reviewer"]["simulate_reviewer"]:
            raise SystemExit(f"{name}: default / reviewer target reports swapped")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--tracker", nargs="+", action="append", required=True,
                   metavar="NAME TARGET_DEFAULT TARGET_REVIEWER BENCH")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    trk = {}
    for t in a.tracker:
        if len(t) < 4:
            raise SystemExit("--tracker NAME TARGET_DEFAULT TARGET_REVIEWER BENCH [BENCH ...]")
        trk[t[0]] = {"default": t[1], "reviewer": t[2], "benches": t[3:]}
    doc = summary(trk)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    for name, d in doc.items():
        tp, fd, fr = d["tracking"]["pooled"], d["following_default"]["pooled"], d["following_reviewer"]["pooled"]
        print(f"{name:10s} frag {tp['fragments_total']:4d} idsw {tp['id_switches']:4d} transfers {tp['identity_transfers']:3d}"
              f" (flagged {tp['flagged_transfers']}, silent {tp['unflagged_transfers']}) | default correct"
              f" {fd['correct_share']:.3f} wrong {fd['wrong']} ({fd['wrong_share']:.2%}) | reviewer correct"
              f" {fr['correct_share']:.3f} wrong {fr['wrong']} ({fr['wrong_share']:.2%}) actions/min"
              f" {fr['actions_per_followed_min']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
