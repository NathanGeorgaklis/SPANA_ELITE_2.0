"""DEV diagnostic (not product code): fit the re-entry position score (target.reentry_position_*) on labelled DEV
re-entries and compare it with appearance only (records/reentry_position_score_2026-11-18).

    python -m tools.dev.reentry_position_fit --config config/default.yaml --out OUT.json \
        --clip LABELS_DIR TRACKS.json EMBEDDINGS.json [--clip ...]

Events and ground truth exactly as tools/dev/reid_reentry_eval.py (consecutive attributed tracks of a labelled player
with a real gap; right = the chosen track's owner is the player). The decision is spana.identity.reid.reentry_choice
with every candidate allowed, evaluated at:
  lens "rank1"    no floor, no margin, no mutual check (the 14 wrong picks of records/reentry_position_mutual_...)
  lens "shipped"  the config's floor / margin / mutual rule (what the follower offers)
for every grid point of (weight, speed_h_per_s, time_scale_s, max_gap_s); weight 0 is appearance only.
Per point: right / wrong / missed (no offer), split by gap (short < 1 s / long) and by whether position could speak
for the true track (no pan / pan: PAN, OUT_OF_PICTURE, LEFT_FRAME or CAMERA_BREAK).
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

from spana.artifacts import read_json
from spana.config import load_config
from spana.identity.reid import ReentryContext, position_fit, reentry_choice
from tools.dev.reid_reentry_eval import clip_events

WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
SPEEDS = (1.0, 2.0, 5.0)
TIME_SCALES = (0.5, 1.0, 2.0)
MAX_GAPS = (1.0, 3.0)
PAN_REASONS = {"PAN", "OUT_OF_PICTURE", "LEFT_FRAME", "CAMERA_BREAK"}


def load(c, window_s):
    ev = clip_events(c[0], c[1], c[2], window_s)["events"]
    tdoc, emb = read_json(c[1]), read_json(c[2])
    return ev, ReentryContext.of(tdoc), emb["tracks"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", nargs=3, action="append", required=True, metavar=("LABELS", "TRACKS", "EMBEDDINGS"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    base = cfg.target
    clips = [(Path(c[0]).parent.name,) + load(c, base.reentry_max_gap_s) for c in a.clip]
    events = []                                           # (clip, event, ctx, sig)
    for name, ev, ctx, sig in clips:
        end = len(ctx.frames) - 1
        for e in ev:
            if e["in_window"] and e["lost_has_signature"]:
                events.append((name, e, ctx, sig, end))
    # event classes (independent of the grid): gap length; can position speak for the true track?
    probe = dataclasses.replace(base, reentry_position_weight=1.0, reentry_position_max_gap_s=1e9)
    for name, e, ctx, sig, end in events:
        _, why = position_fit(ctx, e["from"], ctx.last_obs[e["from"]], e["to"], probe)
        e["pan"] = why in PAN_REASONS
        e["short"] = e["gap_s"] < 1.0
    lenses = {"rank1": dict(reentry_min_similarity=-1.0, reentry_min_margin=0.0, reentry_mutual=False),
              "shipped": {}}
    rows, picks = [], {}
    for w, sp, ts, mg in itertools.product(WEIGHTS, SPEEDS, TIME_SCALES, MAX_GAPS):
        if w == 0.0 and (sp, ts, mg) != (SPEEDS[0], TIME_SCALES[0], MAX_GAPS[0]):
            continue                                      # weight 0: the scales do not matter
        for lens, over in lenses.items():
            tc = dataclasses.replace(base, reentry_position_weight=w, reentry_position_speed_h_per_s=sp,
                                     reentry_position_time_scale_s=ts, reentry_position_max_gap_s=mg, **over)
            out = {}
            for name, e, ctx, sig, end in events:
                ch = reentry_choice(e["from"], ctx.last_obs[e["from"]], end, ctx, sig, tc, cfg.tracking)
                pick = ch["track"] if ch["offer"] else None
                verdict = "missed" if pick is None else \
                    ("right" if e["candidates"].get(pick, {}).get("owner") == e["player"] else "wrong")
                out[(name, e["player"], e["from"], e["to"])] = (verdict, pick)
            key = (w, sp, ts, mg, lens)
            picks[key] = out
            row = {"weight": w, "speed_h_per_s": sp, "time_scale_s": ts, "max_gap_s": mg, "lens": lens}
            for split, sel in (("all", lambda e: True), ("short", lambda e: e["short"]),
                               ("long", lambda e: not e["short"]), ("no_pan", lambda e: not e["pan"]),
                               ("pan", lambda e: e["pan"])):
                vs = [out[(n, e["player"], e["from"], e["to"])][0] for n, e, *_ in events if sel(e)]
                row[split] = {v: vs.count(v) for v in ("right", "wrong", "missed")}
            rows.append(row)
    base_key = lambda lens: (0.0, SPEEDS[0], TIME_SCALES[0], MAX_GAPS[0], lens)      # noqa: E731
    Path(a.out).write_text(json.dumps({
        "events": len(events), "short": sum(e["short"] for _, e, *_ in events),
        "pan": sum(e["pan"] for _, e, *_ in events), "rows": rows,
        "picks": {"|".join(map(str, k)): {"|".join(map(str, ek)): v for ek, v in p.items()} for k, p in picks.items()},
    }, indent=1))
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
