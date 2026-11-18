"""DEV diagnostic (not product code): how accurately do re-ID signatures re-link a player who comes back on a new
track? Decides whether an AUTOMATIC re-entry link could ever be safe (records/reid_reentry_eval_2026-11-18).

    python -m tools.dev.reid_reentry_eval --out OUT.json --window-s 10 \
        --clip LABELS_DIR TRACKS.json EMBEDDINGS.json [--clip ...]

Ground truth (frozen DEV labels, NESTED_V2 attribution as tools/target_score.py):
  track owner   the player a track is attributed to on most of its unambiguous labelled frames (purity reported)
  re-entry      for a player, consecutive attributed tracks A then B where B is born after A's last observation
                (a real gap, not a parallel track); gap = B's birth time - A's last observation time
  candidates    every track born after A's last observation and within --window-s of it, other than A
  right         the chosen candidate's owner is the player (a second track of the same player also counts)
Rule: the follower's own spana.identity.reid.reentry_choice (all candidates allowed), under four variants of the
safety rules (--config's target section otherwise): plain, position gate, mutual best match, both. Reported per
(variant, min_similarity, min_margin): events accepted (coverage), right, wrong; and the exact one-sided 95 % upper
bound on the wrong rate among accepted.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
from pathlib import Path

from scipy.stats import beta

from spana.artifacts import read_json, validate_track_embeddings, validate_tracks
from spana.config import load_config
from spana.identity.reid import ReentryContext, cosine, reentry_choice
from tools.benchmark import load_frozen
from tools.target_score import attributions

SIMS = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
VARIANTS = {"plain": (False, False), "position": (True, False), "mutual": (False, True), "both": (True, True)}
MARGINS = (0.0, 0.1, 0.2)
GAP_BINS = ((0.0, 1.0), (1.0, 3.0), (3.0, 1e9))


def upper95(k: int, n: int) -> float:
    return 1.0 if n == 0 else float(beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0


def clip_events(label_dir: str, trk_p: str, emb_p: str, window_s: float, cfg=None) -> dict:
    _, gt = load_frozen(Path(label_dir))
    tdoc, emb = read_json(trk_p), read_json(emb_p)
    validate_tracks(tdoc)
    validate_track_embeddings(emb, tdoc)
    attr = attributions(gt, tdoc)
    t_of = {fr["frame"]: fr["t"] for fr in tdoc["frames"]}
    born, last_obs = {}, {}
    for fr in tdoc["frames"]:
        for t in fr["tracks"]:
            born.setdefault(t["track_id"], fr["frame"])
            if t["source"] == "OBSERVED":
                last_obs[t["track_id"]] = fr["frame"]
    votes = collections.defaultdict(collections.Counter)
    seq = collections.defaultdict(list)                  # player -> [(frame, track)]
    for f in gt.frames:
        for tid, (p, amb) in attr[f].items():
            if p is not None and not amb:
                votes[tid][p] += 1
                seq[p].append((f, tid))
    owner = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    purity = {tid: c.most_common(1)[0][1] / sum(c.values()) for tid, c in votes.items()}
    sig = emb["tracks"]
    ctx = ReentryContext.of(tdoc)
    end = len(tdoc["frames"]) - 1
    events = []
    for p, s in seq.items():
        order = []
        for _, tid in sorted(s):
            if not order or order[-1] != tid:
                order.append(tid)
        for a, b in zip(order, order[1:]):
            if born[b] <= last_obs.get(a, -1):
                continue                                  # parallel tracks, not a re-entry
            t0 = t_of[last_obs[a]]
            gap = t_of[born[b]] - t0
            cands = [c for c in born if c != a and born[c] > last_obs[a] and t_of[born[c]] - t0 <= window_s]
            last = sig.get(a, {}).get("last")
            sims = {c: cosine(last, sig.get(c, {}).get("first")) for c in cands}
            sims = {c: v for c, v in sims.items() if v is not None}
            picks = {}
            if cfg is not None and last is not None:
                for name, (pos, mut) in VARIANTS.items():
                    for smin in SIMS:
                        for mmin in MARGINS:
                            tc = dataclasses.replace(cfg.target, reentry_max_gap_s=window_s,
                                                     reentry_min_similarity=smin, reentry_min_margin=mmin,
                                                     reentry_position_gate=pos, reentry_mutual=mut)
                            ch = reentry_choice(a, ctx.last_obs[a], end, ctx, sig, tc, cfg.tracking)
                            picks[f"{name}|{smin}|{mmin}"] = None if ch is None else ch["track"]
            events.append({"player": p, "from": a, "to": b, "gap_s": round(gap, 3), "picks": picks,
                           "in_window": gap <= window_s, "lost_has_signature": last is not None,
                           "true_has_signature": b in sims,
                           "candidates": {c: {"similarity": round(v, 4), "owner": owner.get(c),
                                              "purity": round(purity.get(c, 0.0), 3)} for c, v in sims.items()}})
    return {"events": events, "tracks_owned": len(owner),
            "impure_tracks": sum(v < 1.0 for v in purity.values())}


def decide(ev: dict, smin: float, mmin: float, variant: str = None):
    if variant is not None:
        pick = ev["picks"].get(f"{variant}|{smin}|{mmin}")
        if pick is None:
            return None
        return ev["candidates"].get(pick, {}).get("owner") == ev["player"]
    c = ev["candidates"]
    if not c:
        return None
    best = max(c, key=lambda k: (c[k]["similarity"], k))
    others = [v["similarity"] for k, v in c.items() if k != best]
    s = c[best]["similarity"]
    if s < smin - 1e-9 or (others and s - max(others) < mmin - 1e-9):
        return None
    return c[best]["owner"] == ev["player"]


def table(events, label, variant=None):
    rows = []
    usable = [e for e in events if e["in_window"] and e["lost_has_signature"]]
    for smin in SIMS:
        for mmin in MARGINS:
            d = [decide(e, smin, mmin, variant) for e in usable]
            acc = [x for x in d if x is not None]
            wrong = sum(1 for x in acc if not x)
            rows.append({"set": label, "variant": variant or "offline", "min_similarity": smin, "min_margin": mmin, "events": len(usable),
                         "accepted": len(acc), "right": len(acc) - wrong, "wrong": wrong,
                         "coverage": round(len(acc) / max(len(usable), 1), 3),
                         "wrong_rate_upper95": round(upper95(wrong, len(acc)), 3)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, required=True)
    ap.add_argument("--config", required=True, help="spana config (tracking speed bound, target section)")
    ap.add_argument("--clip", nargs=3, action="append", required=True, metavar=("LABELS", "TRACKS", "EMBEDDINGS"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    per_clip = {Path(c[0]).parent.name: clip_events(*c, a.window_s, cfg) for c in a.clip}
    events = [dict(e, clip=k) for k, v in per_clip.items() for e in v["events"]]
    usable = [e for e in events if e["in_window"] and e["lost_has_signature"]]
    rank1 = [e for e in usable if e["candidates"]]
    top = sum(1 for e in rank1 if decide(e, -1.0, 0.0))
    summary = {"events": len(events), "in_window": sum(e["in_window"] for e in events),
               "usable (lost track has a signature)": len(usable),
               "true track has a signature": sum(e["true_has_signature"] for e in usable),
               "rank1_right": top, "rank1_of": len(rank1),
               "impure_tracks": {k: v["impure_tracks"] for k, v in per_clip.items()}}
    rows = table(events, "all")
    for v in VARIANTS:
        rows += table(events, "all", v)
        for lo, hi in GAP_BINS:
            rows += table([e for e in events if lo <= e["gap_s"] < hi], f"gap {lo:g}-{hi:g} s", v)
    Path(a.out).write_text(json.dumps({"summary": summary, "table": rows, "events": events}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["set"] == "all" and r["variant"] != "offline" and r["min_margin"] in (0.0, 0.1) and \
                r["min_similarity"] in (0.5, 0.6, 0.7, 0.75, 0.8):
            print({k: r[k] for k in ("variant", "min_similarity", "min_margin", "accepted", "right", "wrong",
                                     "coverage", "wrong_rate_upper95")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
