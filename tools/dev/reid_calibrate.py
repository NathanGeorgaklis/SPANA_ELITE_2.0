"""DEV calibration of the identity-resolution margin (not product code).

    python -m tools.dev.reid_calibrate --tables TABLES.json --out OUT.json [--config config/reid_calibration.yaml]
                                       [--fusion-config config/reid_fusion.yaml]

TABLES.json = tools/dev/appearance_cue_eval.py --dump-tables (per-test scores of every cue on the DEV clips).
Rival test = a query crop of player A, a reference of A and a reference of B (the IDENTITY_UNCERTAIN situation).
margin = score(query, A) - score(query, B); with a margin m the query is resolved to A if margin >= m, to B if
margin <= -m, else UNKNOWN. A resolution to B is wrong.
m = scanning margins from the largest down: the lowest margin reached while the exact one-sided upper bound of
wrong / resolved stays <= target_false_rate, from the first margin that meets it (fail closed: the bound, not the
observed rate).
Options: colour alone, re-ID alone, combined (per-cue z-normalisation with statistics of the calibration clips,
colour weight from the grid giving the most resolutions on the calibration clips).
Honesty check: leave one clip out - statistics, weight and m fitted on the other clips, applied to the held-out
clip; held-out results pooled. Final values: fitted on all clips (proposal for the integration step).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from tools.dev import cue_fusion as F
from tools.reid_cluster import binom_cdf, upper_bound


def load_cfg(path) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    if c.get("schema") != "spana.reid_calibration_config":
        raise SystemExit(f"{path}: not a spana.reid_calibration_config")
    return c


def to_arrays(table: dict) -> dict:
    """JSON (None = no score) -> numpy (NaN), same structure as tools/dev/cue_fusion tables."""
    def arr(x):
        return np.array([np.nan if v is None else v for v in x], float)
    t = {"match": table["match"], "pairs": {}, "rival": {}}
    for k, kd in table["pairs"].items():
        t["pairs"][k] = {s: {"scores": {c: arr(v) for c, v in sd["scores"].items()}} for s, sd in kd.items()}
    r = table["rival"]
    t["rival"] = {"qa": {c: arr(v) for c, v in r["qa"].items()}, "qb": {c: arr(v) for c, v in r["qb"].items()},
                  "height": arr(r.get("height", [])), "pair": r.get("pair", [])}
    return t


def margins(tables: List[dict], option: str, fit: dict) -> np.ndarray:
    """Rival margins of one option for the given tables, with fitted statistics / weight for the combined option."""
    out = []
    for t in tables:
        qa, qb = t["rival"]["qa"], t["rival"]["qb"]
        if option != "combined":
            out.append(qa[option] - qb[option])
            continue
        c1, c2 = fit["cues"]
        z = lambda c, x: F.normalise(x, fit["stats"][c])                     # noqa: E731
        w = fit["weight"]
        out.append(w * (z(c1, qa[c1]) - z(c1, qb[c1])) + (1 - w) * (z(c2, qa[c2]) - z(c2, qb[c2])))
    return np.concatenate(out) if out else np.zeros(0)


def meets(k: int, n: int, target: float, confidence: float) -> bool:
    """upper_bound(k, n, confidence) <= target, without the bisection: the exact one-sided upper bound is the p at
    which P(X <= k) = 1 - confidence, and P(X <= k) falls as p grows, so bound <= target <=> P(X <= k; target) <= 1 - c."""
    if n <= 0 or k >= n * target:                                          # the bound is >= the observed rate
        return False
    return binom_cdf(k, n, target) <= 1 - confidence


def choose_margin(m: np.ndarray, target: float, confidence: float) -> Optional[float]:
    """Scan the distinct positive |margin| values from the largest down. At the top only a few tests resolve, so the
    bound cannot yet be met (0 wrong needs about 3 / target resolutions); once it is met, keep lowering the margin
    while it stays met and stop at the first failure. Returns the last margin that met it (every margin between it
    and the first one that met it does too), or None if it is never met."""
    m = m[np.isfinite(m)]
    a = np.abs(m)
    keep = a > 0                                                         # a tie (margin 0) is never resolved
    a, wrong = a[keep], (m[keep] < 0)
    order = np.argsort(-a, kind="stable")
    a, wrong = a[order], wrong[order]
    cum_k = np.cumsum(wrong)
    best = None
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[j + 1] == a[i]:                       # all ties of this value resolve together
            j += 1
        n, k = j + 1, int(cum_k[j])
        if meets(k, n, target, confidence):
            best = float(a[i])
        elif best is not None:
            break
        i = j + 1
    return best


def apply(m: np.ndarray, t: Optional[float], confidence: float) -> dict:
    fin = np.isfinite(m)
    n_all = int(len(m))
    if t is None:
        return {"tests": n_all, "resolved": 0, "wrong": 0, "coverage": 0.0, "wrong_rate": None, "upper_bound": None}
    res = fin & (np.abs(m) >= t)
    k = int((fin & (m <= -t)).sum())
    n = int(res.sum())
    return {"tests": n_all, "resolved": n, "wrong": k, "coverage": round(n / n_all, 4) if n_all else None,
            "wrong_rate": round(k / n, 5) if n else None, "upper_bound": upper_bound(k, n, confidence) if n else None}


def fit(tables: List[dict], option: str, cfg: dict, fcfg: dict) -> dict:
    target, conf = cfg["target_false_rate"], cfg["confidence"]
    if option != "combined":
        mm = margins(tables, option, {})
        t = choose_margin(mm, target, conf)
        return {"option": option, "margin": t, "in_sample": apply(mm, t, conf)}
    c1, c2 = cfg["cues"]
    stats = {c: F.norm_stats(np.concatenate([F.score_values(_as_fusion(x), c) for x in tables]), fcfg["normalisation"])
             for c in (c1, c2)}
    best = None
    for w in cfg["weight_grid"]:
        f = {"cues": [c1, c2], "stats": stats, "weight": w}
        mm = margins(tables, "combined", f)
        t = choose_margin(mm, target, conf)
        a = apply(mm, t, conf)
        key = (a["resolved"], -abs(w - cfg["weight_tie_break"]), -w)
        if best is None or key > best[0]:
            best = (key, w, t, a)
    _, w, t, a = best
    return {"option": "combined", "weight": w, "margin": t, "in_sample": a, "cues": [c1, c2],
            "stats": {c: _stats_json(s) for c, s in stats.items()}, "_stats": stats}


def _as_fusion(t: dict) -> dict:
    """The fusion module's table layout (pairs + rival qa / qb) for its score_values()."""
    return {"pairs": {k: {s: {"scores": sd["scores"]} for s, sd in kd.items()} for k, kd in t["pairs"].items()},
            "rival": {"qa": t["rival"]["qa"], "qb": t["rival"]["qb"]}}


def _stats_json(s):
    return {"method": s[0], "mean": s[1], "sd": s[2]} if s[0] == "zscore" else {"method": s[0], "n": len(s[1])}


def held_out(tables: List[dict], option: str, cfg: dict, fcfg: dict) -> dict:
    """Leave one clip out: fit on the others, apply to it; per clip and pooled."""
    per, pooled_m, pooled_t = {}, [], []
    tot = {"tests": 0, "resolved": 0, "wrong": 0}
    for i, t in enumerate(tables):
        others = [x for j, x in enumerate(tables) if j != i]
        f = fit(others, option, cfg, fcfg)
        fx = {"cues": f.get("cues"), "stats": f.get("_stats"), "weight": f.get("weight")}
        mm = margins([t], option, fx)
        a = apply(mm, f["margin"], cfg["confidence"])
        per[t["match"]] = {"fit_on": [x["match"] for x in others], "margin": f["margin"], "weight": f.get("weight"), **a}
        for k in tot:
            tot[k] += a[k]
    ub = upper_bound(tot["wrong"], tot["resolved"], cfg["confidence"]) if tot["resolved"] else None
    return {"per_clip": per, "pooled": {**tot, "coverage": round(tot["resolved"] / tot["tests"], 4) if tot["tests"] else None,
                                        "wrong_rate": round(tot["wrong"] / tot["resolved"], 5) if tot["resolved"] else None,
                                        "upper_bound": ub}}


def by_height(tables, option, f, cfg):
    mm = margins(tables, option, {"cues": f.get("cues"), "stats": f.get("_stats"), "weight": f.get("weight")})
    h = np.concatenate([t["rival"]["height"] for t in tables])
    bins = cfg["height_bins_px"]
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (h >= lo) & (h < hi)
        out[f"{lo}-{hi}"] = apply(mm[sel], f["margin"], cfg["confidence"]) if sel.any() else None
    return out


def calibrate(tables_doc: dict, cfg: dict, fcfg: dict) -> dict:
    tables = [to_arrays(t) for t in tables_doc["tables"]]
    out = {"target_false_rate": cfg["target_false_rate"], "confidence": cfg["confidence"], "options": {}}
    for option in (*cfg["cues"], "combined"):
        final = fit(tables, option, cfg, fcfg)
        res = {"held_out": held_out(tables, option, cfg, fcfg),
               "final": {k: v for k, v in final.items() if not k.startswith("_")},
               "final_by_query_height": by_height(tables, option, final, cfg)}
        out["options"][option] = res
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tables", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config/reid_calibration.yaml")
    p.add_argument("--fusion-config", default="config/reid_fusion.yaml")
    a = p.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    cfg, fcfg = load_cfg(a.config), F.load_fusion_config(a.fusion_config)
    raw = Path(a.tables).read_bytes()
    res = calibrate(json.loads(raw), cfg, fcfg)
    res["inputs"] = {"tables_sha256": hashlib.sha256(raw).hexdigest(),
                     "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                     "fusion_config_sha256": hashlib.sha256(Path(a.fusion_config).read_bytes()).hexdigest()}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    for o, r in res["options"].items():
        print(json.dumps({"option": o, "held_out_pooled": r["held_out"]["pooled"],
                          "final": {k: v for k, v in r["final"].items() if k != "stats"}}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
