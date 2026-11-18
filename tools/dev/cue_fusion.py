# DEV diagnostic module (not product code): combine two appearance cues per test pair and report them next to each
# cue alone. Used by tools/dev/appearance_cue_eval.py; every parameter comes from config/reid_fusion.yaml.
#
# Per match ("table"): the eval's own test items with one score per cue (NaN = the cue has no score):
#   pairs[kind][same|diff]  kind = all | hard; verification pairs (same player >= MIN_GAP_S apart / different players)
#   rival qa / qb            rival test: sim(query, ref of A) and sim(query, ref of B)
# Combined score = w * norm(first cue) + (1 - w) * norm(second cue). Normalisation statistics and the fitted weight for
# match X come from the other matches only (leave-one-match-out); match X is never used to fit or normalise itself.
# Missing score: abstain (NaN: never accepted, never counted correct) or, if configured, the other cue alone.
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

KINDS = ("all", "hard")


def load_fusion_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("schema") != "spana.reid_fusion_config":
        raise SystemExit(f"{path}: not a spana.reid_fusion_config")
    if cfg["normalisation"] not in ("zscore", "rank") or cfg["on_missing"] not in ("abstain", "single_cue") or \
            cfg["weight"]["mode"] not in ("lomo", "fixed") or cfg["weight"]["fit_pairs"] not in KINDS or len(cfg["cues"]) != 2:
        raise SystemExit(f"{path}: invalid fusion config")
    return cfg


def make_table(match: str) -> dict:
    """pairs[kind][same|diff]: scores {cue: [..]}, occluded (either crop), players ("p|q" sorted; "p|p" for same);
    rival: qa / qb {cue: [..]}, occluded (query), pair ("p|q" sorted)."""
    return {"match": match, "pairs": {k: {s: {"scores": {}, "occluded": None, "players": None} for s in ("same", "diff")}
                                      for k in KINDS},
            "rival": {"qa": {}, "qb": {}, "occluded": None, "pair": None}}


# ------------------------------------------------------------------------------------------- normalisation, rule
def score_values(table: dict, cue: str) -> np.ndarray:
    """Every finite score of one cue in one match (pairs and rival sims): an unlabelled reference sample."""
    parts = [table["pairs"][k][s]["scores"][cue] for k in KINDS for s in ("same", "diff")]
    parts += [table["rival"]["qa"][cue], table["rival"]["qb"][cue]]
    x = np.concatenate([np.asarray(p, float) for p in parts]) if parts else np.zeros(0)
    return x[np.isfinite(x)]


def norm_stats(ref: np.ndarray, method: str):
    ref = np.asarray(ref, float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        raise ValueError("empty normalisation reference")
    if method == "zscore":
        sd = float(np.std(ref))
        return ("zscore", float(np.mean(ref)), sd if sd > 0 else 1.0)
    return ("rank", np.sort(ref))


def normalise(x, stats) -> np.ndarray:
    x = np.asarray(x, float)
    if stats[0] == "zscore":
        return (x - stats[1]) / stats[2]
    out = np.searchsorted(stats[1], x, side="right") / len(stats[1])           # empirical CDF of the reference
    return np.where(np.isfinite(x), out, np.nan)


def combine(z1, z2, w: float, on_missing: str) -> np.ndarray:
    z1, z2 = np.asarray(z1, float), np.asarray(z2, float)
    out = w * z1 + (1.0 - w) * z2                                               # NaN if either is missing
    if on_missing == "single_cue":
        out = np.where(np.isnan(z1) & ~np.isnan(z2), z2, out)
        out = np.where(np.isnan(z2) & ~np.isnan(z1), z1, out)
    return out


# ------------------------------------------------------------------------------------------- metrics
def tar_at_far(S, D, far: float) -> Optional[float]:
    """True accepts at the threshold that lets `far` of the different-player pairs through (np.percentile, as the
    eval's tpr_at_1pct_far_hard). Missing scores are never accepted (placed below every score)."""
    S, D = np.asarray(S, float), np.asarray(D, float)
    if len(S) == 0 or len(D) == 0:
        return None
    fin = np.concatenate([S[np.isfinite(S)], D[np.isfinite(D)]])
    if len(fin) == 0:
        return 0.0
    low = float(fin.min()) - 1.0
    S, D = np.where(np.isfinite(S), S, low), np.where(np.isfinite(D), D, low)
    thr = float(np.percentile(D, 100.0 - 100.0 * far))
    return float(np.mean(S > thr))


def rival_stats(qa, qb):
    m = np.asarray(qa, float) - np.asarray(qb, float)
    n = len(m)
    ans = np.isfinite(m)
    if n == 0:
        return {"acc": None, "acc_answered": None, "abstain": None, "n": 0}
    return {"acc": float(np.mean(np.where(ans, m > 0, False))),                 # abstain counts as not correct
            "acc_answered": float(np.mean(m[ans] > 0)) if ans.any() else None,
            "abstain": float(np.mean(~ans)), "n": n}


# ------------------------------------------------------------------------------------------- options per match
def _fused_option(tables: List[dict], i: int, cfg: dict, w: Optional[float]):
    """Combined scores for match i, normalised with statistics of the other matches; weight fixed or fitted there."""
    c1, c2 = cfg["cues"]
    others = [t for j, t in enumerate(tables) if j != i]
    st = {c: norm_stats(np.concatenate([score_values(t, c) for t in others]), cfg["normalisation"]) for c in (c1, c2)}
    fit = None
    if w is None:
        wc = cfg["weight"]
        best = None
        for g in wc["grid"]:
            S, D = [], []
            for t in others:
                p = t["pairs"][wc["fit_pairs"]]
                S.append(combine(normalise(p["same"]["scores"][c1], st[c1]), normalise(p["same"]["scores"][c2], st[c2]),
                                 g, cfg["on_missing"]))
                D.append(combine(normalise(p["diff"]["scores"][c1], st[c1]), normalise(p["diff"]["scores"][c2], st[c2]),
                                 g, cfg["on_missing"]))
            v = tar_at_far(np.concatenate(S), np.concatenate(D), wc["fit_far"])
            key = (v if v is not None else -1.0, -abs(g - wc["fixed"]), -g)
            if best is None or key > best[0]:
                best = (key, g, v)
        w, fit = best[1], {"fit_on": [t["match"] for t in others], "objective": best[2]}
    t = tables[i]
    f = lambda a, b: combine(normalise(a, st[c1]), normalise(b, st[c2]), w, cfg["on_missing"])  # noqa: E731
    opt = {"pairs": {k: {s: f(t["pairs"][k][s]["scores"][c1], t["pairs"][k][s]["scores"][c2]) for s in ("same", "diff")}
                     for k in KINDS},
           "qa": f(t["rival"]["qa"][c1], t["rival"]["qa"][c2]), "qb": f(t["rival"]["qb"][c1], t["rival"]["qb"][c2])}
    info = {"weight_first_cue": w, "normalisation": cfg["normalisation"],
            "stats_from": [o["match"] for o in others], "fit": fit}
    return opt, info


def options(tables: List[dict], cfg: dict):
    """{option: [per-match scores]} for each cue alone and the combined option(s); plus fusion info per match."""
    c1, c2 = cfg["cues"]
    name = f"{c1}+{c2}"
    out = {c: [{"pairs": {k: {s: np.asarray(t["pairs"][k][s]["scores"][c], float) for s in ("same", "diff")}
                          for k in KINDS},
                "qa": np.asarray(t["rival"]["qa"][c], float), "qb": np.asarray(t["rival"]["qb"][c], float)}
               for t in tables] for c in (c1, c2)}
    info = {}
    lomo = cfg["weight"]["mode"] == "lomo"
    out[name], info[name] = [], []
    if lomo:
        out[name + "@fixed"], info[name + "@fixed"] = [], []
    for i in range(len(tables)):
        o, inf = _fused_option(tables, i, cfg, None if lomo else cfg["weight"]["fixed"])
        out[name].append(o), info[name].append(inf)
        if lomo:
            o, inf = _fused_option(tables, i, cfg, cfg["weight"]["fixed"])
            out[name + "@fixed"].append(o), info[name + "@fixed"].append(inf)
    return out, info, name


def abstentions(tables: List[dict], cfg: dict) -> List[dict]:
    """Per match: test items where a cue has no score, by reason (what the combined option abstains on)."""
    c1, c2 = cfg["cues"]
    res = []
    for t in tables:
        items = [(t["pairs"][k][s]["scores"][c1], t["pairs"][k][s]["scores"][c2]) for k in KINDS for s in ("same", "diff")]
        items += [(t["rival"]["qa"][c1], t["rival"]["qa"][c2]), (t["rival"]["qb"][c1], t["rival"]["qb"][c2])]
        a = np.concatenate([np.asarray(x, float) for x, _ in items])
        b = np.concatenate([np.asarray(y, float) for _, y in items])
        reasons = {f"no_{c1}_score": int(np.sum(np.isnan(a) & ~np.isnan(b))),
                   f"no_{c2}_score": int(np.sum(np.isnan(b) & ~np.isnan(a))),
                   "no_score_either": int(np.sum(np.isnan(a) & np.isnan(b)))}
        res.append({"items": len(a), "reasons": reasons})
    return res


# ------------------------------------------------------------------------------------------- report with bootstrap
def _sel(ix: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    return ix if mask is None else ix[mask[ix]]


def _metrics(opt_scores: List[dict], idx: List[dict], fars, masks: Optional[List[dict]] = None) -> dict:
    """Metrics of one option on the items idx (per match index arrays), optionally restricted by per-match masks
    {"rival": bool[], (kind, same|diff): bool[]}. Several matches = pooled, with one threshold."""
    m = {}
    mk = masks or [{} for _ in idx]
    for k in KINDS:
        S = np.concatenate([o["pairs"][k]["same"][_sel(ix[k]["same"], mm.get((k, "same")))]
                            for o, ix, mm in zip(opt_scores, idx, mk)])
        D = np.concatenate([o["pairs"][k]["diff"][_sel(ix[k]["diff"], mm.get((k, "diff")))]
                            for o, ix, mm in zip(opt_scores, idx, mk)])
        for far in fars:
            m[f"tar@{far}_{k}"] = tar_at_far(S, D, far)
    qa, qb = [], []
    for o, ix, mm in zip(opt_scores, idx, mk):
        r = _sel(ix["rival"], mm.get("rival"))
        qa.append(o["qa"][r]), qb.append(o["qb"][r])
    rs = rival_stats(np.concatenate(qa), np.concatenate(qb))
    m.update({"rival_acc": rs["acc"], "rival_acc_answered": rs["acc_answered"], "rival_abstain": rs["abstain"],
              "rival_n": rs["n"]})
    return m


def _full_idx(t: dict) -> dict:
    d = {k: {s: np.arange(len(next(iter(t["pairs"][k][s]["scores"].values())))) for s in ("same", "diff")} for k in KINDS}
    d["rival"] = np.arange(len(t["rival"]["occluded"]))
    return d


def _resample(t: dict, rng) -> dict:
    full = _full_idx(t)
    d = {k: {s: rng.integers(0, len(full[k][s]), len(full[k][s])) if len(full[k][s]) else full[k][s]
             for s in ("same", "diff")} for k in KINDS}
    n = len(full["rival"])
    d["rival"] = rng.integers(0, n, n) if n else full["rival"]
    return d


def _ci(vals, level):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return None
    a = (1.0 - level) / 2.0
    return [round(float(np.quantile(v, a)), 4), round(float(np.quantile(v, 1.0 - a)), 4)]


def report(tables: List[dict], cfg: dict) -> dict:
    rc = cfg["report"]
    fars = rc["fars"]
    opts, info, name = options(tables, cfg)
    c1, c2 = cfg["cues"]
    # subsets: partly blocked (query / either crop occluded); colour-hard player pairs (first cue's rival accuracy on
    # the pair below the cutoff; pairs: different-player pairs of such a player pair, same-player pairs of its players)
    occ = [{"rival": np.asarray(t["rival"]["occluded"], bool),
            **{(k, s): np.asarray(t["pairs"][k][s]["occluded"], bool) for k in KINDS for s in ("same", "diff")}}
           for t in tables]
    colour_hard, pair_acc = [], []
    for i, t in enumerate(tables):
        pairs = np.asarray(t["rival"]["pair"], object)
        acc = {}
        for o in opts:
            m = opts[o][i]["qa"] - opts[o][i]["qb"]
            acc[o] = {p: round(float(np.mean(np.where(np.isfinite(m[pairs == p]), m[pairs == p] > 0, False))), 3)
                      for p in sorted(set(pairs.tolist()))}
        pair_acc.append(acc)
        hard = {p for p, a in acc[c1].items() if a < rc["colour_hard_pair_acc"]}
        hard_players = {q for p in hard for q in p.split("|")}
        mm = {"rival": np.array([p in hard for p in pairs.tolist()], bool)}
        for k in KINDS:
            mm[(k, "diff")] = np.array([p in hard for p in t["pairs"][k]["diff"]["players"]], bool)
            mm[(k, "same")] = np.array([p.split("|")[0] in hard_players for p in t["pairs"][k]["same"]["players"]], bool)
        colour_hard.append(mm)
    subsets = {"all": None, "query_occluded": occ, "colour_hard_pairs": colour_hard}
    groups = [(t["match"], [i]) for i, t in enumerate(tables)] + [("pooled", list(range(len(tables))))]
    rng = np.random.default_rng(rc["seed"])
    boots = [[_resample(t, rng) for t in tables] for _ in range(rc["bootstrap"])]
    out = {"options": list(opts), "combined": name, "fusion": {o: inf for o, inf in info.items()},
           "abstentions": {t["match"]: a for t, a in zip(tables, abstentions(tables, cfg))},
           "rival_pair_accuracy": {t["match"]: pa for t, pa in zip(tables, pair_acc)},
           "colour_hard_pairs": {t["match"]: sorted({p for p, h in zip(t["rival"]["pair"], ch["rival"]) if h})
                                 for t, ch in zip(tables, colour_hard)},
           "groups": {}}
    for gname, members in groups:
        g = {}
        for sname, mask in subsets.items():
            sm = None if mask is None else [mask[i] for i in members]
            pts, cis = {}, {}
            for o in opts:
                sc = [opts[o][i] for i in members]
                pts[o] = _metrics(sc, [_full_idx(tables[i]) for i in members], fars, sm)
                bs = [_metrics(sc, [b[i] for i in members], fars, sm) for b in boots]
                cis[o] = {k: _ci([b[k] for b in bs], rc["ci"]) for k in pts[o] if k != "rival_n"}
            g[sname] = {"point": pts, "ci": cis}
            if sname == "all":                                          # paired difference: combined - better single
                key = f"tar@{rc['decision_far']}_hard"
                singles = [c for c in (c1, c2) if pts[c][key] is not None]
                if singles and pts[name][key] is not None:
                    best = max(singles, key=lambda c: pts[c][key])
                    sc = {o: [opts[o][i] for i in members] for o in (name, best)}
                    diffs = []
                    for b in boots:
                        ix = [b[i] for i in members]
                        a, c = _metrics(sc[name], ix, [rc["decision_far"]])[key], _metrics(sc[best], ix, [rc["decision_far"]])[key]
                        diffs.append(None if a is None or c is None else a - c)
                    g["decision"] = {"metric": key, "better_single": best, "combined": pts[name][key],
                                     "better_single_value": pts[best][key], "combined_ci": cis[name][key],
                                     "paired_diff_ci": _ci(diffs, rc["ci"])}
        out["groups"][gname] = g
    out["verdict"] = verdict(out, cfg)
    return out


def verdict(rep: dict, cfg: dict) -> dict:
    """Promising only if the combined TAR (decision FAR, hard pairs) beats both single cues pooled AND, on no match,
    the combined option's CI upper bound lies below the better single cue's value."""
    c1, c2 = cfg["cues"]
    name = rep["combined"]
    key = f"tar@{cfg['report']['decision_far']}_hard"
    reasons = []
    pooled = rep["groups"]["pooled"]["all"]["point"]
    pv = {o: pooled[o][key] for o in (name, c1, c2)}
    beats = all(v is not None for v in pv.values()) and pv[name] > pv[c1] and pv[name] > pv[c2]
    if not beats:
        reasons.append(f"pooled {key}: {name} {pv[name]} does not beat both {c1} {pv[c1]} and {c2} {pv[c2]}")
    below = []
    for m, g in rep["groups"].items():
        if m == "pooled" or "decision" not in g:
            continue
        d = g["decision"]
        if d["combined_ci"] is not None and d["combined_ci"][1] < d["better_single_value"]:
            below.append(m)
            reasons.append(f"{m}: {name} CI upper {d['combined_ci'][1]} < {d['better_single']} {d['better_single_value']}")
    return {"metric": key, "promising": bool(beats and not below), "pooled": pv, "matches_below": below,
            "reasons": reasons}
