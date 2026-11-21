"""Leave-one-TRAIN-match-out evaluation and margin calibration of a SPANA re-ID model (v4 protocol, DECISIONS 2026-11-20).

    python -m tools.reid_heldout eval      --model MODEL.onnx --crops CROPS_DIR --out FOLD.json [--config config/reid_train.yaml]
    python -m tools.reid_heldout calibrate --folds FOLD.json [FOLD.json ...] --out CAL.json [--config config/reid_train.yaml]

eval: every crop of the held-out match (both crop splits: the whole match was kept out of training), embedded by the
ONNX model (the tools/reid_train.py input convention). Splits by match, never by frame.
  - rank-1 / mAP: each crop a query; gallery = every other crop of the match at least `heldout.min_gap_s` away in time.
  - same player vs teammate AUC: pairs of one player >= min_gap_s apart vs pairs of two players of the same side
    (same kit; side from the TRAIN labels), each sampled up to `heldout.max_pairs` (seed 0). Also vs the other side.
  - teammate rival tests (the DEV rival test with B = every same-side teammate): query of A, the latest crop of A and
    of B at least min_gap_s before it; margin = cos(q, A) - cos(q, B). Clean = all three crops overlap no other
    labelled box above appearance.clean_max_iou of config/default.yaml (TRAIN crops carry no occlusion flag).
calibrate: the margin rule of tools/dev/reid_calibrate.py (target_false_rate, confidence of
config/reid_calibration.yaml) on the pooled out-of-fold teammate rival tests: clean margin floored by the all-tests
margin (clean_subset.margin_floor). Plus leave-one-fold-out: fit on the other folds, apply to the left-out one.
The margin is a TRAIN-fold margin; applied to a model trained on all matches it is "borrowed" (never measured on it).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from tools.dev.reid_calibrate import apply, choose_margin
from tools.reid_crops import load_train_config
from tools.reid_train import load_crops, sha256_file, to_input
from tools.reid_cluster import upper_bound


def _settings(config: str) -> dict:
    cfg = load_train_config(config)
    h = cfg.get("heldout")
    if not h:
        raise SystemExit(f"{config}: no heldout section (schema 1.1.0)")
    app = yaml.safe_load(Path(h["clean_max_iou_from"]).read_text())["appearance"]
    cal = yaml.safe_load(Path(h["calibration_from"]).read_text())
    if cal.get("schema") != "spana.reid_calibration_config":
        raise SystemExit(f"{h['calibration_from']}: not a spana.reid_calibration_config")
    return {"min_gap_s": float(h["min_gap_s"]), "max_pairs": int(h["max_pairs"]),
            "clean_max_iou": float(app["clean_max_iou"]), "target_false_rate": float(cal["target_false_rate"]),
            "confidence": float(cal["confidence"]),
            "margin_floor": (cal.get("clean_subset") or {}).get("margin_floor", "none")}


def embed(model: str | Path, rows: List[dict], bs: int = 64) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out = []
    for k in range(0, len(rows), bs):
        x = to_input([cv2.imread(r["path"]) for r in rows[k:k + bs]])
        out.append(sess.run(None, {"pixel_values": x})[0])
    z = np.concatenate(out).astype(np.float32)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def auc(pos: np.ndarray, neg: np.ndarray) -> Optional[float]:
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.concatenate([pos, neg]).argsort(kind="stable").argsort() + 1.0
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def retrieval(z: np.ndarray, ident: np.ndarray, t: np.ndarray, gap: float, chunk: int = 512) -> dict:
    """rank-1 and mAP; gallery = all crops >= gap away in time from the query; queries without a positive skipped."""
    hits, aps = [], []
    for s in range(0, len(z), chunk):
        sim = z[s:s + chunk] @ z.T
        for r in range(sim.shape[0]):
            q = s + r
            ok = np.abs(t - t[q]) >= gap
            ok[q] = False
            pos = ok & (ident == ident[q])
            if not pos.any():
                continue
            sc, p = sim[r][ok], pos[ok]
            o = np.argsort(-sc, kind="stable")
            p = p[o]
            hits.append(bool(p[0]))
            cum = np.cumsum(p)
            aps.append(float((cum[p] / (np.flatnonzero(p) + 1)).mean()))
    return {"queries": len(hits), "rank1": round(float(np.mean(hits)), 4) if hits else None,
            "mAP": round(float(np.mean(aps)), 4) if aps else None}


def pair_aucs(z, ident, side, t, gap: float, cap: int, seed: int = 0) -> dict:
    """Same player (>= gap apart) vs teammate (same side, other player) and vs the other side; pairs sampled."""
    rng = np.random.default_rng(seed)
    n = len(z)
    want = {"same": [], "mate": [], "other": []}
    draws = 0
    while min(len(v) for v in want.values()) < cap and draws < 200 * cap:
        i, j = rng.integers(0, n, 100_000), rng.integers(0, n, 100_000)
        draws += 100_000
        keep = i != j
        i, j = i[keep], j[keep]
        si, ss = ident[i] == ident[j], side[i] == side[j]
        cls = {"same": si & (np.abs(t[i] - t[j]) >= gap), "mate": ~si & ss, "other": ~ss}
        for k, m in cls.items():
            room = cap - len(want[k])
            if room > 0:
                sel = np.flatnonzero(m)[:room]
                want[k].extend(np.einsum("ij,ij->i", z[i[sel]], z[j[sel]]).tolist())
    v = {k: np.asarray(x) for k, x in want.items()}
    q = lambda x: {"n": int(len(x)), **({f"p{p:02d}": round(float(np.percentile(x, p)), 3)  # noqa: E731
                                         for p in (5, 25, 50, 75, 95)} if len(x) else {})}
    return {"auc_same_vs_teammate": _r(auc(v["same"], v["mate"])),
            "auc_same_vs_other_side": _r(auc(v["same"], v["other"])),
            "same_player": q(v["same"]), "teammate": q(v["mate"]), "other_side": q(v["other"])}


def _r(x):
    return None if x is None else round(x, 4)


def rival_tests(z, rows: List[dict], gap: float, clean_iou: float) -> dict:
    """Teammate rival tests: query of A vs the latest crop of A and of every same-side B >= gap before the query."""
    by_id: Dict[str, List[int]] = {}
    for k in sorted(range(len(rows)), key=lambda k: rows[k]["t"]):
        by_id.setdefault(rows[k]["identity"], []).append(k)
    times = {i: [rows[k]["t"] for k in ks] for i, ks in by_id.items()}
    side_of = {i: rows[ks[0]]["team"] for i, ks in by_id.items()}

    def latest(i, tq):
        n = bisect.bisect_right(times[i], tq - gap)
        return by_id[i][n - 1] if n else None

    margin, clean, pair = [], [], []
    for q, rq in enumerate(rows):
        a = latest(rq["identity"], rq["t"])
        if a is None:
            continue
        for b_id in sorted(by_id):
            if b_id == rq["identity"] or side_of[b_id] != rq["team"]:
                continue
            b = latest(b_id, rq["t"])
            if b is None:
                continue
            margin.append(float(z[q] @ z[a] - z[q] @ z[b]))
            clean.append(max(rows[q]["overlap"], rows[a]["overlap"], rows[b]["overlap"]) <= clean_iou)
            pair.append("|".join(sorted((rq["player"], rows[b]["player"]))))
    return {"margin": margin, "clean": clean, "pair": pair}


def evaluate(a) -> int:
    s = _settings(a.config)
    rows = load_crops([a.crops])
    meta = json.loads((Path(a.crops) / "crops.json").read_text())
    z = embed(a.model, rows)
    ident = np.array([r["identity"] for r in rows])
    side = np.array([r["team"] for r in rows])
    t = np.array([r["t"] for r in rows], float)
    rv = rival_tests(z, rows, s["min_gap_s"], s["clean_max_iou"])
    m, c = np.asarray(rv["margin"]), np.asarray(rv["clean"], bool)
    out = {"schema": "spana.reid_heldout_fold", "schema_version": "1.0.0", "match_id": meta["match_id"],
           "model": str(a.model), "model_sha256": sha256_file(Path(a.model)),
           "crops_index_sha256": meta["index_sha256"], "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
           "settings": s, "crops": len(rows), "identities": int(len(set(ident))),
           "retrieval": retrieval(z, ident, t, s["min_gap_s"]),
           **pair_aucs(z, ident, side, t, s["min_gap_s"], s["max_pairs"]),
           "rival": {"tests": int(len(m)), "clean_tests": int(c.sum()),
                     "accuracy": _r(float((m > 0).mean())) if len(m) else None,
                     "clean_accuracy": _r(float((m[c] > 0).mean())) if c.any() else None},
           "rival_tests": rv}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "rival_tests"}))
    return 0


def _fit(ms: List[np.ndarray], cs: List[np.ndarray], s: dict) -> dict:
    m, c = np.concatenate(ms), np.concatenate(cs)
    all_m = choose_margin(m, s["target_false_rate"], s["confidence"])
    clean_m = choose_margin(m[c], s["target_false_rate"], s["confidence"])
    if s["margin_floor"] == "all_tests":
        final = None if all_m is None or clean_m is None else max(all_m, clean_m)
    else:
        final = clean_m
    return {"all_tests_margin": all_m, "clean_margin": clean_m, "margin": final}


def calibrate(a) -> int:
    s = _settings(a.config)
    folds = [json.loads(Path(f).read_text()) for f in a.folds]
    ms = [np.asarray(f["rival_tests"]["margin"]) for f in folds]
    cs = [np.asarray(f["rival_tests"]["clean"], bool) for f in folds]
    final = _fit(ms, cs, s)
    m, c = np.concatenate(ms), np.concatenate(cs)
    held, tot = {}, {"tests": 0, "resolved": 0, "wrong": 0}
    for i, f in enumerate(folds):
        fo = _fit([x for j, x in enumerate(ms) if j != i], [x for j, x in enumerate(cs) if j != i], s)
        r = apply(ms[i][cs[i]], fo["margin"], s["confidence"])
        held[f["match_id"]] = {**fo, **r}
        for k in tot:
            tot[k] += r[k]
    ub = upper_bound(tot["wrong"], tot["resolved"], s["confidence"]) if tot["resolved"] else None
    out = {"schema": "spana.reid_heldout_calibration", "schema_version": "1.0.0", "settings": s,
           "folds": [{"match_id": f["match_id"], "model_sha256": f["model_sha256"]} for f in folds],
           "final": {**final, "label": "borrowed", "in_sample_clean": apply(m[c], final["margin"], s["confidence"]),
                     "in_sample_all": apply(m, final["margin"], s["confidence"])},
           "held_out": {"per_fold": held, "pooled": {**tot, "upper_bound": ub,
                                                    "passes": bool(ub is not None and ub <= s["target_false_rate"])}}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--crops", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--config", default="config/reid_train.yaml")
    c = sub.add_parser("calibrate")
    c.add_argument("--folds", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--config", default="config/reid_train.yaml")
    a = p.parse_args(argv)
    return evaluate(a) if a.cmd == "eval" else calibrate(a)


if __name__ == "__main__":
    sys.exit(main())
