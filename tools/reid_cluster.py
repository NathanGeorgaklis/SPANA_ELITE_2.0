"""TRAIN re-ID labelling by group-then-confirm: cluster a labelling pack's tracklets, review clusters, audit a sample.

    python -m tools.reid_cluster build --pack PACK_DIR --model SPANA_REID.onnx --out REVIEW_DIR
                                       [--config config/reid_cluster.yaml]
    python -m tools.reid_cluster evaluate --pack PACK_DIR --labels data/reid/TRAIN/<match>/v<N> --model M.onnx
                                       [--min-observed N] [--config ...]

build (TRAIN packs only; DEV / EVAL refused):
- Features per tracklet, from the pack's own thumbnails (no video needed): the mean L2-normalised spana re-ID
  embedding and the mean torso colour histogram (spana.appearance.torso_descriptor) of its thumbnails.
- Tracklets the tracker flagged (identity contest) are excluded (`excluded_tracker_uncertain`): decided as unsure,
  never trained on, not clustered, not audited.
- Agglomerative clustering (average linkage) of the rest; two tracklets observed on the same frame can never share a
  cluster (same-frame rule, enforced as cannot-link); merging stops above `clustering.max_distance`.
- Within a cluster, tracklets are ordered odd ones out first (mean distance to the other members).
- Writes REVIEW_DIR/{index.html, review.js, review.json, thumbs/} and never overwrites.
upgrade: bring an existing review folder to the current format / config without re-clustering (keeps the
reviewer's progress: the page carries it over from the previous config's browser key).

Audit (after the review is locked): tracklets of player clusters with a live cluster-mate, shown alone in a fixed
seeded order (answers form a prefix of it). Freeze rule confirmed_only (Nathan, 2026-11-18): only player labels the
reviewer gave directly (audit answers, single-tracklet clusters) are frozen as players; an unaudited tracklet of a
multi-tracklet player cluster is frozen as unsure (unconfirmed_cluster_label). The cluster disagreement rate is
reported with its exact one-sided upper bound, against a reference (not a gate).

The review page exports the standard `spana.reid_label_export` 1.0.0 (one part per tracklet) plus a
`cluster_review` block; tools/freeze_reid_labels.py re-checks that block (check_cluster_export) and freezes
only directly confirmed player labels (rule confirmed_only).

evaluate: run the same clustering on a pack whose labels are already frozen (TRAIN) and report how often a
tracklet's cluster majority differs from its label (the error rate the audit would see).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import yaml

from spana.reid_labels import ReidLabelError, validate_manifest

REVIEW_SCHEMA = ("spana.reid_cluster_review", "1.2.0")
UNCONFIRMED_STATUS = "unconfirmed_cluster_label"
EXCLUDED_STATUS = "excluded_tracker_uncertain"
PAGE = Path(__file__).resolve().parent / "reid_cluster_review.html"
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "reid_cluster.yaml"
PACK_MARGIN = 0.15     # tools/reid_pack.py MARGIN: thumbnail = box + this margin, clipped to the frame
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_cluster_config(path=DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("schema") != "spana.reid_cluster_config":
        raise SystemExit(f"{path}: not a spana.reid_cluster_config")
    if cfg["clustering"]["linkage"] != "average":
        raise SystemExit(f"{path}: only average linkage is implemented")
    return cfg


def config_sha(path=DEFAULT_CONFIG) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------- features
def is_flagged(t: dict) -> bool:
    return bool(t.get("starts_uncertain") or t.get("uncertain_frames"))


def thumb_box(box: Sequence[float], frame_wh, thumb_shape) -> List[float]:
    """The detector box in thumbnail pixels (tools/reid_pack.py cuts box + PACK_MARGIN, clipped, resized)."""
    x1, y1, x2, y2 = box
    W, H = frame_wh
    mw, mh = (x2 - x1) * PACK_MARGIN, (y2 - y1) * PACK_MARGIN
    xa, ya = int(max(0, x1 - mw)), int(max(0, y1 - mh))
    xb, yb = int(min(W, x2 + mw)), int(min(H, y2 + mh))
    sx, sy = thumb_shape[1] / max(1, xb - xa), thumb_shape[0] / max(1, yb - ya)
    return [(x1 - xa) * sx, (y1 - ya) * sy, (x2 - xa) * sx, (y2 - ya) * sy]


def model_crop(img: np.ndarray, box: Sequence[float], pad: float, size_hw) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    xa, ya = int(max(0, round(x1 - pad * w))), int(max(0, round(y1 - pad * h)))
    xb, yb = int(min(W, round(x2 + pad * w))), int(min(H, round(y2 + pad * h)))
    return cv2.resize(img[ya:yb, xa:xb], (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)


def onnx_embedder(model_path) -> Callable[[List[np.ndarray]], np.ndarray]:
    import onnxruntime as ort
    s = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name

    def embed(crops: List[np.ndarray]) -> np.ndarray:
        out = []
        for k in range(0, len(crops), 32):
            x = np.stack([((c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                          for c in crops[k:k + 32]]).astype(np.float32)
            out.append(s.run(None, {name: x})[0])
        y = np.concatenate(out).astype(np.float64)
        return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return embed


def tracklet_features(pack_dir: Path, pack: dict, tracklets: List[dict], embed: Callable, app_cfg, cfg: dict):
    """Mean embedding (L2-normalised) and mean torso histogram (None if no thumbnail gives one) per tracklet."""
    from spana.appearance import torso_descriptor
    wh = (pack["video"]["width"], pack["video"]["height"])
    fc = cfg["features"]
    E, Hs = [], []
    for t in tracklets:
        crops, hists = [], []
        pos = {f: k for k, f in enumerate(t["frames"])}
        for th in t["thumbs"]:
            img = cv2.imread(str(pack_dir / th["file"]))
            if img is None:
                continue
            b = thumb_box(t["boxes"][pos[th["frame"]]], wh, img.shape)
            crops.append(model_crop(img, b, fc["crop_pad"], fc["size_hw"]))
            h = torso_descriptor(img, b, app_cfg)
            if h is not None:
                hists.append(h)
        if not crops:
            raise ReidLabelError(f"tracklet {t['id']}: no readable thumbnail")
        e = embed(crops).mean(axis=0)
        E.append(e / max(np.linalg.norm(e), 1e-12))
        Hs.append(np.mean(hists, axis=0) if hists else None)
    return np.array(E), Hs


# ------------------------------------------------------------------------------------------- clustering
def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))


def distance_matrix(E: np.ndarray, Hs: List[Optional[np.ndarray]], cfg: dict) -> np.ndarray:
    dc = cfg["distance"]
    De = 1.0 - np.clip(E @ E.T, -1.0, 1.0)
    n = len(E)
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if Hs[i] is None or Hs[j] is None:
                D[i, j] = De[i, j]
            else:
                D[i, j] = (dc["weight_embedding"] * De[i, j] + dc["weight_colour"] * (1.0 - bhattacharyya(Hs[i], Hs[j]))) \
                    / (dc["weight_embedding"] + dc["weight_colour"])
    np.fill_diagonal(D, 0.0)
    return D


def cannot_link(tracklets: List[dict]) -> np.ndarray:
    """Same-frame rule: two tracklets observed on a common frame are two different people."""
    n = len(tracklets)
    owner: Dict[int, List[int]] = {}
    for i, t in enumerate(tracklets):
        for f in t["frames"]:
            owner.setdefault(f, []).append(i)
    CL = np.zeros((n, n), bool)
    for ids in owner.values():
        if len(ids) > 1:
            ix = np.array(ids)
            CL[np.ix_(ix, ix)] = True
    np.fill_diagonal(CL, False)
    return CL


def agglomerate(D: np.ndarray, CL: np.ndarray, max_distance: float) -> List[int]:
    """Average-linkage agglomerative clustering with cannot-link; returns a cluster index per item."""
    n = len(D)
    if n == 0:
        return []
    M = D.astype(float).copy()
    np.fill_diagonal(M, np.inf)
    C = CL.copy()
    size = np.ones(n)
    alive = np.ones(n, bool)
    members = {i: [i] for i in range(n)}
    while True:
        cand = np.where(C | ~alive[:, None] | ~alive[None, :], np.inf, M)
        k = int(np.argmin(cand))
        a, b = divmod(k, n)
        if not np.isfinite(cand[a, b]) or cand[a, b] > max_distance:
            break
        if a > b:
            a, b = b, a
        M[a] = (size[a] * M[a] + size[b] * M[b]) / (size[a] + size[b])      # Lance-Williams, average linkage
        M[:, a] = M[a]
        M[a, a] = np.inf
        C[a] |= C[b]
        C[:, a] = C[a]
        C[a, a] = False
        size[a] += size[b]
        alive[b] = False
        M[b] = M[:, b] = np.inf
        members[a] += members.pop(b)
    labels = [0] * n
    order = sorted(members.values(), key=lambda m: (-len(m), min(m)))     # largest first, deterministic
    for ci, m in enumerate(order):
        for i in m:
            labels[i] = ci
    return labels


def odd_scores(D: np.ndarray, labels: List[int]) -> List[float]:
    """Mean distance of each item to the other members of its cluster (0 for a singleton)."""
    lab = np.array(labels)
    out = []
    for i, c in enumerate(labels):
        others = np.where((lab == c) & (np.arange(len(lab)) != i))[0]
        out.append(float(D[i, others].mean()) if len(others) else 0.0)
    return out


# ------------------------------------------------------------------------------------------- audit
# Audit (records/reid_audit_diagnose_2026-11-18): drawn AFTER the cluster review is locked, from the tracklets whose
# reviewed cluster says a player and that still have a live cluster-mate; shown alone in a fixed seeded order; the
# answers form a prefix of that order. Freeze rule confirmed_only (final_decisions): only directly confirmed player
# labels are frozen. Disagreement (player vs player) is reported with an exact one-sided upper bound.
def fnv1a32(text: str) -> int:
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def audit_order(eligible_ids: Sequence[str], seed: int) -> List[str]:
    """Deterministic draw order (the review page computes the same with the same FNV-1a hash)."""
    return sorted(eligible_ids, key=lambda t: (fnv1a32(f"{seed}:{t}"), t))


def wilson(k: int, n: int, level: float):
    if n == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def upper_bound(k: int, n: int, confidence: float) -> Optional[float]:
    """Exact one-sided (Clopper-Pearson) upper bound: the p at which P(X <= k) = 1 - confidence."""
    if n == 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 1 - confidence:
            lo = mid
        else:
            hi = mid
    return hi


def _decision(d) -> tuple:
    return (d.get("decision"), d.get("player") if d.get("decision") == "player" else None)


def derived_decisions(review: dict, tracklet_ids: Sequence[str]) -> Dict[str, tuple]:
    """Decision per tracklet implied by the clusters as reviewed (before audit answers): excluded -> unsure,
    a tracklet-level override (reject / unsure) wins over its cluster's decision."""
    where: Dict[str, str] = {}
    for cid, c in review["clusters"].items():
        for tid in c["tracklets"]:
            if tid in where:
                raise ReidLabelError(f"tracklet {tid} is in two clusters")
            where[tid] = cid
    excluded = set(review.get("excluded", []))
    out = {}
    for tid in tracklet_ids:
        if tid in excluded:
            out[tid] = ("unsure", None)
        elif tid in review.get("overrides", {}):
            out[tid] = _decision(review["overrides"][tid])
        elif tid in where:
            out[tid] = _decision(review["clusters"][where[tid]])
        else:
            raise ReidLabelError(f"tracklet {tid} is in no cluster")
    return out


def audit_eligible(review: dict) -> List[str]:
    """Tracklets the audit may draw: live (not overridden) in a cluster decided as a player with another live member."""
    over = review.get("overrides", {})
    out = []
    for c in review["clusters"].values():
        live = [t for t in c["tracklets"] if t not in over]
        if c.get("decision") == "player" and len(live) >= 2:
            out += live
    return sorted(out)


def answered_prefix(order: Sequence[str], answers: Dict[str, dict]) -> List[str]:
    """The answered part of the draw order: answers must form a prefix of it (no picking which to answer)."""
    out = []
    for t in order:
        a = answers.get(t)
        if not a or a.get("decision") not in ("player", "reject", "unsure"):
            break
        out.append(t)
    return out


def final_decisions(rv: dict, tracklet_ids: Sequence[str], audited: Sequence[str]):
    """Freeze rule `confirmed_only` (Nathan, 2026-11-18): a player label is frozen only when the reviewer gave it to
    that tracklet directly - an audit answer, or the decision of a single-tracklet cluster. The player label a
    multi-tracklet cluster gives to a tracklet that was not audited is frozen as unsure (status
    unconfirmed_cluster_label). Returns ({tracklet: (decision, player)}, {tracklet: status})."""
    derived = derived_decisions(rv, tracklet_ids)
    over = rv.get("overrides", {})
    size = {}
    for c in rv["clusters"].values():
        live = [t for t in c["tracklets"] if t not in over]
        for t in live:
            size[t] = len(live)
    excluded = set(rv.get("excluded", []))
    aud = set(audited)
    final, status = {}, {}
    for t in tracklet_ids:
        if t in excluded:
            final[t], status[t] = ("unsure", None), EXCLUDED_STATUS
        elif t in aud:
            final[t] = _decision(rv["audit"][t])
        elif derived[t][0] == "player" and size.get(t, 1) > 1:
            final[t], status[t] = ("unsure", None), UNCONFIRMED_STATUS
        else:
            final[t] = derived[t]
    return final, status, derived


def audit_stats(review: dict, derived: Dict[str, tuple], prefix: Sequence[str], cfg: dict) -> dict:
    """Cluster quality, reported (not a gate under confirmed_only): player answers vs the cluster's player label."""
    ac = cfg["audit"]
    comparable = dis = lost = 0
    rows = []
    for tid in prefix:
        a = _decision(review["audit"][tid])
        c = derived[tid]
        if a[0] != "player":
            lost += 1
            rows.append({"tracklet": tid, "answer": list(a), "cluster": list(c), "kind": "lost_data"})
            continue
        comparable += 1
        bad = a != c
        dis += bad
        rows.append({"tracklet": tid, "answer": list(a), "cluster": list(c), "kind": "disagree" if bad else "agree"})
    ub = upper_bound(dis, comparable, ac["confidence"])
    return {"drawn": len(prefix), "comparable": comparable, "disagreements": dis, "lost_data": lost,
            "rate": dis / comparable if comparable else None, "upper_bound": ub, "confidence": ac["confidence"],
            "bound": "exact one-sided (Clopper-Pearson)", "wilson": wilson(dis, comparable, ac["confidence"]),
            "cluster_quality_meets_reference": bool(comparable >= ac["report_min_comparable"] and ub is not None
                                                    and ub <= ac["report_max_disagreement_rate"]),
            "reference": {"min_comparable": ac["report_min_comparable"],
                          "max_disagreement_rate": ac["report_max_disagreement_rate"]}, "rows": rows}


def check_cluster_export(pack: dict, tl: Dict[str, dict], ex: dict, cfg: dict, cfg_sha256: str) -> dict:
    """Freeze gate for a group-then-confirm export (labelling_flow == cluster_confirm), rule confirmed_only.
    Recomputes everything from the pack and the export; raises ReidLabelError, or returns the audit statistics."""
    if cfg["audit"].get("freeze_rule") != "confirmed_only":
        raise ReidLabelError("config audit.freeze_rule must be confirmed_only")
    rv = ex.get("cluster_review") or {}
    if (rv.get("schema"), rv.get("schema_version")) != REVIEW_SCHEMA:
        raise ReidLabelError(f"cluster_confirm export needs a cluster_review block {REVIEW_SCHEMA}")
    if rv.get("config_sha256") != cfg_sha256:
        raise ReidLabelError("cluster_review was made with another config/reid_cluster.yaml")
    flagged = sorted(t for t, v in tl.items() if is_flagged(v)) if cfg["clustering"]["exclude_tracker_flagged"] else []
    if sorted(rv.get("excluded", [])) != flagged:
        raise ReidLabelError("excluded tracklets are not exactly the tracker-flagged ones")
    eligible = sorted(set(tl) - set(flagged))
    in_clusters = sorted(t for c in rv["clusters"].values() for t in c["tracklets"])
    if in_clusters != eligible:
        raise ReidLabelError("clusters do not hold every non-excluded tracklet exactly once")
    over = rv.get("overrides", {})
    for tid, o in over.items():
        if o.get("decision") not in ("reject", "unsure"):
            raise ReidLabelError(f"override for {tid} must be reject or unsure")
    for cid, c in rv["clusters"].items():
        live = [t for t in c["tracklets"] if t not in over]
        if live and c.get("decision") not in ("player", "reject", "unsure"):
            raise ReidLabelError(f"cluster {cid} is undecided")
        if c.get("decision") == "player" and not c.get("player"):
            raise ReidLabelError(f"cluster {cid} is a player cluster without a player")
        seen: Dict[int, str] = {}
        for t in live:
            for f in tl[t]["frames"]:
                if f in seen:
                    raise ReidLabelError(f"CONFLICT: cluster {cid} holds {seen[f]} and {t} on frame {f}")
                seen[f] = t
    order = audit_order(audit_eligible(rv), cfg["audit"]["seed"])
    answers = rv.get("audit", {})
    prefix = answered_prefix(order, answers)
    if sorted(answers) != sorted(prefix):
        raise ReidLabelError("audit answers are not a prefix of the draw order (or name tracklets that cannot be drawn)")
    final, status, derived = final_decisions(rv, sorted(tl), prefix)
    parts: Dict[str, List[dict]] = {}
    for p in ex.get("parts", []):
        parts.setdefault(p.get("tracklet"), []).append(p)
    for tid, t in tl.items():
        ps = parts.get(tid, [])
        if len(ps) != 1 or ps[0].get("first_frame") != t["frames"][0] or ps[0].get("last_frame") != t["frames"][-1]:
            raise ReidLabelError(f"tracklet {tid}: a cluster_confirm export has exactly one whole part per tracklet")
        if _decision(ps[0]) != final[tid]:
            raise ReidLabelError(f"tracklet {tid}: exported decision differs from the confirmed-only rule")
        if ps[0].get("status") != status.get(tid):
            raise ReidLabelError(f"tracklet {tid}: status must be {status.get(tid)!r}")
    st = audit_stats(rv, derived, prefix, cfg)
    singles = sum(1 for t in tl if final[t][0] == "player" and t not in prefix)
    st.update({"freeze_rule": "confirmed_only", "drawable": len(order),
               "player_labels_from_audit": sum(1 for t in prefix if final[t][0] == "player"),
               "player_labels_from_single_tracklet_clusters": singles,
               "unconfirmed_cluster_labels_frozen_unsure": sum(1 for v in status.values() if v == UNCONFIRMED_STATUS)})
    return st


# ------------------------------------------------------------------------------------------- build / evaluate
def _load_pack(pack_dir: Path, train_only: bool = True):
    pack = json.loads((pack_dir / "pack.json").read_text())
    m = validate_manifest(pack["match"])
    if train_only and m["split"] != "TRAIN":
        raise ReidLabelError(f"refusing: cluster labelling is for TRAIN packs only, this pack is {m['split']}")
    tl_bytes = (pack_dir / "tracklets.json").read_bytes()
    if hashlib.sha256(tl_bytes).hexdigest() != pack["tracklets_sha256"]:
        raise ReidLabelError("tracklets.json changed after the pack was made")
    return pack, json.loads(tl_bytes)


def propose(pack_dir: Path, pack: dict, tracklets: List[dict], embed: Callable, cfg: dict, app_cfg) -> dict:
    flagged = [t for t in tracklets if cfg["clustering"]["exclude_tracker_flagged"] and is_flagged(t)]
    fl = {t["id"] for t in flagged}
    use = [t for t in tracklets if t["id"] not in fl]
    E, Hs = tracklet_features(pack_dir, pack, use, embed, app_cfg, cfg) if use else (np.zeros((0, 1)), [])
    D = distance_matrix(E, Hs, cfg) if use else np.zeros((0, 0))
    CL = cannot_link(use)
    labels = agglomerate(D, CL, cfg["clustering"]["max_distance"])
    odd = odd_scores(D, labels)
    clusters: Dict[str, dict] = {}
    width = max(3, len(str(max(labels) + 1))) if labels else 3
    for ci in sorted(set(labels)):
        idx = [i for i, c in enumerate(labels) if c == ci]
        idx.sort(key=lambda i: (-odd[i], use[i]["id"]))                  # odd ones out first
        clusters[f"C{ci + 1:0{width}d}"] = {"tracklets": [use[i]["id"] for i in idx],
                                            "odd": [round(odd[i], 4) for i in idx]}
    return {"clusters": clusters, "excluded": sorted(fl), "labels": labels, "use": [t["id"] for t in use],
            "missing_colour": int(sum(h is None for h in Hs))}


def build(pack_dir, out, cfg_path=DEFAULT_CONFIG, model=None, embed: Optional[Callable] = None,
          app_cfg=None) -> dict:
    from spana.config import load_config
    pack_dir, out = Path(pack_dir), Path(out)
    if out.exists():
        raise ReidLabelError(f"refusing to overwrite {out}")
    cfg = load_cluster_config(cfg_path)
    pack, tracklets = _load_pack(pack_dir)
    embed = embed or onnx_embedder(model)
    app_cfg = app_cfg or load_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml").appearance
    pr = propose(pack_dir, pack, tracklets, embed, cfg, app_cfg)
    data = {
        "schema": REVIEW_SCHEMA[0], "schema_version": REVIEW_SCHEMA[1], "match": pack["match"],
        "tracklets_sha256": pack["tracklets_sha256"], "config_sha256": config_sha(cfg_path),
        "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest() if model else None,
        "excluded_status": EXCLUDED_STATUS, "config": cfg,
        "tracklets": [{"id": t["id"], "track": t["track"], "first_frame": t["frames"][0],
                       "last_frame": t["frames"][-1], "frames": t["frames"], "observed": len(t["frames"]),
                       "thumbs": [th["file"] for th in t["thumbs"]], "flagged": is_flagged(t)} for t in tracklets],
        "clusters": pr["clusters"], "excluded": pr["excluded"],
        "stats": {"tracklets": len(tracklets), "excluded": len(pr["excluded"]), "clustered": len(pr["use"]),
                  "clusters": len(pr["clusters"]), "singletons": sum(len(c["tracklets"]) == 1 for c in pr["clusters"].values()),
                  "missing_colour": pr["missing_colour"]},
    }
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(pack_dir / "thumbs", tmp / "thumbs")
    text = json.dumps(data, separators=(",", ":"))
    (tmp / "review.json").write_text(text)
    (tmp / "review.js").write_text("window.SPANA_CLUSTER_REVIEW = " + text + ";\n")
    shutil.copy(PAGE, tmp / "index.html")
    tmp.rename(out)
    return data


def upgrade(review_dir, cfg_path=DEFAULT_CONFIG) -> dict:
    """Bring an existing review folder to the current review format and config WITHOUT re-clustering: tracklets,
    clusters and exclusions are kept byte for byte; config, config hash and schema are updated, the pre-drawn
    1.0.0 audit sample is dropped, and the page is replaced. Old files are kept as *.<old version>.bak."""
    rd = Path(review_dir)
    data = json.loads((rd / "review.json").read_text())
    old = data.get("schema_version")
    if old == REVIEW_SCHEMA[1] and data.get("config_sha256") == config_sha(cfg_path):
        return data
    cfg = load_cluster_config(cfg_path)                       # 1.0.0 -> drops the build-time audit; 1.1.0 -> rule only
    for name in ("review.json", "review.js", "index.html"):
        bak = rd / f"{name}.{old}.bak"
        if not bak.exists():
            shutil.copy(rd / name, bak)
    if "audit" in data:                       # 1.0.0: these were hidden from the cluster sheets during the review
        data["previously_hidden"] = sorted(data.pop("audit"))
    data.pop("z", None)
    data["stats"].pop("audit", None)
    data.update({"schema_version": REVIEW_SCHEMA[1], "config": cfg, "config_sha256": config_sha(cfg_path),
                 "upgraded_from": old})
    text = json.dumps(data, separators=(",", ":"))
    (rd / "review.json").write_text(text)
    (rd / "review.js").write_text("window.SPANA_CLUSTER_REVIEW = " + text + ";\n")
    shutil.copy(PAGE, rd / "index.html")
    return data


def evaluate(pack_dir, labels_dir, model=None, cfg_path=DEFAULT_CONFIG, min_observed: int = 0,
             embed: Optional[Callable] = None, app_cfg=None) -> dict:
    """Cluster a pack whose labels are frozen (TRAIN) and compare: a tracklet is an error when its cluster's
    majority player differs from its own label (what the audit would count)."""
    from spana.config import load_config
    from spana.reid_labels import load_frozen
    pack_dir = Path(pack_dir)
    cfg = load_cluster_config(cfg_path)
    pack, tracklets = _load_pack(pack_dir)
    lab = load_frozen(Path(labels_dir), "train")
    if lab["frozen"].get("tracklets_sha256") != pack["tracklets_sha256"]:
        raise ReidLabelError("the labels were not made from this pack")
    truth: Dict[str, Optional[str]] = {}
    for r in lab["identities"]:
        truth.setdefault(r["tracklet"], set()).add(r["player"] if r["decision"] == "player" else None)
    tracklets = [t for t in tracklets if len(t["frames"]) >= min_observed]
    embed = embed or onnx_embedder(model)
    app_cfg = app_cfg or load_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml").appearance
    pr = propose(pack_dir, pack, tracklets, embed, cfg, app_cfg)
    single = {t: next(iter(v)) for t, v in truth.items() if len(v) == 1 and None not in v}   # one player, whole
    err = n = 0
    players_in = {}
    for cid, c in pr["clusters"].items():
        known = [single[t] for t in c["tracklets"] if t in single]
        if not known:
            continue
        maj = max(sorted(set(known)), key=known.count)
        players_in.setdefault(maj, []).append(cid)
        n += len(known)
        err += sum(k != maj for k in known)
    return {"tracklets": len(tracklets), "excluded": len(pr["excluded"]), "clustered": len(pr["use"]),
            "clusters": len(pr["clusters"]), "singletons": sum(len(c["tracklets"]) == 1 for c in pr["clusters"].values()),
            "labelled_single_player": n, "errors": err, "error_rate": round(err / n, 4) if n else None,
            "error_ci": wilson(err, n, cfg["audit"]["confidence"]), "players": len(players_in),
            "clusters_per_player_median": float(np.median([len(v) for v in players_in.values()])) if players_in else None,
            "max_distance": cfg["clustering"]["max_distance"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pack", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--config", default=str(DEFAULT_CONFIG))
    u = sub.add_parser("upgrade")
    u.add_argument("--review", required=True)
    u.add_argument("--config", default=str(DEFAULT_CONFIG))
    e = sub.add_parser("evaluate")
    e.add_argument("--pack", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--min-observed", type=int, default=0)
    e.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = p.parse_args(argv)
    if a.cmd == "upgrade":
        d = upgrade(a.review, a.config)
        print(json.dumps({"review": a.review, "schema_version": d["schema_version"], "upgraded_from": d.get("upgraded_from"),
                          "clusters": len(d["clusters"]), "config_sha256": d["config_sha256"]}))
        return 0
    if a.cmd == "build":
        d = build(a.pack, a.out, a.config, a.model)
        print(json.dumps({"out": a.out, **d["stats"]}))
    else:
        print(json.dumps(evaluate(a.pack, a.labels, a.model, a.config, a.min_observed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
