"""Diagnose cluster-review audit disagreements (analysis only: labels, clusters and the audit limit are untouched).

    python -m tools.dev.reid_audit_diagnose --review REVIEW_DIR --pack PACK_DIR --state STATE.json --out OUT_DIR \
        [--model SPANA_REID.onnx] [--light day|night|unknown]

STATE.json = the review page's saved progress (its localStorage value: players, members, cdec, overrides, audit).
Audit tracklets: 1.0.0 reviews = the sample drawn at build; 1.1.0 = the scored prefix of the post-review draw order
(tools.reid_cluster.audit_order / audit_prefix), recomputed from the state.

For every audit tracklet, three cluster labels are compared with the audit answer:
  tool       what the page / freeze gate count: the decision stored on the tracklet's current cluster
             (tools.reid_cluster.derived_decisions; None when the cluster gives no opinion)
  post_edit  what the reviewed clusters say now: the most common final decision of the tracklet's CURRENT live
             cluster-mates (audit and overridden tracklets left out); None when it has no such mates
  pre_edit   what the proposal grouped it with: the most common final decision of its ORIGINAL cluster-mates
             (review.json), labelled as the reviewer labelled them
A tool label that differs from the post-edit label is a stale cluster decision (e.g. the cluster was assigned, then
its other members were split / moved away, leaving the hidden audit tracklet under the old decision).
Rates:
  (a) real conflicts: audit = player and post-edit = player, different players / tracklets where both are players
  (b) as the tool counts: audit not unsure and tool label not None; any (decision, player) difference
Within-tracklet appearance change (with --model): cosine distance between the mean embeddings of the first and
second half of its thumbnails, reported with its percentile among all clustered tracklets of the review.
Writes OUT_DIR/{diagnosis.json, sheet_<tracklet>.jpg, index.html}; never overwrites.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from tools import reid_cluster as RC


def _key(d) -> Optional[tuple]:
    return None if d is None else (d[0], d[1])


def final_decisions(R: dict, S: dict) -> Dict[str, Optional[tuple]]:
    """Final decision per tracklet as the page would export it (audit answers for audit tracklets)."""
    rv = {"clusters": {k: {"tracklets": m, **(S["cdec"].get(k) or {"decision": None, "player": None})}
                       for k, m in S["members"].items()},
          "overrides": S.get("overrides", {}), "excluded": R["excluded"], "audit": S.get("audit", {})}
    der = RC.derived_decisions(rv, [t["id"] for t in R["tracklets"]])
    out = {t: (d if d[0] is not None else None) for t, d in der.items()}
    for t, a in S.get("audit", {}).items():
        if a:
            out[t] = RC._decision(a)
    return out, {t: (d if d[0] is not None else None) for t, d in der.items()}


def majority(labels: List[Optional[tuple]]) -> Optional[tuple]:
    labels = [x for x in labels if x is not None]
    if not labels:
        return None
    c = Counter(labels).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else ("tie", "/".join(sorted(str(x[1] or x[0]) for x, n in c if n == c[0][1])))


def wilson(k, n, level=0.95):
    return RC.wilson(k, n, level)


def classify(ans, tool, post, players, switch_flag) -> str:
    if ans[0] in ("unsure", "reject"):
        return "my answer was unsure or reject"
    if tool is not None and post != tool:
        return "other: stale cluster label (cluster decision no longer matches its members after edits)"
    lab = post or tool
    if lab is None:
        return "other: cluster gives no opinion"
    if lab[0] != "player":
        return "other: cluster was unsure or reject"
    if switch_flag:
        return "tracklet probably switches player partway"
    pa, pc = players.get(ans[1], {}), players.get(lab[1], {})
    if pa.get("side") == pc.get("side") and pa.get("shirt_number") and pa.get("shirt_number") == pc.get("shirt_number"):
        return "name mismatch for the same player"
    return "wrong cluster (same team)" if pa.get("side") == pc.get("side") else "wrong cluster (other team)"


def half_change(pack_dir: Path, pack: dict, t: dict, embed, cfg) -> Optional[float]:
    wh = (pack["video"]["width"], pack["video"]["height"])
    pos = {f: k for k, f in enumerate(t["frames"])}
    crops = []
    for th in t["thumbs"]:
        img = cv2.imread(str(pack_dir / th["file"]))
        if img is not None:
            crops.append(RC.model_crop(img, RC.thumb_box(t["boxes"][pos[th["frame"]]], wh, img.shape),
                                       cfg["features"]["crop_pad"], cfg["features"]["size_hw"]))
    if len(crops) < 2:
        return None
    E = embed(crops)
    h = len(E) // 2
    a, b = E[:h].mean(0), E[h:].mean(0)
    return float(1 - a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def sheet(review_dir: Path, rows: List[tuple], out: Path) -> None:
    """rows: (caption, [thumb paths]) -> one image, 64x128 cells, caption above each row."""
    cells = []
    for cap, files in rows:
        ims = [cv2.resize(cv2.imread(str(review_dir / f)), (64, 128)) for f in files[:12] if (review_dir / f).exists()]
        row = np.hstack(ims + [np.zeros((128, 64, 3), np.uint8)] * (12 - len(ims))) if ims else np.zeros((128, 768, 3), np.uint8)
        bar = np.zeros((20, row.shape[1], 3), np.uint8)
        cv2.putText(bar, cap[:110], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cells += [bar, row]
    cv2.imwrite(str(out), np.vstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 90])


def diagnose(review_dir, pack_dir, state: dict, out_dir, embed=None, light="unknown", cfg_path=RC.DEFAULT_CONFIG) -> dict:
    review_dir, pack_dir, out_dir = Path(review_dir), Path(pack_dir), Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    R = json.loads((review_dir / "review.json").read_text())
    if R.get("schema_version") == "1.0.0" or "audit" in R:
        R.setdefault("audit", [])                                   # 1.0.0: the sample drawn at build time
    else:                                                           # 1.1.0: the scored prefix of the post-review draw
        cfg0 = RC.load_cluster_config(cfg_path)
        rv = {"clusters": {k: {"tracklets": m, **(state["cdec"].get(k) or {"decision": None, "player": None})}
                           for k, m in state["members"].items()}, "overrides": state.get("overrides", {})}
        order = RC.audit_order(RC.audit_eligible(rv), cfg0["audit"]["seed"])
        R["audit"] = RC.audit_prefix(order, state.get("audit", {}), cfg0["audit"]["min_comparable"])[0]
    pack, tracklets = RC._load_pack(pack_dir)
    if pack["tracklets_sha256"] != R["tracklets_sha256"]:
        raise SystemExit("review and pack disagree")
    cfg = RC.load_cluster_config(cfg_path)
    TL = {t["id"]: t for t in tracklets}
    RT = {t["id"]: t for t in R["tracklets"]}
    players = state.get("players", {})
    final, der = final_decisions(R, state)
    audit = set(R["audit"])
    orig = {t: k for k, c in R["clusters"].items() for t in c["tracklets"]}
    cur = {t: k for k, m in state["members"].items() for t in m}
    over = state.get("overrides", {})

    v11 = R.get("schema_version") != "1.0.0"

    def mates(members, t):
        # 1.0.0: audit tracklets were hidden, so they are not evidence; 1.1.0: nothing is hidden, every live mate counts
        return [m for m in members if m != t and (v11 or m not in audit) and m not in over]

    label_of = der if v11 else final        # 1.1.0: what the reviewed clusters say (not the mates' audit answers)

    change, pct = {}, {}
    if embed is not None:
        clustered = [t for t in RT if t not in R["excluded"]]
        vals = {t: half_change(pack_dir, pack, TL[t], embed, cfg) for t in clustered}
        dist = np.array([v for v in vals.values() if v is not None])
        for t in audit:
            change[t] = vals.get(t)
            pct[t] = None if vals.get(t) is None else round(float((dist < vals[t]).mean() * 100), 1)
    out_dir.mkdir(parents=True)
    rows, a_n, a_k, b_n, b_k = [], 0, 0, 0, 0
    for t in R["audit"]:
        ans = _key(RC._decision(state["audit"][t])) if state.get("audit", {}).get(t) else None
        tool = der.get(t)
        post = majority([label_of.get(m) for m in mates(state["members"][cur[t]], t)])
        pre = majority([label_of.get(m) for m in mates(R["clusters"][orig[t]]["tracklets"], t)])
        if ans is None:
            rows.append({"tracklet": t, "answer": None, "note": "unanswered"})
            continue
        if ans[0] != "unsure" and tool is not None:                    # (b) as the page / freeze gate count
            b_n += 1
            b_dis = ans != tool
            b_k += b_dis
        else:
            b_dis = None
        a_dis = None
        if ans[0] == "player" and post is not None and post[0] == "player":   # (a) real assign-vs-assign
            a_n += 1
            a_dis = ans != post
            a_k += a_dis
        disagree = bool(b_dis) or bool(a_dis) or (post is not None and ans != post)
        hts = [b[3] - b[1] for b in TL[t]["boxes"]]
        sw = pct.get(t) is not None and pct[t] >= 95.0
        row = {"tracklet": t, "answer": ans, "tool_label": tool, "post_edit_label": post, "pre_edit_label": pre,
               "original_cluster": orig[t], "current_cluster": cur[t],
               "cluster_decision_stored": state["cdec"].get(cur[t]),
               "live_mates_now": len(mates(state["members"][cur[t]], t)),
               "original_mates": len(mates(R["clusters"][orig[t]]["tracklets"], t)),
               "counted_by_tool_as_disagreement": b_dis, "real_conflict": a_dis, "disagrees": disagree,
               "observed_frames": len(TL[t]["frames"]), "duration_s": round((TL[t]["frames"][-1] - TL[t]["frames"][0]) / pack["video"]["fps"], 2),
               "median_box_height_px": round(float(np.median(hts)), 1), "light": light,
               "within_tracklet_change": None if t not in change or change[t] is None else round(change[t], 4),
               "within_tracklet_change_percentile": pct.get(t)}
        if disagree:
            row["category"] = classify(ans, tool, post, players, sw)
            mates_now = mates(state["members"][cur[t]], t)
            mates_orig = mates(R["clusters"][orig[t]]["tracklets"], t)
            same_ans = [m for m, d in final.items() if d == ans and m != t and m not in audit][:12]
            mid = lambda m: RT[m]["thumbs"][len(RT[m]["thumbs"]) // 2]   # noqa: E731
            sheet(review_dir, [
                (f"AUDIT {t}: you said {ans[1] or ans[0]} | tool counted {tool and (tool[1] or tool[0])} | "
                 f"post-edit {post and (post[1] or post[0])} | pre-edit {pre and (pre[1] or pre[0])}", RT[t]["thumbs"]),
                (f"its cluster NOW ({cur[t]}, stored decision {state['cdec'].get(cur[t])}), {len(mates_now)} live mates",
                 [mid(m) for m in mates_now]),
                (f"its ORIGINAL cluster ({orig[t]}), {len(mates_orig)} mates", [mid(m) for m in mates_orig]),
                (f"tracklets you labelled {ans[1] or ans[0]}", [mid(m) for m in same_ans]),
            ], out_dir / f"sheet_{t}.jpg")
            row["sheet"] = f"sheet_{t}.jpg"
        rows.append(row)
    res = {"audit_tracklets": len(R["audit"]),
           "rate_a_real_conflicts": {"disagreements": a_k, "n": a_n, "rate": round(a_k / a_n, 4) if a_n else None,
                                     "ci95": wilson(a_k, a_n)},
           "rate_b_as_tool_counts": {"disagreements": b_k, "n": b_n, "rate": round(b_k / b_n, 4) if b_n else None,
                                     "ci95": wilson(b_k, b_n)},
           "limit": cfg["audit"]["max_disagreement_rate"],
           "disagreements": [r for r in rows if r.get("disagrees")], "all": rows}
    (out_dir / "diagnosis.json").write_text(json.dumps(res, indent=1, default=str))
    items = "".join(f"<h3>{r['tracklet']}: {r['category']}</h3><p>you: {r['answer']} · tool counted: {r['tool_label']} · "
                    f"post-edit: {r['post_edit_label']} · pre-edit: {r['pre_edit_label']} · {r['duration_s']} s, "
                    f"median box {r['median_box_height_px']} px, {r['light']}, within-tracklet change percentile "
                    f"{r['within_tracklet_change_percentile']}</p><img src='{r['sheet']}' style='max-width:100%'>"
                    for r in res["disagreements"])
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Audit disagreements</title><body style='font:14px system-ui;"
        "background:#111;color:#eee;padding:12px'>"
        f"<h2>Audit disagreements</h2><p>(a) real conflicts: {res['rate_a_real_conflicts']} · "
        f"(b) as the tool counts: {res['rate_b_as_tool_counts']} · limit {res['limit']}</p>{items}</body>")
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    for k in ("review", "pack", "state", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--light", default="unknown", choices=("day", "night", "unknown"))
    a = p.parse_args(argv)
    st = json.loads(Path(a.state).read_text())
    res = diagnose(a.review, a.pack, st, a.out, RC.onnx_embedder(a.model) if a.model else None, a.light)
    print(json.dumps({k: res[k] for k in ("rate_a_real_conflicts", "rate_b_as_tool_counts", "limit")}))
    for r in res["disagreements"]:
        print(json.dumps({k: r.get(k) for k in ("tracklet", "category", "answer", "tool_label", "post_edit_label",
                                                 "pre_edit_label", "duration_s", "median_box_height_px",
                                                 "within_tracklet_change_percentile")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
