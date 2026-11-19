"""DEV analysis only: map rival-test indices of tables_p3.json back to their crops (query, ref A, ref B) on one clip,
re-running the same crop and triplet construction as tools/dev/appearance_cue_eval.py, and save the crops + context.
Usage: python find_test.py TABLES.json CLIP_SPEC OUT_DIR INDEX [INDEX ...]   (run from the spana-calibration dir)"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from tools.dev import appearance_cue_eval as A  # noqa: E402
from spana.config import load_config  # noqa: E402

tables_path, spec, out, *idx = sys.argv[1:]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
cfg = load_config("config/default.yaml")
name, crops, contact, rivals = A.load_clip(spec, cfg)
tab = next(t for t in json.load(open(tables_path))["tables"] if t["match"] == name)
pairs = tab["rival"]["pair"]
by_player = {}
for k, c in enumerate(crops):
    by_player.setdefault(c["player"], []).append(k)
emb = A.Embedder(Path("/Users/nathangeorgaklis/spana-reid-eval/models_p3/spana_reid.onnx"), (224, 112), lambda y: y, None)
feats = emb([c["img"] for c in crops])
video = spec.split(":")[1]
for i in map(int, idx):
    pair = pairs[i]
    start = pairs.index(pair)
    p, q = pair.split("|")
    trip = []
    for Aa, Bb in ((p, q), (q, p)):
        for qk in by_player.get(Aa, []):
            ra = [k for k in by_player[Aa] if crops[qk]["t"] - crops[k]["t"] >= A.MIN_GAP_S]
            rb = [k for k in by_player.get(Bb, []) if crops[qk]["t"] - crops[k]["t"] >= A.MIN_GAP_S]
            if ra and rb:
                trip.append((qk, ra[-1], rb[-1]))
    # the triplet order inside a pair block depends on the set order of (p, q): check both orders against the table
    blk = [j for j in range(len(pairs)) if pairs[j] == pair]
    sa = np.array([tab["rival"]["qa"]["spana"][j] for j in blk])
    cand = []
    for order in (trip, trip[len([1 for t in trip if crops[t[0]]["player"] == p]):] + trip[:len([1 for t in trip if crops[t[0]]["player"] == p])]):
        got = np.array([float(feats[a] @ feats[b]) for a, b, _ in order])
        if len(got) == len(sa) and np.allclose(got, sa, atol=1e-4):
            cand = order
    assert cand, "triplets not reproduced"
    qk, ak, bk = cand[i - start]
    sim_a, sim_b = float(feats[qk] @ feats[ak]), float(feats[qk] @ feats[bk])
    assert abs(sim_a - tab["rival"]["qa"]["spana"][i]) < 1e-4 and abs(sim_b - tab["rival"]["qb"]["spana"][i]) < 1e-4
    info = {"index": i, "pair": pair, "spana_margin": sim_a - sim_b, "sim_query_refA": sim_a, "sim_query_refB": sim_b}
    for role, k in (("query", qk), ("refA", ak), ("refB", bk)):
        c = crops[k]
        info[role] = {"player": c["player"], "frame": c["frame"], "t": round(c["t"], 2), "box": [round(v, 1) for v in c["box"]],
                      "occluded": c["occluded"]}
    # every other labelled player near the query box on the query frame
    qb = np.array(crops[qk]["box"])
    near = []
    for c in crops:
        if c["frame"] == crops[qk]["frame"] and c is not crops[qk]:
            iou = A.iou_matrix(qb[None], np.array(c["box"], float)[None])[0, 0]
            d = float(np.hypot(*(np.array(c["box"][:2]) - qb[:2])))
            if iou > 0 or d < 150:
                near.append({"player": c["player"], "iou": round(float(iou), 3), "box": [round(v, 1) for v in c["box"]]})
    info["near_query"] = near
    print(json.dumps(info))
    (out / f"test_{i}.json").write_text(json.dumps(info, indent=1))
    # contact sheet: query / ref A / ref B crops, plus the frame context around the query
    cap = cv2.VideoCapture(video)
    tiles = []
    for role, k in (("query", qk), ("refA", ak), ("refB", bk)):
        c = crops[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame"])
        ok, img = cap.read()
        x1, y1, x2, y2 = (int(round(v)) for v in c["box"])
        crop = cv2.resize(img[max(0, y1):y2, max(0, x1):x2], (120, 240))
        cv2.putText(crop, f"{role} {c['player']}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(crop, f"f{c['frame']}", (2, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(crop)
        if role == "query":
            ctx = img.copy()
            for c2 in crops:
                if c2["frame"] == c["frame"]:
                    bx = [int(round(v)) for v in c2["box"]]
                    col = (0, 0, 255) if c2 is c else (0, 255, 0)
                    cv2.rectangle(ctx, tuple(bx[:2]), tuple(bx[2:]), col, 2)
                    cv2.putText(ctx, c2["player"], (bx[0], bx[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            H, W = ctx.shape[:2]
            ctx = ctx[max(0, cy - 220):min(H, cy + 220), max(0, cx - 360):min(W, cx + 360)]
            cv2.imwrite(str(out / f"test_{i}_context.jpg"), ctx)
    cv2.imwrite(str(out / f"test_{i}_crops.jpg"), np.hstack(tiles))
