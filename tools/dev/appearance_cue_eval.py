# DEV diagnostic script (not product code). Compares appearance cues on frozen DEV labels:
#   hist     the tracker's torso HSV histogram (spana.appearance.torso_descriptor), Bhattacharyya similarity
#   osnet    OSNet x0.25 person re-ID (MSMT17) ONNX, cosine
#   youtu    OpenCV-Zoo Youtu person re-ID ONNX, cosine
#   dinov2   DINOv2-small (general self-supervised) ONNX CLS token, cosine
#   number   shirt-number OCR (tesseract, digits only) on the torso band
# Pairs: same player (frames >= MIN_GAP_S apart) vs different players. "Hard" pairs: players whose labelled boxes
# overlap (IoU >= 0.30) on some labelled frame (contact pairs: where IDENTITY_UNCERTAIN arises), plus the player
# pairs of unresolved IDENTITY_UNCERTAIN cases. A rival test mirrors identity resolution: query crop of A versus a
# reference crop of A and a reference crop of B, both >= MIN_GAP_S earlier; correct if it is nearer A's.
#   python -m tools.dev.appearance_cue_eval OUT_JSON MODEL_DIR CLIP_SPEC [CLIP_SPEC ...]
#   CLIP_SPEC = label_dir:video:tracks.json:decisions.jsonl   (tracks/decisions: the run whose uncertain cases count)
import collections
import itertools
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from spana.appearance import similarity as hist_sim
from spana.appearance import torso_descriptor
from spana.config import load_config
from spana.evaluation import NESTED_V2, _match
from spana.geometry import iou_matrix
from tools.benchmark import load_frozen

MIN_GAP_S = 1.0
CONTACT_IOU = 0.30
HEIGHT_BINS = ((0, 90), (90, 130), (130, 10_000))
GAP_BINS = ((1.0, 3.0), (3.0, 6.0), (6.0, 100.0))
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class Embedder:
    def __init__(self, path, size_hw, out, fixed_batch=None):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.size_hw, self.out, self.fixed = size_hw, out, fixed_batch

    def __call__(self, crops):
        h, w = self.size_hw
        x = np.stack([((cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] / 255.0 - MEAN) / STD)
                      .transpose(2, 0, 1) for c in crops]).astype(np.float32)
        n = len(x)
        if self.fixed:
            x = np.concatenate([x, np.zeros((-n % self.fixed, *x.shape[1:]), np.float32)])
            outs = [self.s.run(None, {self.inp: x[k:k + self.fixed]})[0] for k in range(0, len(x), self.fixed)]
            y = np.concatenate(outs)[:n]
        else:
            y = np.concatenate([self.s.run(None, {self.inp: x[k:k + 32]})[0] for k in range(0, n, 32)])
        y = self.out(y).reshape(n, -1).astype(np.float64)
        return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def read_number(img_bgr, box):
    """Shirt-number OCR on the torso band; (number or None, confidence)."""
    import pytesseract
    x1, y1, x2, y2 = box; h = y2 - y1
    crop = img_bgr[int(max(0, y1 + 0.12 * h)):int(y1 + 0.55 * h), int(max(0, x1)):int(x2)]
    if crop.size == 0 or crop.shape[0] < 6:
        return None, 0.0
    g = cv2.cvtColor(cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    best = (None, 0.0)
    for img in (cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]):
        d = pytesseract.image_to_data(img, config="--psm 7 -c tessedit_char_whitelist=0123456789",
                                      output_type=pytesseract.Output.DICT)
        for txt, conf in zip(d["text"], d["conf"]):
            if re.fullmatch(r"\d{1,2}", txt.strip() or "") and float(conf) > best[1]:
                best = (txt.strip(), float(conf))
    return best


def load_clip(spec, cfg):
    label_dir, video, trk_path, dec_path = spec.split(":")
    fz, gt = load_frozen(Path(label_dir))
    fps = fz["clip"]["fps"]
    want = set(gt.frames)
    cap = cv2.VideoCapture(video)
    crops = []
    i = 0
    while want:
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            want.discard(i)
            for g, b in gt.players[i].items():
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                c = img[max(0, y1):y2, max(0, x1):x2]
                if c.shape[0] < 16 or c.shape[1] < 6:
                    continue
                crops.append({"clip": fz["clip"]["name"], "frame": i, "t": i / fps, "player": g, "box": list(b),
                              "h": b[3] - b[1], "occluded": g in gt.occluded.get(i, set()), "img": c,
                              "hist": torso_descriptor(img, b, cfg.appearance), "number": read_number(img, b)})
        i += 1
    contact = set()
    for f in gt.frames:
        ids = sorted(gt.players[f])
        if len(ids) < 2:
            continue
        m = iou_matrix(np.array([gt.players[f][g] for g in ids], float), np.array([gt.players[f][g] for g in ids], float))
        for a, b in itertools.combinations(range(len(ids)), 2):
            if m[a, b] >= CONTACT_IOU:
                contact.add(tuple(sorted((ids[a], ids[b]))))
    rivals = uncertain_rival_pairs(gt, trk_path, dec_path)
    return fz["clip"]["name"], crops, contact, rivals


def uncertain_rival_pairs(gt, trk_path, dec_path):
    """Player pairs of IDENTITY_UNCERTAIN tracks never resolved, mapped to labelled players (NESTED_V2)."""
    trk = {f["frame"]: f for f in json.load(open(trk_path))["frames"]}
    resolved = {json.loads(l)["track_id"] for l in open(dec_path) if '"identity_resolved"' in l}
    owner = collections.defaultdict(collections.Counter)
    rivals = collections.defaultdict(set)
    for f in gt.frames:
        obs = [t for t in trk[f]["tracks"] if t["source"] == "OBSERVED"]
        for t in trk[f]["tracks"]:
            rivals[t["track_id"]] |= set(t["identity_rivals"])
        if not obs or not gt.players[f]:
            continue
        m, _, amb = _match(sorted(gt.players[f]), gt.players[f], [t["box"] for t in obs], NESTED_V2,
                           frozenset(gt.occluded.get(f, ())))
        for c, g in m.items():
            if c not in amb:
                owner[obs[c]["track_id"]][g] += 1
    pairs = set()
    for tid, rs in rivals.items():
        if tid in resolved or not owner[tid]:
            continue
        a = owner[tid].most_common(1)[0][0]
        for r in rs:
            if owner[r]:
                b = owner[r].most_common(1)[0][0]
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
    return pairs


def auc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    order = np.concatenate([pos, neg]).argsort().argsort() + 1.0
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(out_json, model_dir, *specs):
    cfg = load_config("config/default.yaml")
    md = Path(model_dir)
    emb = {
        "osnet": Embedder(md / "anriha_osnet_x0_25_msmt17__osnet_x0_25_msmt17.onnx", (256, 128), lambda y: y, 16),
        "youtu": Embedder(md / "opencv_person_reid_youtureid__person_reid_youtu_2021nov.onnx", (256, 128), lambda y: y),
        "dinov2": Embedder(md / "onnx-community_dinov2-small__model.onnx", (224, 112), lambda y: y[:, 0, :]),
    }
    cues = ["hist", *emb]
    clips = [load_clip(s, cfg) for s in specs]
    report = {"cues": cues, "clips": {}}
    for name, crops, contact, rivals in clips:
        imgs = [c["img"] for c in crops]
        feats = {k: e(imgs) for k, e in emb.items()}

        def sim(cue, a, b):
            if cue == "hist":
                return hist_sim(crops[a]["hist"], crops[b]["hist"])
            return float(feats[cue][a] @ feats[cue][b])

        by_player = collections.defaultdict(list)
        for k, c in enumerate(crops):
            by_player[c["player"]].append(k)
        hard = contact | rivals
        res = {"crops": len(crops), "players": len(by_player), "contact_pairs": len(contact),
               "uncertain_rival_pairs": sorted(rivals), "cue": {}}
        # pair sets (subsampled deterministically)
        rng = np.random.default_rng(0)
        same = [(a, b) for ks in by_player.values() for a, b in itertools.combinations(ks, 2)
                if abs(crops[a]["t"] - crops[b]["t"]) >= MIN_GAP_S]
        diff_all = [(a, b) for a, b in itertools.combinations(range(len(crops)), 2)
                    if crops[a]["player"] != crops[b]["player"]]
        diff_hard = [(a, b) for a, b in diff_all if tuple(sorted((crops[a]["player"], crops[b]["player"]))) in hard]
        hard_players = {p for pr in hard for p in pr}
        same_hard = [(a, b) for a, b in same if crops[a]["player"] in hard_players]
        pick = lambda xs, n: [xs[k] for k in rng.choice(len(xs), min(n, len(xs)), replace=False)] if xs else []  # noqa
        same, diff_all, diff_hard, same_hard = pick(same, 6000), pick(diff_all, 6000), pick(diff_hard, 6000), \
            pick(same_hard, 6000)
        # rival test triplets for every hard pair
        triplets = []
        for p, q in hard:
            for A, B in ((p, q), (q, p)):
                for qk in by_player.get(A, []):
                    ra = [k for k in by_player[A] if crops[qk]["t"] - crops[k]["t"] >= MIN_GAP_S]
                    rb = [k for k in by_player.get(B, []) if crops[qk]["t"] - crops[k]["t"] >= MIN_GAP_S]
                    if ra and rb:
                        triplets.append((qk, ra[-1], rb[-1], (p, q)))
        for cue in cues:
            ok = lambda a, b: crops[a]["hist"] is not None and crops[b]["hist"] is not None if cue == "hist" else True  # noqa
            S = [sim(cue, a, b) for a, b in same if ok(a, b)]
            D = [sim(cue, a, b) for a, b in diff_all if ok(a, b)]
            Sh = [sim(cue, a, b) for a, b in same_hard if ok(a, b)]
            Dh = [sim(cue, a, b) for a, b in diff_hard if ok(a, b)]
            r = {"auc_all": auc(S, D), "auc_hard": auc(Sh, Dh), "available": float(np.mean(
                [crops[k]["hist"] is not None for k in range(len(crops))])) if cue == "hist" else 1.0}
            thr = float(np.percentile(Dh, 99)) if Dh else None      # threshold with 1 % false accepts on hard pairs
            r["tpr_at_1pct_far_hard"] = float(np.mean(np.asarray(Sh) > thr)) if Sh and thr is not None else None
            tri = [(t, sim(cue, t[0], t[1]) - sim(cue, t[0], t[2])) for t in triplets
                   if ok(t[0], t[1]) and ok(t[0], t[2])]
            r["rival_test_accuracy"] = float(np.mean([m > 0 for _, m in tri])) if tri else None
            r["rival_test_n"] = len(tri)
            sd = float(np.std(S)) if S else None
            r["rival_margin_over_same_spread_median"] = float(np.median([m for _, m in tri]) / sd) if tri and sd else None
            for nm, sel in [(f"h{lo}-{hi}", lambda t, lo=lo, hi=hi: lo <= crops[t[0]]["h"] < hi) for lo, hi in HEIGHT_BINS] + \
                    [("query_occluded", lambda t: crops[t[0]]["occluded"]), ("query_visible", lambda t: not crops[t[0]]["occluded"])] + \
                    [(f"gap{lo}-{hi}s", lambda t, lo=lo, hi=hi: lo <= crops[t[0]]["t"] - crops[t[1]]["t"] < hi) for lo, hi in GAP_BINS]:
                sub = [m for t, m in tri if sel(t)]
                r[f"rival_acc_{nm}"] = (round(float(np.mean([m > 0 for m in sub])), 3), len(sub)) if sub else None
            r["rival_pair_accuracy"] = {f"{p}|{q}": round(float(np.mean([m > 0 for t, m in tri if t[3] == (p, q)])), 3)
                                        for p, q in sorted(rivals) if any(t[3] == (p, q) for t, _ in tri)}
            res["cue"][cue] = r
        # colour can't separate: hard pairs whose mean torso histograms are similar
        mean_hist = {p: np.mean([crops[k]["hist"] for k in ks if crops[k]["hist"] is not None], axis=0)
                     for p, ks in by_player.items() if any(crops[k]["hist"] is not None for k in ks)}
        res["hard_pairs_colour_similar_ge_0.7"] = sum(1 for p, q in hard if p in mean_hist and q in mean_hist and
                                                     hist_sim(mean_hist[p], mean_hist[q]) >= 0.7)
        res["hard_pairs"] = len(hard)
        # shirt numbers
        reads = [(c["player"], c["number"][0], c["number"][1], c["h"]) for c in crops]
        conf_reads = [(p, n, h) for p, n, cf, h in reads if n is not None and cf >= 60]
        maj = {p: collections.Counter(n for q, n, _ in conf_reads if q == p).most_common(1)[0]
               for p in {p for p, _, _ in conf_reads}}
        consistent = [n == maj[p][0] for p, n, _ in conf_reads if maj[p][1] >= 2]
        res["number"] = {"read_rate_conf60": round(len(conf_reads) / max(1, len(reads)), 4),
                         "read_rate_by_height": {f"{lo}-{hi}": round(np.mean([n is not None and cf >= 60 for _, n, cf, h in reads
                                                                         if lo <= h < hi]), 4) if any(lo <= h < hi for *_, h in reads) else None
                                                 for lo, hi in HEIGHT_BINS},
                         "players_with_>=2_reads": sum(1 for v in maj.values() if v[1] >= 2),
                         "agreement_with_player_majority": round(float(np.mean(consistent)), 3) if consistent else None,
                         "hard_pairs_both_numbered_and_different": sum(
                             1 for p, q in hard if p in maj and q in maj and maj[p][1] >= 2 and maj[q][1] >= 2
                             and maj[p][0] != maj[q][0])}
        report["clips"][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "cue"}))
        for cue in cues:
            r = res["cue"][cue]
            print(f"  {cue:7s} auc_all={r['auc_all'] and round(r['auc_all'],3)} auc_hard={r['auc_hard'] and round(r['auc_hard'],3)} "
                  f"tpr@1%far_hard={r['tpr_at_1pct_far_hard'] and round(r['tpr_at_1pct_far_hard'],3)} "
                  f"rival_acc={r['rival_test_accuracy'] and round(r['rival_test_accuracy'],3)} (n={r['rival_test_n']}) "
                  f"by_h={[r[f'rival_acc_h{lo}-{hi}'] for lo, hi in HEIGHT_BINS]} occ={r['rival_acc_query_occluded']} "
                  f"vis={r['rival_acc_query_visible']} gaps={[r[f'rival_acc_gap{lo}-{hi}s'] for lo, hi in GAP_BINS]} "
                  f"rival_pairs={r['rival_pair_accuracy']}")
    Path(out_json).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
