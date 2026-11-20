"""Detector speed options (DEV only; measurement, not product code). Limits: config/default.yaml `performance.option_*`.

Stage 1, every variant (one loaded window of video frames + the frozen DEV benchmark):
    python -m tools.detector_options speed --video V --start S --frames N --profile-summary P.json \
        --variant medium:1280x1280 --variant medium:960x960 --variant medium:1280x736 \
        --coco-weights small=PATH --variant small:960x544 --out OPTIONS.json

  A variant is `<rf-detr variant>:<W>x<H>` (W, H divisible by 32). The first variant is the reference (the current
  detector). Each variant runs exactly as the product runs the local detector (spana.detection.rfdetr_local
  TorchRunner with config `inference`), built at resolution max(W, H) (RF-DETR resizes the positional embedding to
  that grid at load and interpolates it to the input at run time).
  * Speed: model ms per frame (median, after untimed warm-up frames) and preprocess ms; the projected pipeline
    real-time factor = the profile summary's other per-frame stages + this variant's model + preprocess.
  * The model record's own variant uses the SPANA weights: person boxes (config detector.person_classes) and ball
    boxes >= detection.score_high of the reference reproduced at IoU >= 0.5 per frame (by box height), and the
    frozen DEV benchmark scored with tools/dev_bench_score.score_frames (images from the sha-verified cache) plus
    persons found (class ignored, >= detection.score_high, IoU >= detector_eval.match_iou).
  * Other variants (small / nano) have no SPANA weights: speed only, with the public COCO checkpoint given by
    --coco-weights (sha256 recorded; its class head is rebuilt for our class count, which does not change speed).
  * `stage2_candidates`: variants of the record's own model under `performance.option_realtime_factor_max` whose
    bench persons found drop at most `performance.option_recall_max_drop`, fastest first.

Stage 2, per candidate (the full pipeline on labelled DEV clips):
    python -m tools.detector_options record --model-record models/rfdetr_v1.json --width W --height H --out REC.json
      A DEV model record: the pinned record with another input size (same weights, sha-checked). Square only: the
      product record has one `resolution`.
    python -m tools.detector_options judge --base-dir D --tag TAG [--base base] --out JUDGE.json
      Reads <D>/<clip>_<tag>.bench.json + <D>/target_<tag>_reviewer.json (tools/benchmark.py, tools/target_score.py
      --simulate-reviewer; run_labelled.sh layout) and checks them against the base by `performance.option_*`.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from spana.config import load_config
from spana.detection.rfdetr_local import LocalModelRecord, TorchRunner, decode, preprocess
from spana.evaluation import HEIGHT_BINS_PX
from spana.geometry import iou_matrix

REPO = Path(__file__).resolve().parent.parent
SCHEMA = ("spana.detector_options", "1.0.0")
STRIDE = 32          # RF-DETR medium / small / nano: patch 16 x 2 windows
AGREE_IOU = 0.5      # reproduction of a reference box under another input size (DEV convention, reporting)


@dataclasses.dataclass(frozen=True)
class Variant:
    model: str
    width: int
    height: int

    @property
    def name(self) -> str:
        return f"{self.model}:{self.width}x{self.height}"


def parse_variant(text: str) -> Variant:
    try:
        model, size = text.split(":")
        w, h = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"variant {text!r}: expected <model>:<W>x<H>") from None
    if w <= 0 or h <= 0 or w % STRIDE or h % STRIDE:
        raise SystemExit(f"variant {text!r}: W and H must be positive multiples of {STRIDE}")
    return Variant(model, w, h)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def variant_record(rec: LocalModelRecord, v: Variant, coco: Dict[str, Path]) -> LocalModelRecord:
    """The record TorchRunner builds this variant from: the pinned record at resolution max(W, H), or (another RF-DETR
    variant) the given COCO checkpoint, sha256 computed here."""
    if v.model == rec.variant:
        return dataclasses.replace(rec, resolution=max(v.width, v.height))
    if v.model not in coco:
        raise SystemExit(f"variant {v.name}: no SPANA weights; pass --coco-weights {v.model}=PATH (speed only)")
    p = coco[v.model].resolve()
    return dataclasses.replace(rec, variant=v.model, resolution=max(v.width, v.height), weights_path=str(p),
                               weights_sha256=_sha256(p))


def _by_height(h: float) -> str:
    for lo, hi in zip(HEIGHT_BINS_PX, HEIGHT_BINS_PX[1:]):
        if lo <= h < hi:
            return f"{lo}-{hi if hi < 1_000_000 else 'inf'}"
    return "?"


def reproduced(ref: List[dict], got: List[dict], classes: Sequence[str], score: float, iou_min: float) -> dict:
    """Share of reference boxes (class in `classes`, score >= `score`) with a box of the same group at that score and
    IoU >= iou_min in the same frame, overall and by reference box height (px)."""
    tot: Dict[str, List[int]] = {}
    for r, g in zip(ref, got):
        rb = [d["box"] for d in r if d["class"] in classes and d["score"] >= score]
        gb = [d["box"] for d in g if d["class"] in classes and d["score"] >= score]
        hit = (iou_matrix(np.array(rb), np.array(gb)).max(axis=1) >= iou_min) if rb and gb else np.zeros(len(rb), bool)
        for b, ok in zip(rb, hit):
            for k in ("all", _by_height(b[3] - b[1])):
                t = tot.setdefault(k, [0, 0])
                t[0] += 1
                t[1] += bool(ok)
    return {k: {"ref_boxes": n, "reproduced": m, "share": round(m / n, 4)} for k, (n, m) in sorted(tot.items())}


def persons_found(frames: List[dict], persons: Sequence[str], score: float, iou_min: float) -> dict:
    """Labelled persons (class ignored) matched by a person-class prediction >= score at IoU >= iou_min."""
    n = m = 0
    for f in frames:
        gt = [b["box"] for b in f["gt"] if b["class"] in persons]
        pr = [p["box"] for p in f["pred"] if p["class"] in persons and p["score"] >= score]
        n += len(gt)
        if gt and pr:
            m += int((iou_matrix(np.array(gt), np.array(pr)).max(axis=1) >= iou_min).sum())
    return {"labels": n, "found": m, "share": round(m / n, 4) if n else None}


class Net:
    """One variant: preprocess at W x H, TorchRunner, decode to {class, box, score} dicts."""

    def __init__(self, rec: LocalModelRecord, v: Variant, device: str, precision: str, compile: bool,
                 class_names: Sequence[str]):
        self.v, self.rec = v, rec
        self.run = TorchRunner(rec, device, precision, compile)
        self.names = tuple(class_names)
        self.map = [self.names.index(c) if c else -1 for c in rec.output_classes]

    def __call__(self, img: np.ndarray, floor: float):
        t0 = time.perf_counter()
        x = preprocess(img, self.v.width, self.v.height)
        t1 = time.perf_counter()
        boxes, logits = self.run(x)
        t2 = time.perf_counter()
        dets = decode(boxes, logits, img.shape[1], img.shape[0], floor, self.map)
        out = [{"class": self.names[d.class_id], "box": [round(b, 2) for b in d.box], "score": round(d.score, 4)}
               for d in dets]
        return out, t1 - t0, t2 - t1


def bench_frames(dcfg, cache_dir: Path) -> tuple:
    """(frozen DEV benchmark label frames, pixels by id), pixels read from the cache and verified by sha256."""
    from tools.dev_bench_score import load_frozen
    labels, frozen = load_frozen(dcfg, REPO)
    px = {}
    for e in labels["frames"]:
        f = cache_dir / f"{e['id']}.jpg"
        data = f.read_bytes() if f.is_file() else b""
        if hashlib.sha256(data).hexdigest() != e["image_sha256"]:
            raise SystemExit(f"bench image {e['name']}: not in {cache_dir} or sha256 mismatch (run dev_bench_score once)")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or (img.shape[1], img.shape[0]) != (e["width"], e["height"]):
            raise SystemExit(f"bench image {e['name']}: does not decode at the labelled size")
        px[e["id"]] = img
    return labels, frozen, px


def score_bench(dcfg, net: Net, labels: dict, px: dict, persons: Sequence[str], score_high: float) -> dict:
    from spana.dataset.prefilter import grass_mask
    from tools.dev_bench_score import score_frames
    frames = []
    for e in labels["frames"]:
        pred, _, _ = net(px[e["id"]], dcfg.detector_eval.request_confidence)
        frames.append({"name": e["name"], "id": e["id"], "gt": e["boxes"], "pred": pred, "match_id": e["match_id"]})
    masks = {i: grass_mask(p, dcfg.prelabel.grass) for i, p in px.items()}
    heights = {i: p.shape[0] for i, p in px.items()}
    res = score_frames(dcfg, frames, masks, heights)["overall"]
    keep = {k: res[k] for k in ("map50", "map50_95", "per_class") if k in res}
    keep["ball_recall_by_short_side_px"] = res.get("recall_by_short_side_px")
    keep["persons_found"] = persons_found(frames, persons, score_high, dcfg.detector_eval.match_iou)
    return keep


def cmd_speed(a) -> int:
    from spana.dataset.config import load_dataset_config
    from spana.video import VideoSource
    cfg, dcfg = load_config(a.config), load_dataset_config(a.dataset_config)
    perf, inf = cfg.performance, cfg.inference
    rec = LocalModelRecord.load(a.model_record)
    variants = [parse_variant(v) for v in a.variant]
    if variants[0].model != rec.variant:
        raise SystemExit("the first variant (reference) must be the model record's own variant")
    coco = {}
    for c in a.coco_weights or []:
        k, _, p = c.partition("=")
        coco[k] = Path(p)
    summ = json.loads(Path(a.profile_summary).read_text())
    st = {s["stage"]: s["ms_per_frame"] for s in summ["stages"] if s["pass"] == "detect"}
    other_ms = summ["ms_per_frame"] - st["detect.model"] - st["detect.preprocess"]
    fps = summ["fps"]
    src = VideoSource(a.video)
    imgs = [img for _, _, img in src.frames(a.start, a.start + a.frames)]
    persons = tuple(dcfg.detector.person_classes)
    labels = frozen = px = None
    if not a.no_bench:
        labels, frozen, px = bench_frames(dcfg, REPO / dcfg.paths.image_cache_dir)
    rows, ref_out = [], None
    for v in variants:
        vrec = variant_record(rec, v, coco)
        net = Net(vrec, v, a.device, inf.mps_precision, inf.mps_compile, dcfg.classes)
        outs, pre, mod = [], [], []
        for k, img in enumerate(imgs):
            o, tp, tm = net(img, cfg.detection.score_low)
            if k >= a.warmup:
                pre.append(tp)
                mod.append(tm)
            outs.append(o)
        model_ms, pre_ms = 1000 * statistics.median(mod), 1000 * statistics.median(pre)
        proj_ms = other_ms + model_ms + pre_ms
        rtf = proj_ms / 1000.0 * fps
        row = {"variant": v.name, "weights": "spana" if v.model == rec.variant else "coco (speed only)",
               "weights_sha256": vrec.weights_sha256, "built_at_resolution": vrec.resolution,
               "describe": net.run.describe, "model_ms_per_frame": round(model_ms, 2),
               "preprocess_ms_per_frame": round(pre_ms, 2), "timed_frames": len(mod),
               "projected_pipeline_ms_per_frame": round(proj_ms, 2), "projected_realtime_factor": round(rtf, 3),
               "projected_hours": round(rtf * perf.projection_footage_s / 3600.0, 2),
               "meets_option_target": rtf <= perf.option_realtime_factor_max}
        if v.model == rec.variant:
            if ref_out is None:
                ref_out = outs
            else:
                row["vs_reference"] = {
                    "persons": reproduced(ref_out, outs, persons, cfg.detection.score_high, AGREE_IOU),
                    "ball": reproduced(ref_out, outs, [c for c in dcfg.classes if c not in persons],
                                       cfg.detection.score_high, AGREE_IOU)}
            if labels is not None:
                row["dev_bench"] = score_bench(dcfg, net, labels, px, persons, cfg.detection.score_high)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("variant", "model_ms_per_frame", "projected_realtime_factor")}),
              flush=True)
        del net
    ref_found = (rows[0].get("dev_bench") or {}).get("persons_found", {}).get("share")
    cands = []
    for r in rows[1:]:
        found = (r.get("dev_bench") or {}).get("persons_found", {}).get("share")
        if r["weights"] != "spana" or found is None or ref_found is None:
            continue
        r["bench_persons_found_drop"] = round(ref_found - found, 4)
        if r["meets_option_target"] and ref_found - found <= perf.option_recall_max_drop:
            cands.append(r)
    doc = {"schema": SCHEMA[0], "schema_version": SCHEMA[1], "argv": sys.argv, "video_sha256": src.meta.sha256,
           "start": a.start, "frames": len(imgs), "device": a.device, "inference": dataclasses.asdict(inf),
           "profile_summary": a.profile_summary, "other_stages_ms_per_frame": round(other_ms, 2), "fps": fps,
           "limits": dataclasses.asdict(perf), "config_sha256": cfg.sha256(),
           "dev_bench_labels_sha256": frozen["labels_sha256"] if frozen else None, "variants": rows,
           "stage2_candidates": [r["variant"] for r in sorted(cands, key=lambda r: r["model_ms_per_frame"])]}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print("stage2 candidates:", doc["stage2_candidates"])
    return 0


def cmd_record(a) -> int:
    if a.width != a.height:
        raise SystemExit("the product model record has one square `resolution`; non-square needs a record change")
    if a.width % STRIDE:
        raise SystemExit(f"resolution must be a multiple of {STRIDE}")
    src = Path(a.model_record).resolve()
    rec = LocalModelRecord.load(src)
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    data = json.loads(src.read_text())
    for k in ("weights_path", "onnx_path"):
        data[k] = str(Path(rec.base_dir, data[k]).resolve().relative_to(out.parent, walk_up=True))
    data.update(resolution=a.width, version=f"{rec.version}-dev-r{a.width}",
                name=f"{rec.name} [DEV speed option: input {a.width}]",
                validation=f"DEV speed option only (tools/detector_options.py), derived from {src.name}; the ONNX file "
                           f"is the {rec.resolution} export and is not used by the torch runtime. Not validated.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    LocalModelRecord.load(out).file("weights")
    print(out)
    return 0


def _clip_of(path: str, tag: str) -> str:
    return Path(path).name[: -len(f"_{tag}.bench.json")]


def judge(base: Dict[str, dict], cand: Dict[str, dict], tbase: dict, tcand: dict, perf) -> dict:
    """base / cand: clip -> tools/benchmark.py output; tbase / tcand: tools/target_score.py --simulate-reviewer."""
    if sorted(base) != sorted(cand):
        raise SystemExit(f"clips differ: {sorted(base)} vs {sorted(cand)}")
    def tot(d, key):
        return sum(d[c]["summary"][key] for c in d)
    def recall(d):
        n = sum(d[c]["summary"]["gt_boxes"] for c in d)
        return sum(d[c]["summary"]["detection_recall"] * d[c]["summary"]["gt_boxes"] for c in d) / n
    def wrong(t):
        return sum(c["pooled"]["wrong"] for c in t["clips"])
    def correct_time(t):
        f = sum(c["pooled"]["followed_s"] for c in t["clips"])
        return sum(c["pooled"]["correct_time"] * c["pooled"]["followed_s"] for c in t["clips"]) / f
    sw_b, sw_c = tot(base, "id_switches"), tot(cand, "id_switches")
    checks = {
        "wrong_frames_reviewer": {"base": wrong(tbase), "candidate": wrong(tcand),
                                  "max_increase": perf.option_wrong_frames_max_increase},
        "unflagged_transfers": {"base": tot(base, "unflagged_transfers"), "candidate": tot(cand, "unflagged_transfers"),
                                "max_increase": perf.option_unflagged_transfers_max_increase},
        "id_switches": {"base": sw_b, "candidate": sw_c, "rise": round((sw_c - sw_b) / sw_b, 4) if sw_b else None,
                        "max_rise": perf.option_id_switches_max_rise},
        "detection_recall": {"base": round(recall(base), 4), "candidate": round(recall(cand), 4),
                             "max_drop": perf.option_recall_max_drop},
    }
    checks["wrong_frames_reviewer"]["ok"] = (checks["wrong_frames_reviewer"]["candidate"] - checks["wrong_frames_reviewer"]
                                             ["base"]) <= perf.option_wrong_frames_max_increase
    u = checks["unflagged_transfers"]
    u["ok"] = u["candidate"] - u["base"] <= perf.option_unflagged_transfers_max_increase
    s = checks["id_switches"]
    s["ok"] = sw_c <= sw_b * (1.0 + perf.option_id_switches_max_rise)
    r = checks["detection_recall"]
    r["ok"] = r["base"] - r["candidate"] <= perf.option_recall_max_drop + 1e-12
    per_clip = {c: {k: (base[c]["summary"][k], cand[c]["summary"][k])
                    for k in ("id_switches", "fragments_total", "identity_transfers", "unflagged_transfers",
                              "detection_recall", "track_coverage", "missed_players")} for c in sorted(base)}
    info = {"reviewer_correct_time": (round(correct_time(tbase), 4), round(correct_time(tcand), 4)),
            "identity_transfers": (tot(base, "identity_transfers"), tot(cand, "identity_transfers")),
            "fragments": (tot(base, "fragments_total"), tot(cand, "fragments_total"))}
    return {"checks": checks, "viable": all(c["ok"] for c in checks.values()), "per_clip_base_vs_candidate": per_clip,
            "info_base_vs_candidate": info}


def cmd_judge(a) -> int:
    perf = load_config(a.config).performance
    d = Path(a.base_dir)
    def load(tag):
        return {_clip_of(p, tag): json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(d / f"*_{tag}.bench.json")))}
    base, cand = load(a.base), load(a.tag)
    if not base:
        raise SystemExit(f"no {a.base} bench files in {d}")
    tb = json.loads((d / f"target_{a.base}_reviewer.json").read_text())
    tc = json.loads((d / f"target_{a.tag}_reviewer.json").read_text())
    doc = {"schema": "spana.detector_option_judge", "schema_version": "1.0.0", "argv": sys.argv, "base": a.base,
           "candidate": a.tag, "limits": dataclasses.asdict(perf), **judge(base, cand, tb, tc, perf)}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps({"viable": doc["viable"], **{k: v["ok"] for k, v in doc["checks"].items()}}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(REPO / "config" / "default.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speed")
    s.add_argument("--dataset-config", default=str(REPO / "config" / "roboflow.yaml"))
    s.add_argument("--model-record", default=str(REPO / "models" / "rfdetr_v1.json"))
    s.add_argument("--video", required=True)
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--frames", type=int, required=True)
    s.add_argument("--device", choices=("mps", "cpu"), default="mps")
    s.add_argument("--variant", action="append", required=True, help="<model>:<W>x<H>; first = reference")
    s.add_argument("--coco-weights", action="append", help="<model>=<checkpoint path> for speed-only variants")
    s.add_argument("--profile-summary", required=True, help="tools/identity_speed.py profile output (other stages)")
    s.add_argument("--warmup", type=int, default=3, help="untimed frames per variant (compile happens here)")
    s.add_argument("--no-bench", action="store_true", help="skip the DEV benchmark")
    s.add_argument("--out", required=True)
    r = sub.add_parser("record")
    r.add_argument("--model-record", default=str(REPO / "models" / "rfdetr_v1.json"))
    r.add_argument("--width", type=int, required=True)
    r.add_argument("--height", type=int, required=True)
    r.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--base-dir", required=True)
    j.add_argument("--base", default="base")
    j.add_argument("--tag", required=True)
    j.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"speed": cmd_speed, "record": cmd_record, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
