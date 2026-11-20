"""Speed / agreement probe of the local detector's model stage (DEV only; measurement, not product code).

    python -m tools.detector_speed_probe --video V --start S --frames N --model-record models/rfdetr_v1.json \
        --device mps --variant fp32:1 --variant fp32:4 --variant fp16:1 --out PROBE.json

Each variant is `<precision>:<batch frames>[:compile]` (compile = torch.compile of the network). The first variant is the reference. For every variant: model ms per
frame (median of per-batch wall time / batch, after warm-up batches that are not timed), and the raw-output
agreement with the reference on the same frames (max |logit| and |box| difference, and the share of reference
decoded boxes >= `--score` reproduced at IoU >= `--iou` with the same class). Frames are read once and kept in memory,
so decode is not timed here.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from spana.detection.rfdetr_local import LocalModelRecord, decode, preprocess
from spana.geometry import iou_matrix
from spana.video import VideoSource


def _load_net(rec: LocalModelRecord, device: str):
    import rfdetr
    import torch
    cls = getattr(rfdetr, f"RFDETR{rec.variant.capitalize()}")
    model = cls(pretrain_weights=str(rec.file("weights")), num_classes=rec.num_classes, resolution=rec.resolution,
                device=device)
    return torch, model.model.model.eval().to(device)


def run_variant(torch, net, xs, device: str, precision: str, batch: int, warmup: int, compiled: bool = False):
    dtype = {"fp32": torch.float32, "fp16": torch.float16}[precision]
    net = net.to(dtype)
    fn = torch.compile(net) if compiled else net
    outs, per_frame = [], []
    batches = [xs[i:i + batch] for i in range(0, len(xs), batch)]
    with torch.no_grad():
        for k, b in enumerate(batches):
            t = time.perf_counter()
            x = torch.from_numpy(np.concatenate(b)).to(device=device, dtype=dtype)
            o = fn(x)
            boxes = o["pred_boxes"].float().cpu().numpy()
            logits = o["pred_logits"].float().cpu().numpy()
            dt = time.perf_counter() - t
            if k >= warmup and len(b) == batch:
                per_frame.append(dt / len(b))
            outs.extend(zip(boxes, logits))
    net.to(torch.float32)
    return outs, per_frame


def agreement(ref, got, w: int, h: int, cmap, score: float, iou_min: float) -> dict:
    dl = max(float(np.abs(a[1] - b[1]).max()) for a, b in zip(ref, got))
    db = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(ref, got))
    n_ref = n_hit = 0
    for (rb, rl), (gb, gl) in zip(ref, got):
        r = decode(rb, rl, w, h, score, cmap)
        g = decode(gb, gl, w, h, score, cmap)
        n_ref += len(r)
        if not r or not g:
            continue
        m = iou_matrix(np.array([d.box for d in r]), np.array([d.box for d in g]))
        for i, d in enumerate(r):
            ok = [j for j in range(len(g)) if m[i, j] >= iou_min and g[j].class_id == d.class_id]
            n_hit += bool(ok)
    return {"max_abs_logit_diff": round(dl, 6), "max_abs_box_diff": round(db, 6), "ref_boxes": n_ref,
            "ref_boxes_reproduced": n_hit, "reproduced_share": round(n_hit / n_ref, 6) if n_ref else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--model-record", required=True)
    p.add_argument("--device", choices=("mps", "cpu"), required=True)
    p.add_argument("--variant", action="append", required=True, help="<fp32|fp16>:<batch frames>; first = reference")
    p.add_argument("--warmup", type=int, default=2, help="untimed batches per variant")
    p.add_argument("--score", type=float, default=0.5, help="decoded boxes compared at this score")
    p.add_argument("--iou", type=float, default=0.9)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rec = LocalModelRecord.load(a.model_record)
    src = VideoSource(a.video)
    xs, size = [], None
    for i, t, img in src.frames(a.start, a.start + a.frames):
        xs.append(preprocess(img, rec.resolution))
        size = img.shape[1], img.shape[0]
    torch, net = _load_net(rec, a.device)
    cmap = list(range(rec.num_classes + 1))
    res, ref = [], None
    for v in a.variant:
        prec, b, *opt = v.split(":")
        outs, pf = run_variant(torch, net, xs, a.device, prec, int(b), a.warmup, compiled=opt == ["compile"])
        row = {"variant": v, "model_ms_per_frame_median": round(1000 * statistics.median(pf), 2),
               "model_ms_per_frame_mean": round(1000 * statistics.mean(pf), 2), "timed_batches": len(pf)}
        if ref is None:
            ref = outs
        else:
            row["agreement_vs_" + a.variant[0]] = agreement(ref, outs, size[0], size[1], cmap, a.score, a.iou)
        res.append(row)
        print(json.dumps(row), flush=True)
    doc = {"video_sha256": src.meta.sha256, "start": a.start, "frames": len(xs), "device": a.device,
           "torch": torch.__version__, "resolution": rec.resolution, "argv": sys.argv, "variants": res}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
