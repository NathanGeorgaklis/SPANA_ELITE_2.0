"""Train the SPANA re-ID embedding on crops from tools/reid_crops.py (pilot). Design: records/design/REID_MODEL_DESIGN.md.

    python -m tools.reid_train train  --crops CROPS_DIR [CROPS_DIR ...] --backbone HF_DINOV2_SMALL_DIR --out RUN_DIR
                                      [--config config/reid_train.yaml] [--device mps|cpu|cuda] [--max-minutes M]
    python -m tools.reid_train export --run RUN_DIR --out MODEL.onnx

Model: DINOv2 ViT-S/14 (the patch embedding and the first `model.frozen_blocks` blocks frozen), CLS token
concatenated with `model.parts` horizontal part pools of the patch tokens -> linear -> BN -> L2-normalised
`model.dim`-d embedding. Loss: ArcFace over the training identities + batch-hard triplet. PK batches drawn within
one match, with at least `train.same_team_share` of the identities from one side.

Input = the Embedder convention of tools/dev/appearance_cue_eval.py: RGB, /255, ImageNet mean / std, 224x112.
Crops whose split is "holdout" never enter training; after each epoch a holdout rival check (query of A vs a
reference of A and of a same-side B, both >= `train.check_min_gap_s` away) is logged (closed-set: a sanity check,
not the DEV result).

Resumable: a checkpoint after every epoch; `train` on an existing RUN_DIR continues from it (same config and crops
required). `--max-minutes` stops after the epoch that crosses the budget (Mac background commands stop at 2 h).
The backbone directory must hold the pinned weights: models/dinov2_small.json (sha256 of model.safetensors).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from tools.reid_crops import load_train_config

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PIN = Path(__file__).resolve().parents[1] / "models" / "dinov2_small.json"


# ---------------------------------------------------------------------------------------------- data (no torch)
def load_crops(dirs: Sequence[str | Path]) -> List[dict]:
    """crops.jsonl rows of one or more crop folders; identities are match-local: (match_id, player)."""
    rows = []
    for d in dirs:
        d = Path(d)
        meta = json.loads((d / "crops.json").read_text())
        if meta.get("split") != "TRAIN":
            raise SystemExit(f"{d}: refusing, crops are not from TRAIN labels")
        text = (d / "crops.jsonl").read_text()
        if hashlib.sha256(text.encode()).hexdigest() != meta["index_sha256"]:
            raise SystemExit(f"{d}: crops.jsonl does not match crops.json")
        for line in text.splitlines():
            r = json.loads(line)
            r["path"] = str(d / r["file"])
            r["match"] = meta["match_id"]
            r["identity"] = f"{meta['match_id']}/{r['player']}"
            r["team"] = f"{meta['match_id']}/{r['side']}"
            rows.append(r)
    return rows


class PKSampler:
    """P identities x K crops, all from one match; at least `same_team_share` of the identities from one side."""

    def __init__(self, rows: List[dict], p: int, k: int, same_team_share: float, seed: int):
        self.p, self.k, self.share = p, k, same_team_share
        self.rng = np.random.default_rng(seed)
        self.by_id: Dict[str, List[int]] = {}
        self.team: Dict[str, str] = {}
        self.match: Dict[str, str] = {}
        for i, r in enumerate(rows):
            self.by_id.setdefault(r["identity"], []).append(i)
            self.team[r["identity"]], self.match[r["identity"]] = r["team"], r["match"]
        self.matches = sorted(set(self.match.values()))

    def batch(self) -> List[int]:
        m = self.matches[self.rng.integers(len(self.matches))]
        ids = sorted(i for i in self.by_id if self.match[i] == m)
        teams: Dict[str, List[str]] = {}
        for i in ids:
            teams.setdefault(self.team[i], []).append(i)
        p = min(self.p, len(ids))
        team = sorted(teams)[self.rng.integers(len(teams))]
        n_team = min(len(teams[team]), max(1, int(math.ceil(self.share * p))))
        pick = list(self.rng.choice(teams[team], n_team, replace=False))
        rest = [i for i in ids if i not in pick]
        pick += list(self.rng.choice(rest, p - n_team, replace=False)) if p > n_team else []
        out = []
        for i in pick:
            pool = self.by_id[i]
            out += [int(x) for x in self.rng.choice(pool, self.k, replace=len(pool) < self.k)]
        return out


def augment(img: np.ndarray, a: dict, rng: np.random.Generator) -> np.ndarray:
    """BGR uint8 crop -> augmented BGR uint8 crop of the same size."""
    h, w = img.shape[:2]
    out = img
    if rng.random() < a["flip_p"]:
        out = out[:, ::-1]
    if rng.random() < a["downscale_p"] and h > a["downscale_min_h"]:
        nh = int(rng.integers(a["downscale_min_h"], h))
        nw = max(4, int(round(w * nh / h)))
        out = cv2.resize(cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < a["blur_p"]:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.5)))
    if rng.random() < a["jpeg_p"]:
        q = int(rng.integers(a["jpeg_min_quality"], 95))
        out = cv2.imdecode(cv2.imencode(".jpg", np.ascontiguousarray(out), [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    hsv = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-a["hue_deg"], a["hue_deg"]) / 2.0) % 180.0   # OpenCV hue = deg / 2
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out = (out - out.mean()) * rng.uniform(1 - a["contrast"], 1 + a["contrast"]) + out.mean()
    out = out * rng.uniform(1 - a["brightness"], 1 + a["brightness"])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < a["erase_p"]:
        area = rng.uniform(0.02, a["erase_max_area"]) * h * w
        ratio = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
        eh, ew = int(min(h, round(math.sqrt(area * ratio)))), int(min(w, round(math.sqrt(area / ratio))))
        y, x = int(rng.integers(0, h - eh + 1)), int(rng.integers(0, w - ew + 1))
        out[y:y + eh, x:x + ew] = rng.integers(0, 256, (eh, ew, 3), dtype=np.uint8)
    return out


def to_input(imgs_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """BGR uint8 crops (already model size) -> N x 3 x H x W float32, the appearance_cue_eval Embedder convention."""
    return np.stack([((im[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                     for im in imgs_bgr]).astype(np.float32)


def rival_triplets(rows: List[dict], min_gap_s: float) -> List[tuple]:
    """Holdout rival check: for each query crop of A, the nearest-in-time reference of A and of a same-side B, both
    >= min_gap_s from the query (deterministic: B = every same-side identity with a reference)."""
    out = []
    by_id: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_id.setdefault(r["identity"], []).append(i)
    for q, rq in enumerate(rows):
        def near(ident):
            c = [i for i in by_id[ident] if abs(rows[i]["t"] - rq["t"]) >= min_gap_s]
            return min(c, key=lambda i: abs(rows[i]["t"] - rq["t"])) if c else None
        a = near(rq["identity"])
        if a is None:
            continue
        for ident in sorted(by_id):
            if ident != rq["identity"] and rows[by_id[ident][0]]["team"] == rq["team"]:
                b = near(ident)
                if b is not None:
                    out.append((q, a, b))
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_backbone(d: Path) -> None:
    pin = json.loads(PIN.read_text())
    got = sha256_file(d / pin["weights_file"])
    if got != pin["weights_sha256"]:
        raise SystemExit(f"backbone weights {got} do not match the pin {pin['weights_sha256']} ({PIN})")


# ---------------------------------------------------------------------------------------------- model (torch)
def build(backbone_dir: str | Path, n_ids: int, mc: dict, tc: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import Dinov2Model

    class Embed(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Dinov2Model.from_pretrained(str(backbone_dir))
            d = self.backbone.config.hidden_size
            self.patch = self.backbone.config.patch_size
            self.parts = mc["parts"]
            self.proj = nn.Linear(d * (1 + self.parts), mc["dim"], bias=False)
            self.bn = nn.BatchNorm1d(mc["dim"])
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
            for blk in self.backbone.encoder.layer[:mc["frozen_blocks"]]:
                for p in blk.parameters():
                    p.requires_grad = False

        def forward(self, x):
            hs = self.backbone(pixel_values=x).last_hidden_state
            gh, gw = int(x.shape[2]) // self.patch, int(x.shape[3]) // self.patch   # fixed input size
            tok = hs[:, 1:1 + gh * gw].reshape(x.shape[0], gh, gw, -1)
            bounds = np.linspace(0, gh, self.parts + 1).round().astype(int)
            pools = [tok[:, bounds[i]:bounds[i + 1]].mean(dim=(1, 2)) for i in range(self.parts)]
            z = self.bn(self.proj(torch.cat([hs[:, 0], *pools], dim=1)))
            return F.normalize(z, dim=1)

    class ArcFace(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(n_ids, mc["dim"]) * 0.01)
            self.s, self.m = tc["arcface_s"], tc["arcface_m"]

        def forward(self, z, y):
            cos = F.linear(z, F.normalize(self.w, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            logits = torch.where(F.one_hot(y, cos.shape[1]).bool(), torch.cos(theta + self.m), cos) * self.s
            return F.cross_entropy(logits, y)

    return Embed(), ArcFace()


def batch_hard_triplet(z, y, margin: float):
    import torch
    d = torch.cdist(z, z)
    same = y[:, None] == y[None, :]
    pos = (d * same.float()).max(dim=1).values
    neg = (d + same.float() * 1e6).min(dim=1).values
    return torch.relu(pos - neg + margin).mean()


def embed_all(model, rows, idx, device, bs: int = 64) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            x = torch.from_numpy(to_input([cv2.imread(rows[i]["path"]) for i in idx[k:k + bs]])).to(device)
            out.append(model(x).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def holdout_check(model, rows, hold, triplets, device) -> dict:
    if not triplets:
        return {"holdout_rival_acc": None, "holdout_rival_n": 0}
    pos = {i: k for k, i in enumerate(hold)}
    z = embed_all(model, rows, hold, device)
    m = [float(z[pos[q]] @ z[pos[a]] - z[pos[q]] @ z[pos[b]]) for q, a, b in triplets]
    return {"holdout_rival_acc": round(float(np.mean([x > 0 for x in m])), 4), "holdout_rival_n": len(m)}


def train(a) -> int:
    import torch
    cfg_bytes = Path(a.config).read_bytes()
    cfg = load_train_config(a.config)
    tc, mc, ac = cfg["train"], cfg["model"], cfg["augment"]
    check_backbone(Path(a.backbone))
    rows = load_crops(a.crops)
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    hold = [i for i, r in enumerate(rows) if r["split"] == "holdout"]
    ids = sorted({rows[i]["identity"] for i in tr})
    label = {k: n for n, k in enumerate(ids)}
    hold = [i for i in hold if rows[i]["identity"] in label]
    hold_rows = [rows[i] for i in hold]
    triplets = [(hold[q], hold[x], hold[y]) for q, x, y in rival_triplets(hold_rows, tc["check_min_gap_s"])]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run_meta = {"config_sha256": hashlib.sha256(cfg_bytes).hexdigest(), "config": cfg,
                "crops": [json.loads((Path(d) / "crops.json").read_text())["index_sha256"] for d in a.crops],
                "crop_dirs": [str(d) for d in a.crops], "backbone": json.loads(PIN.read_text()),
                "identities": ids, "train_crops": len(tr), "holdout_crops": len(hold),
                "holdout_triplets": len(triplets)}
    meta_p = out / "run.json"
    if meta_p.exists():
        old = json.loads(meta_p.read_text())
        if {k: old[k] for k in ("config_sha256", "crops", "identities")} != \
                {k: run_meta[k] for k in ("config_sha256", "crops", "identities")}:
            raise SystemExit(f"{out} holds a run with another config or other crops")
    else:
        meta_p.write_text(json.dumps(run_meta, indent=1, sort_keys=True))
    device = torch.device(a.device)
    torch.manual_seed(tc["seed"])
    model, head = build(a.backbone, len(ids), mc, tc)
    model.to(device), head.to(device)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    rest = list(model.proj.parameters()) + list(model.bn.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW([{"params": bb, "lr": tc["lr_backbone"]}, {"params": rest, "lr": tc["lr_head"]}],
                            weight_decay=tc["weight_decay"])
    total = tc["epochs"] * tc["batches_per_epoch"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 *
                                              (1 + math.cos(math.pi * min(s, total) / total)))
    start = 0
    ck = out / "checkpoint.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict(st["model"]), head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        start = st["epoch"]
    sampler = PKSampler([rows[i] for i in tr], tc["p"], tc["k"], tc["same_team_share"], tc["seed"] + start)
    rng = np.random.default_rng(tc["seed"] + 1000 + start)
    log = out / "log.jsonl"
    if start == 0 and not log.exists():
        log.write_text(json.dumps({"epoch": 0, **holdout_check(model, rows, hold, triplets, device)}) + "\n")
    t0 = time.time()
    model.train()
    for ep in range(start, tc["epochs"]):
        la = lt = 0.0
        for _ in range(tc["batches_per_epoch"]):
            b = [tr[i] for i in sampler.batch()]
            x = torch.from_numpy(to_input([augment(cv2.imread(rows[i]["path"]), ac, rng) for i in b])).to(device)
            y = torch.tensor([label[rows[i]["identity"]] for i in b], device=device)
            z = model(x)
            l_arc, l_tri = head(z, y), batch_hard_triplet(z, y, tc["triplet_margin"])
            loss = l_arc + tc["triplet_weight"] * l_tri
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            la, lt = la + float(l_arc.detach()), lt + float(l_tri.detach())
        n = tc["batches_per_epoch"]
        rec = {"epoch": ep + 1, "arcface": round(la / n, 4), "triplet": round(lt / n, 4),
               "minutes": round((time.time() - t0) / 60, 2), **holdout_check(model, rows, hold, triplets, device)}
        tmp = out / "checkpoint.pt.tmp"
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1}, tmp)
        tmp.replace(ck)
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if a.max_minutes and (time.time() - t0) / 60 >= a.max_minutes and ep + 1 < tc["epochs"]:
            print(json.dumps({"stopped": "max_minutes", "epoch": ep + 1}))
            return 0
    print(json.dumps({"done": True, "epochs": tc["epochs"]}))
    return 0


def export(a) -> int:
    import onnxruntime as ort
    import torch
    run = Path(a.run)
    meta = json.loads((run / "run.json").read_text())
    st = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    if st["epoch"] != meta["config"]["train"]["epochs"]:
        raise SystemExit(f"run not finished: epoch {st['epoch']} of {meta['config']['train']['epochs']}")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    model, _ = build(a.backbone, len(meta["identities"]), meta["config"]["model"], meta["config"]["train"])
    model.load_state_dict(st["model"])
    model.eval()
    h, w = meta["config"]["crops"]["size_hw"]
    x = torch.randn(4, 3, h, w)
    torch.onnx.export(model, (x,), str(out), input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    card = {"onnx_sha256": sha256_file(out), "max_abs_diff_torch_onnx": err, "run": meta, "epochs": st["epoch"]}
    Path(str(out) + ".json").write_text(json.dumps(card, indent=1, sort_keys=True))
    print(json.dumps({"onnx": str(out), "sha256": card["onnx_sha256"], "max_abs_diff": err}))
    if err > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--crops", nargs="+", required=True)
    t.add_argument("--backbone", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config", default="config/reid_train.yaml")
    t.add_argument("--device", default="cpu")
    t.add_argument("--max-minutes", type=float, default=0.0)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--backbone", required=True)
    e.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return train(a) if a.cmd == "train" else export(a)


if __name__ == "__main__":
    sys.exit(main())
