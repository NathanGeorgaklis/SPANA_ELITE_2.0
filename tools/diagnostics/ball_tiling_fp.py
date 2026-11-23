"""Diagnose what the TILED ball detector mistook for the ball (analysis only; nothing here changes the pipeline).

Inputs are the outputs of an existing tiled run (records/ball_tiling_v1_2026-11-23): per clip the tiled ball-view
detections, the ball document and the ball path, plus the frozen DEV ball labels and the untiled ball-view
detections of the same clip. DEV only (the labels' split is checked). DEV frames are never written anywhere but the
local contact sheets; nothing here feeds a training set.

    # 1. analyse + re-run the detector on the false-positive frames only + contact sheets (needs the video)
    python -m tools.diagnostics.ball_tiling_fp inspect --labels data/ball_labels/<clip>/v1 --video CLIP.mp4 \
        --tiled-dir RUN/tile --baseline-balldet RUN/<clip>.balldet.json --config TILING_ON.yaml \
        --dataset-config config/roboflow.yaml --model-record models/rfdetr_v1.json --runtime torch --device mps \
        --out OUT_DIR
    # 2. look at OUT_DIR/<clip>_fp_sheet_*.jpg, fill the `category` column of OUT_DIR/<clip>.categories.csv
    # 3. tables
    python -m tools.diagnostics.ball_tiling_fp summarise --out OUT_DIR [--clip <clip> ...]

Definitions (no threshold of its own: "wrong" is tools.ball_score.judge with config/ball_score.yaml; tile geometry,
edge margin and merge IoU are read from the ball document's own ball_config):
- Each stretch of the ball path (confident or not) is judged on the labelled frames where it sighted something:
  REAL (every such sighting is the labelled ball), FALSE (none is), MIXED, UNLABELLED (no labelled frame).
- Accepted false detections: every SEEN frame of a confident FALSE stretch, plus the labelled wrong SEEN frames of
  a confident MIXED stretch. One false OBJECT = one such stretch. Rejected false objects = FALSE stretches the path
  did not accept.
- Per accepted false detection: frame, box, size, score, source (which detector passes produced it: full frame,
  tiles, or several merged by the view's NMS), whether its centre lies in the overlap of two tiles, whether the
  untiled detections of the clip hold a ball box within the scoring radius of it, and its object's persistence
  (sightings, frames, seconds, density, longest run of consecutive sighted frames).
- Bug checks: the re-run must reproduce the saved tiled detection (same box by the view's merge IoU; the IoU
  is recorded); the merged score must equal the best pass score
  (NMS never boosts); a pass box cut by an internal tile edge and overlapping an accepted detection is a seam copy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from spana.artifacts import read_json, refuse_existing, validate_ball, validate_ball_path, write_json_atomic
from spana.ball_labels import BallLabel, load_frozen
from spana.config import BallConfig, ConfigError, typed_from_dict
from spana.dataset.det_metrics import iou
from spana.detection.ball_view import BallView, tile_grid
from spana.reid_labels import split_guard
from tools.ball_score import DEFAULT_CONFIG as SCORE_CONFIG
from tools.ball_score import judge
from tools.ball_score import load_config as load_score_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "ball_fp_diag.yaml"
CONFIG_VERSION = "1.0.0"
CLASSES = ("REAL", "FALSE", "MIXED", "UNLABELLED")
_SAME_CENTRE_PX = 0.5    # numerical: a sighting's centre is its detection's centre, both rounded to 0.1 px


@dataclass(frozen=True)
class DiagConfig:
    schema_version: str
    crop_pad_d: float
    min_crop_px: int
    cell_px: int
    sheet_cols: int
    sheet_rows: int
    crops_per_object: int
    categories: Tuple[str, ...]


def load_config(path=DEFAULT_CONFIG) -> DiagConfig:
    cfg = typed_from_dict(DiagConfig, yaml.safe_load(Path(path).read_text()), "ball_fp_diag")
    if cfg.schema_version != CONFIG_VERSION:
        raise ConfigError(f"ball_fp_diag schema_version {cfg.schema_version!r} != {CONFIG_VERSION!r}")
    if not (cfg.crop_pad_d > 0 and min(cfg.min_crop_px, cfg.cell_px, cfg.sheet_cols, cfg.sheet_rows,
                                       cfg.crops_per_object) >= 1 and cfg.categories
            and len(set(cfg.categories)) == len(cfg.categories)):
        raise ConfigError("ball_fp_diag: sizes must be positive and categories non-empty and unique")
    return cfg


# ------------------------------------------------------------------------------------------- analysis (no video)
def longest_run(indices: Sequence[int]) -> int:
    best = run = 0
    prev = None
    for k in sorted(indices):
        run = run + 1 if prev is not None and k == prev + 1 else 1
        best, prev = max(best, run), k
    return best


def judge_stretches(path: dict, ball: dict, labels: Dict[int, BallLabel], radius_px: float) -> List[dict]:
    """One row per stretch of the path: its persistence and its class against the labels."""
    index = {f["frame"]: k for k, f in enumerate(ball["frames"])}
    sight = {}                                       # (hypothesis, frame) -> sighting
    for f in ball["frames"]:
        for s in f["sightings"]:
            sight[(s["hypothesis"], f["frame"])] = s
    pc = path["ball_path_config"]
    rows = []
    for st in path["stretches"]:
        frames = [f for (h, f) in sight if h == st["hypothesis"] and st["first_frame"] <= f <= st["last_frame"]]
        hits = wrong = 0
        for f in frames:
            lab = labels.get(f)
            if lab is None or lab.state == "UNSURE":
                continue
            outcome, _ = judge(lab, {"state": "OBSERVED", "center": sight[(st["hypothesis"], f)]["center"]}, radius_px)
            hits += outcome.startswith("HIT")
            wrong += outcome.startswith("WRONG")
        cls = "UNLABELLED" if hits + wrong == 0 else "REAL" if wrong == 0 else "FALSE" if hits == 0 else "MIXED"
        why = [] if st["confident"] else \
            [n for n, bad in (("too short", st["duration_s"] < pc["min_stretch_s"]),
                              ("too sparse", st["density"] < pc["min_density"])) if bad]
        rows.append({**st, "class": cls, "labelled_hits": hits, "labelled_wrong": wrong,
                     "longest_run": longest_run([index[f] for f in frames]), "rejected_because": why,
                     "scores": sorted(sight[(st["hypothesis"], f)]["score"] for f in frames)})
    return rows


def _box_of(center: Sequence[float], dets: Sequence[Sequence[float]]) -> Optional[Sequence[float]]:
    """The saved detection row whose centre is the sighting's centre (the sighting was made from it)."""
    best = min(dets, key=lambda r: math.hypot((r[0] + r[2]) / 2 - center[0], (r[1] + r[3]) / 2 - center[1]),
               default=None)
    if best is None or math.hypot((best[0] + best[2]) / 2 - center[0],
                                  (best[1] + best[3]) / 2 - center[1]) > _SAME_CENTRE_PX:
        return None
    return best


def accepted_false(clip: str, path: dict, ball: dict, balldet: dict, stretches: List[dict],
                   labels: Dict[int, BallLabel], radius_px: float, baseline: Optional[dict]) -> List[dict]:
    """One row per accepted false detection (module docstring), without the detector re-run (`source` is added by
    `add_sources`)."""
    cls = {s["stretch"]: s for s in stretches}
    video = ball["video"]
    tiles = tile_grid(video["width"], video["height"], BallConfig(**ball["ball_config"]))
    det = {f["frame"]: f["detections"] for f in balldet["frames"]}
    base = None if baseline is None else {f["frame"]: f["detections"] for f in baseline["frames"]}
    rows = []
    for f in path["frames"]:
        if f["state"] != "SEEN":
            continue
        st = cls[f["stretch"]]
        lab = labels.get(f["frame"])
        scored = lab is not None and lab.state != "UNSURE"
        outcome, dist = judge(lab, {"state": "OBSERVED", "center": f["center"]}, radius_px) if scored else (None, None)
        wrong = bool(outcome and outcome.startswith("WRONG"))
        if not (st["class"] == "FALSE" or (st["class"] == "MIXED" and wrong)):
            continue
        box = _box_of(f["center"], det.get(f["frame"], []))
        cx, cy = f["center"]
        same = None
        if base is not None and f["frame"] in base:
            near = [r for r in base[f["frame"]]
                    if math.hypot((r[0] + r[2]) / 2 - cx, (r[1] + r[3]) / 2 - cy) <= radius_px]
            same = {"present": bool(near), "score": max((r[4] for r in near), default=None)}
        rows.append({"clip": clip, "frame": f["frame"], "stretch": f["stretch"], "stretch_class": st["class"],
                     "center": f["center"], "box": None if box is None else [round(v, 1) for v in box[:4]],
                     "w_px": None if box is None else round(box[2] - box[0], 1),
                     "h_px": None if box is None else round(box[3] - box[1], 1), "score": f["score"],
                     "label": lab.state if lab is not None else None, "scored_wrong": wrong,
                     "distance_px": None if dist is None else round(dist, 1),
                     "in_tile_overlap": sum(x0 <= cx < x0 + s and y0 <= cy < y0 + s for x0, y0, s in tiles) >= 2,
                     "in_untiled_baseline": same, "source": None})
    return rows


def analyse(label_dir: Path, tiled_dir: Path, baseline_balldet: Optional[Path], score_config=SCORE_CONFIG) -> dict:
    rec, labels = load_frozen(label_dir)
    split_guard(rec["split"], "tune")                               # DEV only
    clip = rec["clip"]["name"]
    ball = read_json(tiled_dir / f"{clip}.ball.json")
    path = read_json(tiled_dir / f"{clip}.path.json")
    balldet = read_json(tiled_dir / f"{clip}.balldet.json")
    validate_ball(ball)
    validate_ball_path(path, ball)
    if ball["video"]["sha256"] != rec["clip"]["sha256"] or balldet["video"]["sha256"] != rec["clip"]["sha256"]:
        raise SystemExit(f"refusing: the run outputs in {tiled_dir} are not of the labelled clip {clip}")
    baseline = read_json(baseline_balldet) if baseline_balldet else None
    if baseline is not None and baseline["video"]["sha256"] != rec["clip"]["sha256"]:
        raise SystemExit("refusing: the untiled baseline detections are of another video")
    radius = load_score_config(score_config).hit_radius_frac_height * rec["clip"]["height"]
    stretches = judge_stretches(path, ball, labels, radius)
    return {"tool": "tools.diagnostics.ball_tiling_fp", "clip": clip, "split": rec["split"],
            "label_version": rec["label_version"], "video": ball["video"], "radius_px": round(radius, 2),
            "tile_grid": tile_grid(ball["video"]["width"], ball["video"]["height"], BallConfig(**ball["ball_config"])),
            "ball_config": ball["ball_config"], "ball_path_config": path["ball_path_config"],
            "stretches": stretches,
            "detections": accepted_false(clip, path, ball, balldet, stretches, labels, radius, baseline),
            "source_checks": None}


# ------------------------------------------------------------------------- source of each detection (detector)
class _Recorder:
    """Passes detector calls through and keeps what each call returned (call 0 = full frame, then the tiles)."""

    def __init__(self, detector):
        self._det, self.calls = detector, []

    def __getattr__(self, name):
        return getattr(self._det, name)

    def detect(self, img):
        out = self._det.detect(img)
        self.calls.append(out)
        return out


def tile_names(tiles: Sequence[Tuple[int, int, int]]) -> List[str]:
    if len(tiles) == 2 and tiles[0][1] == tiles[1][1]:
        return ["tile_left", "tile_right"] if tiles[0][0] < tiles[1][0] else ["tile_right", "tile_left"]
    return [f"tile_{i}" for i in range(len(tiles))]


def passes_of(detector, ball_classes: Sequence[str], bcfg: BallConfig, image) -> Tuple[list, List[dict]]:
    """(the view's final ball detections, every ball box of every pass in frame coordinates with `cut` = the view
    drops it for lying within the edge margin of an internal tile edge)."""
    rec = _Recorder(detector)
    view = BallView(rec, ball_classes, bcfg, tile_detector=rec)
    final = view.detect(image)
    h, w = image.shape[:2]
    tiles = tile_grid(w, h, bcfg)
    out = [{"pass": "full", "box": list(d.box), "score": d.score, "cut": False} for d in view._balls(rec.calls[0])]
    for name, (x0, y0, side), dets in zip(tile_names(tiles), tiles, rec.calls[1:]):
        m = bcfg.tile_edge_margin_frac * side
        for d in view._balls(dets):
            x1, y1, x2, y2 = d.box
            cut = (x0 > 0 and x1 <= m) or (y0 > 0 and y1 <= m) or (x0 + side < w and x2 >= side - m) or \
                  (y0 + side < h and y2 >= side - m)
            out.append({"pass": name, "box": [x1 + x0, y1 + y0, x2 + x0, y2 + y0], "score": d.score, "cut": cut})
    return final, out


def source_of(box: Sequence[float], final: list, passes: List[dict], merge_iou: float) -> dict:
    """Which passes produced `box` (a saved accepted detection): the uncut pass boxes the view's NMS would merge
    with it (IoU > merge_iou)."""
    rerun = max(final, key=lambda d: iou(box, d.box), default=None)
    rerun_iou = 0.0 if rerun is None else iou(box, rerun.box)
    reproduced = rerun_iou > merge_iou              # the re-run holds the same ball box, by the view's own merge rule
    mine = [p for p in passes if not p["cut"] and iou(box, p["box"]) > merge_iou]
    names = sorted({p["pass"] for p in mine})
    best = max((p["score"] for p in mine), default=None)
    return {"source": "none" if not names else names[0] if len(names) == 1 else "merged(" + "+".join(names) + ")",
            "pass_scores": {n: round(max(p["score"] for p in mine if p["pass"] == n), 4) for n in names},
            "reproduced": reproduced, "rerun_iou": round(rerun_iou, 3),
            "merge_boost": bool(reproduced and best is not None and rerun.score > best + 1e-6),
            "seam_copy": any(p["cut"] and iou(box, p["box"]) > 0 for p in passes)}


def add_sources(report: dict, detector, ball_classes: Sequence[str], images: Dict[int, "object"]) -> None:
    bcfg = BallConfig(**report["ball_config"])
    cache: Dict[int, tuple] = {}
    for row in report["detections"]:
        if row["box"] is None or row["frame"] not in images:
            continue
        if row["frame"] not in cache:
            cache[row["frame"]] = passes_of(detector, ball_classes, bcfg, images[row["frame"]])
        row["source"] = source_of(row["box"], *cache[row["frame"]], bcfg.nms_iou)
    done = [r["source"] for r in report["detections"] if r["source"]]
    report["source_checks"] = {"detections": len(done), "not_reproduced": sum(not s["reproduced"] for s in done),
                               "merge_boosts": sum(s["merge_boost"] for s in done),
                               "seam_copies": sum(s["seam_copy"] for s in done)}


# ------------------------------------------------------------------------------------------------ contact sheets
def objects_of(report: dict) -> List[dict]:
    """False objects the path accepted: one per stretch, numbered in frame order."""
    by: Dict[str, List[dict]] = {}
    for r in report["detections"]:
        by.setdefault(r["stretch"], []).append(r)
    st = {s["stretch"]: s for s in report["stretches"]}
    out = []
    for n, (sid, rows) in enumerate(sorted(by.items(), key=lambda kv: kv[1][0]["frame"]), 1):
        out.append({"object": n, "stretch": sid, "rows": rows, "first": rows[0]["frame"], "last": rows[-1]["frame"],
                    "persistence": {k: st[sid][k] for k in ("sightings", "frames", "duration_s", "density",
                                                             "longest_run", "class")}})
    return out


def sample(rows: List[dict], n: int) -> List[dict]:
    if len(rows) <= n:
        return rows
    return [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)] if n > 1 else [rows[len(rows) // 2]]


def contact_sheets(report: dict, images: Dict[int, "object"], cfg: DiagConfig, out_dir: Path) -> List[str]:
    import cv2
    import numpy as np
    cells = []
    for obj in objects_of(report):
        for r in sample([x for x in obj["rows"] if x["frame"] in images], cfg.crops_per_object):
            img = images[r["frame"]]
            d = max(r["w_px"] or 0, r["h_px"] or 0, 1.0)
            half = int(round(max(cfg.crop_pad_d * d + d / 2, cfg.min_crop_px / 2)))
            cx, cy = (int(round(v)) for v in r["center"])
            pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            crop = cv2.resize(pad[cy:cy + 2 * half, cx:cx + 2 * half], (cfg.cell_px, cfg.cell_px),
                              interpolation=cv2.INTER_CUBIC)
            k = cfg.cell_px / (2 * half)
            r_px = int(round(max(d / 2 + 3, 6) * k))                    # a ring around the detection, not on it
            cv2.circle(crop, (cfg.cell_px // 2, cfg.cell_px // 2), r_px, (0, 255, 255), 1, cv2.LINE_AA)
            cap = np.full((22, cfg.cell_px, 3), 25, np.uint8)
            cv2.putText(cap, f"#{obj['object']} f{r['frame']} s{r['score']:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(np.vstack([crop, cap]))
    per = cfg.sheet_cols * cfg.sheet_rows
    paths = []
    for n in range(0, len(cells), per):
        chunk = cells[n:n + per]
        chunk += [np.zeros_like(cells[0])] * (-len(chunk) % cfg.sheet_cols)
        rows = [np.hstack(chunk[i:i + cfg.sheet_cols]) for i in range(0, len(chunk), cfg.sheet_cols)]
        p = out_dir / f"{report['clip']}_fp_sheet_{n // per + 1:02d}.jpg"
        cv2.imwrite(str(p), np.vstack(rows))
        paths.append(str(p))
    return paths


def write_categories_template(report: dict, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "object", "stretch", "first_frame", "last_frame", "accepted_frames", "category", "note"])
        for o in objects_of(report):
            w.writerow([report["clip"], o["object"], o["stretch"], o["first"], o["last"], len(o["rows"]), "", ""])


def read_categories(path: Path, cfg: DiagConfig) -> Dict[str, str]:
    out = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            cat = row["category"].strip()
            if cat not in cfg.categories:
                raise SystemExit(f"{path}: object {row['object']} has category {cat!r}; use one of "
                                 f"{list(cfg.categories)}")
            out[row["stretch"]] = cat
    return out


# ------------------------------------------------------------------------------------------------------- tables
def _stats(values: Sequence[float]) -> str:
    if not values:
        return "-"
    return f"{statistics.median(values):g} / {max(values):g}"


def _count(rows: Iterable, key: Callable) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[key(r)] = out.get(key(r), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _table(head: Sequence[str], rows: Sequence[Sequence]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def summarise(report: dict, categories: Optional[Dict[str, str]]) -> str:
    det, objs = report["detections"], objects_of(report)
    st = report["stretches"]
    out = [f"### {report['clip']} (labels v{report['label_version']}, scoring radius {report['radius_px']} px, tiles "
           f"{report['tile_grid']})",
           f"Accepted false detections: {len(det)} frames ({sum(r['scored_wrong'] for r in det)} of them are scored "
           f"wrong SEEN answers on labelled frames) in {len(objs)} false objects."]
    if categories is not None:
        cat = lambda sid: categories.get(sid, "(not labelled)")     # noqa: E731
        by_obj, by_fr = _count(objs, lambda o: cat(o["stretch"])), _count(det, lambda r: cat(r["stretch"]))
        out += ["", "By category:", _table(["category", "objects", "frames"], [[c, by_obj[c], by_fr[c]] for c in by_obj])]
    src = _count(det, lambda r: r["source"]["source"] if r["source"] else "(not re-run)")
    out += ["", "By source (frames):", _table(["source", "frames"], list(src.items()))]
    base = _count(det, lambda r: "no untiled detections for the frame" if r["in_untiled_baseline"] is None else
                  "also in the untiled baseline" if r["in_untiled_baseline"]["present"] else "new with tiling")
    out += ["", "New with tiling vs already in the untiled detections (frames):", _table(["", "frames"], list(base.items()))]
    out += ["", f"In the overlap of two tiles: {sum(r['in_tile_overlap'] for r in det)} of {len(det)} frames."]
    sizes = [max(r["w_px"], r["h_px"]) for r in det if r["w_px"] is not None]
    out += [f"Box size (longer side, px) median / max: {_stats(sizes)}; score median / max: "
            f"{_stats([r['score'] for r in det])}."]
    groups = (("false objects the path ACCEPTED", [s for s in st if s["class"] in ("FALSE", "MIXED") and s["confident"]]),
              ("false objects the path REJECTED", [s for s in st if s["class"] == "FALSE" and not s["confident"]]),
              ("real-ball stretches accepted", [s for s in st if s["class"] == "REAL" and s["confident"]]),
              ("real-ball stretches rejected", [s for s in st if s["class"] == "REAL" and not s["confident"]]))
    out += ["", "Persistence (median / max per group):",
            _table(["group", "stretches", "seconds", "longest run of consecutive frames", "density", "sightings"],
                   [[n, len(g), _stats([s["duration_s"] for s in g]), _stats([s["longest_run"] for s in g]),
                     _stats([s["density"] for s in g]), _stats([s["sightings"] for s in g])] for n, g in groups])]
    rej = _count([s for s in st if s["class"] == "FALSE" and not s["confident"]],
                 lambda s: " and ".join(s["rejected_because"]) or "?")
    out += ["", "Why the rejected false objects were rejected: " + (", ".join(f"{k}: {v}" for k, v in rej.items()) or "-"),
            f"Stretches with no labelled frame (not judged): {sum(s['class'] == 'UNLABELLED' for s in st)}; "
            f"MIXED stretches (a real stretch with wrong frames): "
            f"{sum(s['class'] == 'MIXED' and s['confident'] for s in st)} accepted."]
    if objs:
        out += ["", "Each accepted false object:",
                _table(["#", "frames", "accepted frames", "seconds", "longest run", "density", "class"] +
                       (["category"] if categories is not None else []),
                       [[o["object"], f"{o['first']}-{o['last']}", len(o["rows"]), o["persistence"]["duration_s"],
                         o["persistence"]["longest_run"], o["persistence"]["density"], o["persistence"]["class"]] +
                        ([categories.get(o["stretch"], "(not labelled)")] if categories is not None else [])
                        for o in objs])]
    c = report["source_checks"]
    out += ["", "Tiling / merge bug checks: " + ("not run (no detector re-run)" if c is None else
            f"{c['detections']} detections re-run; not reproduced {c['not_reproduced']}; merged score above the best "
            f"pass score {c['merge_boosts']}; overlapping a pass box cut at a tile seam {c['seam_copies']}.")]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- CLI
def _detector(a: argparse.Namespace, ball_cfg: BallConfig, spana_cfg):
    from spana.dataset.config import load_dataset_config
    from spana.detection.rfdetr_local import LocalModelRecord, LocalRFDETRDetector, make_runner
    dcfg = load_dataset_config(a.dataset_config)
    rec = LocalModelRecord.load(a.model_record)
    conf = min(spana_cfg.detection.score_low, ball_cfg.score_min)        # as `spana.cli detect --ball-out`
    return LocalRFDETRDetector(rec, make_runner(rec, a.runtime, a.device), dcfg.classes, conf), \
        dcfg.detector.ball_classes


def cmd_inspect(a: argparse.Namespace) -> int:
    from spana.config import load_config as load_spana_config
    from spana.video import VideoSource
    cfg, out = load_config(a.diag_config), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = analyse(Path(a.labels), Path(a.tiled_dir), Path(a.baseline_balldet) if a.baseline_balldet else None)
    clip = report["clip"]
    refuse_existing(out / f"{clip}.fp.json", out / f"{clip}.categories.csv")
    spana_cfg = load_spana_config(a.config)
    import dataclasses
    if dataclasses.asdict(spana_cfg.ball) != report["ball_config"]:
        raise SystemExit("refusing: --config is not the config the tiled run was made with (ball section differs)")
    wanted = {r["frame"] for r in report["detections"]}
    images = {}
    if wanted:
        video = VideoSource(a.video, expected_sha256=report["video"]["sha256"])
        images = {i: img for i, _, img in video.frames(min(wanted), max(wanted) + 1) if i in wanted}
        detector, ball_classes = _detector(a, spana_cfg.ball, spana_cfg)
        add_sources(report, detector, ball_classes, images)
    sheets = contact_sheets(report, images, cfg, out) if images else []
    report["contact_sheets"] = sheets
    write_json_atomic(out / f"{clip}.fp.json", report)
    write_categories_template(report, out / f"{clip}.categories.csv")
    print(summarise(report, None))
    print("\ncontact sheets:", *sheets, sep="\n  ")
    print(f"fill the category column of {out / (clip + '.categories.csv')} with one of {list(cfg.categories)}")
    return 0


def cmd_summarise(a: argparse.Namespace) -> int:
    cfg, out = load_config(a.diag_config), Path(a.out)
    files = sorted(out.glob("*.fp.json")) if not a.clip else [out / f"{c}.fp.json" for c in a.clip]
    text = []
    for f in files:
        report = json.loads(f.read_text())
        text.append(summarise(report, read_categories(out / f"{report['clip']}.categories.csv", cfg)))
    print("\n\n".join(text))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect")
    for k in ("labels", "video", "tiled-dir", "config", "dataset-config", "model-record", "out"):
        i.add_argument(f"--{k}", required=True)
    i.add_argument("--baseline-balldet", default=None, help="untiled ball-view detections of the same clip")
    i.add_argument("--runtime", choices=("torch", "onnx"), required=True)
    i.add_argument("--device", choices=("mps", "cpu"), required=True)
    s = sub.add_parser("summarise")
    s.add_argument("--out", required=True)
    s.add_argument("--clip", action="append")
    for q in (i, s):
        q.add_argument("--diag-config", default=str(DEFAULT_CONFIG))
    a = p.parse_args(argv)
    return {"inspect": cmd_inspect, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
