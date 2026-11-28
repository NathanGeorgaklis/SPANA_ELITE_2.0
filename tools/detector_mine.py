"""Mine candidates for detector retraining from TRAIN footage only (docs/DETECTOR_BALL_RETRAINING_PLAN.md, task 1).

Nothing here labels, uploads or trains. It proposes frames for a person to review.

    # ball sizes the detector was trained on (frozen labels; read-only)
    python -m tools.detector_mine scale --labels records/roboflow_version/labels_<stamp>.json

    # spare-ball audit of the existing labels: candidates + contact sheets (needs the project image cache)
    python -m tools.detector_mine spare-audit --labels records/roboflow_version/labels_<stamp>.json \
        --flags records/roboflow_review/review_flags_<stamp>.json --out OUT_DIR

    # mine one TRAIN match video (local detector; one pass over the video; resumable)
    python -m tools.detector_mine mine --video MATCH.mp4 --match-id TRAIN_M00X --config TILING_ON.yaml \
        --dataset-config config/roboflow.yaml --model-record models/rfdetr_v1.json --runtime torch --device mps \
        --out OUT_DIR

    # after the review: what the footage yielded, per category
    python -m tools.detector_mine yield --out OUT_DIR

TRAIN only: `mine` refuses a match that is not on the licence allow-list with split `train` (DEV and EVAL matches are
refused before the video is read). Frames stay on this machine (local detector).

Mining (`mine`). Windows of `window_s` every `window_every_s`, analysed at `analysis_fps`. On each frame the detector
runs once on the full frame and once per ball tile (the tiling config). Two ball paths are built per window from the
same passes: UNTILED (full-frame ball boxes only, as the product runs today) and TILED (full frame + tiles).
Candidate objects for review:
  missed_ball     a hole, or an OPEN gap of at most max_gap_s, in the UNTILED path: the ball was seen just before and
                  just after. The ring marks the position interpolated between the two (a hint, not a label).
  tiled_only      a confident stretch of the TILED path (judged on its own sightings, so also one that is in
                  CONFLICT with the real ball) with no UNTILED SEEN answer on it at all: either a real ball the full
                  frame cannot see, or a look-alike the tiles turn into a "ball".
  tiled_extends   frames a TILED stretch adds to a ball the UNTILED path also has: the tiles see more of it.
"The same ball" = within the scoring radius of config/ball_score.yaml. The reviewer gives each object one choice:
`match ball`, `spare ball`, `nothing there`, or a look-alike category of config/ball_fp_diag.yaml.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from spana.artifacts import BALL_SCHEMA, ball_frame, ball_hypothesis_rows, sha256_doc, validate_ball
from spana.ball.path import ball_path_document
from spana.ball.tracker import BallTracker
from spana.config import ConfigError, typed_from_dict
from spana.dataset.config import PolicyError, authorize, load_dataset_config
from spana.detection.postprocess import nms
from spana.records import Detection
from tools.ball_score import DEFAULT_CONFIG as SCORE_CONFIG
from tools.ball_score import load_config as load_score_config
from tools.diagnostics import ball_tiling_fp as fp

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "config" / "detector_mine.yaml"
CONFIG_VERSION = "1.1.0"
KINDS = ("missed_ball", "tiled_only", "tiled_extends")


@dataclass(frozen=True)
class MineConfig:
    schema_version: str
    window_s: float
    window_every_s: float
    analysis_fps: float
    max_gap_s: float
    frames_per_gap: int
    frames_per_object: int
    min_per_category: int
    audit_flags: Tuple[str, ...]
    audit_choices: Tuple[str, ...]
    ball_choices: Tuple[str, ...]
    review_aliases: Tuple[str, ...]


def load_config(path=DEFAULT_CONFIG) -> MineConfig:
    cfg = typed_from_dict(MineConfig, yaml.safe_load(Path(path).read_text()), "detector_mine")
    if cfg.schema_version != CONFIG_VERSION:
        raise ConfigError(f"detector_mine schema_version {cfg.schema_version!r} != {CONFIG_VERSION!r}")
    if not (0 < cfg.window_s <= cfg.window_every_s and cfg.max_gap_s > 0 and cfg.analysis_fps > 0 and min(
            cfg.frames_per_gap, cfg.frames_per_object, cfg.min_per_category) >= 1
            and cfg.audit_choices and cfg.ball_choices):
        raise ConfigError("detector_mine: need 0 < window_s <= window_every_s, positive counts and choices")
    return cfg


# ----------------------------------------------------------------------------------------------- scale (labels)
def _q(values: Sequence[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def scale_report(labels: dict) -> dict:
    """Ball short sides (px) in the frozen training labels, per split, and the frame sizes."""
    out = {"frames": len(labels["frames"]),
           "frame_sizes": sorted({(f["width"], f["height"]) for f in labels["frames"]}), "splits": {}}
    for split in sorted({f["split"] for f in labels["frames"]}):
        short = [min(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1])
                 for f in labels["frames"] if f["split"] == split for b in f["boxes"] if b["class"] == "ball"]
        out["splits"][split] = {"ball_boxes": len(short)} if not short else {
            "ball_boxes": len(short), **{f"p{int(p * 100):02d}": _q(short, p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    return out


# ------------------------------------------------------------------------------------------- spare-ball audit
def audit_candidates(labels: dict, flags: dict, cfg: MineConfig) -> List[dict]:
    """Ball boxes of the existing labels worth a second look for the "only the match ball is boxed" rule."""
    flagged = {e["name"]: sorted(set(e.get("flags", [])) & set(cfg.audit_flags)) for e in flags["entries"]}
    rows = []
    for f in sorted(labels["frames"], key=lambda f: f["name"]):
        balls = [b["box"] for b in f["boxes"] if b["class"] == "ball"]
        why = flagged.get(f["name"], []) + (["several_ball_boxes"] if len(balls) > 1 else [])
        for k, box in enumerate(balls if why else []):
            rows.append({"n": len(rows) + 1, "name": f["name"], "id": f["id"], "match_id": f["match_id"],
                         "split": f["split"], "why": why, "ball_index": k, "box": box,
                         "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                         "w_px": box[2] - box[0], "h_px": box[3] - box[1], "score": None,
                         "frame": int(Path(f["name"]).stem.rsplit("_f", 1)[-1])})
    return rows


# -------------------------------------------------------------------------------------------------- mining
def window_ranges(n_frames: int, fps: float, cfg: MineConfig) -> List[Tuple[int, int]]:
    length, every = max(1, int(round(cfg.window_s * fps))), max(1, int(round(cfg.window_every_s * fps)))
    return [(s, s + length) for s in range(0, max(0, n_frames - length + 1), every)]


def stride_of(fps: float, cfg: MineConfig) -> int:
    return max(1, int(round(fps / cfg.analysis_fps)))


def _ball_doc(rows: list, trk: BallTracker, video: dict, ball_cfg) -> dict:
    doc = {"schema": BALL_SCHEMA[0], "schema_version": BALL_SCHEMA[1], "provenance": {"tool": "tools.detector_mine"},
           "video": video, "detections_sha256": "0" * 64, "tracks_sha256": None,
           "ball_config": dataclasses.asdict(ball_cfg), "frames": rows,
           "hypotheses": ball_hypothesis_rows(trk.summaries())}
    validate_ball(doc)
    return doc


def window_paths(frames: Sequence[Tuple[int, float, np.ndarray]], detector, ball_classes: Sequence[str], spana_cfg,
                 video: dict) -> Tuple[dict, dict, dict]:
    """(untiled ball path, tiled ball path, tiled ball document) of one window, from one detector run per frame."""
    from spana.tracking.camera import CameraMotionEstimator
    tiled_cfg = spana_cfg.ball
    if tiled_cfg.tile_size_frac <= 0:
        raise SystemExit("refusing: mining needs the tiling config (ball.tile_size_frac > 0)")
    plain_cfg = dataclasses.replace(tiled_cfg, tile_size_frac=0.0)
    cam = CameraMotionEstimator(spana_cfg.camera, video["width"], video["height"], video["fps"])
    trackers = {"untiled": BallTracker(plain_cfg, video["width"], video["height"]),
                "tiled": BallTracker(tiled_cfg, video["width"], video["height"])}
    rows = {"untiled": [], "tiled": []}
    ball_id = list(detector.class_names).index(ball_classes[0])
    for i, t, img in frames:
        final, passes = fp.passes_of(detector, ball_classes, tiled_cfg, img)
        full = [p for p in passes if p["pass"] == "full"]
        keep = nms(np.array([p["box"] for p in full], float), np.array([p["score"] for p in full], float),
                   tiled_cfg.nms_iou) if full else []
        dets = {"untiled": [Detection(tuple(full[k]["box"]), full[k]["score"], ball_id) for k in keep],
                "tiled": list(final)}
        obs = cam.observe(img, [d.box for d in final])
        for name in rows:
            rows[name].append(ball_frame(trackers[name].update(i, t, dets[name], obs), spana_cfg.output))
    docs = {n: _ball_doc(rows[n], trackers[n], video, tiled_cfg if n == "tiled" else plain_cfg) for n in rows}
    untiled, tiled = (ball_path_document(docs[n], spana_cfg.ball_path, {}, sha256_doc(docs[n]))
                      for n in ("untiled", "tiled"))
    return untiled, tiled, docs["tiled"]


def _spread(items: list, n: int) -> list:
    return fp.sample(items, n)


def candidates_of(untiled: dict, tiled: dict, tiled_ball: dict, radius_px: float, cfg: MineConfig) -> List[dict]:
    """Candidate objects of one window (module docstring). Each: {kind, frames: [{frame, center, w_px, h_px, score,
    hint}], ...}."""
    uf = untiled["frames"]
    u_at = {f["frame"]: f for f in uf}
    index = {f["frame"]: k for k, f in enumerate(uf)}
    out: List[dict] = []

    def between(a: dict, b: dict, why: str) -> None:
        inner = [f for f in uf[index[a["frame"]] + 1:index[b["frame"]]]]
        if not inner or b["t"] - a["t"] > cfg.max_gap_s + 1e-9:
            return
        span = b["t"] - a["t"]
        rows = []
        for f in _spread(inner, cfg.frames_per_gap):
            w = (f["t"] - a["t"]) / span
            rows.append({"frame": f["frame"], "center": [a["center"][0] + w * (b["center"][0] - a["center"][0]),
                                                         a["center"][1] + w * (b["center"][1] - a["center"][1])],
                         "w_px": a["diameter"], "h_px": a["diameter"], "score": None, "hint": True})
        out.append({"kind": "missed_ball", "why": why, "first_frame": inner[0]["frame"],
                    "last_frame": inner[-1]["frame"], "frames_available": len(inner), "frames": rows})

    seen = [f for f in uf if f["state"] == "SEEN"]
    for a, b in zip(seen, seen[1:]):                      # holes: consecutive SEEN frames of one stretch, not adjacent
        if a["stretch"] == b["stretch"] and index[b["frame"]] - index[a["frame"]] > 1:
            between(a, b, "hole")
    for g in untiled["gaps"]:
        if g["kind"] == "OPEN":
            between(u_at[g["from"]["frame"]], u_at[g["to"]["frame"]], "open gap")

    # Tiled objects: every CONFIDENT stretch of the tiled path, from its own sightings. Not from SEEN frames: a
    # look-alike that overlaps the real ball in time is CONFLICT in the path and has no SEEN frame at all.
    sight = {(x["hypothesis"], f["frame"]): x for f in tiled_ball["frames"] for x in f["sightings"]}
    for st in sorted((x for x in tiled["stretches"] if x["confident"]), key=lambda x: x["first_frame"]):
        frames = [{"frame": f, **x} for (h, f), x in sorted(sight.items(), key=lambda kv: kv[0][1])
                  if h == st["hypothesis"] and st["first_frame"] <= f <= st["last_frame"]]
        same = [f for f in frames if u_at[f["frame"]]["state"] == "SEEN" and math.hypot(
            u_at[f["frame"]]["center"][0] - f["center"][0], u_at[f["frame"]]["center"][1] - f["center"][1]) <= radius_px]
        extra = frames if not same else [f for f in frames if u_at[f["frame"]]["state"] != "SEEN"]
        if not extra:
            continue
        out.append({"kind": "tiled_only" if not same else "tiled_extends", "why": "tiled stretch",
                    "first_frame": frames[0]["frame"], "last_frame": frames[-1]["frame"],
                    "frames_available": len(extra), "duration_s": st["duration_s"], "density": st["density"],
                    "frames": [{"frame": f["frame"], "center": f["center"], "w_px": f["diameter"],
                                "h_px": f["diameter"], "score": f["score"], "hint": False}
                               for f in _spread(extra, cfg.frames_per_object)]})
    return out


def _train_only(dataset_config: str, match_id: str):
    dcfg = load_dataset_config(dataset_config)
    try:
        authorize(dcfg, match_id, "train", third_party=False)
    except PolicyError as e:
        raise SystemExit(f"{e}\n(mining uses TRAIN matches only; DEV and EVAL footage is never mined)") from e
    return dcfg


def cmd_mine(a: argparse.Namespace) -> int:
    from spana.config import load_config as load_spana_config
    from spana.video import VideoSource
    cfg, dcfg = load_config(a.mine_config), _train_only(a.dataset_config, a.match_id)
    diag = fp.load_config(a.diag_config)
    spana_cfg = load_spana_config(a.config)
    if spana_cfg.ball.tile_size_frac <= 0:                   # before the video is read or the detector is loaded
        raise SystemExit("refusing: mining needs the tiling config (ball.tile_size_frac > 0), e.g. "
                         "records/ball_tiling_v1_2026-11-23/tiling_on.config.yaml")
    out = Path(a.out) / a.match_id
    out.mkdir(parents=True, exist_ok=True)
    video = VideoSource(a.video)
    meta = video.meta.to_dict()
    ranges = window_ranges(video.meta.frame_count_reported, video.meta.fps, cfg)
    if a.max_windows:
        ranges = ranges[:a.max_windows]
    todo = [r for r in ranges if not (out / f"w_{r[0]:07d}.json").is_file()]
    stride = stride_of(video.meta.fps, cfg)
    per = sum(1 for _ in range(0, ranges[0][1] - ranges[0][0], stride)) if ranges else 0
    tiles = len(fp.tile_grid(meta["width"], meta["height"], spana_cfg.ball))
    print(f"{a.match_id}: {len(ranges)} windows of {per} analysed frames ({len(todo)} still to do), "
          f"{1 + tiles} detector passes per frame = {len(todo) * per * (1 + tiles)} detector runs", flush=True)
    radius = load_score_config(a.score_config).hit_radius_frac_height * meta["height"]
    if todo:
        detector, ball_classes = fp._detector(a, spana_cfg.ball, spana_cfg)
        wanted = iter(todo)
        cur = next(wanted)
        buf: List[Tuple[int, float, np.ndarray]] = []
        for i, t, img in video.frames(0, todo[-1][1]):
            if cur is None or i < cur[0]:
                continue
            if i < cur[1]:
                if (i - cur[0]) % stride == 0:
                    buf.append((i, t, img))
                if i < cur[1] - 1:
                    continue
            untiled, tiled, tiled_ball = window_paths(buf, detector, ball_classes, spana_cfg, meta)
            objs = candidates_of(untiled, tiled, tiled_ball, radius, cfg)
            images = {i_: im for i_, _, im in buf}
            cells = [fp.make_cell("?", r, images[r["frame"]], diag) for o in objs for r in o["frames"]]
            np.savez_compressed(out / f"w_{cur[0]:07d}.npz", cells=np.array(cells, np.uint8) if cells
                                else np.zeros((0, diag.cell_px + 22, diag.cell_px, 3), np.uint8))
            summary = lambda p: {k: sum(f["reason"] == k for f in p["frames"])      # noqa: E731
                                 for k in ("CONFIDENT_STRETCH", "HOLE", "GAP", "CONFLICT", "NO_BALL")}
            (out / f"w_{cur[0]:07d}.json").write_text(json.dumps(
                {"match_id": a.match_id, "window": list(cur), "frames_analysed": len(buf),
                 "untiled": summary(untiled), "tiled": summary(tiled), "objects": objs}))
            print(f"  window {cur[0]}-{cur[1] - 1}: untiled seen {summary(untiled)['CONFIDENT_STRETCH']}, tiled seen "
                  f"{summary(tiled)['CONFIDENT_STRETCH']}, candidates {len(objs)}", flush=True)
            buf, cur = [], next(wanted, None)
    return assemble(out, a.match_id, meta, cfg, diag)


def assemble(out: Path, match_id: str, meta: dict, cfg: MineConfig, diag) -> int:
    """Number the objects of all finished windows, write the contact sheets (one set per kind) and the review CSV."""
    objects, cells = [], {k: [] for k in KINDS}
    for wf in sorted(out.glob("w_*.json")):
        w = json.loads(wf.read_text())
        arr = np.load(wf.with_suffix(".npz"))["cells"]
        k = 0
        for o in w["objects"]:
            n = len(objects) + 1
            for r in o["frames"]:
                cell = arr[k].copy()
                k += 1
                import cv2
                cell[-22:] = 25
                cv2.putText(cell, f"#{n} f{r['frame']}" + ("" if r["score"] is None else f" s{r['score']:.2f}"),
                            (4, cell.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cells[o["kind"]].append(cell)
            objects.append({"object": n, "window": w["window"], **o})
    sheets = {k: fp.pack_sheets(cells[k], diag, out.parent, f"{match_id}_{k}") if cells[k] else [] for k in KINDS}
    windows = [json.loads(p.read_text()) for p in sorted(out.glob("w_*.json"))]
    report = {"tool": "tools.detector_mine", "match_id": match_id, "video": meta, "mine_config": dataclasses.asdict(cfg),
              "windows": len(windows), "frames_analysed": sum(w["frames_analysed"] for w in windows),
              "untiled_seen": sum(w["untiled"]["CONFIDENT_STRETCH"] for w in windows),
              "tiled_seen": sum(w["tiled"]["CONFIDENT_STRETCH"] for w in windows),
              "objects": objects, "contact_sheets": sheets}
    (out.parent / f"{match_id}.mine.json").write_text(json.dumps(report))
    review = out.parent / f"{match_id}.review.csv"
    if not review.exists():                                  # never overwrite a review in progress
        with review.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["match_id", "object", "kind", "first_frame", "last_frame", "frames_available", "choice", "note"])
            for o in objects:
                w.writerow([match_id, o["object"], o["kind"], o["first_frame"], o["last_frame"],
                            o["frames_available"], "", ""])
    by = {k: sum(o["kind"] == k for o in objects) for k in KINDS}
    print(f"{match_id}: {len(windows)} windows, {report['frames_analysed']} frames analysed; ball SEEN on "
          f"{report['untiled_seen']} (untiled) / {report['tiled_seen']} (tiled); candidate objects {by}")
    for k in KINDS:
        for p in sheets[k]:
            print("  ", p)
    print(f"review file: {review}")
    return 0


# -------------------------------------------------------------------------------------------------- yield
def choices(cfg: MineConfig, diag) -> List[str]:
    return list(cfg.ball_choices) + list(diag.categories)


def yield_report(out: Path, cfg: MineConfig, diag) -> str:
    allowed = choices(cfg, diag)
    rows = []
    for review in sorted(out.glob("*.review.csv")):
        with review.open(newline="") as fh:
            for r in csv.DictReader(fh):
                c = r["choice"].strip()
                if c and c not in allowed:
                    raise SystemExit(f"{review}: object {r['object']} has choice {c!r}; use one of {allowed}")
                rows.append({**r, "choice": c or "(not reviewed)", "frames_available": int(r["frames_available"])})
    lines = ["| match | kind | choice | objects | frames available |", "|---|---|---|---|---|"]
    keys = sorted({(r["match_id"], r["kind"], r["choice"]) for r in rows})
    for m, k, c in keys:
        sel = [r for r in rows if (r["match_id"], r["kind"], r["choice"]) == (m, k, c)]
        lines.append(f"| {m} | {k} | {c} | {len(sel)} | {sum(r['frames_available'] for r in sel)} |")
    total = {c: sum(r["frames_available"] for r in rows if r["choice"] == c) for c in allowed}
    lines += ["", "| choice | objects (all matches) | frames available | note |", "|---|---|---|---|"]
    for c in allowed:
        n = sum(r["choice"] == c for r in rows)
        thin = c in diag.categories and total[c] < cfg.min_per_category
        lines.append(f"| {c} | {n} | {total[c]} | {'THIN: under ' + str(cfg.min_per_category) + ' frames' if thin else ''} |")
    left = sum(r["choice"] == "(not reviewed)" for r in rows)
    lines.append(f"\nNot reviewed yet: {left} of {len(rows)} objects.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ per-picture review
def parse_review(text: str, cfg: MineConfig, diag) -> Dict[int, dict]:
    """A review as typed by the reviewer: one line per contact sheet, `<kind>_sheet_NN: #7 match ball, #7 sock, ...`,
    one answer per picture in sheet order. Returns {object: {"kind", "choices": [per picture]}}."""
    allowed = choices(cfg, diag)
    alias = {k.strip(): v.strip() for k, _, v in (x.partition("=") for x in cfg.review_aliases)}
    out: Dict[int, dict] = {}
    for line in filter(str.strip, text.splitlines()):
        head, _, rest = line.partition(":")
        kind = re.sub(r"_sheet_?\d*$", "", head.strip())
        if kind not in KINDS:
            raise SystemExit(f"review line starts with {head.strip()!r}; expected one of {list(KINDS)}_sheet_NN")
        for item in filter(None, (x.strip().rstrip(".").strip() for x in rest.split(","))):
            m = re.fullmatch(r"#(\d+)\s+(.+)", item)
            if not m:
                raise SystemExit(f"cannot read {item!r} on the {head.strip()} line")
            c = alias.get(m[2].lower(), m[2].lower())
            if c not in allowed:
                raise SystemExit(f"#{m[1]}: {m[2]!r} is not a choice; use one of {allowed} or {sorted(alias)}")
            o = out.setdefault(int(m[1]), {"kind": kind, "choices": []})
            if o["kind"] != kind:
                raise SystemExit(f"#{m[1]} is answered under both {o['kind']} and {kind}")
            o["choices"].append(c)
    return out


def apply_review(report: dict, review: Dict[int, dict]) -> List[dict]:
    """One row per mined picture with the reviewer's answer. Fails unless the review answers every picture of every
    object exactly once, under the object's own kind: a miscount would attach answers to the wrong frames."""
    problems, rows = [], []
    for o in report["objects"]:
        r = review.get(o["object"])
        if r is None:
            problems.append(f"#{o['object']} ({o['kind']}) has no answer")
        elif r["kind"] != o["kind"]:
            problems.append(f"#{o['object']} is {o['kind']}, answered under {r['kind']}")
        elif len(r["choices"]) != len(o["frames"]):
            problems.append(f"#{o['object']} ({o['kind']}) has {len(o['frames'])} pictures, {len(r['choices'])} answers")
        else:
            rows += [{"match_id": report["match_id"], "object": o["object"], "kind": o["kind"], "frame": f["frame"],
                      "choice": c} for f, c in zip(o["frames"], r["choices"])]
    extra = sorted(set(review) - {o["object"] for o in report["objects"]})
    if extra:
        problems.append(f"answers for objects that do not exist: {extra}")
    if problems:
        raise SystemExit("review does not match the mined pictures:\n  " + "\n  ".join(problems))
    return rows


def picture_table(rows: List[dict], cfg: MineConfig, diag) -> str:
    lines = ["| choice | " + " | ".join(KINDS) + " | pictures | note |", "|---" * (len(KINDS) + 3) + "|"]
    for c in choices(cfg, diag):
        n = [sum(r["choice"] == c and r["kind"] == k for r in rows) for k in KINDS]
        thin = c in diag.categories and c != "unclear" and sum(n) < cfg.min_per_category
        lines.append(f"| {c} | " + " | ".join(map(str, n)) + f" | {sum(n)} | "
                     f"{'THIN: under ' + str(cfg.min_per_category) if thin else ''} |")
    return "\n".join(lines + [f"\n{len(rows)} pictures, {len({(r['match_id'], r['object']) for r in rows})} objects."])


def cmd_review_apply(a: argparse.Namespace) -> int:
    cfg, diag = load_config(a.mine_config), fp.load_config(a.diag_config)
    report = json.loads((Path(a.out) / f"{a.match_id}.mine.json").read_text())
    rows = apply_review(report, parse_review(Path(a.review).read_text(), cfg, diag))
    dest = Path(a.out) / f"{a.match_id}.pictures.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["match_id", "object", "kind", "frame", "choice"])
        w.writeheader()
        w.writerows(rows)
    print(f"{a.match_id}: review matches all {len(rows)} mined pictures; written to {dest}")
    print(picture_table(rows, cfg, diag))
    return 0


# ---------------------------------------------------------------------------------------------------- CLI
def cmd_scale(a: argparse.Namespace) -> int:
    print(json.dumps(scale_report(json.loads(Path(a.labels).read_text())), indent=1))
    return 0


def cmd_spare_audit(a: argparse.Namespace) -> int:
    import cv2
    cfg, diag = load_config(a.mine_config), fp.load_config(a.diag_config)
    dcfg = load_dataset_config(a.dataset_config)
    rows = audit_candidates(json.loads(Path(a.labels).read_text()), json.loads(Path(a.flags).read_text()), cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = REPO / dcfg.paths.image_cache_dir
    missing = sorted({r["name"] for r in rows if not (cache / f"{r['id']}.jpg").is_file()})
    if missing:
        raise SystemExit(f"{len(missing)} of the audit frames are not in the image cache {cache} (first: "
                         f"{missing[:3]}); the cache is filled by the pre-label / detector-eval tools")
    cells = [fp.make_cell(f"#{r['n']}", r, cv2.imread(str(cache / f"{r['id']}.jpg")), diag) for r in rows]
    sheets = fp.pack_sheets(cells, diag, out, "spare_ball_audit") if cells else []
    (out / "spare_ball_audit.json").write_text(json.dumps({"candidates": rows, "contact_sheets": sheets}))
    review = out / "spare_ball_audit.review.csv"
    if not review.exists():
        with review.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["n", "name", "split", "why", "ball_index", "box", "choice", "note"])
            for r in rows:
                w.writerow([r["n"], r["name"], r["split"], "+".join(r["why"]), r["ball_index"], r["box"], "", ""])
    print(f"{len(rows)} ball boxes to review in {len({r['name'] for r in rows})} frames; choices: "
          f"{list(cfg.audit_choices)}")
    print(*sheets, f"review file: {review}", sep="\n")
    return 0


def cmd_yield(a: argparse.Namespace) -> int:
    print(yield_report(Path(a.out), load_config(a.mine_config), fp.load_config(a.diag_config)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scale")
    sc.add_argument("--labels", required=True)
    au = sub.add_parser("spare-audit")
    for k in ("labels", "flags", "out"):
        au.add_argument(f"--{k}", required=True)
    au.add_argument("--dataset-config", default=str(REPO / "config" / "roboflow.yaml"))
    mi = sub.add_parser("mine")
    for k in ("video", "match-id", "config", "dataset-config", "model-record", "out"):
        mi.add_argument(f"--{k}", required=True)
    mi.add_argument("--runtime", choices=("torch", "onnx"), required=True)
    mi.add_argument("--device", choices=("mps", "cpu"), required=True)
    mi.add_argument("--max-windows", type=int, default=0, help="mine only the first N windows (a trial run)")
    mi.add_argument("--score-config", default=str(SCORE_CONFIG))
    yi = sub.add_parser("yield")
    yi.add_argument("--out", required=True)
    ra = sub.add_parser("review-apply")
    for k in ("review", "match-id", "out"):
        ra.add_argument(f"--{k}", required=True)
    for q in (au, mi, yi, ra):
        q.add_argument("--mine-config", default=str(DEFAULT_CONFIG))
        q.add_argument("--diag-config", default=str(fp.DEFAULT_CONFIG))
    a = p.parse_args(argv)
    return {"scale": cmd_scale, "spare-audit": cmd_spare_audit, "mine": cmd_mine, "yield": cmd_yield,
            "review-apply": cmd_review_apply}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
