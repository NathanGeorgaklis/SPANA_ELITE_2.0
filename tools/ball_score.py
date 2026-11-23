"""Score spana.ball (ball tracker) against frozen DEV ball labels (spana.ball_labels 1.0.0). DEV only; report only.

    python -m tools.ball_score --clip data/ball_labels/<clip>/v1 BALL.json [--clip ...] --out REPORT.json \
        [--config config/ball_score.yaml]

BALL.json is a spana.ball document (the tracker's per-frame answer) or a spana.ball_path document (the offline ball
path: its SEEN frames are scored as OBSERVED answers; it has no PREDICTED answers).

Each labelled frame is judged by CENTRE distance (label box centre; box size is not truth) with the radius
`hit_radius_frac_height` x frame height (config/ball_score.yaml). UNSURE frames are not scored.

  label \\ ball    OBSERVED                         PREDICTED                        LOST / AMBIGUOUS
  VISIBLE         HIT_OBSERVED within the radius;  HIT_PREDICTED / WRONG_POSITION    MISS (no answer)
                  else WRONG_OTHER_BALL (within    / WRONG_OTHER_BALL, likewise
                  the radius of an other_balls
                  box) or WRONG_POSITION
  HIDDEN          WRONG_CLAIMED_SEEN               PREDICTED_HIDDEN (fine; distance  ABSTAIN_OK
                                                   to the annotator's point if any)
  OUT             WRONG_PHANTOM                    WRONG_PHANTOM                     ABSTAIN_OK

Reported per clip and pooled: coverage (OBSERVED hits / VISIBLE, OBSERVED + PREDICTED hits / VISIBLE), wrong
answers with the exact one-sided upper bound (OBSERVED, PREDICTED, all answers), hit distances (nearest-rank median
/ p90 / max as a share of the frame height, and in px when every clip has the same height) and every wrong frame.
Refused: a ball document of another video or frame size, one without video sha256 / size, one that does not cover
every scored labelled frame, and the same video passed twice.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from spana.artifacts import read_json, refuse_existing, validate_ball, validate_ball_path, write_json_atomic
from spana.ball.path import PATH_SCHEMA, as_ball_answers
from spana.ball_labels import BallLabel, load_frozen
from spana.config import ConfigError, typed_from_dict
from spana.reid_labels import split_guard
from tools.reid_cluster import upper_bound

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "ball_score.yaml"
CONFIG_VERSION = "1.0.0"
WRONG = ("WRONG_POSITION", "WRONG_OTHER_BALL", "WRONG_CLAIMED_SEEN", "WRONG_PHANTOM")
OUTCOMES = ("HIT_OBSERVED", "HIT_PREDICTED", "MISS", "PREDICTED_HIDDEN", "ABSTAIN_OK") + WRONG
ANSWERS = ("OBSERVED", "PREDICTED")


@dataclass(frozen=True)
class BallScoreConfig:
    schema_version: str
    hit_radius_frac_height: float
    bound_confidence: float


def load_config(path=DEFAULT_CONFIG) -> BallScoreConfig:
    cfg = typed_from_dict(BallScoreConfig, yaml.safe_load(Path(path).read_text()), "ball_score")
    if cfg.schema_version != CONFIG_VERSION:
        raise ConfigError(f"ball_score schema_version {cfg.schema_version!r} != {CONFIG_VERSION!r}")
    if not 0 < cfg.hit_radius_frac_height < 1:
        raise ConfigError("ball_score.hit_radius_frac_height must be in (0, 1)")
    if not 0 < cfg.bound_confidence < 1:
        raise ConfigError("ball_score.bound_confidence must be in (0, 1)")
    return cfg


def _centre(b: Sequence[float]) -> Tuple[float, float]:
    return 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])


def _dist(p: Sequence[float], q: Sequence[float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def judge(label: BallLabel, bf: dict, radius_px: float) -> Tuple[str, Optional[float]]:
    """(outcome, distance in px to the labelled centre / HIDDEN point, or None) for one scored labelled frame."""
    answered = bf["state"] in ANSWERS
    if label.state == "VISIBLE":
        if not answered:
            return "MISS", None
        d = _dist(bf["center"], _centre(label.box))
        if d <= radius_px:
            return ("HIT_OBSERVED" if bf["state"] == "OBSERVED" else "HIT_PREDICTED"), d
        if any(_dist(bf["center"], _centre(o)) <= radius_px for o in label.other_balls):
            return "WRONG_OTHER_BALL", d
        return "WRONG_POSITION", d
    if label.state == "HIDDEN":
        if bf["state"] == "OBSERVED":
            return "WRONG_CLAIMED_SEEN", None
        if bf["state"] == "PREDICTED":
            return "PREDICTED_HIDDEN", None if label.point is None else _dist(bf["center"], label.point)
        return "ABSTAIN_OK", None
    if label.state == "OUT":
        return ("WRONG_PHANTOM", None) if answered else ("ABSTAIN_OK", None)
    raise ValueError(f"frame {label.frame}: {label.state} is not scored")


def _ranks(values: List[float], decimals: int) -> dict:
    """Nearest-rank median / p90 / max: an observed value, never interpolated (even n: the lower middle value;
    p90 = max below 10 values)."""
    s = sorted(values)
    pick = lambda q: s[max(0, math.ceil(q * len(s)) - 1)]      # noqa: E731
    return {"median": round(pick(0.5), decimals), "p90": round(pick(0.9), decimals), "max": round(s[-1], decimals)}


def _dist_stats(rows: List[dict], one_height: bool) -> Optional[dict]:
    """Distances of these rows as a share of their clip's frame height; in px too when all clips have one height
    (px of different resolutions do not pool)."""
    if not rows:
        return None
    out = {"n": len(rows), "frac_height": _ranks([r["distance_frac"] for r in rows], 5)}
    if one_height:
        out["px"] = _ranks([r["distance_px"] for r in rows], 2)
    return out


def summarise(rows: List[dict], cfg: BallScoreConfig, one_height: bool = True) -> dict:
    """Counts and rates of judged frames (one clip, or pooled clips)."""
    counts = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
    by_state = {s: [r for r in rows if r["ball_state"] == s] for s in ANSWERS}
    visible = sum(r["label"] == "VISIBLE" for r in rows)

    def wrong_block(sel: List[dict]) -> dict:
        n, k = len(sel), sum(r["outcome"] in WRONG for r in sel)
        b = upper_bound(k, n, cfg.bound_confidence)
        return {"answers": n, "wrong": k, "rate": None if n == 0 else round(k / n, 5),
                "upper_bound": None if b is None else round(b, 5)}

    answered = by_state["OBSERVED"] + by_state["PREDICTED"]
    return {
        "frames_scored": len(rows),
        "labels": {s: sum(r["label"] == s for r in rows) for s in ("VISIBLE", "HIDDEN", "OUT")},
        "outcomes": counts,
        "coverage_observed": None if visible == 0 else round(counts["HIT_OBSERVED"] / visible, 5),
        "coverage_answered": None if visible == 0 else
        round((counts["HIT_OBSERVED"] + counts["HIT_PREDICTED"]) / visible, 5),
        "wrong_observed": wrong_block(by_state["OBSERVED"]),
        "wrong_predicted": wrong_block(by_state["PREDICTED"]),
        "wrong_answered": wrong_block(answered),
        "hit_distance_observed": _dist_stats([r for r in rows if r["outcome"] == "HIT_OBSERVED"], one_height),
        "hit_distance_predicted": _dist_stats([r for r in rows if r["outcome"] == "HIT_PREDICTED"], one_height),
        "hidden_predicted_distance": _dist_stats([r for r in rows if r["outcome"] == "PREDICTED_HIDDEN"
                                                  and r["distance_px"] is not None], one_height),
    }


def score_clip(rec: dict, labels: Dict[int, BallLabel], ball: dict, cfg: BallScoreConfig) -> dict:
    split_guard(rec["split"], "tune")
    clip = rec["clip"]
    v = ball.get("video")
    if not isinstance(v, dict) or any(k not in v for k in ("sha256", "width", "height")):
        raise SystemExit(f"refusing: the ball document for {clip['name']} has no video sha256 / width / height")
    if ball["video"]["sha256"] != clip["sha256"]:
        raise SystemExit(f"refusing: the ball document is of another video than the labels of {clip['name']}")
    if (ball["video"]["width"], ball["video"]["height"]) != (clip["width"], clip["height"]):
        raise SystemExit(f"refusing: frame size of the ball document differs from the labels of {clip['name']}")
    frames = {f["frame"]: f for f in ball["frames"]}
    scored = [lab for f, lab in sorted(labels.items()) if lab.state != "UNSURE"]
    missing = [lab.frame for lab in scored if lab.frame not in frames]
    if missing:
        raise SystemExit(f"refusing: the ball document does not cover {len(missing)} labelled frames of "
                         f"{clip['name']} (first {missing[:5]}); run the ball pass on every labelled frame")
    radius = cfg.hit_radius_frac_height * clip["height"]
    rows = []
    for lab in scored:
        bf = frames[lab.frame]
        outcome, d = judge(lab, bf, radius)
        rows.append({"frame": lab.frame, "label": lab.state, "ball_state": bf["state"], "outcome": outcome,
                     "distance_px": None if d is None else round(d, 2),
                     "distance_frac": None if d is None else round(d / clip["height"], 6)})
    return {"clip": clip["name"], "video_sha256": clip["sha256"], "height": clip["height"],
            "radius_px": round(radius, 2), "unsure_frames": sum(lab.state == "UNSURE" for lab in labels.values()),
            "summary": summarise(rows, cfg),
            "wrong_frames": [r for r in rows if r["outcome"] in WRONG], "frames": rows}


def pooled(clips: List[dict], cfg: BallScoreConfig) -> dict:
    """All clips' frames together. The same video twice is refused: it would count frames twice and tighten the
    bound."""
    shas = [c["video_sha256"] for c in clips]
    if len(set(shas)) != len(shas):
        raise SystemExit("refusing: the same video is scored more than once")
    return summarise([r for c in clips for r in c["frames"]], cfg, len({c["height"] for c in clips}) == 1)


def load_answers(path: str) -> Tuple[str, dict]:
    """(kind, answers) of a spana.ball document, or of a spana.ball_path document (SEEN = an OBSERVED answer, every
    other frame no answer: the path never predicts)."""
    doc = read_json(path)
    if isinstance(doc, dict) and doc.get("schema") == PATH_SCHEMA[0]:
        validate_ball_path(doc)
        return "spana.ball_path", as_ball_answers(doc)
    validate_ball(doc)
    return "spana.ball", doc


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABEL_DIR", "BALL_JSON"))
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="scoring conventions (config/ball_score.yaml)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    refuse_existing(a.out)
    cfg = load_config(a.config)
    clips = []
    for label_dir, ball_path in a.clip:
        rec, labels = load_frozen(Path(label_dir))
        kind, answers = load_answers(ball_path)
        clips.append({"scored": kind, **score_clip(rec, labels, answers, cfg)})
    report = {"tool": "tools.ball_score", "config": cfg.__dict__, "clips": clips, "pooled": pooled(clips, cfg)}
    write_json_atomic(a.out, report)
    for c in clips + [{"clip": "POOLED", "summary": report["pooled"]}]:
        s = c["summary"]
        print(f"{c['clip'][:40]:<40} scored {s['frames_scored']:>4}  cov obs {s['coverage_observed']}  "
              f"cov ans {s['coverage_answered']}  wrong obs {s['wrong_observed']['wrong']}/"
              f"{s['wrong_observed']['answers']} (<= {s['wrong_observed']['upper_bound']})  wrong all "
              f"{s['wrong_answered']['wrong']}/{s['wrong_answered']['answers']}")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
