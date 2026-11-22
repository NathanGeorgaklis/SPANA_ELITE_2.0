"""Score spana.ball (ball tracker) against the frozen DEV ball labels. DEV only: the labels refuse any other split.

    python -m tools.ball_score --out REPORT.json \
        --clip data/ball_labels/<clip>/v1 BALL.json [BALL_VIEW.json] [--clip ...] [--score-config config/ball_score.yaml]

Per labelled frame that the ball document analysed (labels are every 0.1 s; UNSURE frames are not scored):
  VISIBLE  claim (OBSERVED / PREDICTED) on the ball     -> correct (observed / predicted kept apart)
           claim on a labelled other ball               -> wrong_other_ball
           claim anywhere else                          -> wrong_position
           no claim (LOST / AMBIGUOUS)                  -> missed; with the ball view given, split by the first failing
                                                           layer: no candidate on the ball (detection) or a candidate
                                                           there that the tracker did not use (ball tracking)
  HIDDEN   OBSERVED -> observed_while_hidden (reported, not counted wrong: a part-seen ball can be detected);
           PREDICTED -> predicted_while_hidden; no claim -> quiet
  OUT      any claim -> false_ball (wrong); no claim -> quiet
"On" = centre within max(match_radius_d x label short side, match_radius_min_px) (config/ball_score.yaml).
Reported per clip and pooled: wrong claims / claims on VISIBLE + OUT frames with the exact one-sided bound, coverage
(correct / VISIBLE), position error of correct claims (px and label diameters), detector recall on VISIBLE frames,
and the tracker's state x reason on missed frames.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from spana.artifacts import read_json, refuse_existing, validate_ball, write_json_atomic
from spana.ball_labels import BallLabel, load_frozen
from tools.reid_cluster import upper_bound

DEFAULT_SCORE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "ball_score.yaml"
CLAIMS = ("OBSERVED", "PREDICTED")
WRONG = ("wrong_position", "wrong_other_ball", "false_ball")


def load_score_config(path: str) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    for k in ("match_radius_d", "match_radius_min_px", "bound_confidence"):
        if k not in c:
            raise SystemExit(f"{path}: missing {k}")
    return c


def _centre(b) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _radius(b, cfg: dict) -> float:
    return max(cfg["match_radius_d"] * min(b[2] - b[0], b[3] - b[1]), cfg["match_radius_min_px"])


def _dist(p, q) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def on_box(p, box, cfg: dict) -> bool:
    return _dist(p, _centre(box)) <= _radius(box, cfg)


def _candidates(ball_view: Optional[dict]) -> Optional[Dict[int, List[Tuple[float, float]]]]:
    if ball_view is None:
        return None
    if not ball_view.get("model", {}).get("ball_view"):
        raise SystemExit("the optional third --clip file must be a ball-view spana.detections document")
    return {f["frame"]: [_centre(d[:4]) for d in f["detections"]] for f in ball_view["frames"]}


def score_clip(labels: Dict[int, BallLabel], ball: dict, cfg: dict, ball_view: Optional[dict] = None) -> dict:
    return _score(labels, ball, cfg, ball_view)[0]


def _score(labels: Dict[int, BallLabel], ball: dict, cfg: dict, ball_view: Optional[dict]):
    answers = {f["frame"]: f for f in ball["frames"]}
    cands = _candidates(ball_view)
    out: Counter = Counter()
    missed_why: Counter = Counter()
    err_px: List[float] = []
    err_d: List[float] = []
    for f in sorted(labels):
        lab = labels[f]
        if lab.state == "UNSURE":
            out["unscored_unsure"] += 1
            continue
        a = answers.get(f)
        if a is None:
            out["not_analysed"] += 1
            continue
        claim = a["state"] in CLAIMS and a["center"] is not None
        c = a["center"]
        if lab.state == "VISIBLE":
            out["visible"] += 1
            if cands is not None and any(on_box(p, lab.box, cfg) for p in cands.get(f, [])):
                out["visible_detected"] += 1
            if claim and on_box(c, lab.box, cfg):
                out["correct_" + a["state"].lower()] += 1
                d = _dist(c, _centre(lab.box))
                err_px.append(d)
                err_d.append(d / min(lab.box[2] - lab.box[0], lab.box[3] - lab.box[1]))
            elif claim:
                out["wrong_other_ball" if any(on_box(c, b, cfg) for b in lab.other_balls) else "wrong_position"] += 1
            else:
                out["missed"] += 1
                missed_why[f"{a['state']}/{a['reason']}"] += 1
                if cands is not None:
                    near = any(on_box(p, lab.box, cfg) for p in cands.get(f, []))
                    out["missed_candidate_present" if near else "missed_no_candidate"] += 1
        elif lab.state == "HIDDEN":
            out["hidden"] += 1
            out[{"OBSERVED": "observed_while_hidden", "PREDICTED": "predicted_while_hidden"}.get(a["state"],
                                                                                           "quiet_while_hidden")] += 1
        else:                                                                                      # OUT
            out["out"] += 1
            out["false_ball" if claim else "quiet_while_out"] += 1
    return summarise(out, missed_why, err_px, err_d, cfg, detector=cands is not None), out, missed_why, err_px, err_d


def summarise(out: Counter, missed_why: Counter, err_px: Sequence[float], err_d: Sequence[float], cfg: dict,
              detector: bool) -> dict:
    correct = out["correct_observed"] + out["correct_predicted"]
    wrong = sum(out[k] for k in WRONG)
    claims = correct + wrong
    r = {"counts": dict(sorted(out.items())), "missed_by_state_reason": dict(missed_why.most_common()),
         "claims": claims, "wrong": wrong,
         "wrong_rate": round(wrong / claims, 4) if claims else None,
         "wrong_bound": None if not claims else round(upper_bound(wrong, claims, cfg["bound_confidence"]), 4),
         "coverage": round(correct / out["visible"], 4) if out["visible"] else None,
         "position_error_px": _dist_stats(err_px), "position_error_d": _dist_stats(err_d)}
    if detector:
        r["detector_recall"] = round(out["visible_detected"] / out["visible"], 4) if out["visible"] else None
    return r


def _dist_stats(v: Sequence[float]) -> Optional[dict]:
    if not v:
        return None
    s = sorted(v)
    return {"n": len(s), "median": round(statistics.median(s), 2), "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 2),
            "max": round(s[-1], 2)}


def run(clips: Sequence[Sequence[str]], score_config: str) -> dict:
    cfg = load_score_config(score_config)
    per, pooled, why = {}, Counter(), Counter()
    epx, ed = [], []
    detector_all = True
    for spec in clips:
        if len(spec) not in (2, 3):
            raise SystemExit("--clip takes LABEL_DIR BALL_JSON [BALL_VIEW_JSON]")
        rec, labels = load_frozen(Path(spec[0]))
        ball = read_json(spec[1])
        view = read_json(spec[2]) if len(spec) == 3 else None
        validate_ball(ball, view)                       # with the ball view: also checks ball was made from it
        if ball["video"]["sha256"] != rec["clip"]["sha256"]:
            raise SystemExit(f"{spec[1]} is not from the labelled clip {rec['clip']['name']}")
        detector_all &= view is not None
        r, out, missed_why, err_px, err_d = _score(labels, ball, cfg, view)
        per[rec["clip"]["name"]] = r
        pooled.update(out)
        why.update(missed_why)
        epx.extend(err_px)
        ed.extend(err_d)
    pooled_r = summarise(pooled, why, epx, ed, cfg, detector_all)
    return {"schema": "spana.ball_score", "schema_version": "1.0.0", "split": "DEV", "score_config": cfg,
            "pooled": pooled_r, "clips": per}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", nargs="+", action="append", required=True,
                   metavar="PATH", help="LABEL_DIR BALL_JSON [BALL_VIEW_JSON]")
    p.add_argument("--score-config", default=str(DEFAULT_SCORE_CONFIG))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    refuse_existing(a.out)
    rep = run(a.clip, a.score_config)
    write_json_atomic(Path(a.out), rep)
    q = rep["pooled"]
    print(json.dumps({k: q[k] for k in ("claims", "wrong", "wrong_rate", "wrong_bound", "coverage")}
                     | ({"detector_recall": q["detector_recall"]} if "detector_recall" in q else {})))
    for name, r in rep["clips"].items():
        print(f"  {name[:8]}: wrong {r['wrong']}/{r['claims']} coverage {r['coverage']} "
              f"err median {None if not r['position_error_px'] else r['position_error_px']['median']} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
