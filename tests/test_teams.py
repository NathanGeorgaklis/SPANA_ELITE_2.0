"""Team classification v1 (spana.teams.classifier, spana.teams 1.0.0) on synthetic clips: two painted kits, role
outliers, uncertainty-episode segments, abstention, determinism and artifact validation."""
import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from spana.artifacts import ArtifactError, TEAMS_SCHEMA, sha256_doc, validate_teams
from spana.config import ConfigError, TeamConfig, load_config
from spana.teams.classifier import classify, label_at, model_verdict, segments_of_clip

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(ROOT / "config" / "default.yaml")
NAMES = ["player", "goalkeeper", "referee"]
RED, BLUE, YELLOW, GREEN, PURPLE = (0, 0, 220), (220, 0, 0), (0, 220, 220), (0, 200, 0), (200, 0, 200)
N = 100   # frames at 25 fps = 4 s
ROLES_ON = dataclasses.replace(CFG.team, roles_enabled=True)   # the role rule itself (off in v1)


def spec(tid, slot, colour, cls=0, identity=lambda f: "CERTAIN", start=0, box=None):
    x = 20 + slot * 90
    return {"tid": tid, "box": box or [x, 100.0, x + 60.0, 260.0], "colour": colour, "cls": cls,
            "identity": identity, "start": start}


def clip(specs, n=N):
    """(tracks, detections, frames) of a synthetic clip: every track observed on every frame from its start."""
    tframes, dframes, imgs = [], [], []
    for f in range(n):
        img = np.full((360, 1600, 3), 60, np.uint8)
        rows, dets = [], []
        for s in specs:
            if f < s["start"]:
                continue
            b = s["box"]
            col = s["colour"](f) if callable(s["colour"]) else s["colour"]
            x0 = int(b[0])
            for frac, c in (col if isinstance(col, list) else [(1.0, col)]):   # vertical stripes, left to right
                xe = x0 + int(round(frac * (int(b[2]) - int(b[0]))))
                img[int(b[1]):int(b[3]), x0:max(x0, xe)] = c
                x0 = xe
            rows.append({"track_id": s["tid"], "confirmed": True, "identity": s["identity"](f), "source": "OBSERVED",
                         "detection_index": len(dets), "box": b})
            dets.append([*b, 0.9, s["cls"], 0])
        tframes.append({"frame": f, "t": f / 25.0, "camera": {"status": "FIRST" if f == 0 else "OK"},
                        "tracks": rows, "ended": []})
        dframes.append({"frame": f, "t": f / 25.0, "detections": dets})
        imgs.append((f, img))
    det = {"class_names": NAMES, "frames": dframes}
    tracks = {"video": {"fps": 25.0, "sha256": "v" * 64}, "detections_sha256": sha256_doc(det), "frames": tframes}
    return tracks, det, imgs


def two_kits(extra=()):
    specs = [spec(f"R{i}", i, RED) for i in range(4)] + [spec(f"B{i}", 4 + i, BLUE, start=1) for i in range(4)]
    return specs + list(extra)


def run(specs, tcfg=CFG.team, name="c1"):
    tracks, det, imgs = clip(specs)
    segs = segments_of_clip(name, tracks, det, imgs, CFG.appearance, tcfg)
    model, rows = classify(segs, tcfg, CFG.output.score_decimals)
    doc = {"schema": TEAMS_SCHEMA[0], "schema_version": TEAMS_SCHEMA[1], "provenance": {}, "match_id": "M",
           "inputs": [{"clip": name, "video_sha256": tracks["video"]["sha256"], "tracks_sha256": sha256_doc(tracks),
                       "detections_sha256": tracks["detections_sha256"]}],
           "team_config": dataclasses.asdict(tcfg), "appearance_config": dataclasses.asdict(CFG.appearance),
           "feature": "torso_colour_hs_hist_v1", "model": model, "tracks": rows}
    validate_teams(doc, {name: tracks})
    return doc, tracks


def labels(doc):
    return {t["track_id"]: [s["label"] for s in t["segments"]] for t in doc["tracks"]}


def test_two_kits_and_roles():
    doc, _ = run(two_kits([spec("Y", 8, YELLOW, cls=2), spec("G", 9, GREEN, cls=1), spec("P", 10, PURPLE, cls=0)]),
                 ROLES_ON)
    lab = labels(doc)
    assert doc["model"]["status"] == "FITTED" and doc["model"]["separation"] > 0.9
    assert all(lab[f"R{i}"] == ["TEAM_A"] for i in range(4))   # TEAM_A holds the earliest-starting segment
    assert all(lab[f"B{i}"] == ["TEAM_B"] for i in range(4))
    assert lab["Y"] == ["REFEREE"] and lab["G"] == ["GOALKEEPER"] and lab["P"] == ["UNKNOWN"]
    p = next(t for t in doc["tracks"] if t["track_id"] == "P")["segments"][0]
    assert p["reasons"] == ["COLOUR_OUTLIER_NO_ROLE"]
    assert label_at(doc, "c1", "R0", 50) == ("TEAM_A", None)
    assert label_at(doc, "c1", "nope", 50) == ("UNKNOWN", "OUTSIDE_SEGMENT")


def test_detector_role_never_overrides_a_team_kit():
    doc, _ = run(two_kits([spec("Rref", 8, RED, cls=2)]), ROLES_ON)
    assert labels(doc)["Rref"] == ["TEAM_A"]


def test_role_hint_conflict():
    doc, _ = run(two_kits([spec("Y", 8, YELLOW, cls=2), spec("Y2", 9, YELLOW, cls=1, box=[830.0, 100, 890, 260])]),
                 ROLES_ON)
    assert labels(doc)["Y"] == ["REFEREE"] and labels(doc)["Y2"] == ["GOALKEEPER"]
    mixed = spec("M", 10, YELLOW, cls=0)
    tracks, det, imgs = clip(two_kits([mixed]))
    for k, fr in enumerate(det["frames"]):          # half referee, half goalkeeper hints
        fr["detections"][-1][5] = 2 if k % 2 else 1
    tracks["detections_sha256"] = sha256_doc(det)
    segs = segments_of_clip("c1", tracks, det, imgs, CFG.appearance, ROLES_ON)
    _, rows = classify(segs, ROLES_ON, 4)
    m = next(t for t in rows if t["track_id"] == "M")["segments"][0]
    assert m["label"] == "UNKNOWN" and m["reasons"] == ["ROLE_HINT_CONFLICT"]


def test_low_vote_share():
    doc, _ = run(two_kits([spec("X", 8, lambda f: BLUE if f % 10 < 3 else RED)]))
    x = next(t for t in doc["tracks"] if t["track_id"] == "X")["segments"][0]
    assert x["label"] == "UNKNOWN" and x["reasons"] == ["LOW_VOTE_SHARE"]
    assert x["votes"]["TEAM_A"] == 70 and x["votes"]["TEAM_B"] == 30


def test_segments_split_at_uncertainty_and_team_change_flag():
    unsure = lambda f: "IDENTITY_UNCERTAIN" if 40 <= f < 50 else "CERTAIN"   # noqa: E731
    swap = spec("S", 8, lambda f: RED if f < 45 else BLUE, identity=unsure)
    doc, _ = run(two_kits([swap]))
    s = next(t for t in doc["tracks"] if t["track_id"] == "S")
    assert [(g["frame"], g["frame_end"], g["label"]) for g in s["segments"]] == [(0, 39, "TEAM_A"),
                                                                                 (50, 99, "TEAM_B")]
    assert s["flags"] == ["TEAM_CHANGE"]
    assert label_at(doc, "c1", "S", 45) == ("UNKNOWN", "OUTSIDE_SEGMENT")


def test_unclean_crops_give_no_evidence():
    over = spec("O", 0, RED, box=[30.0, 100, 90, 260])   # overlaps R0 on every frame
    doc, _ = run(two_kits([over]))
    lab = labels(doc)
    assert lab["O"] == ["UNKNOWN"] and lab["R0"] == ["UNKNOWN"]
    o = next(t for t in doc["tracks"] if t["track_id"] == "O")["segments"][0]
    assert o["reasons"] == ["TOO_FEW_CLEAN_FRAMES"] and o["evidence_frames"] == 0


def test_one_kit_abstains():
    doc, _ = run([spec(f"R{i}", i, RED) for i in range(8)])
    assert doc["model"]["status"] == "ABSTAIN"
    assert {g["label"] for t in doc["tracks"] for g in t["segments"]} == {"UNKNOWN"}
    assert {r for t in doc["tracks"] for g in t["segments"] for r in g["reasons"]} == {"MATCH_MODEL_ABSTAIN"}


def test_too_few_segments_abstains():
    doc, _ = run([spec("R0", 0, RED), spec("B0", 1, BLUE)])
    assert doc["model"] == {**doc["model"], "status": "ABSTAIN", "reason": "TOO_FEW_SEGMENTS"}


def test_deterministic_and_order_independent():
    specs = two_kits([spec("Y", 8, YELLOW, cls=2)])
    a, _ = run(specs)
    b, _ = run(list(reversed(specs)))
    assert sha256_doc([a["model"], a["tracks"]]) == sha256_doc([b["model"], b["tracks"]])


def test_validation_refuses_bad_documents():
    doc, tracks = run(two_kits())
    bad = copy.deepcopy(doc)
    bad["tracks"][0]["segments"][0]["reasons"] = ["LOW_VOTE_SHARE"]
    with pytest.raises(ArtifactError):
        validate_teams(bad)
    bad = copy.deepcopy(doc)
    bad["tracks"][0]["flags"] = ["TEAM_CHANGE"]
    with pytest.raises(ArtifactError):
        validate_teams(bad)
    t2 = copy.deepcopy(tracks)
    t2["frames"][10]["tracks"][0]["identity"] = "IDENTITY_UNCERTAIN"
    bad = copy.deepcopy(doc)
    bad["inputs"][0]["tracks_sha256"] = sha256_doc(t2)
    with pytest.raises(ArtifactError, match="not confirmed and CERTAIN"):
        validate_teams(bad, {"c1": t2})
    with pytest.raises(ArtifactError):
        validate_teams(doc, {"c1": t2})                   # hash mismatch


def test_config_fails_closed():
    for k, v in (("min_vote_share", 0.5), ("feature", "reid"), ("role_conflict_share", 0.7),
                 ("goalkeeper_class", "referee")):
        with pytest.raises(ConfigError):
            dataclasses.replace(CFG.team, **{k: v}).validate()
    assert isinstance(CFG.team, TeamConfig)


ORANGE = [(0.3, RED), (1.0, YELLOW)]   # torso 30 % red: nearer red than blue, beyond outlier_distance from both


def test_roles_off_outliers_stay_unknown_never_nearest_team():
    """v1 default (decision 2026-11-22): referees / keepers / any colour outlier answer UNKNOWN, even with a strong
    detector role hint and even when one team is clearly nearer."""
    assert CFG.team.roles_enabled is False
    doc, _ = run(two_kits([spec("Y", 8, YELLOW, cls=2), spec("G", 9, GREEN, cls=1), spec("O", 10, ORANGE, cls=0)]))
    assert doc["model"]["status"] == "FITTED"
    for tid in ("Y", "G", "O"):
        seg = next(t for t in doc["tracks"] if t["track_id"] == tid)["segments"][0]
        assert seg["label"] == "UNKNOWN" and seg["reasons"] == ["COLOUR_OUTLIER"], tid
        assert min(seg["dist"].values()) > CFG.team.outlier_distance
    o = next(t for t in doc["tracks"] if t["track_id"] == "O")["segments"][0]
    assert o["dist"]["TEAM_A"] < o["dist"]["TEAM_B"]          # a nearest team exists; it is still not answered
    assert labels(doc)["R0"] == ["TEAM_A"] and labels(doc)["B0"] == ["TEAM_B"]
    bad = copy.deepcopy(doc)
    seg = next(t for t in bad["tracks"] if t["track_id"] == "Y")["segments"][0]
    seg.update(label="REFEREE", reasons=[])
    with pytest.raises(ArtifactError, match="roles_enabled false"):
        validate_teams(bad)


def test_overlapping_clusters_abstain():
    """Match-level rule (decision 2026-11-22): separation must exceed team.min_separation_radius_ratio x the larger
    cluster radius. Numbers: TRAIN_M001 (0.4385 vs radius 0.4497) abstains; the DEV matches (0.80-0.98 vs <= 0.45)
    do not."""
    t = CFG.team
    assert t.min_separation_radius_ratio == 1.0
    assert model_verdict(0.4385, (0.4446, 0.4497), (207, 370), t) == "CLUSTERS_OVERLAP"
    assert model_verdict(0.8015, (0.4384, 0.3431), (8, 11), t) is None
    assert model_verdict(0.4497, (0.4497, 0.1), (5, 5), t) == "CLUSTERS_OVERLAP"     # equal is not enough
    assert model_verdict(0.25, (0.1, 0.1), (5, 5), t) == "CLUSTERS_NOT_SEPARATED"
    assert model_verdict(0.9, (0.3, 0.3), (5, 2), t) == "CLUSTER_TOO_SMALL"
    assert model_verdict(0.4385, (0.4446, 0.4497), (207, 370),
                         dataclasses.replace(t, min_separation_radius_ratio=0.9)) is None
