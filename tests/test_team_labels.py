"""DEV side labels for team classification: contact sheets (tools.team_label_sheet) and freezing
(tools.freeze_team_labels) on a synthetic frozen DEV label version."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from spana.labels import LabelError
from spana.reid_labels import ReidLabelError
from spana.video import sha256_file
from tools.freeze_team_labels import freeze
from tools.team_label_sheet import build_sheets, load_cfg, pick_crops, load_frozen_dev

W, H, N = 320, 240, 12
CFG = {"schema_version": "1.0.0", "crops_per_id": 3, "clean_max_iou": 0.05, "thumb_height_px": 60,
       "cell_width_px": 40, "context_width_px": 80, "rows_per_page": 2}


def make(tmp_path, split="DEV"):
    video = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (W, H))
    for _ in range(N):
        img = np.full((H, W, 3), 60, np.uint8)
        img[50:150, 20:60] = (0, 0, 220)
        img[50:150, 120:160] = (220, 0, 0)
        vw.write(img)
    vw.release()
    lines = []
    for f in range(0, N, 3):
        lines.append(f"{f + 1},1,20,50,40,100,1,1,1")
        lines.append(f"{f + 1},2,120,50,40,100,1,1,1")
        lines.append(f"{f + 1},3,130,60,40,100,1,1,0.5")      # overlaps id 2: never clean
    gt = ("\n".join(lines) + "\n").encode()
    d = tmp_path / "labels" / "CLIP_X" / "v1"
    d.mkdir(parents=True)
    (d / "gt.txt").write_bytes(gt)
    rec = {"split": split, "frames_labelled": list(range(0, N, 3))}
    (d / "labels_frozen.json").write_text(json.dumps(rec))
    clip = {"name": "CLIP_X", "sha256": sha256_file(video), "width": W, "height": H, "fps": 10.0}
    (d / "FROZEN.json").write_text(json.dumps({
        "split": split, "clip": clip, "label_version": 1, "gt_sha256": hashlib.sha256(gt).hexdigest(),
        "labels_record_sha256": hashlib.sha256((d / "labels_frozen.json").read_bytes()).hexdigest()}))
    return d, video


def test_pick_crops_prefers_clean(tmp_path):
    d, _ = make(tmp_path)
    _, gt = load_frozen_dev(d)
    picks = pick_crops(gt, 3, 0.05)
    assert list(picks) == ["1", "2", "3"]
    assert [f for f, _, _ in picks["1"]] == [0, 6, 9]
    assert all(c for _, _, c in picks["1"]) and not any(c for _, _, c in picks["3"])


def test_sheets_and_template_then_freeze(tmp_path):
    d, video = make(tmp_path)
    out = build_sheets("DEV_GX", [(d, str(video))], CFG, tmp_path / "out")
    pngs = [p for p in out if p.suffix == ".png"]
    assert len(pngs) == 2 and cv2.imread(str(pngs[0])) is not None
    tpl_path = tmp_path / "out" / "CLIP_X.sides.json"
    tpl = json.loads(tpl_path.read_text())
    assert tpl["sides"] == {"1": None, "2": None, "3": None} and tpl["match_id"] == "DEV_GX"
    with pytest.raises(LabelError, match="sides must be one of"):
        tpl["annotator"] = "nathan"
        tpl_path.write_text(json.dumps(tpl))
        freeze(tpl_path, d)
    tpl["sides"] = {"1": "A", "2": "B", "3": "REF"}
    tpl_path.write_text(json.dumps(tpl))
    dest = freeze(tpl_path, d)
    fz = json.loads((dest / "FROZEN.json").read_text())
    assert dest.name == "teams_v1" and fz["counts"]["A"] == 1 and fz["counts"]["REF"] == 1 and fz["ids"] == 3
    assert (dest / "sides.json").read_bytes() == tpl_path.read_bytes()
    with pytest.raises(LabelError, match="never overwritten"):
        freeze(tpl_path, d)


def test_freeze_refusals(tmp_path):
    d, video = make(tmp_path)
    build_sheets("DEV_GX", [(d, str(video))], CFG, tmp_path / "out")
    p = tmp_path / "out" / "CLIP_X.sides.json"
    base = json.loads(p.read_text())
    base.update(annotator="nathan", sides={"1": "A", "2": "B", "3": "GK_A"})

    def attempt(**kw):
        doc = {**base, **kw}
        p.write_text(json.dumps(doc))
        return freeze(p, d)
    with pytest.raises(LabelError, match="missing"):
        attempt(sides={"1": "A", "2": "B"})
    with pytest.raises(LabelError, match="annotator"):
        attempt(annotator=" ")
    with pytest.raises(LabelError, match="other label version|another label version"):
        attempt(gt_sha256="0" * 64)
    with pytest.raises(LabelError, match="different clips"):
        attempt(clip={"name": "CLIP_X", "sha256": "1" * 64})
    with pytest.raises(ReidLabelError):
        attempt(split="EVAL")
    (d / "gt.txt").write_bytes((d / "gt.txt").read_bytes() + b"\n")
    with pytest.raises(LabelError, match="frozen hash"):
        attempt()
    assert not (d.parent / "teams_v1").exists()


def test_config_is_strict(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(json.dumps({**CFG, "extra": 1}))
    with pytest.raises(LabelError):
        load_cfg(str(p))
    p.write_text(json.dumps(CFG))
    assert load_cfg(str(p)) == CFG
