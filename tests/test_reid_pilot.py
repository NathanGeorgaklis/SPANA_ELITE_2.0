"""Re-ID pilot tooling: crop extraction from frozen TRAIN labels, PK batches, augmentation, input convention, holdout
rival check, and (with torch + transformers) a tiny end-to-end train / resume / ONNX export."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spana.reid_labels import REQUIRED_PERMISSIONS, RECOMMENDED_PERMISSIONS, ReidLabelError, manifest_template
from spana.video import sha256_file
from tools import reid_crops, reid_train

FPS, W, H, N = 10.0, 320, 240, 40
COLOURS = {1: (200, 60, 60), 2: (60, 200, 60), 3: (60, 60, 200)}     # MOT id -> BGR


def _box(pid, f):
    return (20 + 90 * (pid - 1) + f, 60.0, 70 + 90 * (pid - 1) + f, 180.0)


def make_fixture(root: Path, split="TRAIN"):
    video = root / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 40, np.uint8)
        for pid, c in COLOURS.items():
            x1, y1, x2, y2 = (int(v) for v in _box(pid, f))
            img[y1:y2, x1:x2] = c
        w.write(img)
    w.release()
    lab = root / "labels"
    lab.mkdir()
    m = manifest_template("M_TEST_001", split, sha256_file(str(video)))
    m.update(footage_licence="test", rights_holder="test", consent_basis="test", training_permitted=True,
             agreement_reference="test", permissions={k: True for k in REQUIRED_PERMISSIONS + RECOMMENDED_PERMISSIONS})
    players = {"A7": {"id": 1, "shirt_number": "7", "side": "A"}, "A9": {"id": 2, "shirt_number": "9", "side": "A"},
               "B4": {"id": 3, "shirt_number": "4", "side": "B"}}
    mot = []
    for f in range(0, N, 2):                                    # labelled every 2nd frame, 1-based in MOT
        for pid in COLOURS:
            x1, y1, x2, y2 = _box(pid, f)
            mot.append(f"{f + 1},{pid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,1,1")
    files = {"manifest.json": json.dumps(m), "players.json": json.dumps(players), "boxes.mot.txt": "\n".join(mot) + "\n",
             "identities.jsonl": ""}
    for k, v in files.items():
        (lab / k).write_text(v)
    (lab / "FROZEN.json").write_text(json.dumps({
        "split": split, "match_id": "M_TEST_001", "label_version": 1,
        "files": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()}}))
    return video, lab


class CropTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_parse_mot_is_one_based(self):
        b = reid_crops.parse_mot("1,4,10,20,30,40,1,1,1\n3,4,0,0,5,5,1,1,1\n")
        self.assertEqual(sorted(b), [0, 2])
        self.assertEqual(b[0], [(4, (10.0, 20.0, 40.0, 60.0))])

    def test_sample_gap_height_holdout_overlap(self):
        boxes = {f: [(1, (0, 0, 10, 50)), (2, (5, 0, 15, 50)), (3, (100, 0, 110, 20))] for f in range(0, 20)}
        rows = reid_crops.sample(boxes, fps=10.0, min_gap_s=0.5, min_h=30, holdout_tail=0.25)
        self.assertNotIn(3, {r["id"] for r in rows})                     # too small
        f1 = [r["frame"] for r in rows if r["id"] == 1]
        self.assertEqual(f1, [0, 5, 10, 15])                              # every 0.5 s
        self.assertEqual({r["split"] for r in rows if r["frame"] >= 15}, {"holdout"})
        self.assertEqual({r["split"] for r in rows if r["frame"] < 15}, {"train"})
        self.assertAlmostEqual(rows[0]["overlap"], 1 / 3, places=3)

    def test_cut_crop_pads_clips_and_resizes(self):
        img = np.zeros((100, 100, 3), np.uint8)
        c = reid_crops.cut_crop(img, (-10, 10, 30, 90), 0.05, (224, 112))
        self.assertEqual(c.shape, (224, 112, 3))

    def test_main_cuts_train_crops_and_refuses_dev(self):
        video, lab = make_fixture(self.d)
        out = self.d / "crops"
        reid_crops.main(["--labels", str(lab), "--video", str(video), "--out", str(out)])
        meta = json.loads((out / "crops.json").read_text())
        rows = [json.loads(x) for x in (out / "crops.jsonl").read_text().splitlines()]
        self.assertEqual(meta["players"], 3)
        self.assertEqual(meta["crops"], len(rows))
        self.assertEqual({r["player"] for r in rows}, {"A7", "A9", "B4"})
        r = next(r for r in rows if r["player"] == "B4")
        img = cv2.imread(str(out / r["file"]))
        self.assertEqual(img.shape, (224, 112, 3))
        centre = img[100:120, 40:70].reshape(-1, 3).mean(axis=0)
        self.assertLess(np.abs(centre - np.array(COLOURS[3])).max(), 25)  # the right player's colour
        self.assertEqual(hashlib.sha256((out / r["file"]).read_bytes()).hexdigest(), r["sha256"])
        with self.assertRaises(SystemExit):                                # never overwritten
            reid_crops.main(["--labels", str(lab), "--video", str(video), "--out", str(out)])
        d2 = self.d / "dev"
        d2.mkdir()
        video2, lab2 = make_fixture(d2, split="DEV")
        with self.assertRaises(ReidLabelError):
            reid_crops.main(["--labels", str(lab2), "--video", str(video2), "--out", str(d2 / "c")])


def _rows(n_players=6, per=8):
    rows = []
    for p in range(n_players):
        for k in range(per):
            rows.append({"identity": f"M/{p}", "team": f"M/{'A' if p % 2 else 'B'}", "match": "M", "t": float(k),
                         "split": "train"})
    return rows


class TrainDataTests(unittest.TestCase):
    def test_pk_batches(self):
        s = reid_train.PKSampler(_rows(), p=4, k=3, same_team_share=0.5, seed=0)
        for _ in range(20):
            b = s.batch()
            self.assertEqual(len(b), 12)
            ids = [s_ for s_ in {_rows()[i]["identity"] for i in b}]
            self.assertEqual(len(ids), 4)
            teams = [_rows()[i]["team"] for i in b]
            self.assertGreaterEqual(max(teams.count("M/A"), teams.count("M/B")), 6)

    def test_augment_keeps_shape_and_is_seeded(self):
        cfg = reid_crops.load_train_config("config/reid_train.yaml")["augment"]
        img = np.random.default_rng(1).integers(0, 255, (224, 112, 3), dtype=np.uint8)
        a = reid_train.augment(img, cfg, np.random.default_rng(3))
        b = reid_train.augment(img, cfg, np.random.default_rng(3))
        self.assertEqual(a.shape, img.shape)
        self.assertEqual(a.dtype, np.uint8)
        self.assertTrue(np.array_equal(a, b))

    def test_input_matches_eval_embedder_convention(self):
        from tools.dev.appearance_cue_eval import MEAN, STD
        img = np.random.default_rng(2).integers(0, 255, (224, 112, 3), dtype=np.uint8)
        ref = ((img[:, :, ::-1] / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        self.assertLess(np.abs(reid_train.to_input([img])[0] - ref).max(), 1e-5)

    def test_rival_triplets_same_team_and_gap(self):
        rows = _rows(n_players=4, per=4)
        tri = reid_train.rival_triplets(rows, min_gap_s=1.0)
        self.assertTrue(tri)
        for q, a, b in tri:
            self.assertEqual(rows[q]["identity"], rows[a]["identity"])
            self.assertNotEqual(rows[q]["identity"], rows[b]["identity"])
            self.assertEqual(rows[q]["team"], rows[b]["team"])
            self.assertGreaterEqual(abs(rows[q]["t"] - rows[a]["t"]), 1.0)

    def test_load_crops_refuses_non_train_and_edited_index(self):
        d = Path(tempfile.mkdtemp())
        try:
            text = json.dumps({"player": "A7", "side": "A", "file": "A7/0000000.jpg", "t": 0.0, "split": "train"}) + "\n"
            (d / "crops.jsonl").write_text(text)
            meta = {"split": "TRAIN", "match_id": "M", "index_sha256": hashlib.sha256(text.encode()).hexdigest()}
            (d / "crops.json").write_text(json.dumps(meta))
            self.assertEqual(reid_train.load_crops([d])[0]["identity"], "M/A7")
            (d / "crops.json").write_text(json.dumps({**meta, "split": "DEV"}))
            with self.assertRaises(SystemExit):
                reid_train.load_crops([d])
            (d / "crops.json").write_text(json.dumps({**meta, "index_sha256": "0" * 64}))
            with self.assertRaises(SystemExit):
                reid_train.load_crops([d])
        finally:
            shutil.rmtree(d)


def _torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_torch(), "torch / transformers not installed")
class TrainEndToEndTests(unittest.TestCase):
    """A tiny random DINOv2 (no download): train 2 epochs in two calls (resume), export ONNX, outputs agree."""

    def test_train_resume_export(self):
        from transformers import Dinov2Config, Dinov2Model
        d = Path(tempfile.mkdtemp())
        try:
            video, lab = make_fixture(d)
            crops = d / "crops"
            reid_crops.main(["--labels", str(lab), "--video", str(video), "--out", str(crops)])
            bb = d / "bb"
            Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                                     intermediate_size=64, patch_size=14, image_size=56)).save_pretrained(bb)
            pin = d / "pin.json"
            pin.write_text(json.dumps({"weights_file": "model.safetensors",
                                       "weights_sha256": reid_train.sha256_file(bb / "model.safetensors")}))
            cfg = reid_crops.load_train_config("config/reid_train.yaml")
            cfg["model"]["frozen_blocks"] = 1
            cfg["train"].update(epochs=2, batches_per_epoch=2, p=3, k=2, check_min_gap_s=0.3)
            cfg_p = d / "cfg.yaml"
            import yaml
            cfg_p.write_text(yaml.safe_dump(cfg))
            old_pin, reid_train.PIN = reid_train.PIN, pin
            try:
                run = d / "run"
                args = ["train", "--crops", str(crops), "--backbone", str(bb), "--out", str(run), "--config", str(cfg_p)]
                reid_train.main(args + ["--max-minutes", "1e-9"])                   # stops after epoch 1
                self.assertEqual(len((run / "log.jsonl").read_text().splitlines()), 2)
                with self.assertRaises(SystemExit):                                 # unfinished: no export
                    reid_train.main(["export", "--run", str(run), "--backbone", str(bb), "--out", str(d / "m.onnx")])
                reid_train.main(args)                                               # resumes, finishes epoch 2
                log = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
                self.assertEqual([r["epoch"] for r in log], [0, 1, 2])
                self.assertGreater(log[-1]["holdout_rival_n"], 0)
                reid_train.main(["export", "--run", str(run), "--backbone", str(bb), "--out", str(d / "m.onnx")])
                card = json.loads((d / "m.onnx.json").read_text())
                self.assertLess(card["max_abs_diff_torch_onnx"], 1e-3)
                from tools.dev.appearance_cue_eval import Embedder
                e = Embedder(d / "m.onnx", (224, 112), lambda y: y)
                z = e([np.zeros((150, 60, 3), np.uint8), np.full((90, 40, 3), 200, np.uint8)])
                self.assertEqual(z.shape, (2, cfg["model"]["dim"]))
                self.assertTrue(np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5))
                bad = d / "bad.json"
                bad.write_text(json.dumps({"weights_file": "model.safetensors", "weights_sha256": "0" * 64}))
                reid_train.PIN = bad
                with self.assertRaises(SystemExit):                                 # unpinned weights refused
                    reid_train.main(["train", "--crops", str(crops), "--backbone", str(bb), "--out", str(d / "r2"),
                                     "--config", str(cfg_p)])
            finally:
                reid_train.PIN = old_pin
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
