"""Group-then-confirm re-ID labelling (tools/reid_cluster.py, tools/reid_cluster_review.html, freeze gate):
same-frame rule, exclusion of tracker-flagged tracklets, odd-ones-out order, deterministic audit, Wilson CI, freeze
refusals, M001-style manual exports unaffected, and the review page end to end (Node Playwright, if available)."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from spana.reid_labels import ReidLabelError, load_frozen, manifest_template
from tools import reid_cluster as RC
from tools.freeze_reid_labels import freeze

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 360
RED, RED2, BLUE, GREEN = (40, 40, 220), (70, 70, 190), (220, 60, 40), (40, 200, 40)   # A / B: similar kits
# tracklet: (player, kit, frames, flagged); A and B wear the same kit and are on the pitch at the same time
TRACKLETS = {
    "T000001-1": ("A", RED, range(0, 40, 4), False), "T000002-1": ("B", RED2, range(0, 40, 4), False),
    "T000003-1": ("A", RED, range(100, 140, 4), False), "T000004-1": ("B", RED2, range(100, 140, 4), False),
    "T000005-1": ("C", BLUE, range(0, 40, 4), False), "T000006-1": ("C", BLUE, range(200, 240, 4), False),
    "T000007-1": ("C", BLUE, range(300, 340, 4), False), "T000008-1": ("D", GREEN, range(0, 40, 4), True),
}
X = {"A": 100, "B": 250, "C": 400, "D": 520}


def make_pack(root: Path, split="TRAIN") -> Path:
    pd = root / f"pack_{split}"
    (pd / "thumbs").mkdir(parents=True)
    tl = []
    for tid, (pl, kit, frames, flagged) in TRACKLETS.items():
        frames = list(frames)
        boxes = [[float(X[pl]), 120.0, float(X[pl] + 40), 260.0] for _ in frames]
        thumbs = []
        for k, f in enumerate(frames[::3]):
            img = np.full((H, W, 3), 90, np.uint8)
            x1, y1, x2, y2 = (int(v) for v in boxes[0])
            img[y1:y2, x1:x2] = kit
            mw, mh = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            c = img[int(y1 - mh):int(y2 + mh), int(x1 - mw):int(x2 + mw)]
            c = cv2.resize(c, (round(c.shape[1] * 112 / c.shape[0]), 112))
            name = f"thumbs/{tid}_{k}.jpg"
            cv2.imwrite(str(pd / name), c)
            thumbs.append({"frame": f, "file": name})
        tl.append({"id": tid, "track": tid[:7], "frames": frames, "boxes": boxes, "starts_uncertain": flagged,
                   "after_gap": False, "uncertain_frames": 0, "confirmed": True, "thumbs": thumbs})
    tl_bytes = json.dumps(tl, separators=(",", ":")).encode()
    (pd / "tracklets.json").write_bytes(tl_bytes)
    m = manifest_template("M_TEST_CL", split, "1" * 64)
    m.update(footage_licence="synthetic", rights_holder="SPANA", consent_basis="synthetic", training_permitted=True,
             agreement_reference="synthetic", permissions={k: True for k in m["permissions"]})
    pack = {"schema": "spana.reid_pack", "schema_version": "1.0.0", "match": m,
            "video": {"name": "v.mp4", "sha256": "1" * 64, "fps": 25.0, "width": W, "height": H},
            "tracks_sha256": "2" * 64, "tracklets_sha256": hashlib.sha256(tl_bytes).hexdigest(),
            "proposal": {"max_gap_s": 0.5, "min_observed": 1, "thumbs": 8},
            "tracklets": [{"id": t["id"], "track": t["track"], "first_frame": t["frames"][0],
                           "last_frame": t["frames"][-1], "thumbs": t["thumbs"], "frames": t["frames"]} for t in tl]}
    (pd / "pack.json").write_text(json.dumps(pack))
    return pd


def colour_embed(crops):
    """Fake re-ID model: the mean colour of the crop's centre (identical kits -> identical embeddings)."""
    v = np.array([c[c.shape[0] // 3:2 * c.shape[0] // 3, c.shape[1] // 4:3 * c.shape[1] // 4].reshape(-1, 3).mean(0)
                  for c in crops], float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def write_cfg(root: Path, **audit) -> Path:
    cfg = RC.load_cluster_config()
    cfg["audit"].update(audit)
    p = root / "cluster.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def export_from(review: dict, assign: dict, audit_answers=None, overrides=None, clusters=None, players=None) -> dict:
    """What the review page exports: assign = {cluster: (decision, player)}; audit answers default to the truth."""
    clusters = clusters or {k: list(c["tracklets"]) for k, c in review["clusters"].items()}
    overrides = overrides or {}
    audit_answers = audit_answers if audit_answers is not None else \
        {t: {"decision": "player", "player": TRACKLETS[t][0]} for t in review["audit"]}
    rv = {"schema": RC.REVIEW_SCHEMA[0], "schema_version": RC.REVIEW_SCHEMA[1], "config_sha256": review["config_sha256"],
          "model_sha256": None, "excluded": review["excluded"], "overrides": overrides, "audit": audit_answers,
          "clusters": {k: {"tracklets": m, "decision": assign.get(k, (None, None))[0],
                           "player": assign.get(k, (None, None))[1]} for k, m in clusters.items()}}
    derived = RC.derived_decisions(rv, [t["id"] for t in review["tracklets"]])
    parts = []
    for t in review["tracklets"]:
        d = RC._decision(audit_answers[t["id"]]) if t["id"] in audit_answers else derived[t["id"]]
        p = {"tracklet": t["id"], "first_frame": t["first_frame"], "last_frame": t["last_frame"], "decision": d[0],
             "player": d[1]}
        if t["id"] in review["excluded"]:
            p["status"] = RC.EXCLUDED_STATUS
        parts.append(p)
    players = players or {k: {"side": "A" if k != "C" else "B", "shirt_number": None} for k in "ABC"}
    return {"schema": "spana.reid_label_export", "schema_version": "1.0.0", "match_id": review["match"]["match_id"],
            "split": "TRAIN", "tracklets_sha256": review["tracklets_sha256"], "annotator": "QA", "label_version": 1,
            "reason": "", "exported_utc": "2026-11-18T00:00:00Z", "players": players, "parts": parts,
            "labelling_flow": "cluster_confirm", "cluster_review": rv}


def truth_assign(review: dict) -> dict:
    out = {}
    for k, c in review["clusters"].items():
        who = {TRACKLETS[t][0] for t in c["tracklets"]}
        out[k] = ("player", who.pop()) if len(who) == 1 else ("unsure", None)
    return out


class Core(unittest.TestCase):
    def test_same_frame_rule_beats_identical_features(self):
        tl = [{"frames": [0, 4, 8]}, {"frames": [8, 12]}, {"frames": [20, 24]}]
        CL = RC.cannot_link(tl)
        self.assertTrue(CL[0, 1] and CL[1, 0] and not CL[0, 2])
        labels = RC.agglomerate(np.zeros((3, 3)), CL, 0.5)              # identical features
        self.assertNotEqual(labels[0], labels[1])
        self.assertIn(labels[2], (labels[0], labels[1]))

    def test_threshold_and_odd_ones_out(self):
        D = np.array([[0, .1, .2, .9], [.1, 0, .15, .9], [.2, .15, 0, .9], [.9, .9, .9, 0]])
        labels = RC.agglomerate(D, np.zeros((4, 4), bool), 0.3)
        self.assertEqual(labels[:3], [labels[0]] * 3)
        self.assertNotEqual(labels[3], labels[0])
        odd = RC.odd_scores(D, labels)
        self.assertEqual(max(range(3), key=lambda i: odd[i]), 2)        # the farthest member first in review
        self.assertEqual(odd[3], 0.0)

    def test_audit_sample_deterministic_and_wilson(self):
        cfg = RC.load_cluster_config()
        ids = [f"T{i:06d}-1" for i in range(400)]
        a = RC.audit_sample(ids, cfg)
        self.assertEqual(a, RC.audit_sample(list(reversed(ids)), cfg))
        self.assertEqual(len(a), 20)                                     # 5 % of 400
        self.assertEqual(len(RC.audit_sample(ids[:50], cfg)), 10)       # minimum
        lo, hi = RC.wilson(0, 20, 0.95)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1611, places=3)

    def test_thumb_box_recovers_the_player(self):
        img = np.full((H, W, 3), 90, np.uint8)
        img[120:260, 100:140] = RED
        c = img[99:281, 94:146]
        th = cv2.resize(c, (round(c.shape[1] * 112 / c.shape[0]), 112))
        b = RC.thumb_box([100, 120, 140, 260], (W, H), th.shape)
        crop = RC.model_crop(th, b, 0.0, (224, 112))
        self.assertLess(np.abs(crop[40:180, 20:90].reshape(-1, 3).mean(0) - RED).max(), 12)


class BuildAndFreeze(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.pack = make_pack(self.td)
        self.cfg = write_cfg(self.td, min_tracklets=3, fraction=0.05)
        self.review = RC.build(self.pack, self.td / "review", self.cfg, embed=colour_embed)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _freeze(self, ex: dict, name="ex.json"):
        p = self.td / name
        p.write_text(json.dumps(ex))
        return freeze(str(self.pack), str(p), root=self.td / "repo", cluster_config=self.cfg)

    def test_build(self):
        r = self.review
        self.assertEqual(r["excluded"], ["T000008-1"])                  # tracker-flagged
        self.assertTrue(all("T000008-1" not in c["tracklets"] for c in r["clusters"].values()))
        tl = {t: set(v[2]) for t, v in TRACKLETS.items()}
        for c in r["clusters"].values():                                # no cluster holds two tracklets on one frame
            ts = c["tracklets"]
            self.assertFalse(any(tl[a] & tl[b] for i, a in enumerate(ts) for b in ts[i + 1:]))
        cs = [set(c["tracklets"]) for c in r["clusters"].values()]
        self.assertIn({"T000005-1", "T000006-1", "T000007-1"}, cs)       # player C: one cluster
        self.assertEqual(len(r["audit"]), 3)
        self.assertNotIn("T000008-1", r["audit"])
        self.assertEqual(r["audit"], RC.audit_sample(r["audit"] and [t for t in TRACKLETS if t != "T000008-1"],
                                                     RC.load_cluster_config(self.cfg)))
        self.assertTrue((self.td / "review" / "index.html").exists() and (self.td / "review" / "thumbs").is_dir())
        with self.assertRaises(ReidLabelError):                         # never overwrites
            RC.build(self.pack, self.td / "review", self.cfg, embed=colour_embed)
        dev = make_pack(self.td, split="DEV")
        with self.assertRaises(ReidLabelError):                         # TRAIN only
            RC.build(dev, self.td / "review_dev", self.cfg, embed=colour_embed)

    def test_freeze_accepts_a_correct_review_and_records_the_audit(self):
        ex = export_from(self.review, truth_assign(self.review))
        dest = self._freeze(ex)
        fz = json.loads((dest / "FROZEN.json").read_text())
        self.assertEqual(fz["labelling_flow"], "cluster_confirm")
        self.assertEqual(fz["cluster_audit"]["disagreements"], 0)
        lab = load_frozen(dest, "train")
        rows = {r["tracklet"]: r for r in lab["identities"]}
        self.assertEqual(rows["T000008-1"]["decision"], "unsure")         # excluded: decided, not trained on
        self.assertEqual(json.loads((dest / "export.json").read_text())["parts"][-1]["status"], RC.EXCLUDED_STATUS)

    def test_freeze_refusals(self):
        good = export_from(self.review, truth_assign(self.review))
        audited = self.review["audit"]

        def refused(ex, msg):
            with self.assertRaises(ReidLabelError, msg=msg):
                self._freeze(ex, f"bad_{msg}.json")

        bad = copy.deepcopy(good)                                         # audit says otherwise -> disagreement
        cid = next(k for k, c in bad["cluster_review"]["clusters"].items() if audited[0] in c["tracklets"])
        bad = export_from(self.review, {**truth_assign(self.review), cid: ("reject", None)})
        refused(bad, "above_limit")
        bad = copy.deepcopy(good)
        del bad["cluster_review"]["audit"][audited[0]]
        refused(bad, "audit_incomplete")
        cl = {k: list(c["tracklets"]) for k, c in self.review["clusters"].items()}
        a_cl = next(k for k, m in cl.items() if "T000001-1" in m)
        b_cl = next(k for k, m in cl.items() if "T000002-1" in m)
        cl[a_cl].append("T000002-1")
        cl[b_cl].remove("T000002-1")
        refused(export_from(self.review, truth_assign(self.review), clusters=cl), "conflict")
        bad = copy.deepcopy(good)
        bad["parts"][-1].pop("status")
        refused(bad, "excluded_status")
        bad = copy.deepcopy(good)
        bad["cluster_review"]["config_sha256"] = "0" * 64
        refused(bad, "config")
        bad = copy.deepcopy(good)
        p = next(p for p in bad["parts"] if p["tracklet"] not in audited and p["tracklet"] != "T000008-1")
        p["decision"], p["player"] = "reject", None
        refused(bad, "parts_differ")

    def test_manual_export_still_freezes(self):
        ex = export_from(self.review, truth_assign(self.review))
        for k in ("labelling_flow", "cluster_review"):
            ex.pop(k)
        for p in ex["parts"]:
            p.pop("status", None)
        self._freeze(ex, "manual.json")

    def test_evaluate_against_frozen_labels(self):
        dest = self._freeze(export_from(self.review, truth_assign(self.review)))
        r = RC.evaluate(self.pack, dest, cfg_path=self.cfg, embed=colour_embed)
        self.assertEqual(r["errors"], 0)
        self.assertGreater(r["labelled_single_player"], 0)


NODE_SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const [dir, out] = process.argv.slice(2);
  const b = await chromium.launch(); const page = await b.newPage({ acceptDownloads: true });
  await page.goto('file://' + dir + '/index.html');
  const R = await page.evaluate(() => window.SPANA_CLUSTER_REVIEW);
  const truth = JSON.parse(process.argv[4]);
  await page.fill('#annotator', 'QA');
  for (const [k, side] of [['A', 'A'], ['B', 'A'], ['C', 'B']]) { await page.fill('#pkey', k); await page.selectOption('#pside', side); await page.click('#padd'); }
  const player = async k => page.locator('#players .player').filter({ hasText: new RegExp('^' + k + ' ') }).click();
  for (const [cid, c] of Object.entries(R.clusters)) {
    const shown = c.tracklets.filter(t => !R.audit.includes(t)); if (!shown.length) continue;
    const who = [...new Set(shown.map(t => truth[t]))];
    const card = page.locator(`.card[data-cid="${cid}"]`);
    if (who.length === 1) { await player(who[0]); await card.locator('button', { hasText: 'Assign cluster' }).click(); }
    else await card.locator('button', { hasText: 'Unsure cluster' }).click();
  }
  // CONFLICT: move a tracklet into a cluster holding another tracklet on the same frame -> flagged, export refused
  const cl = t => Object.keys(R.clusters).find(k => R.clusters[k].tracklets.includes(t));
  const shown = R.tracklets.filter(t => !R.audit.includes(t.id) && !R.excluded.includes(t.id));
  let pair = null;
  for (const x of shown) for (const y of shown) if (!pair && cl(x.id) !== cl(y.id) && x.frames.some(f => y.frames.includes(f))) pair = [x.id, y.id];
  await page.locator('.row', { hasText: pair[1] }).locator('select').selectOption(cl(pair[0]));
  if (!(await page.locator('.card.conflict').count())) throw new Error('no CONFLICT shown');
  await page.click('#export');
  if (!(await page.textContent('#status')).includes('CONFLICT')) throw new Error('export not refused on CONFLICT');
  await page.locator('.row', { hasText: pair[1] }).locator('button', { hasText: 'Unsure' }).click();
  if (await page.locator('.card.conflict').count()) throw new Error('CONFLICT not resolved by removing the tracklet');
  await page.click('#tabAudit');
  for (let i = 0; i < R.audit.length; i++) {
    await player(truth[R.audit[i]]);
    await page.locator('.card').nth(i).locator('button', { hasText: 'Assign active player' }).click();
  }
  const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#export')]);
  await dl.saveAs(out); console.log(await page.textContent('#status')); await b.close();
})().catch(e => { console.error(e); process.exit(1); });
"""


def _node_playwright() -> bool:
    if shutil.which("node") is None:
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0


@unittest.skipUnless(_node_playwright(), "node + playwright not available")
class ReviewPage(unittest.TestCase):
    def test_page_export_freezes(self):
        td = Path(tempfile.mkdtemp())
        try:
            pack = make_pack(td)
            cfg = write_cfg(td, min_tracklets=3)
            RC.build(pack, td / "review", cfg, embed=colour_embed)
            js = td / "drive.js"
            js.write_text(NODE_SCRIPT)
            out = td / "export.json"
            truth = json.dumps({t: v[0] for t, v in TRACKLETS.items()})
            r = subprocess.run(["node", str(js), str(td / "review"), str(out), truth], capture_output=True, text=True,
                               timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            ex = json.loads(out.read_text())
            self.assertEqual(ex["labelling_flow"], "cluster_confirm")
            dest = freeze(str(pack), str(out), root=td / "repo", cluster_config=cfg)
            self.assertEqual(json.loads((dest / "FROZEN.json").read_text())["cluster_audit"]["disagreements"], 0)
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
