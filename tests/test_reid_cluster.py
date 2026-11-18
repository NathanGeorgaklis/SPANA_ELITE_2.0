"""Group-then-confirm re-ID labelling (tools/reid_cluster.py, tools/reid_cluster_review.html, freeze gate):
same-frame rule, exclusion of tracker-flagged tracklets, odd-ones-out order, the audit (drawn after the review, player
vs player only, exact upper bound, shortest prefix), freeze refusals, upgrade of a 1.0.0 review, M001-style manual
exports unaffected, and the review page end to end incl. carrying over saved progress (Node Playwright)."""
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


def truth_answer(t):
    return {"decision": "player", "player": TRACKLETS[t][0]}


def export_from(review: dict, assign: dict, cfg: dict, answer=truth_answer, audit_answers=None, overrides=None,
                clusters=None, players=None, n_answers=3) -> dict:
    """What the review page exports: assign = {cluster: (decision, player)}; the audit answers the first n_answers
    tracklets of the draw order with `answer` (or uses audit_answers as given); rule confirmed_only."""
    clusters = clusters or {k: list(c["tracklets"]) for k, c in review["clusters"].items()}
    overrides = overrides or {}
    rv = {"schema": RC.REVIEW_SCHEMA[0], "schema_version": RC.REVIEW_SCHEMA[1], "config_sha256": review["config_sha256"],
          "model_sha256": None, "excluded": review["excluded"], "overrides": overrides,
          "clusters": {k: {"tracklets": m, "decision": assign.get(k, (None, None))[0],
                           "player": assign.get(k, (None, None))[1]} for k, m in clusters.items()}}
    if audit_answers is None:
        order = RC.audit_order(RC.audit_eligible(rv), cfg["audit"]["seed"])
        audit_answers = {t: answer(t) for t in order[:n_answers]}
    rv["audit"] = audit_answers
    ids = [t["id"] for t in review["tracklets"]]
    final, status, _ = RC.final_decisions(rv, ids, [t for t in audit_answers])
    parts = []
    for t in review["tracklets"]:
        d = final[t["id"]]
        p = {"tracklet": t["id"], "first_frame": t["first_frame"], "last_frame": t["last_frame"], "decision": d[0],
             "player": d[1]}
        if t["id"] in status:
            p["status"] = status[t["id"]]
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

    def test_upper_bound_matches_the_60_rule(self):
        self.assertAlmostEqual(RC.upper_bound(0, 60, 0.95), 1 - 0.05 ** (1 / 60), places=6)   # 0.0487: passes 5 %
        self.assertLess(RC.upper_bound(0, 60, 0.95), 0.05)
        self.assertGreater(RC.upper_bound(1, 60, 0.95), 0.05)                                 # one error fails
        self.assertGreater(RC.wilson(0, 60, 0.95)[1], 0.05)              # why the gate is not Wilson
        self.assertLess(RC.upper_bound(2, 200, 0.95), RC.upper_bound(2, 100, 0.95))
        self.assertIsNone(RC.upper_bound(0, 0, 0.95))

    def test_order_and_prefix(self):
        ids = [f"T{i:06d}-1" for i in range(50)]
        o = RC.audit_order(ids, 0)
        self.assertEqual(o, RC.audit_order(list(reversed(ids)), 0))
        self.assertNotEqual(o, sorted(ids))
        self.assertEqual(RC.fnv1a32("0:T000001-1"), RC.fnv1a32("0:T000001-1"))
        ans = {o[0]: {"decision": "player", "player": "A"}, o[1]: {"decision": "reject"},
               o[2]: {"decision": "unsure"}, o[3]: {"decision": "player", "player": "B"}}
        self.assertEqual(RC.answered_prefix(o, ans), o[:4])
        self.assertEqual(RC.answered_prefix(o, {o[1]: ans[o[1]]}), [])     # a gap: nothing before it is answered
        self.assertEqual(RC.answered_prefix(o[:4], ans), o[:4])

    def test_eligible_needs_a_player_cluster_with_a_live_mate(self):
        rv = {"overrides": {"c": {"decision": "reject"}},
              "clusters": {"X": {"tracklets": ["a", "b"], "decision": "player", "player": "P"},
                           "Y": {"tracklets": ["c", "d"], "decision": "player", "player": "Q"},     # one live left
                           "Z": {"tracklets": ["e", "f"], "decision": "unsure", "player": None}}}
        self.assertEqual(RC.audit_eligible(rv), ["a", "b"])

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
        self.cfg = write_cfg(self.td)
        self.c = RC.load_cluster_config(self.cfg)
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
        self.assertNotIn("audit", r)                                     # drawn after the review, not at build
        self.assertTrue((self.td / "review" / "index.html").exists() and (self.td / "review" / "thumbs").is_dir())
        with self.assertRaises(ReidLabelError):                         # never overwrites
            RC.build(self.pack, self.td / "review", self.cfg, embed=colour_embed)
        dev = make_pack(self.td, split="DEV")
        with self.assertRaises(ReidLabelError):                         # TRAIN only
            RC.build(dev, self.td / "review_dev", self.cfg, embed=colour_embed)

    def _rows(self, dest):
        return {r["tracklet"]: r for r in load_frozen(dest, "train")["identities"]}

    def test_only_confirmed_player_labels_are_frozen(self):
        ex = export_from(self.review, truth_assign(self.review), self.c, n_answers=3)
        dest = self._freeze(ex)
        fz = json.loads((dest / "FROZEN.json").read_text())
        self.assertEqual(fz["labelling_flow"], "cluster_confirm")
        a = fz["cluster_audit"]
        self.assertEqual((a["freeze_rule"], a["comparable"], a["disagreements"]), ("confirmed_only", 3, 0))
        self.assertEqual(a["drawable"], 7)
        self.assertEqual(a["unconfirmed_cluster_labels_frozen_unsure"], 4)
        rows = self._rows(dest)
        audited = set(ex["cluster_review"]["audit"])
        for t in TRACKLETS:
            if t == "T000008-1":
                self.assertEqual(rows[t]["decision"], "unsure")              # excluded
            elif t in audited:
                self.assertEqual((rows[t]["decision"], rows[t]["player"]), ("player", TRACKLETS[t][0]))
            else:
                self.assertEqual(rows[t]["decision"], "unsure")              # unconfirmed cluster label
        st = {p["tracklet"]: p.get("status") for p in json.loads((dest / "export.json").read_text())["parts"]}
        self.assertEqual(st["T000008-1"], RC.EXCLUDED_STATUS)
        self.assertEqual(sorted(t for t, v in st.items() if v == RC.UNCONFIRMED_STATUS), sorted(set(TRACKLETS) - audited - {"T000008-1"}))

    def test_answers_win_over_clusters_and_disagreements_do_not_block(self):
        wrong = {}

        def answer(t):                                                   # the first drawn: a different player
            if not wrong:
                wrong[t] = "C" if TRACKLETS[t][0] != "C" else "A"
                return {"decision": "player", "player": wrong[t]}
            return truth_answer(t)
        dest = self._freeze(export_from(self.review, truth_assign(self.review), self.c, answer=answer, n_answers=7))
        a = json.loads((dest / "FROZEN.json").read_text())["cluster_audit"]
        self.assertEqual((a["comparable"], a["disagreements"], a["unconfirmed_cluster_labels_frozen_unsure"]), (7, 1, 0))
        self.assertFalse(a["cluster_quality_meets_reference"])          # reported, not a gate
        (t, p), = wrong.items()
        self.assertEqual(self._rows(dest)[t]["player"], p)               # your answer is what is frozen

    def test_reject_answer_is_lost_data(self):
        first = {}

        def answer(t):
            if not first:
                first[t] = True
                return {"decision": "reject", "player": None}
            return truth_answer(t)
        dest = self._freeze(export_from(self.review, truth_assign(self.review), self.c, answer=answer, n_answers=4))
        a = json.loads((dest / "FROZEN.json").read_text())["cluster_audit"]
        self.assertEqual((a["comparable"], a["disagreements"], a["lost_data"], a["drawn"]), (3, 0, 1, 4))
        self.assertEqual(self._rows(dest)[next(iter(first))]["decision"], "reject")

    def test_freeze_refusals(self):
        good = export_from(self.review, truth_assign(self.review), self.c)

        def refused(ex, msg):
            with self.assertRaises(ReidLabelError, msg=msg):
                self._freeze(ex, f"bad_{msg}.json")

        bad = copy.deepcopy(good)
        rv = bad["cluster_review"]
        first = RC.audit_order(RC.audit_eligible(rv), 0)[0]
        rv["audit"].pop(first)                                            # answered later ones, skipped the first
        refused(bad, "audit_not_prefix")
        bad = copy.deepcopy(good)
        p = next(p for p in bad["parts"] if p.get("status") == RC.UNCONFIRMED_STATUS)
        p.update(decision="player", player=TRACKLETS[p["tracklet"]][0])
        p.pop("status")
        refused(bad, "unconfirmed_label_as_player")
        bad = copy.deepcopy(good)
        next(p for p in bad["parts"] if p.get("status") == RC.UNCONFIRMED_STATUS).pop("status")
        refused(bad, "unconfirmed_status_missing")
        bad = copy.deepcopy(good)
        bad["cluster_review"]["schema_version"] = "1.1.0"
        refused(bad, "old_audit_format")
        cl = {k: list(c["tracklets"]) for k, c in self.review["clusters"].items()}
        a_cl = next(k for k, m in cl.items() if "T000001-1" in m)
        b_cl = next(k for k, m in cl.items() if "T000002-1" in m)
        cl[a_cl].append("T000002-1")
        cl[b_cl].remove("T000002-1")
        refused(export_from(self.review, truth_assign(self.review), self.c, clusters=cl), "conflict")
        bad = copy.deepcopy(good)
        bad["parts"][-1].pop("status")
        refused(bad, "excluded_status")
        bad = copy.deepcopy(good)
        bad["cluster_review"]["config_sha256"] = "0" * 64
        refused(bad, "config")

    def test_upgrade_from_1_1_keeps_everything_but_the_rule(self):
        rd = self.td / "review"
        old = json.loads((rd / "review.json").read_text())
        old.update(schema_version="1.1.0", config_sha256="e" * 64)
        (rd / "review.json").write_text(json.dumps(old))
        new = RC.upgrade(rd, self.cfg)
        self.assertEqual(new["schema_version"], RC.REVIEW_SCHEMA[1])
        self.assertNotIn("previously_hidden", new)
        self.assertEqual((new["clusters"], new["tracklets"], new["excluded"]), (old["clusters"], old["tracklets"], old["excluded"]))

    def test_upgrade_keeps_clusters_and_marks_previously_hidden(self):
        rd = self.td / "review"
        old = json.loads((rd / "review.json").read_text())
        old.update(schema_version="1.0.0", config_sha256="f" * 64, audit=["T000003-1", "T000006-1"], z=1.96)
        (rd / "review.json").write_text(json.dumps(old))
        (rd / "review.js").write_text("old")
        new = RC.upgrade(rd, self.cfg)
        self.assertEqual(new["schema_version"], RC.REVIEW_SCHEMA[1])
        self.assertEqual(new["previously_hidden"], ["T000003-1", "T000006-1"])
        self.assertEqual(new["clusters"], old["clusters"])
        self.assertEqual(new["tracklets"], old["tracklets"])
        self.assertNotIn("audit", new)
        self.assertEqual(new["config_sha256"], RC.config_sha(self.cfg))
        self.assertTrue((rd / "review.json.1.0.0.bak").exists() and (rd / "review.js.1.0.0.bak").read_text() == "old")
        self.assertIn("window.SPANA_CLUSTER_REVIEW", (rd / "review.js").read_text())
        self.assertEqual(RC.upgrade(rd, self.cfg), new)                  # idempotent

    def test_manual_export_still_freezes(self):
        ex = export_from(self.review, truth_assign(self.review), self.c)
        for k in ("labelling_flow", "cluster_review"):
            ex.pop(k)
        for p in ex["parts"]:
            p.pop("status", None)
        self._freeze(ex, "manual.json")

    def test_evaluate_against_frozen_labels(self):
        dest = self._freeze(export_from(self.review, truth_assign(self.review), self.c))
        r = RC.evaluate(self.pack, dest, cfg_path=self.cfg, embed=colour_embed)
        self.assertEqual(r["errors"], 0)
        self.assertGreater(r["labelled_single_player"], 0)


NODE_SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const [dir, out] = process.argv.slice(2);
  const truth = JSON.parse(process.argv[4]);
  const b = await chromium.launch(); const page = await b.newPage({ acceptDownloads: true });
  await page.goto('file://' + dir + '/index.html');
  const R = await page.evaluate(() => window.SPANA_CLUSTER_REVIEW);
  const cid0 = Object.keys(R.clusters).find(k => new Set(R.clusters[k].tracklets.map(t => truth[t])).size === 1);
  // progress saved under earlier configs: a 1.0.0 state (no lock) and a locked post-review state -> the locked one
  // is carried over with its audit answers; a 1.0.0 state alone is carried over without its audit
  await page.evaluate(([R, cid0, who]) => {
    const members = Object.fromEntries(Object.entries(R.clusters).map(([k, c]) => [k, c.tracklets.slice()]));
    const base = { players: { [who]: { side: 'A', shirt_number: null } }, members, cdec: { [cid0]: { decision: 'player', player: who } }, overrides: {}, next: 99 };
    localStorage.setItem('spana_reid_cluster_' + R.tracklets_sha256 + '_' + 'f'.repeat(64),
      JSON.stringify({ ...base, audit: { [R.clusters[cid0].tracklets[0]]: { decision: 'reject', player: null } } }));
  }, [R, cid0, truth[R.clusters[cid0].tracklets[0]]]);
  await page.reload();
  if (!(await page.textContent('#status')).includes('carried over')) throw new Error('earlier progress not carried over');
  if (!(await page.locator(`.card[data-cid="${cid0}"] h3`).textContent()).includes(truth[R.clusters[cid0].tracklets[0]])) throw new Error('decision lost');
  if ((await page.textContent('#lock')).includes('Unlock')) throw new Error('a 1.0.0 state must not arrive locked');
  await page.evaluate(([R, cid0, who]) => {
    localStorage.removeItem('spana_reid_cluster_' + R.tracklets_sha256 + '_' + R.config_sha256);
    const members = Object.fromEntries(Object.entries(R.clusters).map(([k, c]) => [k, c.tracklets.slice()]));
    localStorage.setItem('spana_reid_cluster_' + R.tracklets_sha256 + '_' + 'e'.repeat(64), JSON.stringify({
      players: { [who]: { side: 'A', shirt_number: null } }, members, cdec: { [cid0]: { decision: 'player', player: who } },
      overrides: {}, next: 99, locked: true, audit: { x: { decision: 'unsure', player: null } } }));
  }, [R, cid0, truth[R.clusters[cid0].tracklets[0]]]);
  await page.reload();
  if (!(await page.textContent('#lock')).includes('Unlock')) throw new Error('locked post-review state not carried over');
  if (!(await page.textContent('#status')).includes('audit answers were carried over')) throw new Error('audit answers not carried over');
  await page.evaluate(() => localStorage.clear()); await page.reload();
  // a fresh review
  await page.fill('#annotator', 'QA');
  for (const [k, side] of [['A', 'A'], ['B', 'A'], ['C', 'B']]) { await page.fill('#pkey', k); await page.selectOption('#pside', side); await page.click('#padd'); }
  const player = async k => page.locator('#players .player').filter({ hasText: new RegExp('^' + k + ' ') }).click();
  const members = Object.fromEntries(Object.entries(R.clusters).map(([k, c]) => [k, c.tracklets.slice()]));
  const decideAll = async (skip) => {
    for (const [cid, m] of Object.entries(members)) {
      const L = m.filter(t => t !== skip); if (!L.length) continue;
      const card = page.locator(`.card[data-cid="${cid}"]`);
      const who = [...new Set(L.map(t => truth[t]))];
      if (who.length === 1) { await player(who[0]); await card.locator('button', { hasText: 'Assign cluster' }).click(); }
      else await card.locator('button', { hasText: 'Unsure cluster' }).click();
    }
  };
  await decideAll(null);
  // CONFLICT: move a tracklet into a cluster holding another tracklet on the same frame -> flagged, lock refused;
  // the move clears both clusters' decisions
  const cl = t => Object.keys(members).find(k => members[k].includes(t));
  const shown = R.tracklets.filter(t => !R.excluded.includes(t.id));
  let pair = null;
  for (const x of shown) for (const y of shown) if (!pair && cl(x.id) !== cl(y.id) && x.frames.some(f => y.frames.includes(f))) pair = [x.id, y.id];
  const from = cl(pair[1]), to = cl(pair[0]);
  await page.locator('.row', { hasText: pair[1] }).locator('select').selectOption(to);
  members[from] = members[from].filter(t => t !== pair[1]); members[to].push(pair[1]);
  if (!(await page.locator('.card.conflict').count())) throw new Error('no CONFLICT shown');
  if (!(await page.locator(`.card[data-cid="${to}"] h3`).textContent()).includes('undecided')) throw new Error('move did not clear the decision');
  await page.click('#lock');
  if ((await page.textContent('#lock')).includes('Unlock')) throw new Error('locked despite CONFLICT / undecided');
  await page.locator('.row', { hasText: pair[1] }).locator('button', { hasText: 'Unsure' }).click();
  if (await page.locator('.card.conflict').count()) throw new Error('CONFLICT not resolved by removing the tracklet');
  await page.selectOption('#filter', 'all');
  await decideAll(pair[1]);
  await page.click('#lock');
  if (!(await page.textContent('#lock')).includes('Unlock')) throw new Error('not locked');
  for (let i = 0; i < 2; i++) {                                        // answer 2, leave the rest unconfirmed
    const card = page.locator('.card[data-audit]').first();             // newest drawn tracklet on top
    const t = await card.getAttribute('data-audit');
    await player(truth[t]); await card.locator('button', { hasText: 'Assign active player' }).click();
  }
  page.once('dialog', d => d.accept());                                 // "N tracklets will be frozen as unsure"
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
    def test_page_lock_audit_export_freezes(self):
        td = Path(tempfile.mkdtemp())
        try:
            pack = make_pack(td)
            cfg = write_cfg(td)
            RC.build(pack, td / "review", cfg, embed=colour_embed)
            js = td / "drive.js"
            js.write_text(NODE_SCRIPT)
            out = td / "export.json"
            truth = json.dumps({t: v[0] for t, v in TRACKLETS.items()})
            r = subprocess.run(["node", str(js), str(td / "review"), str(out), truth], capture_output=True, text=True,
                               timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            ex = json.loads(out.read_text())
            self.assertEqual(ex["cluster_review"]["schema_version"], RC.REVIEW_SCHEMA[1])
            dest = freeze(str(pack), str(out), root=td / "repo", cluster_config=cfg)   # JS draw order == Python
            a = json.loads((dest / "FROZEN.json").read_text())["cluster_audit"]
            self.assertEqual((a["comparable"], a["disagreements"], a["freeze_rule"]), (2, 0, "confirmed_only"))
            self.assertGreater(a["unconfirmed_cluster_labels_frozen_unsure"], 0)
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
