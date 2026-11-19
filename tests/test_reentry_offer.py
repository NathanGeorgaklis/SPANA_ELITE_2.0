"""Re-entry offers by re-ID (target.reentry_offer, spana.target 2.4.0) and track embeddings (spana.track_embeddings).

After the followed track is lost (here: it left the picture) and no predecessor relink applies, the follower offers
(CANDIDATE, REENTRY_REID) the returning track whose first re-ID signature is closest to the lost track's last one,
when it reaches the minimum similarity and beats every other returning track by the margin. Never TARGET without
the user (Nathan, 2026-11-18: option 1, built for testing in the app)."""
from __future__ import annotations

import copy
import dataclasses
import math
import unittest

from spana.artifacts import ArtifactError, sha256_doc, validate_target, validate_track_embeddings
from spana.identity.follower import Selection, confirm_relink, follow, reject_candidate, target_document
from tests.test_target_follow import A, CFG, N, tracks_doc

FPS = 10.0
SEL = Selection.track_at_frame("t1", 1)
T = CFG.target


def cfg(**target):
    base = dict(reentry_offer=True, reentry_max_gap_s=10.0, reentry_min_similarity=0.6, reentry_min_margin=0.1,
                reentry_signature="mean")         # these tests use averaged 1.0.0 signatures
    return dataclasses.replace(CFG, target=dataclasses.replace(T, **{**base, **target}))


def unit(*v):
    n = math.sqrt(sum(x * x for x in v))
    return [round(x / n, 5) for x in v]


def returning(born_at: int = 8, others: dict = None, length: int = 14):
    """t1 followed from frame 1, leaves the picture after frame 3 (ended on frame 4); t2 (and `others`) appear at
    born_at (fresh, then confirmed)."""
    others = others or {}
    frames = [{"t1": N}, {"t1": A}, {"t1": A}, {"t1": A}]
    names = ["t2", *others]
    for n in range(4, length):
        if n < born_at:
            frames.append({})
        elif n == born_at:
            frames.append({k: N for k in names})
        else:
            frames.append({k: A for k in names})
    return tracks_doc(frames, ended={4: [("t1", "LEFT_FRAME")]})


def emb_doc(tracks: dict, sigs: dict) -> dict:
    """sigs: track -> (first, last) unit vectors; sample frames = the track's confirmed observed frames."""
    obs = {}
    for fr in tracks["frames"]:
        for t in fr["tracks"]:
            if t["source"] == "OBSERVED" and t["confirmed"]:
                obs.setdefault(t["track_id"], []).append(fr["frame"])
    doc = {"schema": "spana.track_embeddings", "schema_version": "1.0.0", "provenance": {}, "video": tracks["video"],
           "tracks_sha256": sha256_doc(tracks), "model": {"name": "test", "version": "t", "sha256": "0" * 64,
                                                          "embedding_dim": 3},
           "reid_config": {}, "tracks": {}}
    for tid, (first, last) in sigs.items():
        fs = obs[tid]
        doc["tracks"][tid] = {"first": first, "last": last, "first_frames": fs[:1], "last_frames": fs[-1:]}
    validate_track_embeddings(doc, tracks)
    return doc


ME = unit(1, 0, 0)
ME_AGAIN = unit(0.95, 0.31, 0)            # cos 0.95 with ME
SOMEONE = unit(0.1, 0, 1)                  # cos 0.10
LOOKALIKE = unit(0.9, 0.44, 0)             # cos 0.90: within the margin of ME_AGAIN


def answers(frames):
    return [(f["state"], f["reason"], f["track_id"]) for f in frames]


class Offer(unittest.TestCase):
    def test_no_embeddings_no_offer(self):
        frames, info = follow(returning(), SEL, cfg())
        self.assertFalse(info["relinks"])
        self.assertTrue(all(f["state"] != "CANDIDATE" for f in frames))

    def test_returning_player_is_offered_never_answered(self):
        tr = returning(others={"t3": None})
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN), "t3": (SOMEONE, SOMEONE)})
        frames, info = follow(tr, SEL, cfg(), (), emb)
        offers = [f for f in frames if f["state"] == "CANDIDATE"]
        self.assertTrue(offers)
        self.assertTrue(all((f["reason"], f["track_id"]) == ("REENTRY_REID", "t2") for f in offers))
        self.assertFalse(any(f["state"] == "TARGET" and f["track_id"] != "t1" for f in frames))
        r = info["relinks"][0]
        self.assertEqual((r["evidence"]["kind"], r["outcome"]), ("reid", "PENDING"))
        self.assertAlmostEqual(r["evidence"]["similarity"], 0.95, places=2)
        self.assertAlmostEqual(r["evidence"]["second_combined"], 0.10, places=2)
        doc = target_document(tr, SEL, cfg(), ["test"], (), emb)
        validate_target(doc, tr, emb)

    def test_confirmed_offer_becomes_the_target(self):
        tr = returning()
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN)})
        r = follow(tr, SEL, cfg(), (), emb)[1]["relinks"][0]
        frames, info = follow(tr, SEL, cfg(), [confirm_relink(r, r["proposed_frame"] + 1)], emb)
        self.assertEqual(answers(frames)[-1], ("TARGET", "TRACK_OBSERVED", "t2"))
        self.assertEqual(info["relinks"][0]["outcome"], "CONFIRMED")
        doc = target_document(tr, SEL, cfg(), ["test"], [confirm_relink(r, r["proposed_frame"] + 1)], emb)
        validate_target(doc, tr, emb)

    def test_auto_confirm_never_applies_to_reid(self):
        tr = returning()
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN)})
        frames, _ = follow(tr, SEL, cfg(relink_auto_confirm=True), (), emb)
        self.assertFalse(any(f["state"] == "TARGET" and f["track_id"] == "t2" for f in frames))

    def test_two_lookalikes_no_offer(self):
        tr = returning(others={"t3": None})
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN), "t3": (LOOKALIKE, LOOKALIKE)})
        self.assertFalse(follow(tr, SEL, cfg(), (), emb)[1]["relinks"])

    def test_not_similar_enough_no_offer(self):
        tr = returning()
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (SOMEONE, SOMEONE)})
        self.assertFalse(follow(tr, SEL, cfg(), (), emb)[1]["relinks"])

    def test_too_late_no_offer(self):
        tr = returning(born_at=12, length=16)          # born 0.9 s after the loss
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN)})
        self.assertFalse(follow(tr, SEL, cfg(reentry_max_gap_s=0.5), (), emb)[1]["relinks"])
        self.assertTrue(follow(tr, SEL, cfg(), (), emb)[1]["relinks"])

    def test_track_without_a_signature_is_never_offered(self):
        tr = returning()
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (None, None)})
        self.assertFalse(follow(tr, SEL, cfg(), (), emb)[1]["relinks"])

    def test_declined_offer_is_not_repeated(self):
        tr = returning()
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN)})
        r = follow(tr, SEL, cfg(), (), emb)[1]["relinks"][0]
        frames, info = follow(tr, SEL, cfg(), [reject_candidate(r, r["proposed_frame"])], emb)
        self.assertEqual(len(info["relinks"]), 1)
        self.assertFalse(any(f["state"] == "CANDIDATE" and f["frame"] > r["proposed_frame"] for f in frames))

    def test_switch_off_no_offer(self):
        tr = returning()
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN)})
        self.assertFalse(follow(tr, SEL, cfg(reentry_offer=False), (), emb)[1]["relinks"])


def two_lost():
    """t1 (followed) and t4 both leave the picture after frame 3; t2 appears at frame 8."""
    frames = [{"t1": N, "t4": N}, {"t1": A, "t4": A}, {"t1": A, "t4": A}, {"t1": A, "t4": A}]
    for n in range(4, 14):
        frames.append({} if n < 8 else {"t2": N} if n == 8 else {"t2": A})
    return tracks_doc(frames, ended={4: [("t1", "LEFT_FRAME"), ("t4", "LEFT_FRAME")]})


class SafetyRules(unittest.TestCase):
    def test_mutual_refuses_a_track_that_matches_another_lost_player_better(self):
        tr = two_lost()
        other = unit(0.96, 0.28, 0)                  # t4 looks even more like t2 than t1 does
        emb = emb_doc(tr, {"t1": (ME, ME), "t4": (other, other), "t2": (ME_AGAIN, ME_AGAIN)})
        self.assertTrue(follow(tr, SEL, cfg(reentry_mutual=False, reentry_min_margin=0.0), (), emb)[1]["relinks"])
        self.assertFalse(follow(tr, SEL, cfg(reentry_mutual=True, reentry_min_margin=0.0), (), emb)[1]["relinks"])

    def test_mutual_keeps_a_clear_two_way_match(self):
        tr = two_lost()
        emb = emb_doc(tr, {"t1": (ME, ME), "t4": (SOMEONE, SOMEONE), "t2": (ME_AGAIN, ME_AGAIN)})
        r = follow(tr, SEL, cfg(reentry_mutual=True, reentry_min_margin=0.0), (), emb)[1]["relinks"]
        self.assertEqual([x["to_track"] for x in r], ["t2"])

    def test_reachable(self):
        from spana.identity.reid import reachable
        tc = CFG.tracking
        ident = {"status": "OK", "affine": [[1, 0, 0], [0, 1, 0]]}
        pan = {"status": "OK", "affine": [[1, 0, -50], [0, 1, 0]]}       # camera pans: scene moves 50 px left
        frames = [{"t": k / 10, "camera": ident} for k in range(3)] + [{"t": 0.3, "camera": pan}]
        lost = [100, 100, 140, 200]                                         # 100 px tall, centre (120, 150)
        near = [52, 100, 92, 200]                                           # where the pan carried him
        far = [600, 100, 640, 200]
        self.assertTrue(reachable(frames, lost, 0, 3, near, tc, 1280, 720))
        self.assertFalse(reachable(frames, lost, 0, 3, far, tc, 1280, 720))
        cut = frames[:3] + [{"t": 0.3, "camera": {"status": "CUT", "affine": [[1, 0, 0], [0, 1, 0]]}}]
        self.assertIsNone(reachable(cut, lost, 0, 3, far, tc, 1280, 720))  # across a cut: cannot judge
        out = frames[:3] + [{"t": 0.3, "camera": {"status": "OK", "affine": [[1, 0, -500], [0, 1, 0]]}}]
        self.assertIsNone(reachable(out, lost, 0, 3, far, tc, 1280, 720))  # carried out of the picture
        self.assertIsNone(reachable(frames, lost, 0, 3, far, tc, None, None))


class Validation(unittest.TestCase):
    def setUp(self):
        self.tr = returning(others={"t3": None})
        self.emb = emb_doc(self.tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN), "t3": (SOMEONE, SOMEONE)})
        self.doc = target_document(self.tr, SEL, cfg(), ["test"], (), self.emb)

    def test_tampered_similarity_is_refused(self):
        d = copy.deepcopy(self.doc)
        ev = d["relinks"][0]["evidence"]
        ev["similarity"] = round(ev["similarity"] - 0.01, 6)          # consistent scores, wrong appearance
        ev["combined"] = round(ev["combined"] - 0.01, 6)
        validate_target(d, self.tr)                                   # structurally fine without the embeddings
        with self.assertRaises(ArtifactError):
            validate_target(d, self.tr, self.emb)

    def test_reid_offer_needs_the_policy_and_embeddings(self):
        for edit in (lambda d: d["target_config"].update(reentry_offer=False),
                     lambda d: d.update(embeddings_sha256=None),
                     lambda d: d["relinks"][0]["evidence"].update(similarity=0.5)):
            d = copy.deepcopy(self.doc)
            edit(d)
            with self.assertRaises(ArtifactError):
                validate_target(d, self.tr)

    def test_candidate_reason_must_match_the_offer_kind(self):
        d = copy.deepcopy(self.doc)
        f = next(f for f in d["frames"] if f["state"] == "CANDIDATE")
        f["reason"] = "RELINK_PREDECESSOR"
        with self.assertRaises(ArtifactError):
            validate_target(d, self.tr)

    def test_embedding_samples_must_be_clean_observations(self):
        e = copy.deepcopy(self.emb)
        e["tracks"]["t2"]["first_frames"] = [0]                      # t2 does not exist on frame 0
        with self.assertRaises(ArtifactError):
            validate_track_embeddings(e, self.tr)
        e = copy.deepcopy(self.emb)
        e["tracks"]["t2"]["first"] = [1.0, 1.0, 0.0]                 # not unit length
        with self.assertRaises(ArtifactError):
            validate_track_embeddings(e)


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------ position as a score (target.reentry_position_*)
from spana.identity.reid import ReentryContext, reentry_choice  # noqa: E402

IDENT = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def scene(births, pan_px: float = 0.0, left_frame: bool = False, n: int = 20):
    """Hand-made tracks rows (10 fps, 1280 x 720): t1 observed on frames 0-3 at x 100-140 (100 px tall), then lost;
    births = {track: (frame, x)}: a candidate first seen on that frame at that x. pan_px: camera translation per
    frame from frame 4 on (the scene moves left). left_frame: t1 ends LEFT_FRAME on frame 4."""
    frames = []
    for k in range(n):
        cam = {"status": "FIRST" if k == 0 else "OK", "inliers": 0, "hist_distance": None,
               "affine": [[1.0, 0.0, -pan_px if k >= 4 else 0.0], [0.0, 1.0, 0.0]]}
        tracks = [{"track_id": "t1", "source": "OBSERVED", "box": [100.0, 100.0, 140.0, 200.0]}] if k < 4 else []
        for tid, (f, x) in births.items():
            if k >= f:
                tracks.append({"track_id": tid, "source": "OBSERVED", "box": [x, 100.0, x + 40.0, 200.0]})
        ended = [["t1", "LEFT_FRAME"]] if (left_frame and k == 4) else []
        frames.append({"frame": k, "t": k / 10.0, "camera": cam, "tracks": tracks, "ended": ended})
    return {"video": {"width": 1280, "height": 720}, "frames": frames}


def tc(**kw):
    base = dict(reentry_offer=True, reentry_max_gap_s=10.0, reentry_min_similarity=0.6, reentry_min_margin=0.0,
                reentry_mutual=False, reentry_position_gate=False, reentry_position_weight=0.3,
                reentry_position_base_h=0.35, reentry_position_speed_h_per_s=2.0, reentry_position_time_scale_s=1.0,
                reentry_position_max_gap_s=3.0, reentry_position_max_pan_frac=0.25, reentry_signature="mean")
    return dataclasses.replace(T, **{**base, **kw})


def choose(doc, sigs, **kw):
    ctx = ReentryContext.of(doc)
    sig = {k: {"first": v, "last": v} for k, v in sigs.items()}
    return reentry_choice("t1", 3, len(doc["frames"]) - 1, ctx, sig, tc(**kw), CFG.tracking)


NEAR_LOOK = unit(0.80, 0.60, 0)          # cos 0.80 with ME
FAR_LOOK = unit(0.85, 0.527, 0)          # cos 0.85 with ME: slightly more similar


class PositionScore(unittest.TestCase):
    def test_close_in_place_and_time_outranks_a_slightly_more_similar_far_late_track(self):
        doc = scene({"t2": (4, 102.0), "t3": (15, 900.0)})
        sigs = {"t1": ME, "t2": NEAR_LOOK, "t3": FAR_LOOK}
        self.assertEqual(choose(doc, sigs, reentry_position_weight=0.0)["track"], "t3")   # appearance alone
        ch = choose(doc, sigs)
        self.assertEqual((ch["offer"], ch["track"], ch["position_informative"]), (True, "t2", True))
        self.assertGreater(ch["position"], 0.5)
        self.assertEqual(ch["candidates"]["t3"]["position"], 0.0)

    def test_pan_or_exit_ranks_by_appearance_alone(self):
        sigs = {"t1": ME, "t2": NEAR_LOOK, "t3": FAR_LOOK}
        # pan: the scene moves 400 px right on frame 4 onwards (t2's birth: PAN; later: carried out of the picture)
        for doc, why in ((scene({"t2": (5, 102.0), "t3": (15, 900.0)}, pan_px=-400.0), {"PAN", "OUT_OF_PICTURE"}),
                         (scene({"t2": (5, 102.0), "t3": (15, 900.0)}, left_frame=True), {"LEFT_FRAME"})):
            with_pos, without = choose(doc, sigs), choose(doc, sigs, reentry_position_weight=0.0)
            self.assertEqual((with_pos["track"], with_pos["offer"]), (without["track"], without["offer"]))
            self.assertEqual(with_pos["track"], "t3")
            self.assertTrue(all(c["position"] == 0.0 and c["position_reason"] in why
                                for c in with_pos["candidates"].values()))
            if "PAN" in why:
                self.assertEqual(with_pos["candidates"]["t2"]["position_reason"], "PAN")

    def test_an_exit_silences_position_for_every_candidate(self):
        # t2 appears near at once (position would favour it), t1 then ends LEFT_FRAME before t3 returns: the decision
        # is appearance-only, so the pan / exit return t3 is not out-ranked by position.
        doc = scene({"t2": (4, 102.0), "t3": (15, 900.0)})
        doc["frames"][6]["ended"] = [["t1", "LEFT_FRAME"]]
        sigs = {"t1": ME, "t2": NEAR_LOOK, "t3": FAR_LOOK}
        ch = choose(doc, sigs)
        self.assertEqual(ch["track"], choose(doc, sigs, reentry_position_weight=0.0)["track"])
        self.assertEqual(ch["track"], "t3")
        self.assertTrue(all(c["position"] == 0.0 for c in ch["candidates"].values()))

    def test_silenced_decision_is_appearance_only_in_the_two_way_check_too(self):
        # t1 (followed) leaves the picture; t4 was lost right where t2 will appear; t2 looks most like t1. Position
        # would favour t4 for t2 in the reverse check, but the decision is silenced by the exit: appearance decides.
        doc = scene({"t2": (8, 300.0)}, left_frame=True)
        for k in range(4):
            doc["frames"][k]["tracks"].append({"track_id": "t4", "source": "OBSERVED", "box": [300.0, 100.0, 340.0, 200.0]})
        sigs = {"t1": ME, "t2": ME_AGAIN, "t4": unit(0.9, 0.44, 0)}       # t4 ~ t2: 0.98 vs t1 ~ t2: 0.95
        sig = {k: {"first": v, "last": v} for k, v in sigs.items()}
        ctx = ReentryContext.of(doc)
        on = reentry_choice("t1", 3, len(doc["frames"]) - 1, ctx, sig, tc(reentry_mutual=True), CFG.tracking)
        off = reentry_choice("t1", 3, len(doc["frames"]) - 1, ReentryContext.of(doc), sig,
                             tc(reentry_mutual=True, reentry_position_weight=0.0), CFG.tracking)
        self.assertEqual((on["offer"], on["reason"]), (off["offer"], off["reason"]))

    def test_weight_zero_is_appearance_only_whatever_the_scales(self):
        doc = scene({"t2": (4, 102.0), "t3": (15, 900.0)})
        sigs = {"t1": ME, "t2": NEAR_LOOK, "t3": FAR_LOOK}
        a = choose(doc, sigs, reentry_position_weight=0.0)
        b = choose(doc, sigs, reentry_position_weight=0.0, reentry_position_speed_h_per_s=50.0,
                   reentry_position_time_scale_s=99.0, reentry_position_max_gap_s=0.01)
        self.assertEqual(a, b)
        self.assertTrue(all(c["combined"] == c["appearance"] and c["position_reason"] == "WEIGHT_ZERO"
                            for c in a["candidates"].values()))

    def test_margin_still_abstains_when_the_top_two_are_close(self):
        doc = scene({"t2": (4, 102.0), "t3": (15, 900.0)})
        close = unit(0.99, 0.14, 0)                            # appearance ~0.99: combined within 0.1 of t2's
        ch = choose(doc, {"t1": ME, "t2": NEAR_LOOK, "t3": close}, reentry_min_margin=0.10)
        self.assertEqual((ch["offer"], ch["reason"]), (False, "MARGIN"))
        self.assertLess(ch["margin"], 0.10)

    def test_offers_and_abstentions_are_logged_with_their_scores(self):
        tr = returning()
        emb = emb_doc(tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN)})
        log = follow(tr, SEL, cfg(), (), emb)[1]["reentry_log"]
        offer = next(e for e in log if e["offer"])
        self.assertIsNone(offer["reason"])
        self.assertEqual(offer["track"], "t2")
        self.assertEqual(offer["position_reason"], "NO_FRAME_SIZE" if CFG.target.reentry_position_weight > 0
                         else "WEIGHT_ZERO")
        self.assertEqual({c["track"] for c in offer["candidates"]}, {"t2"})
        doc = target_document(tr, SEL, cfg(), ["test"], (), emb)
        self.assertEqual(doc["reentry_log"], log)
        validate_target(doc, tr, emb)
