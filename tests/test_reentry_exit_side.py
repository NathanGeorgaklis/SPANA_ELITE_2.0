"""Exit side (target.reentry_exit_edge, spana.target 2.7.0; records/reentry_exit_side_2026-11-19).

When the followed player LEFT THE PICTURE, a returning track must be born at the side they left by (within
reentry_exit_edge_max_frac of the frame). Nathan's FRESH_08 review found an offer of another player, born on the far
side of the picture, while ours was off screen; on DEV every true off-screen return was born <= 0.011 from the exit
side and every wrong offer 0.62-0.98 away. No effect while the player is still in the picture or after a camera cut."""
from __future__ import annotations

import copy
import dataclasses
import unittest

from spana.artifacts import ArtifactError, validate_target, validate_tracks
from spana.identity.follower import follow, target_document
from spana.identity.reid import ReentryContext, exit_side, reentry_choice
from tests.test_reentry_offer import ME, ME_AGAIN, SEL, SOMEONE, cfg, emb_doc, returning, scene, tc, unit
from tests.test_target_follow import CFG

NEAR_ME = unit(0.8, 0.6, 0)                # cos 0.80 with ME


def choose(doc, sigs, **kw):
    sig = {k: {"first": v, "last": v} for k, v in sigs.items()}
    return reentry_choice("t1", 3, len(doc["frames"]) - 1, ReentryContext.of(doc), sig,
                          tc(**{"reentry_exit_edge": True, "reentry_exit_edge_max_frac": 0.05, **kw}), CFG.tracking)


class Geometry(unittest.TestCase):
    def test_left_frame_side_and_distance(self):
        doc = scene({"t2": (8, 0.0), "t3": (8, 900.0)}, left_frame=True)      # t1 at x 100-140, 1280 wide
        ctx = ReentryContext.of(doc)
        self.assertEqual(exit_side(ctx, "t1", 3, "t2"), {"status": "off_screen", "side": "left", "edge_dist": 0.0})
        self.assertAlmostEqual(exit_side(ctx, "t1", 3, "t3")["edge_dist"], 900 / 1280, places=5)

    def test_carried_out_of_the_picture_by_the_camera(self):
        # the scene moves 100 px left per frame from frame 4: t1's centre (x 120) is beyond the left side by frame 6
        doc = scene({"t2": (6, 0.0), "t3": (6, 1240.0)}, pan_px=100.0)
        ctx = ReentryContext.of(doc)
        self.assertEqual(exit_side(ctx, "t1", 3, "t2")["side"], "left")
        self.assertAlmostEqual(exit_side(ctx, "t1", 3, "t3")["edge_dist"], 1240 / 1280, places=5)

    def test_in_picture_and_camera_cut_say_nothing(self):
        doc = scene({"t3": (8, 900.0)})
        self.assertEqual(exit_side(ReentryContext.of(doc), "t1", 3, "t3"), {"status": "in_picture"})
        doc = scene({"t3": (8, 900.0)}, left_frame=True)
        doc["frames"][6]["camera"]["status"] = "CUT"
        self.assertEqual(exit_side(ReentryContext.of(doc), "t1", 3, "t3"), {"status": "unknown"})


class Rule(unittest.TestCase):
    def test_a_lookalike_on_the_far_side_is_not_offered_the_return_at_the_exit_is(self):
        doc = scene({"t2": (8, 900.0), "t3": (9, 0.0)}, left_frame=True)
        sigs = {"t1": ME, "t2": ME_AGAIN, "t3": NEAR_ME}
        off = choose(doc, sigs, reentry_exit_edge=False)
        self.assertEqual((off["offer"], off["track"]), (True, "t2"))
        on = choose(doc, sigs)
        self.assertEqual((on["offer"], on["track"], on["exit_excluded"]), (True, "t3", ["t2"]))
        self.assertNotIn("t2", on["candidates"])                   # not ranked, not counted for the margin

    def test_only_far_side_tracks_no_offer(self):
        ch = choose(scene({"t2": (8, 900.0)}, left_frame=True), {"t1": ME, "t2": ME_AGAIN})
        self.assertEqual((ch["offer"], ch["reason"], ch["exit_excluded"]), (False, "NO_CANDIDATE", ["t2"]))

    def test_threshold_is_the_config_value(self):
        doc = scene({"t2": (8, 100.0)}, left_frame=True)              # 100 / 1280 = 0.078 from the left side
        sigs = {"t1": ME, "t2": ME_AGAIN}
        self.assertFalse(choose(doc, sigs)["offer"])
        self.assertTrue(choose(doc, sigs, reentry_exit_edge_max_frac=0.1)["offer"])

    def test_no_effect_while_in_the_picture_or_after_a_cut(self):
        sigs = {"t1": ME, "t2": ME_AGAIN}
        self.assertEqual(choose(scene({"t2": (8, 900.0)}), sigs)["track"], "t2")
        doc = scene({"t2": (8, 900.0)}, left_frame=True)
        doc["frames"][6]["camera"]["status"] = "CUT"
        self.assertEqual(choose(doc, sigs)["track"], "t2")


def sized(tracks):
    tracks["video"].update(width=1280, height=720)
    validate_tracks(tracks)
    return tracks


ON = dict(reentry_exit_edge=True, reentry_exit_edge_max_frac=0.05)


class Follower(unittest.TestCase):
    """returning(): t1 (x 0-40) ends LEFT_FRAME on frame 4; t2 is born on frame 8 at x 0 (the exit side) and t3 at
    x 100 (0.078 of the width away)."""

    def setUp(self):
        self.tr = sized(returning(others={"t3": None}))
        self.emb = emb_doc(self.tr, {"t1": (ME, ME), "t2": (SOMEONE, SOMEONE), "t3": (ME_AGAIN, ME_AGAIN)})

    def test_far_track_is_excluded_and_logged(self):
        off = follow(self.tr, SEL, cfg(), (), self.emb)[1]
        self.assertEqual(off["relinks"][0]["to_track"], "t3")
        on = follow(self.tr, SEL, cfg(**ON), (), self.emb)[1]
        self.assertFalse(on["relinks"])
        self.assertEqual(on["reentry_log"][0]["exit_excluded"], ["t3"])
        doc = target_document(self.tr, SEL, cfg(**ON), ["test"], (), self.emb)
        self.assertEqual(doc["schema_version"], "2.7.0")
        validate_target(doc, self.tr, self.emb)

    def test_validator_refuses_an_offer_that_breaks_the_exit_side(self):
        doc = target_document(self.tr, SEL, cfg(), ["test"], (), self.emb)       # rule off: t3 offered
        validate_target(doc, self.tr, self.emb)
        bad = copy.deepcopy(doc)
        bad["target_config"]["reentry_exit_edge"] = True
        with self.assertRaises(ArtifactError):
            validate_target(bad, self.tr, self.emb)

    def test_exit_side_offer_still_made(self):
        emb = emb_doc(self.tr, {"t1": (ME, ME), "t2": (ME_AGAIN, ME_AGAIN), "t3": (SOMEONE, SOMEONE)})
        r = follow(self.tr, SEL, cfg(**ON), (), emb)[1]["relinks"][0]
        self.assertEqual(r["to_track"], "t2")
        validate_target(target_document(self.tr, SEL, cfg(**ON), ["test"], (), emb), self.tr, emb)

    def test_config_refuses_a_fraction_outside_0_1(self):
        from spana.config import ConfigError
        with self.assertRaises(ConfigError):
            dataclasses.replace(CFG.target, reentry_exit_edge_max_frac=1.5).validate()


if __name__ == "__main__":
    unittest.main()
