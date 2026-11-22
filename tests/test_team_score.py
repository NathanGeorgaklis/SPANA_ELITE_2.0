"""tools.team_score: permutation, wrong answers (role answers on team players, OTHER), roles as counts, frozen
side-label hash check."""
import hashlib
import json

import pytest

from tools.team_score import _perm, load_sides, roles, summarise


def u(side, pred, player="1", reason="", clip="c"):
    return {"clip": clip, "player": player, "side": side, "pred": pred, "reason": reason, "frame": 0, "track": "T"}


def test_permutation_follows_agreement_and_is_fixed():
    units = [u("A", "TEAM_B")] * 5 + [u("B", "TEAM_A")] * 4 + [u("A", "TEAM_A")]
    perm, counts = _perm(units)
    assert perm == {"A": "TEAM_B", "B": "TEAM_A"} and counts == {"A->TEAM_A": 1, "A->TEAM_B": 9}
    assert _perm([])[0] == {"A": "TEAM_A", "B": "TEAM_B"}


def test_wrong_answers_and_bound():
    perm = {"A": "TEAM_A", "B": "TEAM_B"}
    units = ([u("A", "TEAM_A")] * 90 + [u("B", "GOALKEEPER", player="7")] * 2 + [u("REF", "REFEREE", player="9")]
             + [u("GK_B", "TEAM_B", player="8")] + [u("OTHER", "TEAM_A", player="5")]
             + [u("B", "UNKNOWN", reason="LOW_VOTE_SHARE")] * 5)
    s = summarise(units, perm, gt_boxes=120)
    assert s["answered"] == 95 and s["wrong"] == 4 and s["units"] == 100
    assert s["role_answers_on_team_players"] == {"frames": 2, "ids": ["7"]}
    assert s["wrong_ids"] == {"5": 1, "7": 2, "8": 1}
    assert s["unknown_reasons"] == {"LOW_VOTE_SHARE": 5}
    assert s["coverage"] == 0.95 and s["coverage_of_gt_boxes"] == round(95 / 120, 4)
    assert 4 / 95 < s["wrong_bound_95"] < 0.11
    assert summarise([u("A", "TEAM_A")] * 600, perm, 600)["wrong_bound_95"] == pytest.approx(0.00498, abs=1e-5)


def test_roles_are_counts_per_id():
    units = [u("REF", "REFEREE", player="9")] * 3 + [u("REF", "UNKNOWN", player="9", reason="COLOUR_OUTLIER_NO_ROLE")]
    r = roles(units, {"c": {"9": "REF", "1": "A"}})
    assert r == [{"clip": "c", "id": "9", "side": "REF", "truth": "REFEREE", "units": 4,
                  "answered": {"REFEREE": 3, "UNKNOWN": 1}, "unknown_reasons": {"COLOUR_OUTLIER_NO_ROLE": 1}}]


def test_load_sides_checks_hash(tmp_path):
    v1 = tmp_path / "clip" / "v1"
    t = tmp_path / "clip" / "teams_v1"
    v1.mkdir(parents=True)
    t.mkdir()
    raw = json.dumps({"sides": {"1": "A", "2": "REF"}}).encode()
    (t / "sides.json").write_bytes(raw)
    (t / "FROZEN.json").write_text(json.dumps({"split": "DEV", "sides_sha256": hashlib.sha256(raw).hexdigest()}))
    assert load_sides(v1)[1] == {"1": "A", "2": "REF"}
    (t / "sides.json").write_bytes(raw + b" ")
    with pytest.raises(SystemExit, match="frozen hash"):
        load_sides(v1)
    (t / "FROZEN.json").write_text(json.dumps({"split": "EVAL", "sides_sha256": "x"}))
    with pytest.raises(Exception):
        load_sides(v1)
