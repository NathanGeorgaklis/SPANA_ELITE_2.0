"""QA metrics table over the synthetic scenario catalogue (identity transfers, switches, duplicates,
fragments, wrong recoveries, coverage). Run: python -m tools.qa_scenarios"""
from __future__ import annotations

import numpy as np

from spana.records import CameraObservation, CameraStatus, Detection
from tests.metrics import evaluate
from tests.synth import Player, kit, run

H = 80.0


def scenarios():
    yield "normal_movement", lambda: [Player(f"P{k}", (300 + 200 * k, 400 + 60 * k), (120 - 40 * k, 30 * k))
                                      for k in range(5)], {}
    yield "missed_detection_2f", lambda: [Player("A", (300, 500), (250, 0),
                                                 visible=lambda t: not (1.0 <= t < 1.07))], {}
    yield "occlusion_distinct", lambda: [Player("A", (700, 500), (0, 0), height=90, appearance=kit(0)),
                                         Player("B", (500, 495), (200, 0), appearance=kit(1),
                                                visible=lambda t: not (0.9 <= t < 1.3))], {}
    yield "occlusion_same_look", lambda: [Player("A", (700, 500), (0, 0), height=90, appearance=kit(0)),
                                          Player("B", (500, 495), (200, 0), appearance=kit(0),
                                                 visible=lambda t: not (0.9 <= t < 1.3))], {}
    yield "crossing_same_look", lambda: [Player("A", (400, 500), (300, 0), appearance=kit(0)),
                                         Player("B", (1000, 510), (-300, 0), appearance=kit(0))], {}
    yield "crossing_hidden_distinct", lambda: [Player("A", (400, 500), (300, 0), appearance=kit(0)),
                                               Player("B", (1000, 505), (-300, 0), appearance=kit(1),
                                                      visible=lambda t: not (0.85 <= t < 1.15))], {}
    yield "crossing_hidden_same_look", lambda: [Player("A", (400, 500), (300, 0), appearance=kit(0)),
                                                Player("B", (1000, 505), (-300, 0), appearance=kit(0),
                                                       visible=lambda t: not (0.85 <= t < 1.15))], {}
    yield "lookalike_emerges", lambda: [Player("A", (700, 500), (0, 0), height=90, appearance=kit(0)),
                                        Player("B", (600, 495), (200, 0), appearance=kit(1),
                                               visible=lambda t: t < 0.5),
                                        Player("C", (724, 497), (150, 0), appearance=kit(1),
                                               visible=lambda t: t >= 0.9)], {}
    yield "new_player_near_lost", lambda: [Player("B", (800, 500), (0, 0), visible=lambda t: t < 1.0),
                                           Player("A", (880, 500), (0, 0), visible=lambda t: t >= 1.6)], {}
    yield "enter_frame", lambda: [Player("A", (-60, 500), (300, 0), visible=lambda t: -60 + 300 * t > 10)], {}
    yield "crowd_8_random", lambda: crowd(8), {}
    yield "crowd_15_random", lambda: crowd(15, seed=2), {}
    yield "front_swap_same_look", lambda: [Player("A", (400, 500), (300, 0), appearance=kit(0),
                                                  visible=lambda t: not (1.0 <= t < 1.2)),
                                           Player("B", (1000, 505), (-300, 0), appearance=kit(0),
                                                  visible=lambda t: not (0.8 <= t < 1.0))], {}


def crowd(n, seed=1):
    rng = np.random.default_rng(seed)
    return [Player(f"P{k}", (600 + rng.uniform(-200, 200), 500 + rng.uniform(-60, 60)),
                   (rng.uniform(-250, 250), rng.uniform(-40, 40)), height=rng.uniform(70, 90),
                   appearance=kit(int(k % 2)),
                   visible=(lambda t, a=rng.uniform(0.5, 2.0): not (a <= t < a + 0.15))) for k in range(n)]


def main() -> None:
    print(f"{'scenario':28s} fps transfers unflagged switches dup_frames fragments wrong_rec recov coverage")
    tot = dict(transfers=0, unflagged=0, wrong=0, dup=0)
    for name, make, kw in scenarios():
        for fps in (10, 25, 60):
            m = evaluate(run(make(), fps=fps, duration_s=3, **kw))
            tot["transfers"] += m.identity_transfers; tot["wrong"] += m.wrong_recoveries
            tot["unflagged"] += m.unflagged_transfers
            tot["dup"] += m.duplicate_frames
            print(f"{name:28s} {fps:3d} {m.identity_transfers:9d} {m.unflagged_transfers:9d} {m.id_switches:8d} "
                  f"{m.duplicate_frames:10d} "
                  f"{m.total_fragments:9d} {m.wrong_recoveries:9d} {m.recoveries:5d} {m.coverage:8.3f}")
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
