import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
from tests.metrics import evaluate
from tests.synth import Player, kit, run
import diag, adv
def hid(a, b): return lambda t: not (a <= t < b)
def occluder_drop(la, lb, emerge=1.3, drop=0.1):
    return lambda: [Player("A", (700, 500), (0, 0), height=90, appearance=la, visible=hid(emerge, emerge+drop)),
                    Player("B", (500, 495), (200, 0), appearance=lb, visible=hid(0.9, emerge))]
adv.S["occluder_drops_as_B_emerges_diff"] = occluder_drop(kit(0), kit(1))
adv.S["occluder_drops_as_B_emerges_same"] = occluder_drop(kit(0), kit(0))
adv.S["occluder_drops_1frame_diff_e1.2"] = occluder_drop(kit(0), kit(1), emerge=1.2, drop=0.04)
# lateral offset sweep for unknown-appearance third player after short gap (B missed, no occluder)
def short_gap_none(off_h, gap=0.1):
    H=80
    return lambda: [Player("B", (500, 500), (200, 0), appearance=kit(1), visible=lambda t: t < 1.0),
                    Player("C", (500, 500+off_h*H), (200, 0), appearance=None, visible=lambda t, g=gap: t >= 1.0+g)]
for off in (0.0, 0.5, 1.0, 1.3):
    adv.S[f"short_gap_None_C_off{off}h"] = short_gap_none(off)
# all appearance None (tiny/distant): B missed 0.15s, C (untracked) appears 1.0h away
for off in (0.5, 1.0):
    adv.S[f"short_gap_allNone_off{off}h"] = (lambda o=off: [Player("B", (500, 500), (200, 0), visible=lambda t: t < 1.0),
                     Player("C", (500, 500+o*80), (200, 0), visible=lambda t: t >= 1.1)])
if __name__ == "__main__":
    for name in [k for k in adv.S if k.startswith(("occluder_drops","short_gap"))]:
        row=[]
        for fps in adv.FPS:
            m = evaluate(run(adv.S[name](), fps=fps, duration_s=3))
            row.append(f"{m.identity_transfers}/{m.wrong_recoveries}/{m.duplicate_frames}/{m.total_fragments}")
        print(f"{name:36s}", "  ".join(row))
    print(diag.diag("occluder_drops_as_B_emerges_diff", 10))
