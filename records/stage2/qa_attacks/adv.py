import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
import numpy as np
from spana.records import CameraObservation, CameraStatus, Detection
from tests.metrics import evaluate
from tests.synth import Player, kit, run, cfg

def hid(a, b): return lambda t: not (a <= t < b)
def vis_from(a): return lambda t: t >= a

S = {}
# 1 occluded player B (kit1) behind static A (kit0); third player C emerges exactly on B's extrapolated path
def third_on_path(look_c, hide_end=None, cstart=0.9):
    def f():
        return [Player("A", (700, 500), (0, 0), height=90, appearance=kit(0)),
                Player("B", (500, 495), (200, 0), appearance=kit(1), visible=lambda t: t < 0.9),
                Player("C", (500, 495), (200, 0), appearance=look_c, visible=lambda t, c=cstart: t >= c)]
    return f
S["third_on_path_samelook_gap0.4"] = third_on_path(kit(1), cstart=1.3)
S["third_on_path_samelook_gap0.9"] = third_on_path(kit(1), cstart=1.8)
S["third_on_path_None_gap0.1"] = third_on_path(None, cstart=1.0)
S["third_on_path_None_gap0.4"] = third_on_path(None, cstart=1.3)
S["third_on_path_diff_gap0.1"] = third_on_path(kit(2), cstart=1.0)
# 2 two players both occluded by same occluder, emerging swapped
def two_behind(lb, lc, swap):
    def f():
        ps = [Player("A", (700, 500), (0, 0), height=100, appearance=kit(0))]
        vb = (150, 0); vc = (150, 0)
        ps.append(Player("B", (520, 490), vb, appearance=lb, visible=hid(0.9, 1.4)))
        ps.append(Player("C", (480, 505), vc, appearance=lc, visible=hid(1.0, 1.5)))
        return ps
    return f
S["two_behind_same"] = two_behind(kit(1), kit(1), False)
S["two_behind_diff"] = two_behind(kit(1), kit(2), False)
S["two_behind_None"] = two_behind(None, None, False)
# 2b two players behind occluder, then velocity swap (B emerges where C would be)
def two_swap(lb, lc):
    def f():
        class P2(Player):
            pass
        A = Player("A", (700, 500), (0, 0), height=100, appearance=kit(0))
        B = Player("B", (560, 500), (150, 0), appearance=lb, visible=hid(0.8, 1.4))
        C = Player("C", (840, 500), (-150, 0), appearance=lc, visible=hid(0.8, 1.4))
        return [A, B, C]
    return f
S["two_cross_behind_same"] = two_swap(kit(1), kit(1))
S["two_cross_behind_diff"] = two_swap(kit(1), kit(2))
S["two_cross_behind_None"] = two_swap(None, None)
# 3 occluder-switch at crossing: B hidden behind A, then A hidden behind B (front swap)
def front_swap(la, lb):
    def f():
        A = Player("A", (400, 500), (300, 0), appearance=la, visible=hid(1.0, 1.2))
        B = Player("B", (1000, 505), (-300, 0), appearance=lb, visible=hid(0.8, 1.0))
        return [A, B]
    return f
S["front_swap_same"] = front_swap(kit(0), kit(0))
S["front_swap_diff"] = front_swap(kit(0), kit(1))
S["front_swap_None"] = front_swap(None, None)
# 3b slow crossing, front swap, low relative speed
def front_swap_slow(la, lb):
    def f():
        A = Player("A", (600, 500), (60, 0), appearance=la, visible=hid(1.0, 1.3))
        B = Player("B", (800, 503), (-60, 0), appearance=lb, visible=hid(1.3, 1.6))
        return [A, B]
    return f
S["front_swap_slow_same"] = front_swap_slow(kit(0), kit(0))
S["front_swap_slow_None"] = front_swap_slow(None, None)
# 4 occluder track leaves the frame / terminated while B occluded
def occluder_leaves(lb, lc):
    def f():
        A = Player("A", (60, 500), (-300, 0), height=90, appearance=kit(0))   # leaves left edge
        B = Player("B", (0, 495), (60, 0), appearance=lb, visible=hid(0.05, 0.6))
        return [A, B]
    return f
# 4b B occluded by A with same kit as B; A track dies (cut? no). A walks out of image -> margin protection gone; C same look emerges.
def margin_gone():
    A = Player("A", (150, 500), (-500, 0), height=90, appearance=kit(1))   # same kit as B, leaves frame fast
    B = Player("B", (100, 495), (100, 0), appearance=kit(1), visible=lambda t: t < 0.2)
    C = Player("C", (100, 495), (100, 0), appearance=kit(1), visible=lambda t: t >= 0.8)
    return [A, B, C]
S["margin_gone_samekit"] = margin_gone
# 5 long occlusion near 1.0s with velocity reversal
def reversal(look_same):
    def f():
        A = Player("A", (700, 500), (0, 0), height=110, appearance=kit(0))
        def bpos(t):  pass
        B1 = Player("B", (600, 495), (150, 0), appearance=kit(1), visible=lambda t: t < 0.8)
        # after hidden ~0.95s, a player emerges at mirrored position moving back (reversal); is it B? geometry: B reversed
        B2 = Player("B", (600+150*0.8*2, 495), (-150, 0), appearance=kit(1), visible=lambda t: t >= 1.75)
        return [A, B1, B2]
    return f
S["reversal_true_B_0.95s"] = reversal(True)
# 6 appearance None for some players in a crowd with 2 kits
def crowd(n, seed, none_frac=0.0, drop=0.15, H=(70, 90), speed=250, cx=900, spread=300):
    rng = np.random.default_rng(seed)
    ps = []
    for k in range(n):
        app = None if rng.uniform() < none_frac else kit(int(k % 2))
        starts = sorted(rng.uniform(0.3, 2.5, 3))
        durs = rng.uniform(0.03, drop, 3)
        vis = (lambda t, s=starts, d=durs: not any(a <= t < a + b for a, b in zip(s, d)))
        ps.append(Player(f"P{k}", (cx + rng.uniform(-spread, spread), 500 + rng.uniform(-80, 80)),
                         (rng.uniform(-speed, speed), rng.uniform(-50, 50)), height=rng.uniform(*H),
                         appearance=app, visible=vis))
    return ps
for n in (10, 15):
    for seed in range(3):
        S[f"crowd{n}_s{seed}"] = (lambda n=n, s=seed: crowd(n, s))
        S[f"crowd{n}_s{seed}_none50"] = (lambda n=n, s=seed: crowd(n, s, none_frac=0.5))
        S[f"crowd{n}_s{seed}_longdrop"] = (lambda n=n, s=seed: crowd(n, s, drop=0.8))
        S[f"crowd{n}_s{seed}_tiny"] = (lambda n=n, s=seed: crowd(n, s, H=(18, 26), speed=60, spread=150))
FPS = (10, 25, 60, 120)
if __name__ == "__main__":
    sel = sys.argv[1:] 
    print(f"{'scenario':34s} " + " ".join(f"fps{f}:T/W/D/F" for f in FPS))
    for name, mk in S.items():
        if sel and not any(s in name for s in sel): continue
        row = []
        for fps in FPS:
            m = evaluate(run(mk(), fps=fps, duration_s=3))
            row.append(f"{m.identity_transfers}/{m.wrong_recoveries}/{m.duplicate_frames}/{m.total_fragments}")
        flag = " <<<" if any(r.split('/')[0] != '0' or r.split('/')[1] != '0' or r.split('/')[2] != '0' for r in row) else ""
        print(f"{name:34s} " + "  ".join(f"{r:>13s}" for r in row) + flag)
