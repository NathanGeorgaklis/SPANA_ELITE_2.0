import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np, json
from tests.metrics import evaluate
from tests.synth import Player, kit, run
import adv
# dangling occluder: A covers B then A's detection vanishes (A track LOST), A prediction leaves frame
def hid(a,b): return lambda t: not (a<=t<b)
A = Player("A", (1800, 500), (0, 0), height=90, appearance=kit(0), visible=lambda t: t < 1.0)
B = Player("B", (1700, 495), (100, 0), appearance=kit(1), visible=lambda t: t < 0.9)
for fps in (25,):
    r = run([A, B], fps=fps, duration_s=2)
    dang = 0; occ_of_unobserved = 0
    for ft in r.frames:
        live = {o.track_id: o for o in ft.tracks}
        for o in ft.tracks:
            if o.occluder is not None:
                if o.occluder not in live: dang += 1
                elif live[o.occluder].detection_index is None: occ_of_unobserved += 1
    print("dangling occluder frames", dang, "occluder itself unobserved frames", occ_of_unobserved)
# determinism
def sig(r): return json.dumps([[d for d in ft.decisions] for ft in r.frames], sort_keys=True, default=str)
a = sig(run(adv.S["crowd15_s2"](), fps=25, duration_s=3)); b = sig(run(adv.S["crowd15_s2"](), fps=25, duration_s=3))
print("deterministic crowd15:", a == b)
# scale invariance: x2 everything
def scaled(ps, s):
    return [Player(p.gid, (p.start[0]*s, p.start[1]*s), (p.velocity[0]*s, p.velocity[1]*s), height=p.height*s,
                   aspect=p.aspect, visible=p.visible, score=p.score, appearance=p.appearance) for p in ps]
for name in ("crowd15_s2", "crowd10_s1", "front_swap_same", "third_on_path_samelook_gap0.4"):
    row=[]
    for s in (0.5, 1.0, 2.0):
        m = evaluate(run(scaled(adv.S[name](), s), fps=25, duration_s=3, frame_w=int(1920*s*2), frame_h=int(1080*s*2)))
        row.append((s, m.identity_transfers, m.wrong_recoveries, m.total_fragments))
    print(name, row)
