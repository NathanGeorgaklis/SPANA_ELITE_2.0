import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
import numpy as np
from tests.metrics import evaluate
from tests.synth import Player, kit, run
from spana.records import CameraObservation, CameraStatus
def hid(a, b): return lambda t: not (a <= t < b)
def occl(lb, score=lambda t: 0.9, third=False):
    ps = [Player("A", (900, 500), (0, 0), height=90, appearance=kit(0)),
          Player("B", (700, 495), (200, 0), appearance=lb, visible=hid(0.9, 1.3), score=score)]
    if third:
        ps.append(Player("C", (900, 495), (60, 0), appearance=kit(1), visible=lambda t: t >= 1.3))
    return ps
def pan_run(ps, fps, pan_px_s, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed); per = pan_px_s / fps
    err = [0.0] + list(rng.normal(0, noise_px, 10000))
    def cam(i):
        if i == 0: return CameraObservation.identity(CameraStatus.FIRST)
        return CameraObservation(CameraStatus.OK, ((1, 0, -per + err[i]), (0, 1, 0)), 100, 0.01)
    return run(ps, fps=fps, duration_s=3, camera=cam, frame_w=40000,
               world_to_image=lambda i, b: b - np.array([1, 0, 1, 0]) * per * i)
print("pan during occlusion (distinct): fps pan noise -> T/W/D/F recov")
for fps in (10, 25, 60, 120):
    for pan, noise in ((400, 0), (800, 0), (400, 3), (400, 8)):
        m = evaluate(pan_run(occl(kit(1)), fps, pan, noise))
        print(f"  {fps:3d} {pan} {noise}: {m.identity_transfers}/{m.wrong_recoveries}/{m.duplicate_frames}/{m.total_fragments} rec={m.recoveries}")
print("pan + same-look third emerging off path")
for fps in (10, 25, 60, 120):
    m = evaluate(pan_run(occl(kit(1), third=True), fps, 400, 3))
    print(f"  {fps}: {m.identity_transfers}/{m.wrong_recoveries}/{m.duplicate_frames}/{m.total_fragments}")
print("score flicker around score_high while emerging / during occlusion")
for fps in (10, 25, 60, 120):
    for name, sc in (("alt", lambda t: 0.45 if int(t*1000) % 2 else 0.55),
                     ("low_after", lambda t: 0.45 if 1.3 <= t < 1.6 else 0.9),
                     ("rand", lambda t: 0.3 + 0.4 * ((np.sin(t * 997.0) + 1) / 2))):
        for lb in (kit(1), kit(0)):
            m = evaluate(run(occl(lb, score=sc), fps=fps, duration_s=3))
            print(f"  {fps:3d} {name:9s} {'diff' if lb is not None and lb[0]!=kit(0)[0] else 'same'}: {m.identity_transfers}/{m.wrong_recoveries}/{m.duplicate_frames}/{m.total_fragments}")
