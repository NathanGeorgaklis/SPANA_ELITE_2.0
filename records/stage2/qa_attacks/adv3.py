import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
import numpy as np
from tests.metrics import evaluate
from tests.synth import Player, kit, run
from spana.records import Detection, PositionSource
# (a) real second player standing behind A, box nested inside A's box
print("nested real player: hC, dx(px) -> C coverage, spawn_refused count")
for hc in (50, 60, 70, 80):
    for dx in (0, 5, 10):
        A = Player("A", (700, 500), (0, 0), height=90, appearance=kit(0))
        C = Player("C", (700+dx, 500+(90-hc)/2 - 2), (0, 0), height=hc, appearance=kit(1))
        res = run([A, C], fps=25, duration_s=2)
        m = evaluate(res)
        ref = sum(1 for ft in res.frames for d in ft.decisions if d["kind"] == "spawn_refused")
        a=A.box_at(0); c=C.box_at(0)
        iw=max(0,min(a[2],c[2])-max(a[0],c[0])); ih=max(0,min(a[3],c[3])-max(a[1],c[1]))
        ioa=iw*ih/((c[2]-c[0])*(c[3]-c[1]))
        print(f"  hC={hc} dx={dx} ioa={ioa:.2f} coverage={m.coverage:.3f} refused={ref} T/W/D={m.identity_transfers}/{m.wrong_recoveries}/{m.duplicate_frames}")
# (b) partial-body false detection of A shifted sideways: does it spawn a duplicate confirmed track?
print("partial-body false box of A (upper 60%), shifted sideways by frac of width")
for sh in (0.0, 0.1, 0.2, 0.3):
    A = Player("A", (700, 500), (100, 0), height=90, appearance=kit(0))
    def extra(i, t, sh=sh):
        b = A.box_at(t); w = b[2]-b[0]; h = b[3]-b[1]
        return [Detection((b[0]+sh*w, b[1], b[2]+sh*w, b[1]+0.6*h), 0.6, 0)]
    res = run([A], fps=25, duration_s=2, extra=extra)
    conf_ids = {o.track_id for ft in res.frames for o in ft.tracks if o.confirmed}
    m = evaluate(res)
    print(f"  shift={sh}: confirmed tracks={len(conf_ids)} dup_frames(metric)={m.duplicate_frames} transfers={m.identity_transfers}")
