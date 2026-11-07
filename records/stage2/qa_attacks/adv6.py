import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
import numpy as np
from tests.metrics import evaluate
from tests.synth import Player, kit, run
print("single player, iid score: P(score>=0.5)=p; fraction of frames with a CONFIRMED observed track")
for p in (0.9, 0.7, 0.5):
    row=[]
    for fps in (10,25,60,120):
        rng = np.random.default_rng(0); n=int(3*fps); s=np.where(rng.uniform(size=n)<p, 0.8, 0.3)
        A = Player("A", (500,500),(150,0), score=lambda t,s=s,fps=fps: s[min(int(round(t*fps)), len(s)-1)])
        r = run([A], fps=fps, duration_s=3)
        conf = sum(1 for ft in r.frames if any(o.confirmed and o.detection_index is not None for o in ft.tracks))
        m = evaluate(r)
        row.append(f"fps{fps}: conf={conf/len(r.frames):.2f} frag={m.total_fragments}")
    print(f" p={p}: ", " | ".join(row))
