import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "..")); sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
from tests.metrics import evaluate
from tests.synth import Player, kit, run
import adv4, diag, adv
for fps in (10,25,60,120):
    r = run(adv4.occl(kit(1), third=True), fps=fps, duration_s=3)
    m = evaluate(r); print("nopan", fps, m.identity_transfers, m.wrong_recoveries, m.duplicate_frames, m.total_fragments)
adv.S["third_nopan"] = lambda: adv4.occl(kit(1), third=True)
print(diag.diag("third_nopan", 60)[:6])
# when are B and C separated?
B = adv4.occl(kit(1), third=True)[1]; C = adv4.occl(kit(1), third=True)[2]
for t in (1.3, 1.4, 1.5, 1.6):
    print(t, "B-C dx px", (B.box_at(t)[0]-C.box_at(t)[0]))
