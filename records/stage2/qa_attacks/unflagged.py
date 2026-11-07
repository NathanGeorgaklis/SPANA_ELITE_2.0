import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
import adv
from tests.synth import run
from tests.metrics import evaluate
tot = {"T": 0, "U": 0, "W": 0, "D": 0, "runs": 0}
for name, make in adv.S.items():
    for fps in adv.FPS:
        m = evaluate(run(make(), fps=fps, duration_s=3))
        tot["runs"] += 1; tot["T"] += m.identity_transfers; tot["U"] += m.unflagged_transfers
        tot["W"] += m.wrong_recoveries; tot["D"] += m.duplicate_frames
        if m.unflagged_transfers or m.wrong_recoveries or m.duplicate_frames:
            print(f"{name:34s} fps={fps:3d} T={m.identity_transfers} U={m.unflagged_transfers} W={m.wrong_recoveries} D={m.duplicate_frames}")
print(tot)
