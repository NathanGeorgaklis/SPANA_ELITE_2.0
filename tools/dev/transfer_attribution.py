# DEV diagnostic script (not product code). For each identity transfer of a benchmark report, prints per labelled
# frame around it the track box against both players' labelled boxes: IoU, and the foot point (bottom-centre)
# offset in the player's body heights, plus the players' mutual IoU and occlusion flags. Separates real
# identity transfers from attribution ambiguity between overlapping players.
#   python tools/dev/transfer_attribution.py LABEL_DIR TRACKS BENCHMARK [--all]
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from tools.benchmark import load_frozen  # noqa: E402

SPAN = 4      # labelled frames shown on each side


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)


def foot(t, b):
    h = b[3] - b[1]
    return ((t[0] + t[2]) - (b[0] + b[2])) / 2 / h, (t[3] - b[3]) / h


def main(label_dir, trk_path, bench, *flags):
    fz, gt = load_frozen(Path(label_dir))
    trk = {f["frame"]: {t["track_id"]: t for t in f["tracks"]} for f in json.load(open(trk_path))["frames"]}
    tr = [t for t in json.load(open(bench))["events"]["transfers"] if "--all" in flags or not t["flagged"]]
    lf = gt.frames
    for t in tr:
        i = lf.index(t["frame"])
        print(f"\n=== {t['track_id']} {t['from']}->{t['to']} at {t['frame']} (flagged={t['flagged']})")
        for f in lf[max(0, i - SPAN): i + SPAN + 1]:
            o = trk.get(f, {}).get(t["track_id"])
            if not o:
                print(f"  f{f} track not present"); continue
            P, occ = gt.players[f], gt.occluded.get(f, set())
            cells = []
            for g in (t["from"], t["to"]):
                if g in P:
                    dx, dy = foot(o["box"], P[g])
                    cells.append(f"{g}{'(occ)' if g in occ else ''} iou {iou(o['box'], P[g]):.2f} foot {dx:+.2f},{dy:+.2f}h")
                else:
                    cells.append(f"{g} unlabelled")
            m = iou(P[t["from"]], P[t["to"]]) if t["from"] in P and t["to"] in P else None
            print(f"  f{f} {o['source'][:3]} | " + " | ".join(cells) + (f" | players' IoU {m:.2f}" if m is not None else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
