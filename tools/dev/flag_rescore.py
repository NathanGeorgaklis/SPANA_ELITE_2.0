"""Old vs new transfer flagging on the same runs (not product code; 2026-11-20 rule change in tools/benchmark.py).

    python -m tools.dev.flag_rescore --out OUT.json --run SET CLIP OLD.bench.json NEW.bench.json [--run ...]

Matches transfer events by (track, frame, from, to); lists every event whose flag changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _k(e):
    return e["track_id"], e["frame"], e["from"], e["to"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs=4, action="append", required=True, metavar=("SET", "CLIP", "OLD", "NEW"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    sets: dict = {}
    for name, clip, old, new in a.run:
        o, n = (json.loads(Path(x).read_text()) for x in (old, new))
        oe = {_k(e): e for e in o["events"]["transfers"]}
        ne = {_k(e): e for e in n["events"]["transfers"]}
        if set(oe) != set(ne):
            raise SystemExit(f"{name} {clip}: transfer events differ between old and new scoring")
        flips = [{"clip": clip, **ne[k], "old_flagged": oe[k]["flagged"]} for k in sorted(oe)
                 if oe[k]["flagged"] != ne[k]["flagged"]]
        so, sn = o["summary"], n["summary"]
        s = sets.setdefault(name, {"transfers": 0, "old_silent": 0, "new_silent": 0, "recovery_caused": 0,
                                   "recovery_caused_silent": 0, "link_caused": 0, "link_caused_silent": 0,
                                   "on_linked_tracks": 0, "flips": []})
        s["transfers"] += sn["identity_transfers"]
        s["old_silent"] += so["unflagged_transfers"]
        s["new_silent"] += sn["unflagged_transfers"]
        s["recovery_caused"] += sn["recovery_caused_transfers"]
        s["recovery_caused_silent"] += sn["recovery_caused_silent"]
        s["link_caused"] += sn["link_caused_transfers"]
        s["link_caused_silent"] += sn["link_caused_silent"]
        s["on_linked_tracks"] += sn["transfers_on_linked_tracks"]
        s["flips"] += flips
    Path(a.out).write_text(json.dumps(sets, indent=1, sort_keys=True))
    for name, s in sets.items():
        print(f"{name:14s} transfers {s['transfers']:3d} silent old {s['old_silent']:3d} new {s['new_silent']:3d} | "
              f"recovery-caused {s['recovery_caused']} ({s['recovery_caused_silent']} silent) link-caused "
              f"{s['link_caused']} ({s['link_caused_silent']} silent) on linked tracks {s['on_linked_tracks']} | "
              f"flipped {len(s['flips'])}")
        for f in s["flips"]:
            print(f"   {f['clip']} {f['track_id']} frame {f['frame']} {f['from']}->{f['to']} flagged "
                  f"{f['old_flagged']} -> {f['flagged']} (recovery {f['recovery_caused']}, link {f['link_caused']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
