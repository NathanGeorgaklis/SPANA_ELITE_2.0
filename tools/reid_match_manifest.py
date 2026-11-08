"""Create the per-match manifest template that every re-ID labelling pack requires.

    python -m tools.reid_match_manifest --video MATCH.mp4 --match-id M2026_001 --split TRAIN --out MATCH.reid_match.json

Fills match_id, split and the video sha256. You must then fill footage_licence, rights_holder and consent_basis,
and set training_permitted to true only if the rights and consent cover ML training. tools.reid_pack refuses an
incomplete manifest; a TRAIN match also needs training_permitted = true.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spana.reid_labels import SPLITS, ReidLabelError, manifest_template
from spana.video import sha256_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--match-id", required=True)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 1
    try:
        m = manifest_template(a.match_id, a.split, sha256_file(a.video))
    except ReidLabelError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"wrote {out}: now fill footage_licence, rights_holder, consent_basis and training_permitted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
