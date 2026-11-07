"""Download the weights named by a model record and verify the pinned sha256. Refuses on mismatch.

    python tools/fetch_model.py models/yolox_s.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def main(record_path: str) -> int:
    rec_p = Path(record_path)
    rec = json.loads(rec_p.read_text())
    dest = rec_p.parent / rec["weights_path"]
    if dest.exists():
        print(f"exists: {dest}")
    else:
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(rec["source"], tmp)
        tmp.rename(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != rec["sha256"]:
        print(f"SHA256 MISMATCH {digest} != {rec['sha256']} -- refusing; delete {dest}", file=sys.stderr)
        return 1
    print(f"verified {dest} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
