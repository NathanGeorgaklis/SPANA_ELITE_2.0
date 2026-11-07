# SPANA Elite 2.0

Football match analysis: a player uploads a full match, selects themselves, and the system tracks them, detects
their actions and outcomes, and produces event data, statistics and clips.

Fresh build started 2026-11-07. No code from earlier SPANA builds is used.

## Pipeline and status

| # | Layer | Package | Status |
|---|---|---|---|
| 1 | Video input + provenance | `spana/video.py`, `spana/provenance.py` | built |
| 2 | Player detection | `spana/detection/` | built (interface + pinned ONNX backend). **No model pinned yet** |
| 3 | Multi-player tracking | `spana/tracking/` | built, synthetic-tested only |
| 4 | Target identity | — | next |
| 5 | Ball tracking | — | |
| 6 | Team classification | — | |
| 7 | Pitch / context | — | |
| 8 | Possession / touches | — | |
| 9 | Event segmentation → classification → outcomes | — | |
| 10 | Statistics (reducer over validated events only) | — | |
| 11 | Clips (from validated events only) | — | |

Each layer consumes the previous layer's **versioned artifact** (`spana/artifacts.py`, currently
`spana.detections` / `spana.tracks` 1.1.0), never its internals. Stage records, including negative results,
are in `records/`.

## Run

```bash
pip install -r requirements.txt
python -m spana.cli detect --video match.mp4 --config config/default.yaml --model-record models/<model>.json --out det.json
python -m spana.cli track  --video match.mp4 --config config/default.yaml --detections det.json --out tracks.json
python -m pytest tests -q
```

A model record (`ModelRecord` in `spana/detection/onnx_backend.py`) pins name, source, exact version, sha256,
licence, preprocessing and output interpretation. The weights are hash-checked before loading.

## Rules the code enforces

- Config fails closed: no code defaults, unknown or missing keys rejected. Times in seconds, lengths in body
  heights or frame fractions. All values in `config/default.yaml` are **uncalibrated priors**.
- Tracker states keep observation and inference apart: OBSERVED / PREDICTED / HELD.
- A camera cut or unestimated camera motion ends motion continuity (a new epoch). Re-linking across epochs
  belongs to the identity layer.
- A lost track is re-matched by motion only within `rematch_max_s` and only when the pairing is exclusive.
  Every bridged gap is recorded (`gaps_bridged`, `last_gap_s`) and must be verified by the identity layer.
- Ambiguity is reported, not hidden: `contested` (association ambiguity) and `max_overlap_iou` / `overlap_with`
  (physical overlap) are separate fields.
- Every association, spawn and end is written to the decision log (`*.decisions.jsonl`).
- Artifacts are canonical JSON, validated, written atomically and never overwritten.
- DEV and EVAL footage are kept strictly separate. There is no footage in this repository yet.
