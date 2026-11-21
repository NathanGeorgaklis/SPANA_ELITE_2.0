# appearance_cue_eval: deterministic rival-test order (2026-11-22)

CODE TASK (Nathan), evaluation. Branch `evaluation/cue-eval-deterministic`. DEV FRESH_02 / 05 / 07 / 08 frozen labels
only; no EVAL; no model, margin or config change.

**Finding.** `tools/dev/appearance_cue_eval.py` built the rival test triplets by iterating `hard`, a set of player-id
pairs. Player ids are strings, so the order follows Python's hash seed: every rerun wrote the same tests in a
different order, and `--dump-tables` files differed byte-wise (noted in `records/reid_v5_dev_eval_2026-11-21`).
**Fix.** Triplets are built by `rival_triplets()`, iterating `sorted(hard)`. Nothing else in the tool depended on set
order (the other set uses are counts or membership; pair subsampling is seeded).

## DEV check (`run_check.sh`, outputs `~/spana-cue-det/out`, HASHES.txt)
Same inputs as `records/reid_v5_dev_eval_2026-11-21` (labels, videos, the main tracker's tracks and decision logs,
models v2 / v5).
- v2 dumped twice, `PYTHONHASHSEED` 1 and 2: tables (sha256 `42b55b14...`), report and calibration all
  **byte-identical**.
- Same test multiset as the v5 record for every clip (FRESH_05 53, FRESH_07 2,106, FRESH_02 1,377, FRESH_08 2,246);
  only the order of FRESH_07 / 02 / 08 changed (FRESH_05 order was already the same).
- `tools/dev/reid_calibrate.py` vs the v5 record: every value equal (v2 max difference 0.0, v5 2.2e-16 from float
  summation order), apart from `inputs.tables_sha256`. The adopted v2 numbers and the v5 verdict are unchanged.
- Table hashes written from now on differ from earlier records (new canonical order); earlier records stay valid
  as recorded.
