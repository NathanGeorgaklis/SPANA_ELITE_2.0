# Football detector person view vs YOLOX baseline: tracking on DEV (2026-11-16)

- Scope: DEV only. EVAL was not used. No threshold or model change: default.yaml and config/roboflow.yaml as on
  branch `tracking/person-detections` (commit 9b3ac67).
- Clips: FRESH_02 (DEV_G1_SYDNEY_UNITED_APIA) and FRESH_05 (DEV_G2_NWS_SPIRIT_SD_RAIDERS), the two allow-listed DEV
  clips with frozen MOT labels (`data/labels/<clip>/v1`, frozen 2026-11-08). FRESH_07 / FRESH_08 are Game 3 (not on
  the allow-list) and were not used.
- Both backends ran on the same frames (every decodable frame: FRESH_02 391, FRESH_05 781; same video sha256 and the
  same `detection_config`), then the same tracker and the same scorer (`tools/benchmark.py`, NESTED_V2):
  - YOLOX: `spana.cli detect` with the pinned YOLOX-s (`models/yolox_s.json`, weights sha256 c5c2d13e…), local.
    Re-run today so both sides use today's tracker code. The 2026-11-08 records used older code.
  - Football detector: `spana.cli detect --backend roboflow`, i.e. detector v1 (`spana-elite-2-rfdetr-medium-t1`)
    through `PersonView` (player + goalkeeper + referee, class-agnostic NMS at detection.nms_iou, requested at
    detection.score_low 0.10). Hosted inference: 1,172 calls.

## Results (NESTED_V2)

| metric | FRESH_02 YOLOX | FRESH_02 football | FRESH_05 YOLOX | FRESH_05 football |
|---|---|---|---|---|
| detection recall | 0.9725 | **0.9804** | 0.9889 | **0.9978** |
| missed players (labelled boxes) | 28 | **20** | 5 | **1** |
| recall, occluded players | 0.741 | **0.815** | 1.000 | 1.000 |
| duplicate detections | 126 | **17** | 84 | **3** |
| track coverage | 0.9597 | **0.9705** | 0.9800 | **0.9978** |
| ID switches | 28 | **25** | 22 | **8** |
| fragments | 28 | **24** | 21 | **8** |
| duplicate track frames | 16 | **5** | 3 | **0** |
| identity transfers (unflagged) | 3 (0) | 3 (0) | 0 (0) | 0 (0) |
| labelled boxes | 1018 | 1018 | 450 | 450 |

Missed by box height: FRESH_02 YOLOX 48-96 px 6, >= 96 px 22; football 6 and 14. FRESH_05: 5 and 1, all >= 96 px.

## Findings
1. The football detector's person view is better than YOLOX on every tracking metric on both clips. It never does
   worse.
2. The largest gains are duplicates: duplicate detections fall 126 → 17 and 84 → 3, and duplicate track frames
   16 → 5 and 3 → 0. ID switches / fragments fall 22 → 8 / 21 → 8 on FRESH_05.
3. The kit-colour role confusion is still in the class hints (FRESH_05: 1,902 of 5,886 person boxes say
   "referee"; the yellow-kit team). The tracker does not use the class, so it does not hurt tracking.
4. The remaining FRESH_02 misses are mostly small or occluded players in the 8/9/10 contact cluster. The 3
   identity transfers are flagged, as before.
5. Limits: two short clips (15.7 s and 17.7 s), 18 + 12 players. Both matches come from the same competition and
   broadcast style as the benchmark. This compares detectors; tracker settings were tuned on YOLOX output and were
   not re-tuned here.

## Artifacts (runs/person_view_compare/, not committed)
Detections, tracks and decision logs: full sha256 in `HASHES.txt` (written by `shasum -a 256`). Scorer reports are
committed here: `F0[25]_{rf,yolox}.bench.json`.
