# Re-ID tracklet labelling: how to prepare and label a match

Tools:
- `tools/reid_match_manifest.py` makes the match manifest template.
- `tools/reid_pack.py` makes the labelling pack: tracker tracklets and thumbnails.
- `tools/reid_labeller.html` is the labelling page, copied into each pack.
- `tools/freeze_reid_labels.py` freezes the export.

Library: `spana/reid_labels.py` holds the manifest rules, the split guard (EVAL is never used for train or tune) and
hash-checked loading of frozen labels.

## 1. Manifest: split, rights and consent (required)
```
python -m tools.reid_match_manifest --video MATCH.mp4 --match-id <MATCH_ID> --split TRAIN --out <MATCH_ID>.reid_match.json
```
Then fill in `footage_licence`, `rights_holder`, `consent_basis` and `agreement_reference`. Set
`training_permitted` and each entry of `permissions` to true only where the written permission says so:

| permission | the agreement must grant |
|---|---|
| `copy_and_process` | copying and processing the footage |
| `ml_development` | developing, training, testing and improving ML / computer-vision models |
| `commercial_use` | that this use may be commercial |
| `player_consent` | consent or authority covering the players' images for tracking / re-ID development |
| `minors_covered` | parent / guardian consent or another legal basis for any minors (true also if none appear) |
| `retain_derived_data` | keeping crops, annotations, embeddings and training outputs |
| `model_survives_footage_removal` | recommended, not required: continued use of the trained model if the footage is later removed |

A TRAIN match is refused unless all six required permissions, `training_permitted` and `agreement_reference` are
set (manifest schema 1.1.0). The split is decided per match, before labelling, and never changes.

## 2. Detect and track (tracker proposes tracklets)
```
python -m spana.cli detect --video MATCH.mp4 --config config/default.yaml --model-record models/yolox_s.json \
    --stride 2 --start <F0> --stop <F1> --out <MATCH_ID>.det.json
python -m spana.cli track --video MATCH.mp4 --config config/default.yaml --detections <MATCH_ID>.det.json \
    --out <MATCH_ID>.trk.json
```
Detection takes about 2.2 s per analysed frame on the 2-CPU workspace. A 10-minute window at 25 fps with
`--stride 2` is about 7,500 frames, about 4.5 h. A full match needs a faster machine.

## 3. Pack
```
python -m tools.reid_pack --video MATCH.mp4 --tracks <MATCH_ID>.trk.json --manifest <MATCH_ID>.reid_match.json \
    --out <MATCH_ID>_reid_pack
```
Tracklets are cut automatically where a track was unobserved for more than 0.5 s, and where it entered
IDENTITY_UNCERTAIN.

## 4. Label (Chrome, offline)
Open `<MATCH_ID>_reid_pack/index.html` in Chrome.
1. Enter your name as annotator.
2. Add players on the left: key, side, optional shirt number. Keys are match-local, e.g. `A07`, `B11`, `GK_A1`,
   `REF1`. Use the side for keepers and referees.
3. For each card:
   - **Split (✂)** between two thumbnails if the tracklet changes person. "Join with previous" undoes it.
   - **Assign** the active player to each part. Assigning the same player to several tracklets links them.
   - **Reject** a part that is not a person or is a box mixing two people.
   - **Unsure** a part that is a person you cannot identify.
4. Use "Show: undecided" to finish the rest. "Show: conflicts" lists any player placed in two places at once.
5. Click **Export frozen labels**. The page refuses while any part is undecided or conflicting. Progress is saved in
   the browser.

## 5. Freeze and commit
```
python -m tools.freeze_reid_labels --pack <MATCH_ID>_reid_pack --export <MATCH_ID>_reid_v1.json --video MATCH.mp4
```
This writes `data/reid/<SPLIT>/<MATCH_ID>/v1/`, containing `manifest.json`, `players.json`, `identities.jsonl`,
`boxes.mot.txt` (MOT 1.1) and `export.json`, plus `FROZEN.json` with the sha256 of each file. It never overwrites.
A correction is v2 with a reason.
