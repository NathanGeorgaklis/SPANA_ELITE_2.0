# TRAIN_M001_MORETON_CITY: preparation status (2026-11-08)

## Match
- Source: `Downloads/videoplayback (1).mp4`, chosen by Nathan.
- FQTV broadcast featuring Moreton City Excelsior, 1280×720, 60 fps, 147 min.
- sha256 `e8bd4d5fd8105a862ad8388f09473732b47481d6bda89ce10480ca18c810c519`.
- **Not a DEV match.** The DEV matches are G1 SUD v APL, G2 NWS v SDR and G3 SFC v STC (FRESH_* clips). A
  contact sheet confirms this one is a different broadcast, competition and teams.
- Proposed 10-minute window: 3000–3600 s of the file. Play was confirmed at 3000 s; the full window is not yet
  inspected.
- Draft manifest: `Downloads/SPANA_reid/TRAIN_M001/TRAIN_M001_MORETON_CITY.reid_match.json`, split TRAIN.
  `validate_manifest` refuses it, as intended, until the rights fields are filled.

## Blocked: written permission details not recorded
- Nathan answered "written permission exists" but supplied the list of what the permission must cover, not the
  agreement itself. That list is now enforced as `permissions` in manifest schema 1.1.0.
- Nothing has been copied, cropped, detected or tracked from this match. Processing the footage needs
  `copy_and_process` first.

## Needed to proceed
1. **The agreement details:** `rights_holder`, `footage_licence`, `consent_basis`, `agreement_reference` (document
   name or number and date), and true/false for each permission, as the document states it.
2. **Then:**
   - cut the window on the Mac (stream copy);
   - `spana.cli detect --stride 4` (15 Hz analysis), then `track`;
   - `tools.reid_pack`.
   - Expected time: about 9,000 analysed frames × about 2 s, so about 5 h on the 2-CPU workspace. A faster
     machine is recommended.
