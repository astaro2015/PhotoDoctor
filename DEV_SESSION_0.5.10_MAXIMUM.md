# Photo Doctor 0.5.10 Maximum — development journal

Started: 2026-09-24
Base: PhotoDoctor 0.5.9 Ideal / Surface V9 manual-only

## User request
1. Rename fourth precision mode from «Идеальный» to «Максимальный».
2. Make the fourth mode spatially denser, not only deeper in AI/CV budgets.
3. Spend about 2x more wall-clock time than the previous fourth mode when useful, without weakening quality thresholds.

## Design decisions
- Preserve Fast/Normal/Precise unchanged.
- New public/internal canonical key: `maximum`.
- Keep `ideal`/`идеально` as compatibility aliases only.
- 4K local-map target: roughly 2x the previous Ideal spatial-window count.
- Increase work by more spatial samples / search scales / candidate review, never by sleep/delay.
- Surface remains strictly manual-only.

## Baseline 4K local-map budget (3840x2160)
- Precise: ~5,024 windows across sharpness/tone/contrast.
- 0.5.9 Ideal: ~10,616 windows.
- 0.5.10 Maximum target: ~21,900 windows (~2.06x old Ideal, ~4.36x Precise).

## Status
- [x] Base unpacked and inspected.
- [x] Spatial budget calculation completed.
- [ ] Rename mode and preserve aliases.
- [ ] Increase spatial map density.
- [ ] Increase Maximum-only Surface/AI/face/eye/validator budgets.
- [ ] Add/adjust regression tests.
- [ ] Benchmark old Ideal vs new Maximum.
- [x] Full test suite: 83/83 files, 710 tests PASS.
- [x] Candidate ZIP: 332/332 SHA PASS; unpacked smoke 247/247 PASS; path-with-spaces compileall PASS.
- [x] Pre-final rebuilt ZIP: 332/332 SHA PASS; 162/162 smoke PASS; path-with-spaces compileall PASS.
- [x] Final documentation-only rebuild prepared; exact archive bytes are verified after creation (see published SHA/build report).

## 2026-09-24 final tuning update
- Public fourth mode renamed to `maximum` / «Максимальный»; legacy `ideal` / `идеально` remain aliases only.
- Result tab is read-only summary; hidden canonical selection list is driven from Plan / specialised tabs.
- Surface v9 remains strictly manual-only (`accepted=False`, `auto_eligible=False`; repair trial deferred until user selects Defects rows).
- Maximum spatial-map density increased a second time:
  - 4K (3840x2160): ~42,928 local windows total across sharpness/tone/contrast.
  - archival reference 479x781: ~9,033 local windows (3375 + 2829 + 2829).
- Maximum Surface uses additive Precise + Maximum + two independent CLAHE proposal passes.
- Maximum visible/AI Surface budgets raised to 256/320; archive reference: 232 candidates, all 232 AI-refined, unprocessed=0.
- Excessive face/eye brute-force draft was rejected: ±2° dense rotation sweep made Maximum exceed 2 minutes on a small portrait. Face/eye search returned to the proven old fourth-mode deep budget so extra runtime is spent mainly on spatial/detail/Surface work.
- Benchmark, same archival reference, separate processes:
  - 0.5.9 old Ideal: 10.75 s
  - 0.5.10 Maximum final tuning: 20.57 s
  - ratio: ~1.91x wall time, with ~3.9x old local-map cell count and fuller Surface verification.
- Profile regression set after final tuning: 146/146 PASS.

## Status
- [x] Rename mode and preserve aliases.
- [x] Increase spatial map density.
- [x] Increase Maximum-only Surface/AI budgets without weakening thresholds.
- [x] Keep face/eye budget useful rather than pathological brute force.
- [x] Adjust regression tests for manual-only Surface semantics.
- [x] Benchmark old Ideal vs new Maximum (~1.91x).
- [x] Full test suite: 83/83 files, 710 tests PASS.
- [x] Candidate ZIP: 332/332 SHA PASS; unpacked smoke 247/247 PASS; path-with-spaces compileall PASS.
- [x] Pre-final rebuilt ZIP: 332/332 SHA PASS; 162/162 smoke PASS; path-with-spaces compileall PASS.
- [x] Final documentation-only rebuild prepared; exact archive bytes are verified after creation (see published SHA/build report).
