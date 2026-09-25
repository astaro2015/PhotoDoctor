# Photo Doctor 0.5.9 — Result summary / Surface manual-only / Ideal mode

Started: 2026-09-24
Base: PhotoDoctor_0.5.8_PERFORMANCE_STABILITY_FINAL_2026-09-24.zip
Base app version: 0.5.8
Base algorithm: 0.5.8-surface-v8

## User requirements
1. First tab (Результат): remove correction-selection checkboxes; keep read-only summary only.
2. Improve scratch/crack detection, search more thoroughly/completely, but Surface repair must ALWAYS require explicit user approval.
3. Add 4th precision mode "Идеальный", computationally ~2–3x more thorough than "Точно" without weakening thresholds or reducing quality.

## Plan
- [x] Make Result tab read-only summary; preserve selection backend and Plan tab controls.
- [x] Surface candidates manual-only at correction level and per-candidate level; no auto-seeding.
- [x] Improve Surface recall without auto-application.
- [x] Add Ideal precision profile and propagate through face/eye/AI/validator/analyzer/CLI/UI.
- [x] Add regression tests.
- [x] Profile Ideal vs Precise; verify expected ~2–3x analysis effort, not necessarily wall-time.
- [x] Full test suite.
- [x] compileall, clean zip, SHA, unpack into path with spaces, smoke tests.

## Work log
- Base extracted and relevant UI/precision/Surface paths located.
0ff8cb9b174a75e6d30d0e386d825e2235c0e02f556c2d4ccdaa0fbcde552c3c  /mnt/data/photodoctor_059_work/PhotoDoctor_0.5.8_PERFORMANCE_STABILITY_FINAL_2026-09-24.zip

## 2026-09-24 — continuation checkpoint
- Result tab converted to read-only summary: visible `fix_summary_list` has no checkboxes; hidden canonical selection list remains only as backend for Plan/preview compatibility.
- Surface v9 made hard manual-only at two layers: Validator returns `accepted=False`, `auto_eligible=False`; GUI Defects tab seeds treatment only from persisted human `user_label=defect` and never from AI verdict.
- Added Ideal precision profile (6144px deep-analysis budget, denser local maps, deeper face/eye passes, larger Surface/AI budgets).
- Surface v9 search expanded. Important regression found: sensitive extra Hough lines increased parallel-neighbor penalty for legacy V8 candidates. Fixed by tagging Hough provenance (`legacy_v8` / `sensitive_v9`) and excluding sensitive-only lines from repetition penalties applied to legacy/primary candidates. Verified old curved-scratch top candidate restored exactly (quality 0.6164, same geometry).
- Normal Surface AI verification budget raised to 64, Precise to up to 96, Ideal up to 128. Archive regression now has `unprocessed_count=0` while Surface remains manual-only.
- Tests updated to assert the new semantics rather than old auto-selection behavior.

## 2026-09-24 — manual-only hardening + Ideal budget checkpoint
- Added hard API guard: `surface_candidate_auto_repair_eligible()` now always returns False.
- Preserved a separate `_surface_candidate_repairable_for_manual_preview()` technical gate so explicit human selection can still preview a real crack, including a high-confidence line crossing an eye/face area.
- Removed unused GUI import of the old auto-repair helper.
- Ideal Surface budgets increased to 128 visible / 288 deep candidates; native Surface verifier budget to 192 candidates (Precise remains 64 / 144 / 96 respectively).
- Ideal Hough retained-line budget raised to 144 (Precise 72).
- Added regression proving 4K local-map budget is 2.0–3.0x Precise; current ratio ~2.11x and deep pixel budget = 2.25x.
- Version metadata partially advanced to 0.5.9; fixed Windows FixedFileInfo tuple to 0.5.9 and restored `__author__ = "Привалов Олег"`.
- Surface method diagnostic renamed to `context_aware_morphology_v9_deep_recall_manual_only`.
- Targeted suite after hardening: 172/172 PASS.

## 2026-09-24 — deeper Surface + additive Ideal checkpoint
- Increased V9 recall budgets while keeping text/fabric hard negatives. Normal now uses more morphology scales and a more sensitive low-contrast/broad-emulsion search; Precise and Ideal deepen further.
- A new text-like false positive surfaced after lowering recall thresholds. Added a precision-independent hard-negative guard for tiny primary line fragments with extreme edge texture + weak local ridge contrast. Surface suite returned to green.
- Added `merge_surface_detections()` and changed Ideal Surface to be additive: run Precise search + Ideal search, then evidence-rank and IoU-deduplicate. Ideal no longer loses a distinct candidate merely because its own component/Hough geometry changed.
- Real archival reference after changes:
  - Normal: 62 raw / 48 refined, 0 unprocessed, Surface accepted=False/auto_eligible=False.
  - Precise: 61 raw / 61 refined, 0 unprocessed, Surface accepted=False/auto_eligible=False.
  - Ideal: 76 raw / 76 refined, 0 unprocessed, Surface accepted=False/auto_eligible=False.
  - Measured wall time on this small reference: Precise 5.06 s, Ideal 10.26 s => 2.03x.
- Added regressions for faint interrupted bright scratch on gradient and additive Ideal merge preserving a Precise-only candidate.
- Renamed first-tab group to `Резюме исправлений`.
- Targeted Surface/Precision/Validator/GUI suite after these changes: 130/130 PASS.

## 2026-09-24 — final source validation checkpoint
- Added GUI regression ensuring Plan -> "Выбрать всё доступное" cannot arm Surface unless the human has selected at least one Surface candidate.
- One full-suite source-string failure was found and fixed: restored explicit wording that the "Лечить" column controls local repair/map while the checkbox is human-only. Behaviour was already correct.
- Final resumable per-file source suite completed: 83/83 test files, 709 tests PASS, 0 failures.
- Code freeze after this checkpoint; only release documentation/manifests/packaging may change.


## 2026-09-24 — unpacked candidate verification checkpoint
- Candidate archive unpacked to `/mnt/data/Photo Doctor 0.5.9 unpacked/PhotoDoctor_ALMAZ` (path contains spaces).
- Internal source manifest: 327/327 PASS; no cache directories or bytecode were present in the archive.
- Imported versions from unpacked copy: app 0.5.9, author Привалов Олег, algorithm 0.5.9-surface-v9, Validator 0.2.2.
- Precision registry confirmed in unpacked copy: Быстро / Нормально / Точно / Идеальный.
- Expanded smoke from the unpacked candidate: 278/278 PASS in 24.65 s. Scope included precision, Surface V9, Context Meta, feedback/history/refinement, Validator, GUI read-only/manual-only sources, packaging, CLI, preview cache, DB batch/cache, faces/eyes and startup/preview order.
- `python -m compileall -q src tests scripts training tools` from unpacked path-with-spaces copy: PASS.
- Code remains frozen; only reports/manifest/final packaging change after this checkpoint.
