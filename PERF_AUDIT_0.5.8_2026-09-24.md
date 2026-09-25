# Photo Doctor 0.5.8 Surface V8 — performance/stability audit

Date: 2026-09-24
Goal: audit current 0.5.7 from multiple angles, improve speed only where output/quality is preserved, keep recovery log.

## Baseline
- Source archive SHA-256: `5d3ecb70db1389cfb7b33c2c187465c66fa086c9f964301537f9ec145b0992ae`
- Source: PhotoDoctor_0.5.7_ALMAZ_SURFACE_V8_CRACK_TRACK_FIX_2026-09-23.zip
- No algorithm/quality changes are allowed unless explicitly justified by output-equivalence tests.

## Work log
1. Unpacked release into isolated audit worktree.
2. Began source audit: analyzer, service, surface, caches, preview and batch paths.

3. Baseline on `Grandmother(1).jpg` after hardware scheduler benchmark:
   - selected profile: serial OpenCV 5 threads, pipeline 1 worker, Validator 4 workers × OpenCV 1 thread.
   - scheduler calibration: 3.62 s.
   - normal cold-ish: 5.158 s.
   - normal warm: 5.004 s.
   - precise warm: 6.801 s.

4. Rejected optimization: raising native Surface AI batch cap above 16. Large NumPy UNet batches caused severe memory pressure/timeouts in the constrained test environment. Reverted; current cap remains 16.

5. Implemented and retained: batched Context Meta forest evaluation across Surface candidates. Scalar vs batch predictions are bit-identical in regression test; isolated 48-candidate Context Meta runtime improved ~1.74x in this environment.
6. Rejected optimization: im2col/matmul replacement for native Surface UNet convolution. It was 2–4x slower and introduced ~5e-6 numeric drift. Original einsum path retained.

7. Rejected optimization: parallel Haar cascade families. Coordinates/confidence stayed identical, but runtime worsened from ~1.52 s (serial families, OpenCV 5 threads) to ~2.32–2.67 s. Reverted.

8. Retained Surface allocation optimization: reuse existing full-frame gray float copy, one LAB float copy, and one low-contrast eligibility mask across bright/dark Hough branches. Detector output hashes were exactly identical on both small and 4K reference images; 4K Surface isolated runtime improved ~2.74 s -> ~2.33 s in one controlled run (small-image timing was noise-bound).
9. Rejected optimization: reuse upstream PCA/linearity inside `_component_context_features`. Candidate sets stayed the same but stored floating geometry drifted ~1e-8..1e-7 because the context crop's translated float32 coordinates reproduce slightly different rounding. Reverted to preserve exact candidate payloads.

10. Retained local-contrast allocation simplification after broad equivalence audit.
    - Removed redundant RGB→gray→float32/255→*255→uint8 round-trips; the module now works directly on the uint8 gray image it actually consumes.
    - Compared OLD vs NEW on: archive reference, 4K upscaled reference, deterministic 4K gradient/object image, deterministic 4K noise/object image, deterministic 4K heavy texture image, and a tiny textured image; both Normal and Precise modes.
    - Exact decision equivalence in every case: classification, informative/total/map counts, map geometry, every cell status, local-correction candidate indices, and candidate strengths.
    - Scalar summary fields were exact. Maximum observed auxiliary float drift was 2.28e-7 in confidence / 8.72e-8 in structure_persistence on one small run; no threshold or decision changed. Most 4K cases were numerically exact.
    - Representative 4K Normal isolated speedups: ~1.35x gradient, ~1.47x noise, ~1.51x texture, ~1.37x archive reference. Precise gains are smaller/noise-bound because window processing dominates.
    - Comparison artifact: /tmp/local_contrast_compare.json

11. Retained Validator Surface-mask reuse optimization after byte-exact image audit.
    - `_linked_surface_repair_mask`: one float grayscale frame is now reused by crack-track linking and adaptive ridge-width growth instead of converting RGB to gray twice.
    - `_bounded_surface_heal` accepts an optional already-built repair mask; post-AI validation reuses the exact candidate mask for 90% and optional 100% trial instead of rebuilding it.
    - The optional 100% retry reuses the already-measured before-contrast instead of recomputing the original residual.
    - OLD vs NEW compared with the same 26 confirmed Surface candidates on the archive image and its 3840×2344 4K copy.
    - Repair masks were byte-identical, healed RGB outputs at 45/90/100% were byte-identical (max diff 0), and effectiveness triplets were exactly equal.
    - 4K `_linked_surface_repair_mask`: ~1.36s -> ~1.01s (~1.34x).
    - 4K heal using reused mask: ~2.9–4.8x faster depending on strength in isolated measurements.
    - Simulated post-AI Surface validation flow: ~1.64s -> ~1.13s (~1.45x) with identical result/metrics.
    - Regression subset: 46/46 PASS (`surface_history`, `validator`, `surface`).

12. Retained conditional full-frame grayscale cache for native Surface AI feature preparation.
    - Per-crop RGB→GRAY and slicing a precomputed full-frame GRAY were byte-identical for every tested feature/prior tensor and produced exactly equal prediction dictionaries.
    - 43-candidate 4K feature-preparation median improved ~1.115s -> ~1.023s (~1.09x); small-image gain was ~2% and whole-model runtime remains dominated by NumPy UNet forward.
    - To avoid wasting a full-frame conversion on simple images, the cache is used only for >=8 candidates; 1–7 candidates retain the old crop-local path.
    - Regression subset: 27/27 PASS.

13. Retained persistent GUI analysis-cache reuse for unchanged files.
    - Added `AnalysisDatabase.load_current_result(ImageInfo, precision)`, which first requires the existing size+mtime+quick-hash+algorithm+normalization cache fingerprint to be current, then reconstructs `AnalysisResult` from persisted metrics.
    - `PhotoAnalysisWorker` now loads that persistent result after decoding the image/manual markup and returns immediately instead of rerunning analysis. Any source change, algorithm/precision mismatch, or manual face/eye edit fails closed and triggers the normal full analysis.
    - Fresh vs cached metrics are JSON-semantically identical. Only informational model-spec tuples (`input_size`, `labels`) naturally round-trip through SQLite JSON as lists; no decision/correction payload changes.
    - 4K cached-open benchmark (JPEG decode + fingerprint + SQLite metric load): median ~0.073s; JPEG decode ~0.064s, DB/fingerprint ~0.008s. Existing full 4K analysis is ~21s in this environment.
    - `is_current` fingerprint check: ~0.00019s median with OS cache in the 100-call benchmark.
    - Database/GUI cache regression subset: 98/98 PASS, including file-change and manual-override invalidation.

14. Retained Validator summary-only local-contrast probe.
    - Validator requested only `local_contrast_score` but previously ran the full local-contrast analyzer, including the detailed localization map that was immediately discarded.
    - Added `local_contrast_summary_score`, using the exact same fixed summary windows, per-window measurements, unique-area aggregation and score math while skipping only the unused fine map.
    - Exact score equality was verified on structured high/low contrast, flat and deterministic noisy inputs; regression subset 29/29 PASS.
    - Validator-size 1024px isolated median: ~0.082s -> ~0.032s (~2.57x).
    - 4K isolated scalar-only use: ~0.400s -> ~0.190s (~2.11x).

15. Full-result equivalence check against the saved pre-audit 0.5.7 baselines passed.
    - Re-ran the same archival reference in Normal and Precise after accepted optimizations.
    - Normal: decision plan, recommendation validation, Surface refinement, faces, eyes, and local-contrast raw payloads were exactly equal to baseline (apart from the deliberately placeholder-vs-real image path in the saved fixture).
    - Precise: all decisions/candidates/strengths remained equal. Only 23 auxiliary local-contrast confidence/structure floats drifted by <=1.72e-7; four copied Validator `source_confidence` values inherited <=1.52e-7 drift. No status, threshold, selected region, action, or correction parameter changed.
    - Current small-reference timings in this run: Normal ~5.32s, Precise ~6.77s; small-image timing is dominated by face/AI stages and remains effectively baseline-level. The accepted speedups target large images and repeated/cached workflows.

16. Startup calibration audit: no change needed.
    - CPU/Validator benchmarking is already persisted in QSettings and reused when performance-profile version + hardware signature match.
    - Full calibration runs only on first use/profile-version/hardware change; subsequent starts call `apply_performance_profile` directly.
    - Rejected any startup changes: current behavior is already the safe fast path.

17. Clean-original vs optimized multi-scene audit passed.
    - Compared untouched 0.5.7 source against the audit worktree on four extra Normal-mode scenes: low-contrast gradient/lines, deterministic noise, heavy fabric-like texture, and small 320x240 input.
    - At 1e-6 float tolerance there were 0 meaningful JSON differences in all four cases. Decision Plan, recommendation validation, Surface refinement/candidates, faces and eyes were exact.
    - Baseline -> optimized wall times: gradient 5.513 -> 5.505 s; noise 8.749 -> 8.593 s; small 0.594 -> 0.526 s; texture 6.861 -> 6.618 s.

18. Persistent analysis cache made external-model-state aware.
    - Risk found during audit: source/algorithm/manual-markup fingerprints were sufficient for classical analysis but could return stale AI/ALMAZ decisions after a model install/removal/replacement.
    - `_cache_algorithm_version` now also carries a cheap model-directory inventory fingerprint (path + regular-file name/size/mtime_ns). Large model payloads are NOT re-hashed on every file open.
    - Any model/manifest file install/removal/change invalidates the cached analysis and fails closed to a full analysis; after saving under the new state, normal cache reuse resumes.
    - Database/GUI regression subset including the new model-state invalidation case: 99/99 PASS.

19. Batch/tab workflow and memory audit passed.
    - Synthetic 100-file first Fast pass: 100/100 succeeded in ~11.88 s; immediate second batch reused 100/100 cached entries in ~0.008 s at batch-scan level.
    - RSS: ~114.1 MiB start -> ~148.0 MiB after first 100; cached second pass stayed ~148.0 MiB.
    - Four further independent 50-file batches: RSS 147.0 -> 147.2 -> 147.2 -> 147.2 MiB after GC. No monotonic per-file leak observed; the initial ~35 MiB is library/allocator warm-up.
    - GUI source audit confirms only the current full-resolution RGB plus one corrected preview are retained. Normal/Precise session cache retains metrics/results, not another full-resolution source image per mode.

20. Retained two-phase cache freshness check.
    - `AnalysisDatabase.is_current` now queries path/size/mtime/algorithm/model-state first and computes the 128 KiB quick hash only if a DB row could actually be current.
    - New/unseen or obvious-stat-changed files therefore avoid pointless content reads; same-size+same-mtime replacements still require and pass through quick-hash verification.
    - Added regression that replaces contents while restoring identical size+mtime; cache correctly fails closed.
    - Database/batch regression: 19/19 PASS.

21. Retained quick-hash reuse on persistent GUI cache hits.
    - `PhotoAnalysisWorker` computes the source quick hash once before loading manual vision overrides + persistent analysis cache and passes the same verified fingerprint to both lookups.
    - Cache-hit payload carries that fingerprint into `_analysis_completed`, eliminating additional duplicate 128 KiB reads there.
    - Full fresh-analysis safety path remains conservative; this optimization targets unchanged cached opens only.
    - Added regression that spies on `AnalysisDatabase.quick_hash`: manual override + current-result lookup with a supplied fingerprint performs 0 additional hash reads.
    - GUI/database/profile subset: 131/131 PASS.

22. Retained deferred pre-AI Surface preview only in the main `analyze_file` pipeline.
    - Audit found that classical `validate_recommendations` rendered a bounded Surface heal for the first <=8 classical candidates before Surface AI, then `refresh_surface_validation_after_refinement` always rebuilt/revalidated the final executable Surface set after AI. AIRouter explicitly excludes `recommendation_validation`, so the expensive preliminary heal does not affect Surface AI routing.
    - Added opt-in `defer_surface_refinement=True`; standalone Validator keeps its historical full probe by default. `analyze_file` opts into deferral because it always performs the post-AI synchronization later in the same call.
    - Full-result deep diff on 4K Normal and archive Precise showed no changed decision/status/candidate/strength. Differences were only live available-RAM metadata, JSON tuple/list representation in the saved baseline, and pre-existing local-map float noise <=~1.72e-7.
    - Isolated 4K initial Validator median: ~0.591 s -> ~0.519 s (~1.14x). The whole-pipeline gain is smaller because validators run concurrently.
    - Surface/Validator regression subset: 78/78 PASS.

23. Strengthened external-model cache invalidation without full model re-hashing.
    - Metadata-only inventory could miss a manual in-place model replacement if file name, size and mtime were deliberately preserved.
    - Added `photodoctor.ai.model_fingerprint.sampled_file_digest`: small files are hashed completely; large files sample beginning + middle + end (16 KiB each) with BLAKE2b.
    - Analysis cache model-state signature now includes this sampled content token in addition to path/name/size/mtime.
    - SR/restoration runtime status caches and backend cache keys use the same sampled token. A same-stat manual replacement therefore forces fresh full SHA/manifest validation before execution instead of reusing stale ready/backend state.
    - Same-size/same-mtime replacement regression now invalidates correctly.
    - Cost check: sampling five sparse 512 MiB model files took ~0.57 ms median total on the current filesystem; model payloads are not read in full.
    - Database + SR/restoration runtime subset: 33/33 PASS.

24. Release target and final source verification.
    - Release version bumped to Photo Doctor 0.5.8 / ALGORITHM_VERSION 0.5.8-surface-v8. Surface V8 math/thresholds/candidate limits were not reduced; cache key/version changed for the performance/stability release.
    - Direct neighbouring-process 4K Normal check: clean 0.5.7 ~19.64 s vs optimized ~16.79 s (~14.5% faster). Earlier same-session profiles showed the same direction with roughly 14–20% whole-analysis improvement depending on system load.
    - Final full source test pass after ALL code changes (including sampled model fingerprints): 83/83 test files, 702 tests PASS, 0 failures.

24. Final 0.5.8 source regression pass completed.
    - Final pass was restarted after the last model-cache hardening change rather than reusing the earlier full run.
    - 83/83 test files PASS, 702 tests PASS, 0 failures.
    - The per-file runner is resumable via FULL_TEST_0.5.8_FINAL.log; duplicate entries caused by one externally-killed wrapper were normalized before the final count.
    - No further algorithm/performance changes are permitted after this point; remaining work is compile/package/unpacked verification only.

25. Final release-candidate verification from an unpacked path with spaces.
    - Internal manifest: 321/321 SHA-256 PASS.
    - Imported versions: Photo Doctor 0.5.8 / 0.5.8-surface-v8 / Validator 0.2.2.
    - Targeted unpacked smoke covering database/model cache, SR/restoration runtime, Local Contrast, Surface/Context/history/refinement, Validator, packaging, preview-cache and GUI source/startup: 235/235 PASS.
    - compileall from unpacked candidate: PASS.
    - Candidate code is frozen; final repack changes verification text/manifest only.

25. Clean-archive candidate verification completed.
    - Internal SHA manifest: 322/322 PASS after unpacking into a path containing spaces.
    - Archive junk check: 0 __pycache__, 0 .pytest_cache, 0 .pyc/.pyo.
    - Imported from unpacked copy: Photo Doctor 0.5.8, ALGORITHM_VERSION 0.5.8-surface-v8, VALIDATOR_VERSION 0.2.2.
    - Targeted unpacked smoke across DB/cache, AI/SR/restoration runtime, GUI startup, local contrast, precision, preview cache, Surface/Context/History/Refinement, Validator and packaging: 165/165 PASS.
    - compileall from unpacked path with spaces: PASS.
    - Code frozen after this point; only documentation and SHA manifest are repacked for the final artifact.
