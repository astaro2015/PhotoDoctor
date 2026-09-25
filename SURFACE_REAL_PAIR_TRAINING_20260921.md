# Photo Doctor — Surface defects real-pair study
Date: 2026-09-21
App target: 0.5.3.18

## Source material
User-provided pack: 10 images plus one separate `Grandmother.jpg`.
Five archive images contain damaged/restored side-by-side pairs suitable for weak supervision.
The original images are NOT embedded in the application or release archive.

Provenance for derived labels:
- source_tier: user_feedback / user_supplied_reference
- label_origin: weak_real_pair
- expert_verified: false
- training_lane: hard_mining / calibration_research
- license/use_scope: private-local-only
- may_train_expert_base: false

## Weak real-pair extraction
Pairs were split into damaged/restored halves and aligned with SIFT + RANSAC homography.
High-confidence defect labels require a morphology response present in the damaged half and strongly reduced/absent after alignment to the restored half.
Hard negatives are surface candidates detected on the restored/clean half.

Derived calibration set:
- 330 candidate rows
- 262 weak defect labels
- 68 weak natural-detail labels
- five independent source groups

Validation is GROUPED BY SOURCE IMAGE. Candidate patches from one source never appear in both train and validation folds.

## Finding: current Surface AI v1 is not calibrated for real old-photo defects
The embedded v1 model was trained primarily on procedural/synthetic candidates. On this weak-real set its scores do not transfer reliably across source photographs.
A real-pair calibration experiment was therefore NOT promoted to production: cross-source generalization was unstable (OOF AUC about 0.66 for a small calibration model).

Production rule for 0.5.3.18:
- v1 remains advisory/experimental;
- a v1 `defect` prediction cannot rescue a low-quality classical candidate;
- final displayed verification combines local context, geometry, texture risk and AI evidence;
- user D/N/U feedback remains the preferred real-data source for a future trainable v2.

## Classical detector comparison on supplied material
Candidate counts, old 0.5.3.17 -> context-aware 0.5.3.18:
- Grandmother.jpg: 156 -> 81 in direct detector comparison (about -48%)
- 14847150.jpg: 300 -> 45 (-85%)
- 2fdcc048eda46d4af418cc95469337b0.jpg: 45 -> 38 (-16%)
- 3319af230574d3523e52ea7894fc0fa3.jpg: 51 -> 19 (-63%)
- 34275618126.988754.jpg: 317 -> 151 (-52%)
- 804e9d312d99cae66a476e5e9e2784f9.jpg: 150 -> 69 (-54%)
- ec4de1192f0092ff9f6b19d50610fd86.jpg: 28 -> 8 (-71%)
- f5cf5ebb8fe81e24195451e73f6854ef.jpg: 84 -> 13 (-85%)
- fe5cdae09923a8c1557ff87f0501fb84.jpg: 123 -> 54 (-56%)
- i.webp: 104 -> 60 (-42%)
- timelapse-1.jpg: 59 -> 21 (-64%)

The purpose is not merely fewer boxes: the removed pool is dominated by text strokes, hair/fabric-like details, repeated parallel structures and low-context-contrast candidates.

## Grandmother.jpg precise-mode check
In the integrated 0.5.3.18 precise path:
- surface candidates: 78
- raw synthetic-AI proposals among evaluated regions: defect=11, natural=1, uncertain=2
- final context verification: confirmed defect=3, uncertain=11

Visual inspection of the three confirmed regions showed actual crack/tear structures. Many additional true cracks remain deliberately `uncertain`; precision is currently preferred over automatic over-confirmation.

## Direction for Surface AI v2
Do not simply retrain the current 96x96 binary classifier on a few duplicated photographs.
Next model should move toward pixel/region segmentation and use:
- real D/N/U user labels;
- real paired damaged/restored examples as weak supervision;
- strict source-group split / leakage prevention;
- hard negatives from clean/restored faces, hair, fabric, text and edges;
- per-source and per-defect-type metrics;
- promotion only after real-photo validation beats v1/context baseline without increasing false-defect rate.
