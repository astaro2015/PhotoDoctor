# Photo Doctor 0.5.13 — flat Overview + camera metadata

Base: 0.5.12 UI_REDESIGN_PRIVALOV FINAL
Date: 2026-09-24

User-approved direction continued from the previous chat:
- make Overview flatter and denser instead of nesting QGroupBox rectangles;
- remove duplicate Overview buttons already available globally / as tabs;
- remove the ambiguous "Потенциал улучшения" score from the main Overview;
- show useful photo/camera information directly on Overview;
- prevent layout collisions on Windows scaling by relying on content-driven heights and a vertical scroll area;
- keep existing analysis/correction math and Surface manual-only policy unchanged.

Implemented:
1. Overview now uses a flat QScrollArea with plain section headings and separators.
2. Four framed score cards replaced by one compact line: Quality / Analysis confidence / Problems found.
3. "Потенциал улучшения" removed from Overview; underlying summary/diagnostics remain untouched.
4. "О фотографии" is no longer a QGroupBox. Added:
   - pixel dimensions;
   - megapixels;
   - source format;
   - ICC -> sRGB / sRGB indication;
   - file size;
   - camera make/model;
   - lens when present;
   - aperture;
   - shutter speed;
   - ISO;
   - focal length;
   - EXIF original date when present.
5. "Резюме исправлений" QGroupBox removed; flat "Исправления" section retained.
6. Duplicate Overview buttons "Открыть исправления" and "Предпросмотр исправлений" removed. The global preview button and the Corrections tab remain the canonical controls.
7. Long analysis details remain collapsed behind one button, but the details container is now an unframed QWidget instead of another QGroupBox.
8. Existing vertical main QSplitter remains unchanged and continues to let the user resize photo vs. lower panel.
9. Overview itself scrolls only when content does not fit, which protects 125–150% Windows scaling from overlap/clipping.
10. App version bumped to 0.5.13. ALGORITHM_VERSION intentionally remains 0.5.10-surface-v9-maximum because analysis math did not change.
