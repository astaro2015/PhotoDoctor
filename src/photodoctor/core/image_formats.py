from __future__ import annotations

HEIF_EXTENSIONS = frozenset({".heic", ".heif", ".hif"})
SUPPORTED_INPUT_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".jpe", ".jfif",
    ".png", ".webp",
    ".heic", ".heif", ".hif",
    ".tif", ".tiff", ".bmp", ".avif",
})
INPUT_FILE_DIALOG_FILTER = (
    "Изображения (*.jpg *.jpeg *.jpe *.jfif *.png *.webp "
    "*.heic *.heif *.hif *.tif *.tiff *.bmp *.avif)"
)
SUPPORTED_INPUT_LABEL = "JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP, AVIF"

try:
    from pillow_heif import register_heif_opener as _register_heif_opener
except ImportError:  # Source-only environments may not have runtime dependencies installed yet.
    _register_heif_opener = None

_heif_plugin_registered = False


def ensure_heif_plugin() -> bool:
    """Register pillow-heif as a Pillow opener exactly once.

    The project dependency installs pillow-heif for end users. Keeping the import
    lazy/fault-tolerant lets source-only checks still run in restricted build
    environments and gives a useful error when the dependency is missing.
    """
    global _heif_plugin_registered
    if _heif_plugin_registered:
        return True
    if _register_heif_opener is None:
        return False
    _register_heif_opener()
    _heif_plugin_registered = True
    return True
