from __future__ import annotations

import importlib
import sys

REQUIRED = (
    "photodoctor",
    "cv2",
    "PIL",
    "numpy",
    "PySide6",
    "pillow_heif",
    "onnxruntime",
)


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print(f"ERROR Python 3.12 required, got {sys.version.split()[0]}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if failures:
        print("ERROR runtime import check failed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 3

    import photodoctor

    print(f"OK Photo Doctor {photodoctor.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
