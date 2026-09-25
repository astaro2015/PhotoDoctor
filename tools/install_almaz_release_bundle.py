from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photodoctor.ai.almaz_release_bundle import install_release_bundle


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Install one verified ALMAZ release-bundle ZIP")
    p.add_argument("bundle")
    p.add_argument("--model-dir", default="")
    args = p.parse_args(argv)
    print("[ALMAZ RELEASE] проверяю структуру ZIP и SHA…", flush=True)
    result = install_release_bundle(args.bundle, model_dir=(args.model_dir or None))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
