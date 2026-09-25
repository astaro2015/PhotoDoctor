from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import time
from typing import Callable, Iterable
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class VerifiedDownload:
    path: Path
    source_url: str
    sha256: str
    size: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str | Path, *, expected_sha256: str, expected_size: int | None = None) -> None:
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Файл не найден: {p}")
    if expected_size is not None and p.stat().st_size != int(expected_size):
        raise RuntimeError(
            f"Размер файла не совпадает: {p.stat().st_size} != {int(expected_size)}"
        )
    actual = sha256_file(p).lower()
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or actual != expected:
        raise RuntimeError(f"SHA-256 не совпадает: {actual} != {expected}")


def _stdlib_download(url: str, target: Path, *, timeout: float = 60.0) -> None:
    request = Request(url, headers={"User-Agent": "PhotoDoctor-ALMAZ/0.5.4"})
    with urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def download_verified(
    urls: Iterable[str],
    target: str | Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    attempts_per_url: int = 2,
    downloader: Callable[[str, Path], None] | None = None,
) -> VerifiedDownload:
    """Download from ordered mirrors and accept only the pinned artifact.

    The final *target* is replaced atomically only after exact size/SHA checks.
    Failed partial downloads are deleted, so a cancelled preparation can be
    safely retried without trusting a stale .part file.
    """
    target = Path(target).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    fetch = downloader or (lambda url, path: _stdlib_download(url, path))

    # Reuse an already-correct artifact without touching the network.
    if target.is_file():
        try:
            verify_file(target, expected_sha256=expected_sha256, expected_size=expected_size)
            return VerifiedDownload(target, "existing", expected_sha256.lower(), target.stat().st_size)
        except RuntimeError:
            target.unlink(missing_ok=True)

    errors: list[str] = []
    for raw_url in urls:
        url = str(raw_url).strip()
        if not url:
            continue
        for attempt in range(1, max(1, int(attempts_per_url)) + 1):
            partial.unlink(missing_ok=True)
            try:
                fetch(url, partial)
                verify_file(partial, expected_sha256=expected_sha256, expected_size=expected_size)
                partial.replace(target)
                return VerifiedDownload(target, url, expected_sha256.lower(), target.stat().st_size)
            except Exception as exc:
                partial.unlink(missing_ok=True)
                errors.append(f"{url} (попытка {attempt}): {exc}")
                if attempt < attempts_per_url:
                    time.sleep(0.4 * attempt)

    raise RuntimeError("Не удалось получить проверенный файл ни из одного источника:\n" + "\n".join(errors))
