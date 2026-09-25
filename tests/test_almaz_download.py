from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
import sys
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from almaz_download import download_verified, verify_file


def test_download_verified_falls_back_and_commits_only_verified_file(tmp_path):
    payload = b"verified-almaz-payload"
    sha = hashlib.sha256(payload).hexdigest()
    calls: list[str] = []
    def fetch(url: str, target: Path) -> None:
        calls.append(url)
        if "bad" in url:
            raise OSError("network fail")
        target.write_bytes(payload)
    out = tmp_path / "model.pth"
    result = download_verified(
        ("https://bad.invalid/model", "https://mirror.invalid/model"), out,
        expected_sha256=sha, expected_size=len(payload), attempts_per_url=1, downloader=fetch,
    )
    assert out.read_bytes() == payload
    assert result.source_url.endswith("mirror.invalid/model")
    assert calls == ["https://bad.invalid/model", "https://mirror.invalid/model"]
    assert not (tmp_path / "model.pth.part").exists()


def test_download_verified_rejects_wrong_hash_and_removes_partial(tmp_path):
    def fetch(_url: str, target: Path) -> None:
        target.write_bytes(b"wrong")
    with pytest.raises(RuntimeError):
        download_verified(
            ("https://one.invalid/model",), tmp_path / "model.pth",
            expected_sha256="0" * 64, expected_size=5, attempts_per_url=1, downloader=fetch,
        )
    assert not (tmp_path / "model.pth").exists()
    assert not (tmp_path / "model.pth.part").exists()


def test_verify_file_checks_size_before_hash(tmp_path):
    p = tmp_path / "x.bin"; p.write_bytes(b"123")
    with pytest.raises(RuntimeError, match="Размер"):
        verify_file(p, expected_sha256=hashlib.sha256(b"123").hexdigest(), expected_size=4)
