from __future__ import annotations

import numpy as np
from PIL import Image

from photodoctor.cli import main


def test_cli_analyze_json(tmp_path, capsys):
    p = tmp_path / "a.png"
    Image.fromarray(np.full((32, 32, 3), 120, np.uint8), "RGB").save(p)
    code = main(["analyze", str(p), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert 'метрики' in out or 'metrics' in out
    assert 'Яркость' in out or 'brightness' in out


def test_cli_batch(tmp_path, capsys):
    p = tmp_path / "a.png"
    Image.fromarray(np.full((32, 32, 3), 120, np.uint8), "RGB").save(p)
    code = main(["batch", str(tmp_path), "--db", str(tmp_path / "db.sqlite3")])
    out = capsys.readouterr().out
    assert code == 0
    assert 'Успешно: 1' in out


def test_cli_accepts_legacy_ideal_precision_alias(tmp_path, capsys):
    p = tmp_path / "ideal.png"
    Image.fromarray(np.full((40, 48, 3), 128, np.uint8), "RGB").save(p)
    code = main(["analyze", str(p), "--точность", "идеально", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert out


def test_cli_accepts_maximum_precision(tmp_path, capsys):
    p = tmp_path / "maximum.png"
    from PIL import Image
    Image.new("RGB", (64, 64), (120, 120, 120)).save(p)
    code = main(["analyze", str(p), "--точность", "максимально", "--json"])
    assert code == 0
