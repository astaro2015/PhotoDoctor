# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve()
SRC = ROOT / "src"
MAIN = SRC / "photodoctor" / "__main__.py"
MODEL_DIR = SRC / "photodoctor" / "ai" / "models"
ASSET_DIR = SRC / "photodoctor" / "assets"
VERSION_INFO = ROOT / "version_info.txt"

cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all("cv2")
heif_datas, heif_binaries, heif_hiddenimports = collect_all("pillow_heif")
ort_datas, ort_binaries, ort_hiddenimports = collect_all("onnxruntime")
model_datas = [
    (str(path), "photodoctor/ai/models")
    for path in sorted(MODEL_DIR.iterdir())
    if path.is_file() and path.suffix.lower() in {".npz", ".txt"}
]
asset_datas = [
    (str(path), "photodoctor/assets")
    for path in sorted(ASSET_DIR.iterdir())
    if path.is_file() and path.suffix.lower() in {".png", ".ico"}
]

a = Analysis(
    [str(MAIN)],
    pathex=[str(SRC)],
    binaries=cv2_binaries + heif_binaries + ort_binaries,
    datas=cv2_datas + heif_datas + ort_datas + model_datas + asset_datas,
    hiddenimports=cv2_hiddenimports + heif_hiddenimports + ort_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PhotoDoctor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ASSET_DIR / "photodoctor.ico"),
    version=str(VERSION_INFO),
)
