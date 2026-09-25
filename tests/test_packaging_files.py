from pathlib import Path

from photodoctor import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_version_bumped():
    assert __version__ == "0.5.16"


def test_windows_entrypoints_exist():
    for name in ["START_PhotoDoctor.bat", "INSTALL_AND_RUN.bat", "RUN_PhotoDoctor.bat", "BUILD_EXE.bat"]:
        p = ROOT / name
        assert p.is_file()
        assert p.stat().st_size > 40


def test_powershell_install_is_per_user_and_signed():
    text = (ROOT / "scripts" / "install_and_run.ps1").read_text(encoding="utf-8")
    assert "InstallAllUsers=0" in text
    assert "Get-AuthenticodeSignature" in text
    assert "Python Software Foundation" in text
    assert "Start-Process -FilePath $VenvPythonw" in text


def test_build_script_uses_spec_and_captures_real_pyinstaller_output():
    text = (ROOT / "scripts" / "build_exe.ps1").read_text(encoding="utf-8")
    assert "PhotoDoctor.spec" in text
    assert "@('-m', 'PyInstaller', '--noconfirm', '--clean', $Spec)" in text
    assert "2>&1" in text
    assert "Write-CommandOutput" in text
    assert "PhotoDoctor.exe" in text


def test_build_script_does_not_treat_pyinstaller_stderr_info_as_failure():
    text = (ROOT / "scripts" / "build_exe.ps1").read_text(encoding="ascii")
    assert "$PreviousErrorActionPreference = $ErrorActionPreference" in text
    assert "$ErrorActionPreference = 'Continue'" in text
    assert "$ExitCode = $LASTEXITCODE" in text
    assert "ForEach-Object" in text
    assert 'if ($ExitCode -ne 0)' in text


def test_pyinstaller_spec_is_onefile_windowed_and_collects_cv2_and_models():
    text = (ROOT / "PhotoDoctor.spec").read_text(encoding="utf-8")
    assert 'collect_all("cv2")' in text
    assert 'collect_all("pillow_heif")' in text
    assert '"photodoctor/ai/models"' in text
    assert 'name="PhotoDoctor"' in text
    assert 'console=False' in text
    assert 'a.binaries' in text
    assert 'a.datas' in text



def test_build_dependencies_pin_pyinstaller_for_reproducibility():
    text = (ROOT / "requirements-build.txt").read_text(encoding="ascii")
    assert "pyinstaller==6.21.0" in text


def test_build_log_writer_does_not_use_powershell_tee_object_default_encoding():
    text = (ROOT / "scripts" / "build_exe.ps1").read_text(encoding="ascii")
    assert "UTF8Encoding($false)" in text
    assert "Tee-Object" not in text
    assert "AppendAllText" in text

def test_run_repairs_broken_environment():
    text = (ROOT / "RUN_PhotoDoctor.bat").read_text(encoding="utf-8")
    assert 'scripts\\verify_env.py' in text
    assert 'goto INSTALL' in text
    assert 'INSTALL_AND_RUN.bat' in text


def test_powershell_scripts_are_ascii_for_windows_powershell_51():
    for name in ["install_and_run.ps1", "build_exe.ps1"]:
        data = (ROOT / "scripts" / name).read_bytes()
        data.decode("ascii")


def test_install_script_avoids_fragile_inline_python_and_extras_path():
    text = (ROOT / "scripts" / "install_and_run.ps1").read_text(encoding="ascii")
    assert ' -c ' not in text
    assert '$Root[gui]' not in text
    assert 'requirements-user.txt' in text
    assert 'verify_env.py' in text


def test_user_requirements_install_project_and_gui():
    text = (ROOT / "requirements-user.txt").read_text(encoding="ascii")
    assert '-e .[ai]' in text
    assert 'PySide6' in text


def test_input_format_dependencies_are_declared():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"Pillow>=11.3,<13"' in pyproject
    assert '"pillow-heif>=1.7,<2"' in pyproject


def test_verify_env_has_no_shell_quoting_dependency():
    text = (ROOT / "scripts" / "verify_env.py").read_text(encoding="utf-8")
    for name in ["photodoctor", "cv2", "PIL", "numpy", "PySide6", "pillow_heif"]:
        assert f'"{name}"' in text


def test_installer_recreates_invalid_virtual_environment():
    text = (ROOT / "scripts" / "install_and_run.ps1").read_text(encoding="ascii")
    assert 'Existing virtual environment is invalid. Recreating it.' in text
    assert 'Remove-Item -LiteralPath $VenvDir -Recurse -Force' in text
    assert 'Test-Python312 $VenvPython' in text


def test_brightness_recalibration_invalidates_old_photo_analysis_cache():
    from photodoctor.core.versioning import ALGORITHM_VERSION, NORMALIZATION_VERSION
    assert __version__ == "0.5.16"
    assert ALGORITHM_VERSION == "0.5.10-surface-v9-maximum"
    assert NORMALIZATION_VERSION == "0.2.0"


def test_native_surface_v1_weights_are_packaged_by_spec():
    weights = ROOT / "src" / "photodoctor" / "ai" / "models" / "native_surface_refiner_v1.npz"
    assert weights.is_file()
    assert weights.stat().st_size > 1_000_000
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"photodoctor.ai" = ["models/*.npz", "models/*.txt"]' in pyproject


def test_surface_context_meta_v2_weights_are_packaged():
    weights = ROOT / "src" / "photodoctor" / "ai" / "models" / "native_surface_context_meta_v2.npz"
    assert weights.is_file()
    assert weights.stat().st_size > 20_000
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"photodoctor.ai" = ["models/*.npz", "models/*.txt"]' in pyproject
    spec = (ROOT / "PhotoDoctor.spec").read_text(encoding="utf-8")
    assert 'MODEL_DIR = SRC / "photodoctor" / "ai" / "models"' in spec
    spec = (ROOT / "PhotoDoctor.spec").read_text(encoding="utf-8")
    assert 'MODEL_DIR = SRC / "photodoctor" / "ai" / "models"' in spec
    assert 'path.suffix.lower() in {".npz", ".txt"}' in spec


def test_pyproject_version_matches_package_version():
    import re
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == __version__


def test_series_algorithm_has_own_version():
    from photodoctor.core.versioning import SERIES_ALGORITHM_VERSION
    assert SERIES_ALGORITHM_VERSION == "0.4.0"


def test_windows_metadata_author_and_icon_are_packaged():
    import photodoctor
    assert photodoctor.__author__ == "Привалов Олег"
    assert (ROOT / "AUTHORS.txt").is_file()
    icon = ROOT / "src" / "photodoctor" / "assets" / "photodoctor.ico"
    png = ROOT / "src" / "photodoctor" / "assets" / "photodoctor.png"
    assert icon.is_file() and icon.stat().st_size > 10_000
    assert png.is_file() and png.stat().st_size > 10_000
    spec = (ROOT / "PhotoDoctor.spec").read_text(encoding="utf-8")
    assert 'icon=str(ASSET_DIR / "photodoctor.ico")' in spec
    assert 'version=str(VERSION_INFO)' in spec
    version_info = (ROOT / "version_info.txt").read_text(encoding="utf-8")
    assert "Привалов Олег" in version_info
    assert "0.5.16" in version_info


def test_almaz_onnx_runtime_is_installed_and_packaged():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ai = ["onnxruntime>=1.20,<2"]' in pyproject
    req = (ROOT / "requirements-user.txt").read_text(encoding="ascii")
    assert '-e .[ai]' in req
    verify = (ROOT / "scripts" / "verify_env.py").read_text(encoding="utf-8")
    assert '"onnxruntime"' in verify
    spec = (ROOT / "PhotoDoctor.spec").read_text(encoding="utf-8")
    assert 'collect_all("onnxruntime")' in spec
    assert 'ort_binaries' in spec and 'ort_hiddenimports' in spec
