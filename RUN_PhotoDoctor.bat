@echo off
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PYW=.venv\Scripts\pythonw.exe"
set "VERIFY=%~dp0scripts\verify_env.py"

if not exist "%PY%" goto INSTALL
if not exist "%PYW%" goto INSTALL
if not exist "%VERIFY%" goto INSTALL

"%PY%" "%VERIFY%" >nul 2>&1
if errorlevel 1 goto INSTALL

start "Photo Doctor" /D "%~dp0" "%PYW%" -m photodoctor
exit /b 0

:INSTALL
call "%~dp0INSTALL_AND_RUN.bat"
exit /b %ERRORLEVEL%
