@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_and_run.ps1"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo Photo Doctor: установка или запуск завершились с ошибкой.
  echo Подробности: "%~dp0setup.log"
  echo.
  pause
)
exit /b %RC%
