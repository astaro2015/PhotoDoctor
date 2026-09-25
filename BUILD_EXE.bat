@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_exe.ps1"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo Сборка PhotoDoctor.exe завершилась с ошибкой.
  echo Подробности: "%~dp0build.log"
  echo.
  pause
  exit /b %RC%
)
echo.
echo Готово: "%~dp0dist\PhotoDoctor.exe"
pause
exit /b 0
