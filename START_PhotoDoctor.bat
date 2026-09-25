@echo off
setlocal
cd /d "%~dp0"
call "%~dp0RUN_PhotoDoctor.bat"
exit /b %ERRORLEVEL%
