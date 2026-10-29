@echo off
title Modulo installer
if not exist "%~dp0installer\install-windows.ps1" (
  echo Please extract the whole zip first, then run this again.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install-windows.ps1"
echo.
pause
