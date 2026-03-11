@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%run_train_all.ps1"

if not exist "%PS_SCRIPT%" (
  echo [FAIL] PowerShell script not found: "%PS_SCRIPT%"
  exit /b 2
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
exit /b %ERRORLEVEL%
