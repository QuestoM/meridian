@echo off
setlocal
echo Starting Kairos Optimizer...
echo.

cd /d "%~dp0"

echo Starting Kairos API on http://127.0.0.1:8000 ...
start "Kairos API" cmd /k "python -m uvicorn kairos_api.server:app --host 127.0.0.1 --port 8000"

echo Starting Kairos dashboard on http://127.0.0.1:3000 ...
cd /d "%~dp0tv-break-dashboard"
npm run dev -- --port 3000

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Error starting Kairos dashboard.
  echo Check Python and Node dependencies.
  pause
)
