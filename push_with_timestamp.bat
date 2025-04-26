@echo off
REM Get timestamp in yyyy-MM-dd_HH-mm-ss format
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set timestamp=%%i

git add -A
git commit -m "%timestamp%"
git push origin main
