@echo off
echo Starting TV Commercial Break Dashboard...
echo.

:: Navigate to the project directory
cd /d C:\Intel\questo\projects\meridian\tv-break-dashboard

:: Start the React application
echo Starting React development server...
npm start

:: If the server fails to start, keep the window open
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Error starting development server.
  echo Please check that you have all required packages installed.
  pause
)
