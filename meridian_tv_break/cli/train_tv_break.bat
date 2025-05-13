@echo off
call %~dp0..\..\meridian_env\Scripts\activate.bat
python -m meridian_tv_break.data_transform %*
if %ERRORLEVEL% NEQ 0 goto :error
python -m meridian_tv_break.train_model %*
if %ERRORLEVEL% NEQ 0 goto :error
goto :EOF

:error
echo Error during execution! Exiting...
exit /b %ERRORLEVEL%
